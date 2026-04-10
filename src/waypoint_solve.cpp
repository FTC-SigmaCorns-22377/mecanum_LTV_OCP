// waypoint_solve.cpp
// Terminal-targeting IPM solve for waypoint problems.
// See waypoint_solve.h for rationale.

#include "waypoint_solve.h"
#include "ipm_solver.h"
#include "heading_lookup.h"
#include "mpc_types.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <time.h>

#ifdef MPC_USE_NEON
#ifdef __aarch64__
#include <arm_neon.h>
#else
#include "neon_sim.h"
#endif
#endif

QPSolution ipm_solve_terminal(
    const EulerDynamicsData& euler,
    const HeadingLookupData& hld,
    const RefNode* ref_window,
    const double x0[NX],
    const double x_f[NX],
    const MPCConfig& config,
    const HeadingScheduleConfig& sched_config,
    const IpmSolverConfig& ipm_config,
    IpmWorkspace& ws,
    const double* theta_sched_override)
{
    const int N = config.N;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // 1. Generate heading schedule
    double theta_sched[N_MAX + 1];
    if (theta_sched_override != nullptr) {
        std::memcpy(theta_sched, theta_sched_override, (N + 1) * sizeof(double));
    } else {
        generate_heading_schedule(x0, ref_window, N, config.dt, sched_config, theta_sched);
    }

    // 2. Build consistent 6-state reference using Euler dynamics
    //    x_ref[k+1] = A_euler · x_ref[k] + B_d(θ_k) · u_ref[k]
    double x_ref[(N_MAX + 1) * NX];
    double u_ref_stacked[N_MAX * NU];
    std::memcpy(x_ref, ref_window[0].x_ref, NX * sizeof(double));

    for (int k = 0; k < N; ++k) {
        std::memcpy(u_ref_stacked + k * NU, ref_window[k].u_ref, NU * sizeof(double));

        const double* xk = x_ref + k * NX;
        double* xn = x_ref + (k + 1) * NX;

        // Rotate B_body by heading
        double ct = std::cos(theta_sched[k]);
        double st = std::sin(theta_sched[k]);
        double Bl[12];
        for (int j = 0; j < 4; ++j) {
            Bl[0 * 4 + j] =  ct * euler.B_body[0 * 4 + j] - st * euler.B_body[1 * 4 + j];
            Bl[1 * 4 + j] =  st * euler.B_body[0 * 4 + j] + ct * euler.B_body[1 * 4 + j];
            Bl[2 * 4 + j] = euler.B_body[2 * 4 + j];
        }

        // Bu = Bl · u_ref[k]
        double Bu[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                Bu[i] += Bl[i * 4 + j] * ref_window[k].u_ref[j];

        // Euler propagation: p_{k+1} = p_k + dt·v_k, v_{k+1} = D·v_k + Bl·u_k
        for (int i = 0; i < 3; ++i) {
            xn[i]     = xk[i] + config.dt * xk[3 + i];
            xn[3 + i] = euler.D_diag[i] * xk[3 + i] + Bu[i];
        }
    }

    // Terminal-constraint patch: override the endpoint of the consistent Euler
    // reference with the explicit target x_f. This makes the Riccati terminal
    // costate initialise as p_N = -Qf·x_f instead of -Qf·xr_euler[N], which
    // correctly targets x_f even when x_f has nonzero velocity (or when the
    // Euler sim of a constant LQR reference drifts away from x_f).
    std::memcpy(x_ref + N * NX, x_f, NX * sizeof(double));

    // 3. Extract Q/Qf diagonals
    double Q_diag[6], Qf_diag[6], R_diag[4];
    for (int i = 0; i < NX; ++i) Q_diag[i]  = config.Q[i + NX * i];
    for (int i = 0; i < NX; ++i) Qf_diag[i] = config.Qf[i + NX * i];
    for (int i = 0; i < NU;  ++i) R_diag[i]  = config.R[i + NU * i];

    double D_diag_d[3] = { euler.D_diag[0], euler.D_diag[1], euler.D_diag[2] };
    double B_body_d[12];
    for (int i = 0; i < 12; ++i) B_body_d[i] = euler.B_body[i];

    // 4. Initialize u_bar (warm-start or u_ref clamp)
    float* u_bar = ws.u_bar;
    if (ws.warm_valid && ws.prev_N == N) {
        // Shift previous solution by 1 step
        std::memcpy(u_bar, ws.u_prev + NU, (N - 1) * NU * sizeof(float));
        std::memcpy(u_bar + (N - 1) * NU, ws.u_prev + (N - 1) * NU, NU * sizeof(float));
    } else {
        for (int i = 0; i < N * NU; ++i)
            u_bar[i] = (float)std::clamp(u_ref_stacked[i],
                                         (double)config.u_min,
                                         (double)config.u_max);
    }

    // Clip to strict interior
    float margin = ipm_config.interior_margin;
    float lo = -1.0f + margin, hi = 1.0f - margin;
    for (int i = 0; i < N * NU; ++i)
        u_bar[i] = std::clamp(u_bar[i], lo, hi);

    // 5. IPM barrier iterations
    int total_iters = 0;
    float mu = ipm_config.mu_init;

    // Convert theta schedule to float once — doesn't change across outer iterations.
    float theta_f[N_MAX];
    for (int i = 0; i < N; ++i) theta_f[i] = (float)theta_sched[i];

    // Precompute tip thresholds in velocity-change-per-step units.
    // B_body already carries the dt factor, so dv_tip = a_tip * dt is the
    // max |Δv_body_i| = |B_body[i,:]·u + (D_i-1)·vb_i| allowed per step.
    const bool use_tip = (config.a_tip_x > 0.0 || config.a_tip_y > 0.0);
    const float dv_tip_x = (float)(config.a_tip_x * config.dt);
    const float dv_tip_y = (float)(config.a_tip_y * config.dt);

    // IIR low-pass filter parameters for sustained-acceleration constraint.
    // tau > 0: f_k = gamma*f_{k-1} + (1-gamma)*|a_body_k|, barrier on f_k <= dv_tip.
    // tau = 0: per-step instantaneous two-sided barrier (original behaviour).
    const bool use_iir = use_tip && (config.a_tip_tau > 1e-9);
    const float tip_gamma = use_iir ? expf(-(float)(config.dt / config.a_tip_tau)) : 0.0f;
    const float tip_alpha = 1.0f - tip_gamma;

    for (int outer = 0; outer < ipm_config.max_outer_iters && mu >= ipm_config.mu_min; ++outer) {
        // --- Tipping constraint setup ---
        // Forward-simulate body-frame velocities using current u_bar, and
        // precompute B_body·u_bar dot products (body-frame, theta=0) per step.
        // For IIR mode, also accumulate the low-pass filtered |a_body| per step.
        // Re-evaluated each outer iteration as u_bar converges.
        float vb_x[N_MAX] = {}, vb_y[N_MAX] = {};
        float dot_bx[N_MAX] = {}, dot_by[N_MAX] = {};
        float filtered_ax[N_MAX] = {}, filtered_ay[N_MAX] = {};
        if (use_tip) {
            float vfx = (float)x0[3], vfy = (float)x0[4];
            float f_x = 0.0f, f_y = 0.0f;
            for (int k = 0; k < N; ++k) {
                // Rotate field-frame velocity → body frame
                float ct = cosf(theta_f[k]), st = sinf(theta_f[k]);
                vb_x[k] =  ct * vfx + st * vfy;
                vb_y[k] = -st * vfx + ct * vfy;

                // Body-frame B·u at step k (B_body is at theta=0, body frame)
                float bx = 0, by = 0;
                for (int jj = 0; jj < NU; ++jj) {
                    bx += euler.B_body[0 * 4 + jj] * u_bar[k * NU + jj];
                    by += euler.B_body[1 * 4 + jj] * u_bar[k * NU + jj];
                }
                dot_bx[k] = bx;
                dot_by[k] = by;

                if (use_iir) {
                    float damp_x_k = (euler.D_diag[0] - 1.0f) * vb_x[k];
                    float damp_y_k = (euler.D_diag[1] - 1.0f) * vb_y[k];
                    f_x = tip_gamma * f_x + tip_alpha * std::abs(bx + damp_x_k);
                    f_y = tip_gamma * f_y + tip_alpha * std::abs(by + damp_y_k);
                    filtered_ax[k] = f_x;
                    filtered_ay[k] = f_y;
                }

                // Propagate field-frame velocity for next step
                float Bu_fx = 0, Bu_fy = 0;
                for (int jj = 0; jj < NU; ++jj) {
                    float Bl0 =  ct * euler.B_body[0*4+jj] - st * euler.B_body[1*4+jj];
                    float Bl1 =  st * euler.B_body[0*4+jj] + ct * euler.B_body[1*4+jj];
                    Bu_fx += Bl0 * u_bar[k * NU + jj];
                    Bu_fy += Bl1 * u_bar[k * NU + jj];
                }
                vfx = euler.D_diag[0] * vfx + Bu_fx;
                vfy = euler.D_diag[1] * vfy + Bu_fy;
            }
        }

        // Compute barrier terms and form R_eff, ur_eff
        for (int i = 0; i < N * NU; ++i) {
            int k = i / NU;
            int j = i % NU;
            float u = u_bar[i];
            float slack_lo = u + 1.0f;
            float slack_hi = 1.0f - u;
            float W = mu / (slack_lo * slack_lo) + mu / (slack_hi * slack_hi);
            float g = -mu / slack_lo + mu / slack_hi;

            // Tipping constraint barriers.
            if (use_tip) {
                float damp_x = (euler.D_diag[0] - 1.0f) * vb_x[k];
                float damp_y = (euler.D_diag[1] - 1.0f) * vb_y[k];
                if (dv_tip_x > 0.0f) {
                    if (use_iir) {
                        // One-sided barrier on IIR-filtered |a_body_x|.
                        // ∂f_k/∂u[k,j] ≈ tip_alpha·B[0,j]·sign(dv_x_k)  (diagonal approx)
                        float fax  = filtered_ax[k];
                        float slack = std::max(dv_tip_x - fax, 1e-6f);
                        float dv_x  = dot_bx[k] + damp_x;
                        float sgn   = (dv_x >= 0.0f) ? 1.0f : -1.0f;
                        float Bj_eff = tip_alpha * euler.B_body[0 * 4 + j] * sgn;
                        W += mu * Bj_eff * Bj_eff / (slack * slack);
                        g += mu * Bj_eff / slack;
                    } else {
                        // Original per-step two-sided barrier.
                        float slo = std::max(dot_bx[k] + damp_x + dv_tip_x, 1e-6f);
                        float shi = std::max(dv_tip_x - dot_bx[k] - damp_x, 1e-6f);
                        float Bj  = euler.B_body[0 * 4 + j];
                        W += mu * Bj * Bj * (1.0f / (slo * slo) + 1.0f / (shi * shi));
                        g += mu * Bj * (-1.0f / slo + 1.0f / shi);
                    }
                }
                if (dv_tip_y > 0.0f) {
                    if (use_iir) {
                        float fay  = filtered_ay[k];
                        float slack = std::max(dv_tip_y - fay, 1e-6f);
                        float dv_y  = dot_by[k] + damp_y;
                        float sgn   = (dv_y >= 0.0f) ? 1.0f : -1.0f;
                        float Bj_eff = tip_alpha * euler.B_body[1 * 4 + j] * sgn;
                        W += mu * Bj_eff * Bj_eff / (slack * slack);
                        g += mu * Bj_eff / slack;
                    } else {
                        float slo = std::max(dot_by[k] + damp_y + dv_tip_y, 1e-6f);
                        float shi = std::max(dv_tip_y - dot_by[k] - damp_y, 1e-6f);
                        float Bj  = euler.B_body[1 * 4 + j];
                        W += mu * Bj * Bj * (1.0f / (slo * slo) + 1.0f / (shi * shi));
                        g += mu * Bj * (-1.0f / slo + 1.0f / shi);
                    }
                }
            }

            float R_j   = (float)R_diag[j];
            float R_eff = R_j + W;
            ws.R_eff[i]  = R_eff;
            ws.ur_eff[i] = (R_j * (float)u_ref_stacked[i] + W * u - g) / R_eff;
        }

        // x_ref already has x_f patched into slot N — copy all N+1 nodes
        float xr_f[(N_MAX + 1) * 6];
        for (int i = 0; i < (N + 1) * NX; ++i) xr_f[i] = (float)x_ref[i];

        float x0_f[6];
        for (int i = 0; i < NX; ++i) x0_f[i] = (float)x0[i];

        float u_new[N_MAX * NU];

#ifdef MPC_USE_NEON
        std::array<float, 6>  Q_f, Qf_f;
        std::array<float, 3>  D_f;
        std::array<float, 12> B_f;
        for (int i = 0; i < 6;  ++i) { Q_f[i] = (float)Q_diag[i]; Qf_f[i] = (float)Qf_diag[i]; }
        for (int i = 0; i < 3;  ++i) D_f[i] = euler.D_diag[i];
        for (int i = 0; i < 12; ++i) B_f[i]  = euler.B_body[i];

        riccati_6state_neon(ws.stage_data, Q_f, Qf_f, ws.R_eff, D_f,
                            euler.dt, B_f, N, theta_f, xr_f, ws.ur_eff,
                            x0_f, u_new);
#else
        double R_eff_d[N_MAX * NU], ur_eff_d[N_MAX * NU];
        for (int i = 0; i < N * NU; ++i) {
            R_eff_d[i]  = ws.R_eff[i];
            ur_eff_d[i] = ws.ur_eff[i];
        }
        double u_new_d[N_MAX * NU];
        // The scalar Riccati reads xr[N*NX] for terminal costate — we patched
        // x_ref[N*NX] = x_f above, so the terminal cost correctly targets x_f.
        riccati_6state_scalar(Q_diag, Qf_diag, R_eff_d, D_diag_d,
                              config.dt, B_body_d, N,
                              theta_sched, x_ref, ur_eff_d, x0, u_new_d);
        for (int i = 0; i < N * NU; ++i) u_new[i] = (float)u_new_d[i];
#endif

        // Clip to strict interior and update u_bar
        for (int i = 0; i < N * NU; ++i)
            u_bar[i] = std::clamp(u_new[i], lo, hi);

        total_iters++;
        mu *= ipm_config.mu_factor;
    }

    // 6. Final clip to exact bounds
    for (int i = 0; i < N * NU; ++i)
        u_bar[i] = std::clamp(u_bar[i], (float)config.u_min, (float)config.u_max);

    // 7. Store for warm-start
    std::memcpy(ws.u_prev, u_bar, N * NU * sizeof(float));
    ws.warm_valid = true;
    ws.prev_N = N;

    // 8. Build QPSolution (float32 → double)
    QPSolution sol;
    std::memset(&sol, 0, sizeof(sol));
    for (int i = 0; i < N * NU; ++i)
        sol.U[i] = (double)u_bar[i];
    std::memcpy(sol.u0, sol.U, NU * sizeof(double));
    sol.n_iterations = total_iters;

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    sol.solve_time_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);
    return sol;
}
