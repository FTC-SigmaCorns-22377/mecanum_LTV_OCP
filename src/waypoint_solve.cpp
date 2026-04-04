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
    IpmWorkspace& ws)
{
    const int N = config.N;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // 1. Generate heading schedule
    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref_window, N, config.dt, sched_config, theta_sched);

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

    for (int outer = 0; outer < ipm_config.max_outer_iters && mu >= ipm_config.mu_min; ++outer) {
        // Compute barrier terms and form R_eff, ur_eff
        for (int i = 0; i < N * NU; ++i) {
            float u = u_bar[i];
            float slack_lo = u + 1.0f;
            float slack_hi = 1.0f - u;
            float W = mu / (slack_lo * slack_lo) + mu / (slack_hi * slack_hi);
            float g = -mu / slack_lo + mu / slack_hi;
            int j = i % NU;
            float R_j   = (float)R_diag[j];
            float R_eff = R_j + W;
            ws.R_eff[i]  = R_eff;
            ws.ur_eff[i] = (R_j * (float)u_ref_stacked[i] + W * u - g) / R_eff;
        }

        // Convert to float for Riccati kernel
        float theta_f[N_MAX];
        for (int i = 0; i < N; ++i) theta_f[i] = (float)theta_sched[i];

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
