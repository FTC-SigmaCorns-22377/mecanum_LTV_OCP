// test_riccati_tracking.cpp -- Validate sparse affine Riccati recursion
//                              against HPIPM OCP Riccati solver.
//
// Tests:
// 1. Scalar reference Riccati vs HPIPM on straight-line trajectory
// 2. Scalar reference Riccati vs HPIPM on turning trajectory
// 3. Consistent reference recovery (Euler-discretized sparse model)
// 4. NEON riccati_tracking vs scalar reference (real NEON on ARM, neon_sim on x86)

#include "mpc_types.h"
#include "heading_lookup.h"
#include "mecanum_model.h"
#include "blas_dispatch.h"
#include "qp_solvers.h"
#include "neon_kernels.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ModelParams make_params()
{
    ModelParams p{};
    p.mass            = 10.0;
    p.inertia         = 0.5;
    p.damping_linear  = 2.0;
    p.damping_angular = 0.3;
    p.wheel_radius    = 0.05;
    p.lx              = 0.15;
    p.ly              = 0.15;
    p.stall_torque    = 6.0;
    p.free_speed      = 435.0;
    compute_mecanum_jacobian(p);
    return p;
}

static MPCConfig make_config(double dt = 0.02, int N = 10)
{
    MPCConfig cfg{};
    cfg.N     = N;
    cfg.dt    = dt;
    cfg.u_min = -1.0;
    cfg.u_max =  1.0;

    std::memset(cfg.Q, 0, sizeof(cfg.Q));
    cfg.Q[0 + NX * 0] = 10.0;
    cfg.Q[1 + NX * 1] = 10.0;
    cfg.Q[2 + NX * 2] =  5.0;
    cfg.Q[3 + NX * 3] =  1.0;
    cfg.Q[4 + NX * 4] =  1.0;
    cfg.Q[5 + NX * 5] =  0.5;

    std::memset(cfg.R, 0, sizeof(cfg.R));
    for (int i = 0; i < NU; ++i)
        cfg.R[i + NU * i] = 0.1;

    for (int i = 0; i < NX * NX; ++i)
        cfg.Qf[i] = 2.0 * cfg.Q[i];

    return cfg;
}

static void build_straight_ref(RefNode* path, int n_path, double dt)
{
    for (int k = 0; k < n_path; ++k) {
        std::memset(&path[k], 0, sizeof(RefNode));
        path[k].t        = k * dt;
        path[k].x_ref[0] = k * dt * 0.5;
        path[k].x_ref[3] = 0.5;
        path[k].theta    = 0.0;
        path[k].omega    = 0.0;
    }
}

static void build_turning_ref(RefNode* path, int n_path, double dt, double omega_ref)
{
    for (int k = 0; k < n_path; ++k) {
        std::memset(&path[k], 0, sizeof(RefNode));
        path[k].t        = k * dt;
        path[k].x_ref[2] = omega_ref * k * dt;
        path[k].x_ref[5] = omega_ref;
        path[k].theta    = path[k].x_ref[2];
        path[k].omega    = omega_ref;
    }
}

// ---------------------------------------------------------------------------
// Scalar reference implementation of affine Riccati recursion (double precision)
// Same math as neon riccati_tracking, but portable and double precision.
// ---------------------------------------------------------------------------
static void riccati_tracking_scalar(
    const double Q_diag[6],
    const double R_diag[4],
    const double A_diag[3],
    const double* B0_row,      // 3x4 row-major body-frame B
    int N,
    const double* theta,
    const double* xr_upper,    // [N+1][3]
    const double* ur,           // [N][4]
    const double* c_upper,     // [N][3]
    const double* x0_upper,
    double u_min, double u_max,
    double* u_star)
{
    double P[6];  // {P00, P01, P02, P11, P12, P22}
    double p[3];

    // Terminal
    P[0] = Q_diag[0]; P[1] = 0; P[2] = 0;
    P[3] = Q_diag[1]; P[4] = 0; P[5] = Q_diag[2];
    const double* xr_N = xr_upper + N * 3;
    p[0] = -Q_diag[0] * xr_N[0];
    p[1] = -Q_diag[1] * xr_N[1];
    p[2] = -Q_diag[2] * xr_N[2];

    double K_store[N_MAX * 12];
    double v_store[N_MAX * 4];
    double B_store[N_MAX * 12];

    for (int k = N - 1; k >= 0; k--) {
        const double* xr_k = xr_upper + k * 3;
        const double* ur_k = ur + k * 4;
        const double* c_k  = c_upper + k * 3;

        double ct = std::cos(theta[k]);
        double st = std::sin(theta[k]);
        double B[12];
        for (int j = 0; j < 4; j++) {
            B[0*4+j] =  ct * B0_row[0*4+j] - st * B0_row[1*4+j];
            B[1*4+j] =  st * B0_row[0*4+j] + ct * B0_row[1*4+j];
            B[2*4+j] = B0_row[2*4+j];
        }
        std::memcpy(B_store + k * 12, B, sizeof(B));

        double Pf[9] = {P[0],P[1],P[2], P[1],P[3],P[4], P[2],P[4],P[5]};

        double PB[12];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) {
                double s = 0;
                for (int m = 0; m < 3; m++) s += Pf[i*3+m] * B[m*4+j];
                PB[i*4+j] = s;
            }

        double L[3];
        for (int i = 0; i < 3; i++) {
            L[i] = p[i];
            for (int m = 0; m < 3; m++) L[i] += Pf[i*3+m] * c_k[m];
        }

        double S[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                double s = 0;
                for (int m = 0; m < 3; m++) s += B[m*4+i] * PB[m*4+j];
                S[i*4+j] = s + (i == j ? R_diag[i] : 0.0);
            }

        double Lch[16] = {};
        for (int i = 0; i < 4; i++) {
            double s = S[i*4+i];
            for (int k2 = 0; k2 < i; k2++) s -= Lch[i*4+k2] * Lch[i*4+k2];
            Lch[i*4+i] = std::sqrt(s);
            double inv = 1.0 / Lch[i*4+i];
            for (int j = i+1; j < 4; j++) {
                s = S[j*4+i];
                for (int k2 = 0; k2 < i; k2++) s -= Lch[j*4+k2] * Lch[i*4+k2];
                Lch[j*4+i] = s * inv;
            }
        }

        double Sinv[16] = {};
        for (int col = 0; col < 4; col++) {
            double y[4] = {};
            for (int i = 0; i < 4; i++) {
                double s = (i == col) ? 1.0 : 0.0;
                for (int j = 0; j < i; j++) s -= Lch[i*4+j] * y[j];
                y[i] = s / Lch[i*4+i];
            }
            for (int i = 3; i >= 0; i--) {
                double s = y[i];
                for (int j = i+1; j < 4; j++) s -= Lch[j*4+i] * Sinv[j*4+col];
                Sinv[i*4+col] = s / Lch[i*4+i];
            }
        }

        double invBt[12];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++) {
                double s = 0;
                for (int m = 0; m < 4; m++) s += Sinv[i*4+m] * B[j*4+m];
                invBt[i*3+j] = s;
            }

        double PA[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                PA[i*3+j] = Pf[i*3+j] * A_diag[j];

        double K[12];
        for (int w = 0; w < 4; w++)
            for (int s = 0; s < 3; s++) {
                double val = 0;
                for (int i = 0; i < 3; i++) val += invBt[w*3+i] * PA[i*3+s];
                K[w*3+s] = val;
            }
        std::memcpy(K_store + k * 12, K, sizeof(K));

        double z[4];
        for (int i = 0; i < 4; i++) {
            z[i] = 0;
            for (int j = 0; j < 3; j++) z[i] += invBt[i*3+j] * L[j];
        }

        double v[4];
        for (int i = 0; i < 4; i++) {
            double sinv_rur = 0;
            for (int j = 0; j < 4; j++) sinv_rur += Sinv[i*4+j] * R_diag[j] * ur_k[j];
            v[i] = z[i] - sinv_rur;
        }
        std::memcpy(v_store + k * 4, v, sizeof(v));

        double ATPB[12];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                ATPB[i*4+j] = A_diag[i] * PB[i*4+j];

        double d[3];
        for (int i = 0; i < 3; i++) {
            d[i] = 0;
            for (int m = 0; m < 4; m++) d[i] += ATPB[i*4+m] * z[m];
        }

        double sc[6];
        for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++) {
                double s = 0;
                for (int m = 0; m < 4; m++) s += ATPB[i*4+m] * K[m*3+j];
                int idx = (i == 0) ? j : (i == 1) ? 3 + (j-1) : 5;
                sc[idx] = s;
            }

        double ATPA[6];
        for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++) {
                int idx = (i == 0) ? j : (i == 1) ? 3 + (j-1) : 5;
                ATPA[idx] = PA[i*3+j] * A_diag[i];
            }

        double Qsum[3] = { Q_diag[0]+Q_diag[3], Q_diag[1]+Q_diag[4], Q_diag[2]+Q_diag[5] };
        P[0] = ATPA[0] + Qsum[0] - sc[0];
        P[1] = ATPA[1] - sc[1];
        P[2] = ATPA[2] - sc[2];
        P[3] = ATPA[3] + Qsum[1] - sc[3];
        P[4] = ATPA[4] - sc[4];
        P[5] = ATPA[5] + Qsum[2] - sc[5];

        for (int i = 0; i < 3; i++)
            p[i] = -Qsum[i] * xr_k[i] + A_diag[i] * L[i] - d[i];
    }

    // Forward pass
    double x[3] = { x0_upper[0], x0_upper[1], x0_upper[2] };
    for (int k = 0; k < N; k++) {
        const double* K = K_store + k * 12;
        const double* v = v_store + k * 4;
        const double* B = B_store + k * 12;
        const double* c_k = c_upper + k * 3;

        for (int w = 0; w < 4; w++) {
            double u = -v[w];
            for (int s = 0; s < 3; s++) u -= K[w*3+s] * x[s];
            u_star[k*4+w] = std::max(u_min, std::min(u_max, u));
        }

        double Bu[3] = {};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                Bu[i] += B[i*4+j] * u_star[k*4+j];
        for (int i = 0; i < 3; i++)
            x[i] = A_diag[i] * x[i] + Bu[i] + c_k[i];
    }
}

// ---------------------------------------------------------------------------
// Extract sparse Riccati inputs from heading-lookup data
// ---------------------------------------------------------------------------
struct RiccatiInputs {
    double A_diag[3];
    double B0_row[12];
    double Q_diag[6];
    double R_diag[4];
    double theta[N_MAX];
    double xr_upper[(N_MAX+1) * 3];
    double ur[N_MAX * 4];
    double c_upper[N_MAX * 3];
};

static void extract_riccati_inputs(
    const HeadingLookupData& data,
    const RefNode* ref_window,
    const double x0[NX],
    const MPCConfig& config,
    const HeadingScheduleConfig& sched_config,
    RiccatiInputs& out)
{
    const int N = config.N;

    out.A_diag[0] = data.A_d[0 + NX * 0];
    out.A_diag[1] = data.A_d[1 + NX * 1];
    out.A_diag[2] = data.A_d[2 + NX * 2];

    for (int i = 0; i < NX; i++) out.Q_diag[i] = config.Q[i + NX * i];
    for (int i = 0; i < NU; i++) out.R_diag[i] = config.R[i + NU * i];

    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref_window, N, config.dt, sched_config, theta_sched);
    std::memcpy(out.theta, theta_sched, N * sizeof(double));

    double B_list[N_MAX * NX * NU];
    heading_lookup_build_B_list(data, theta_sched, N, B_list);

    // B0_row: body-frame B at theta=0 (upper 3 rows, col-major→row-major)
    double B_at_zero[NX * NU];
    for (int i = 0; i < NX * NU; i++)
        B_at_zero[i] = data.B_d0[i] + data.B_dc[i];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            out.B0_row[i * 4 + j] = B_at_zero[i + NX * j];

    // Build dynamically-consistent reference with full 6x6 model
    double x_ref[(N_MAX + 1) * NX];
    std::memcpy(x_ref, ref_window[0].x_ref, NX * sizeof(double));
    double temp_Ax[NX], temp_Bu[NX];
    for (int k = 0; k < N; k++) {
        mpc_linalg::gemv(NX, NX, data.A_d, x_ref + k * NX, temp_Ax);
        mpc_linalg::gemv(NX, NU, B_list + k * NX * NU, ref_window[k].u_ref, temp_Bu);
        for (int i = 0; i < NX; i++)
            x_ref[(k+1) * NX + i] = temp_Ax[i] + temp_Bu[i];
    }

    for (int k = 0; k <= N; k++)
        for (int i = 0; i < 3; i++)
            out.xr_upper[k * 3 + i] = x_ref[k * NX + i];

    for (int k = 0; k < N; k++)
        std::memcpy(out.ur + k * 4, ref_window[k].u_ref, NU * sizeof(double));

    // c_upper for the SPARSE model:
    // c_k = xr_{k+1} - A_diag * xr_k - B_u * ur_k
    for (int k = 0; k < N; k++) {
        const double* B_k = B_list + k * NX * NU;
        for (int i = 0; i < 3; i++) {
            double sparse_Ax = out.A_diag[i] * x_ref[k * NX + i];
            double Bu = 0;
            for (int j = 0; j < NU; j++) Bu += B_k[i + NX * j] * ref_window[k].u_ref[j];
            out.c_upper[k * 3 + i] = x_ref[(k+1) * NX + i] - sparse_Ax - Bu;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: Scalar Riccati vs HPIPM (straight line)
// ---------------------------------------------------------------------------
static bool test_straight_line()
{
#ifdef MPC_USE_HPIPM
    std::printf("Test 1: Scalar Riccati vs HPIPM (straight line) ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    HeadingLookupData data;
    heading_lookup_precompute(params, config.dt, data);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    SolverContext ctx;
    solver_context_init(ctx, config.N * NU);
    QPSolution sol_hpipm = heading_lookup_solve_ocp(data, path, x0, config, sched, ctx);
    solver_context_free(ctx);

    RiccatiInputs ri;
    extract_riccati_inputs(data, path, x0, config, sched, ri);

    double u_riccati[N_MAX * 4];
    double x0_upper[3] = { x0[0], x0[1], x0[2] };
    riccati_tracking_scalar(ri.Q_diag, ri.R_diag, ri.A_diag, ri.B0_row,
        config.N, ri.theta, ri.xr_upper, ri.ur, ri.c_upper,
        x0_upper, config.u_min, config.u_max, u_riccati);

    double u0_diff = 0.0, u_max_diff = 0.0;
    for (int j = 0; j < NU; j++) {
        double d = std::fabs(u_riccati[j] - sol_hpipm.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }
    for (int k = 0; k < config.N * NU; k++) {
        double d = std::fabs(u_riccati[k] - sol_hpipm.U[k]);
        if (d > u_max_diff) u_max_diff = d;
    }

    std::printf("u0_diff=%.3e  full_diff=%.3e", u0_diff, u_max_diff);
    std::printf("  riccati=[%.4f,%.4f,%.4f,%.4f] hpipm=[%.4f,%.4f,%.4f,%.4f]",
                u_riccati[0], u_riccati[1], u_riccati[2], u_riccati[3],
                sol_hpipm.u0[0], sol_hpipm.u0[1], sol_hpipm.u0[2], sol_hpipm.u0[3]);

    bool ok = u0_diff < 0.15;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
#else
    std::printf("Test 1: Scalar Riccati vs HPIPM (straight line) ... SKIPPED (no HPIPM)\n");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Test 2: Scalar Riccati vs HPIPM (turning)
// ---------------------------------------------------------------------------
static bool test_turning()
{
#ifdef MPC_USE_HPIPM
    std::printf("Test 2: Scalar Riccati vs HPIPM (turning) ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 50;
    RefNode path[50];
    build_turning_ref(path, n_path, config.dt, 1.0);

    HeadingLookupData data;
    heading_lookup_precompute(params, config.dt, data);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[2] += 0.05;

    SolverContext ctx;
    solver_context_init(ctx, config.N * NU);
    QPSolution sol_hpipm = heading_lookup_solve_ocp(data, path, x0, config, sched, ctx);
    solver_context_free(ctx);

    RiccatiInputs ri;
    extract_riccati_inputs(data, path, x0, config, sched, ri);

    double u_riccati[N_MAX * 4];
    double x0_upper[3] = { x0[0], x0[1], x0[2] };
    riccati_tracking_scalar(ri.Q_diag, ri.R_diag, ri.A_diag, ri.B0_row,
        config.N, ri.theta, ri.xr_upper, ri.ur, ri.c_upper,
        x0_upper, config.u_min, config.u_max, u_riccati);

    double u0_diff = 0.0;
    for (int j = 0; j < NU; j++) {
        double d = std::fabs(u_riccati[j] - sol_hpipm.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }

    std::printf("u0_diff=%.3e", u0_diff);
    std::printf("  riccati=[%.4f,%.4f,%.4f,%.4f] hpipm=[%.4f,%.4f,%.4f,%.4f]",
                u_riccati[0], u_riccati[1], u_riccati[2], u_riccati[3],
                sol_hpipm.u0[0], sol_hpipm.u0[1], sol_hpipm.u0[2], sol_hpipm.u0[3]);

    bool ok = u0_diff < 0.25;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
#else
    std::printf("Test 2: Scalar Riccati vs HPIPM (turning) ... SKIPPED (no HPIPM)\n");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Test 3: Consistent reference recovery (Euler-discretized sparse model)
// Uses continuous A_c (exactly sparse) + Euler discretization so A_diag is exact.
// ---------------------------------------------------------------------------
static bool test_consistent_reference()
{
    std::printf("Test 3: Consistent reference recovery (Euler) ... ");

    ModelParams params = make_params();
    double dt = 0.02;
    int N = 10;

    double Ac[NX * NX] = {};
    double Bc[NX * NU] = {};
    continuous_dynamics(0.0, params, Ac, Bc);

    // Euler: A_d = I + dt*Ac, B_d = dt*Bc
    double A_d[NX * NX] = {};
    double B_d[NX * NU] = {};
    for (int i = 0; i < NX; i++) A_d[i + NX * i] = 1.0;
    for (int i = 0; i < NX * NX; i++) A_d[i] += dt * Ac[i];
    for (int i = 0; i < NX * NU; i++) B_d[i] = dt * Bc[i];

    // Sparse A diagonal (Ac upper-left is zero → {1,1,1})
    double A_diag[3] = { A_d[0], A_d[1 + NX], A_d[2 + 2*NX] };

    // B0_row: upper 3 rows of B_d, col-major→row-major
    double B0_row[12];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < NU; j++)
            B0_row[i * 4 + j] = B_d[i + NX * j];

    // Build consistent ref with full Euler model
    double x_ref[(N_MAX + 1) * NX] = {};
    double u_ref[N_MAX * NU] = {};
    for (int k = 0; k < N; k++) {
        u_ref[k*4+0] = 0.1; u_ref[k*4+1] = 0.05;
        u_ref[k*4+2] = 0.05; u_ref[k*4+3] = 0.1;
    }
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < NX; i++) {
            double ax = 0, bu = 0;
            for (int j = 0; j < NX; j++) ax += A_d[i + NX * j] * x_ref[k*NX + j];
            for (int j = 0; j < NU; j++) bu += B_d[i + NX * j] * u_ref[k*NU + j];
            x_ref[(k+1)*NX + i] = ax + bu;
        }
    }

    double xr_upper[(N_MAX + 1) * 3];
    for (int k = 0; k <= N; k++)
        for (int i = 0; i < 3; i++)
            xr_upper[k * 3 + i] = x_ref[k * NX + i];

    // c_upper for sparse model
    double c_upper[N_MAX * 3];
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < 3; i++) {
            double sparse_Ax = A_diag[i] * x_ref[k * NX + i];
            double Bu = 0;
            for (int j = 0; j < NU; j++) Bu += B_d[i + NX * j] * u_ref[k * NU + j];
            c_upper[k * 3 + i] = x_ref[(k+1) * NX + i] - sparse_Ax - Bu;
        }
    }

    double theta[N_MAX] = {};
    double Q_diag[6] = { 10, 10, 5, 1, 1, 0.5 };
    double R_diag[4] = { 0.1, 0.1, 0.1, 0.1 };

    double u_riccati[N_MAX * 4];
    double x0_upper[3] = { x_ref[0], x_ref[1], x_ref[2] };
    riccati_tracking_scalar(Q_diag, R_diag, A_diag, B0_row,
        N, theta, xr_upper, u_ref, c_upper,
        x0_upper, -1.0, 1.0, u_riccati);

    double max_diff = 0.0;
    for (int k = 0; k < N * NU; k++) {
        double d = std::fabs(u_riccati[k] - u_ref[k]);
        if (d > max_diff) max_diff = d;
    }

    std::printf("max_diff=%.3e", max_diff);
    std::printf("  u0=[%.4f,%.4f,%.4f,%.4f] uref=[%.4f,%.4f,%.4f,%.4f]",
                u_riccati[0], u_riccati[1], u_riccati[2], u_riccati[3],
                u_ref[0], u_ref[1], u_ref[2], u_ref[3]);

    bool ok = max_diff < 0.01;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4: NEON riccati_tracking vs scalar reference
// On ARM: real NEON intrinsics. On x86: neon_sim.h scalar simulation.
// ---------------------------------------------------------------------------
static bool test_neon_vs_scalar()
{
    std::printf("Test 4: NEON riccati_tracking vs scalar ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    HeadingLookupData data;
    heading_lookup_precompute(params, config.dt, data);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    RiccatiInputs ri;
    extract_riccati_inputs(data, path, x0, config, sched, ri);

    // Scalar reference (double)
    double u_scalar[N_MAX * 4];
    double x0d[3] = { x0[0], x0[1], x0[2] };
    riccati_tracking_scalar(ri.Q_diag, ri.R_diag, ri.A_diag, ri.B0_row,
        config.N, ri.theta, ri.xr_upper, ri.ur, ri.c_upper,
        x0d, config.u_min, config.u_max, u_scalar);

    // NEON (real or simulated via neon_sim.h)
    std::array<float, 6> Q_f;
    std::array<float, 4> R_f;
    std::array<float, 3> A_f;
    std::array<float, 12> B0_f;
    for (int i = 0; i < 6; i++) Q_f[i] = (float)ri.Q_diag[i];
    for (int i = 0; i < 4; i++) R_f[i] = (float)ri.R_diag[i];
    for (int i = 0; i < 3; i++) A_f[i] = (float)ri.A_diag[i];
    for (int i = 0; i < 12; i++) B0_f[i] = (float)ri.B0_row[i];

    float theta_f[N_MAX], xr_f[(N_MAX+1)*3], ur_f[N_MAX*4], c_f[N_MAX*3];
    float x0_f[3], u_neon[N_MAX * 4];
    for (int i = 0; i < config.N; i++) theta_f[i] = (float)ri.theta[i];
    for (int i = 0; i < (config.N+1)*3; i++) xr_f[i] = (float)ri.xr_upper[i];
    for (int i = 0; i < config.N*4; i++) ur_f[i] = (float)ri.ur[i];
    for (int i = 0; i < config.N*3; i++) c_f[i] = (float)ri.c_upper[i];
    for (int i = 0; i < 3; i++) x0_f[i] = (float)x0d[i];

    float workspace[N_MAX * 28];
    riccati_tracking(workspace, Q_f, R_f, A_f, B0_f,
                     config.N, theta_f, xr_f, ur_f, c_f, x0_f, u_neon);

    double max_diff = 0.0;
    for (int k = 0; k < config.N * 4; k++) {
        double d = std::fabs((double)u_neon[k] - u_scalar[k]);
        if (d > max_diff) max_diff = d;
    }

    std::printf("max_diff=%.3e", max_diff);
    std::printf("  neon=[%.4f,%.4f,%.4f,%.4f] scalar=[%.4f,%.4f,%.4f,%.4f]",
                u_neon[0], u_neon[1], u_neon[2], u_neon[3],
                u_scalar[0], u_scalar[1], u_scalar[2], u_scalar[3]);

    bool ok = max_diff < 0.01;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
int main()
{
    int n_pass = 0, n_fail = 0;
    auto run = [&](bool result) { result ? n_pass++ : n_fail++; };

    run(test_straight_line());
    run(test_turning());
    run(test_consistent_reference());
    run(test_neon_vs_scalar());

    std::printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
