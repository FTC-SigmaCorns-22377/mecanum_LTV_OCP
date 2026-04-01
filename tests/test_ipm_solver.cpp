// test_ipm_solver.cpp -- Validate 6-state block-sparse Riccati and IPM solver.
//
// Tests:
// 1. Scalar 6-state Riccati: consistent reference recovery
// 2. NEON 6-state Riccati vs scalar reference
// 3. IPM convergence: all outputs in [-1, 1]
// 4. IPM vs HPIPM comparison (if HPIPM available)
// 5. Warm-start verification
// 6. Timing benchmark

#include "mpc_types.h"
#include "ipm_solver.h"
#include "heading_lookup.h"
#include "mecanum_model.h"
#include "qp_solvers.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <time.h>

// ---------------------------------------------------------------------------
// Helpers (same as test_riccati_tracking.cpp)
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

// ---------------------------------------------------------------------------
// Test 1: Scalar 6-state Riccati — consistent reference recovery
// When started on the reference with correct feedforward, the Riccati
// should recover u_ref as the optimal input.
// ---------------------------------------------------------------------------
static bool test_consistent_reference()
{
    std::printf("Test 1: Scalar 6-state Riccati consistent reference ... ");

    ModelParams params = make_params();
    double dt = 0.02;
    int N = 10;

    EulerDynamicsData euler;
    euler_dynamics_precompute(params, dt, euler);

    // Build a reference trajectory using Euler dynamics
    double x_ref[(N_MAX + 1) * NX] = {};
    double u_ref[N_MAX * NU] = {};
    for (int k = 0; k < N; ++k) {
        u_ref[k * 4 + 0] = 0.1;
        u_ref[k * 4 + 1] = 0.05;
        u_ref[k * 4 + 2] = 0.05;
        u_ref[k * 4 + 3] = 0.1;
    }

    // Forward simulate with Euler dynamics at theta=0
    double theta[N_MAX] = {};
    for (int k = 0; k < N; ++k) {
        double* xk = x_ref + k * NX;
        double* xn = x_ref + (k + 1) * NX;
        double Bu[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                Bu[i] += euler.B_body[i * 4 + j] * u_ref[k * 4 + j];
        for (int i = 0; i < 3; ++i) {
            xn[i]     = xk[i] + dt * xk[3 + i];
            xn[3 + i] = euler.D_diag[i] * xk[3 + i] + Bu[i];
        }
    }

    double Q_diag[6] = { 10, 10, 5, 1, 1, 0.5 };
    double Qf_diag[6] = { 20, 20, 10, 2, 2, 1 };
    double R_diag[N_MAX * 4];
    for (int i = 0; i < N * 4; ++i) R_diag[i] = 0.1;

    double D_diag_d[3] = { euler.D_diag[0], euler.D_diag[1], euler.D_diag[2] };
    double B_body_d[12];
    for (int i = 0; i < 12; ++i) B_body_d[i] = euler.B_body[i];

    double u_out[N_MAX * 4];
    riccati_6state_scalar(Q_diag, Qf_diag, R_diag, D_diag_d, dt,
                          B_body_d, N, theta, x_ref, u_ref,
                          x_ref, u_out);

    double max_diff = 0.0;
    for (int k = 0; k < N * NU; ++k) {
        double d = std::fabs(u_out[k] - u_ref[k]);
        if (d > max_diff) max_diff = d;
    }

    std::printf("max_diff=%.3e", max_diff);
    std::printf("  u0=[%.4f,%.4f,%.4f,%.4f] uref=[%.4f,%.4f,%.4f,%.4f]",
                u_out[0], u_out[1], u_out[2], u_out[3],
                u_ref[0], u_ref[1], u_ref[2], u_ref[3]);

    bool ok = max_diff < 0.01;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: NEON 6-state Riccati vs scalar reference
// ---------------------------------------------------------------------------
static bool test_neon_vs_scalar()
{
#ifdef MPC_USE_NEON
    std::printf("Test 2: NEON 6-state Riccati vs scalar ... ");

    ModelParams params = make_params();
    double dt = 0.02;
    int N = 10;

    EulerDynamicsData euler;
    euler_dynamics_precompute(params, dt, euler);

    // Build reference
    double x_ref_d[(N_MAX + 1) * NX] = {};
    double u_ref_d[N_MAX * NU] = {};
    for (int k = 0; k < N; ++k) {
        u_ref_d[k * 4 + 0] = 0.1;  u_ref_d[k * 4 + 1] = 0.05;
        u_ref_d[k * 4 + 2] = 0.05; u_ref_d[k * 4 + 3] = 0.1;
    }
    double theta_d[N_MAX];
    for (int k = 0; k < N; ++k) theta_d[k] = 0.1 * k;  // varying heading

    // Forward simulate
    for (int k = 0; k < N; ++k) {
        double* xk = x_ref_d + k * NX;
        double* xn = x_ref_d + (k + 1) * NX;
        double ct = std::cos(theta_d[k]), st = std::sin(theta_d[k]);
        double Bl[12];
        for (int j = 0; j < 4; ++j) {
            Bl[0*4+j] =  ct*euler.B_body[0*4+j] - st*euler.B_body[1*4+j];
            Bl[1*4+j] =  st*euler.B_body[0*4+j] + ct*euler.B_body[1*4+j];
            Bl[2*4+j] = euler.B_body[2*4+j];
        }
        double Bu[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) Bu[i] += Bl[i*4+j] * u_ref_d[k*4+j];
        for (int i = 0; i < 3; ++i) {
            xn[i]   = xk[i] + dt * xk[3+i];
            xn[3+i] = euler.D_diag[i] * xk[3+i] + Bu[i];
        }
    }

    // Add perturbation to initial state
    double x0_d[NX];
    std::memcpy(x0_d, x_ref_d, NX * sizeof(double));
    x0_d[0] += 0.01; x0_d[1] -= 0.005;

    double Q_diag[6] = { 10, 10, 5, 1, 1, 0.5 };
    double Qf_diag[6] = { 20, 20, 10, 2, 2, 1 };
    double R_eff_d[N_MAX * 4];
    for (int i = 0; i < N * 4; ++i) R_eff_d[i] = 0.1;

    double D_d[3] = { euler.D_diag[0], euler.D_diag[1], euler.D_diag[2] };
    double B_d[12];
    for (int i = 0; i < 12; ++i) B_d[i] = euler.B_body[i];

    // Scalar reference
    double u_scalar[N_MAX * 4];
    riccati_6state_scalar(Q_diag, Qf_diag, R_eff_d, D_d, dt, B_d,
                          N, theta_d, x_ref_d, u_ref_d, x0_d, u_scalar);

    // NEON
    std::array<float, 6> Q_f, Qf_f;
    std::array<float, 3> D_f;
    std::array<float, 12> B_f;
    for (int i = 0; i < 6; ++i) { Q_f[i] = (float)Q_diag[i]; Qf_f[i] = (float)Qf_diag[i]; }
    for (int i = 0; i < 3; ++i) D_f[i] = euler.D_diag[i];
    for (int i = 0; i < 12; ++i) B_f[i] = euler.B_body[i];

    float theta_f[N_MAX], xr_f[(N_MAX+1)*6], ur_f[N_MAX*4];
    float x0_f[6], R_eff_f[N_MAX*4], u_neon[N_MAX*4];
    float ws[N_MAX * 40];
    for (int i = 0; i < N; ++i) theta_f[i] = (float)theta_d[i];
    for (int i = 0; i < (N+1)*6; ++i) xr_f[i] = (float)x_ref_d[i];
    for (int i = 0; i < N*4; ++i) ur_f[i] = (float)u_ref_d[i];
    for (int i = 0; i < N*4; ++i) R_eff_f[i] = 0.1f;
    for (int i = 0; i < 6; ++i) x0_f[i] = (float)x0_d[i];

    riccati_6state_neon(ws, Q_f, Qf_f, R_eff_f, D_f, (float)dt, B_f,
                        N, theta_f, xr_f, ur_f, x0_f, u_neon);

    double max_diff = 0.0;
    for (int k = 0; k < N * 4; ++k) {
        double d = std::fabs((double)u_neon[k] - u_scalar[k]);
        if (d > max_diff) max_diff = d;
    }

    std::printf("max_diff=%.3e", max_diff);
    std::printf("  neon=[%.4f,%.4f,%.4f,%.4f] scalar=[%.4f,%.4f,%.4f,%.4f]",
                u_neon[0], u_neon[1], u_neon[2], u_neon[3],
                u_scalar[0], u_scalar[1], u_scalar[2], u_scalar[3]);

    bool ok = max_diff < 0.02;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
#else
    std::printf("Test 2: NEON 6-state Riccati vs scalar ... SKIPPED (no NEON)\n");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Test 3: IPM convergence — constraints active, all u in [-1,1]
// ---------------------------------------------------------------------------
static bool test_ipm_convergence()
{
    std::printf("Test 3: IPM convergence (constraints) ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    EulerDynamicsData euler;
    euler_dynamics_precompute(params, config.dt, euler);

    HeadingLookupData hld;
    heading_lookup_precompute(params, config.dt, hld);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    // Start with large offset to force saturation
    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.5;  // large position error
    x0[1] -= 0.3;

    IpmSolverConfig ipm_cfg = ipm_solver_default_config();
    IpmWorkspace ws;
    ipm_workspace_init(ws);

    QPSolution sol = ipm_solve(euler, hld, path, x0, config, sched, ipm_cfg, ws);

    // Check all outputs in [-1, 1]
    bool all_feasible = true;
    for (int i = 0; i < config.N * NU; ++i) {
        if (sol.U[i] < -1.0 - 1e-6 || sol.U[i] > 1.0 + 1e-6) {
            all_feasible = false;
            break;
        }
    }

    // Count active constraints
    int n_active = 0;
    for (int i = 0; i < config.N * NU; ++i)
        if (std::fabs(sol.U[i]) > 0.99) n_active++;

    std::printf("feasible=%s  n_active=%d  iters=%d  time=%.1fus",
                all_feasible ? "yes" : "NO",
                n_active, sol.n_iterations, sol.solve_time_ns / 1000.0);
    std::printf("  u0=[%.4f,%.4f,%.4f,%.4f]",
                sol.u0[0], sol.u0[1], sol.u0[2], sol.u0[3]);

    bool ok = all_feasible;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4: IPM vs HPIPM comparison
// ---------------------------------------------------------------------------
static bool test_ipm_vs_hpipm()
{
#ifdef MPC_USE_HPIPM
    std::printf("Test 4: IPM vs HPIPM comparison ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    EulerDynamicsData euler;
    euler_dynamics_precompute(params, config.dt, euler);

    HeadingLookupData hld;
    heading_lookup_precompute(params, config.dt, hld);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    // HPIPM solve
    SolverContext ctx;
    solver_context_init(ctx, config.N * NU);
    QPSolution sol_hpipm = heading_lookup_solve_ocp(hld, path, x0, config, sched, ctx);
    solver_context_free(ctx);

    // IPM solve
    IpmSolverConfig ipm_cfg = ipm_solver_default_config();
    IpmWorkspace ws;
    ipm_workspace_init(ws);
    QPSolution sol_ipm = ipm_solve(euler, hld, path, x0, config, sched, ipm_cfg, ws);

    double u0_diff = 0.0;
    for (int j = 0; j < NU; ++j) {
        double d = std::fabs(sol_ipm.u0[j] - sol_hpipm.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }

    std::printf("u0_diff=%.3e", u0_diff);
    std::printf("  ipm=[%.4f,%.4f,%.4f,%.4f]  hpipm=[%.4f,%.4f,%.4f,%.4f]",
                sol_ipm.u0[0], sol_ipm.u0[1], sol_ipm.u0[2], sol_ipm.u0[3],
                sol_hpipm.u0[0], sol_hpipm.u0[1], sol_hpipm.u0[2], sol_hpipm.u0[3]);

    // Expect some difference due to Euler vs RK4 discretization
    bool ok = u0_diff < 0.25;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
#else
    std::printf("Test 4: IPM vs HPIPM comparison ... SKIPPED (no HPIPM)\n");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Test 5: Warm-start verification
// ---------------------------------------------------------------------------
static bool test_warm_start()
{
    std::printf("Test 5: Warm-start verification ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    EulerDynamicsData euler;
    euler_dynamics_precompute(params, config.dt, euler);

    HeadingLookupData hld;
    heading_lookup_precompute(params, config.dt, hld);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;

    IpmSolverConfig ipm_cfg = ipm_solver_default_config();
    IpmWorkspace ws;
    ipm_workspace_init(ws);

    // First solve (cold)
    QPSolution sol1 = ipm_solve(euler, hld, path, x0, config, sched, ipm_cfg, ws);

    // Second solve (warm — shift by 1 step)
    x0[0] += 0.001;
    QPSolution sol2 = ipm_solve(euler, hld, path + 1, x0, config, sched, ipm_cfg, ws);

    bool warm_used = ws.warm_valid;
    std::printf("warm_valid=%s  t1=%.1fus  t2=%.1fus",
                warm_used ? "yes" : "no",
                sol1.solve_time_ns / 1000.0, sol2.solve_time_ns / 1000.0);

    bool ok = warm_used;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 6: Timing benchmark
// ---------------------------------------------------------------------------
static bool test_timing()
{
    std::printf("Test 6: Timing benchmark ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    EulerDynamicsData euler;
    euler_dynamics_precompute(params, config.dt, euler);

    HeadingLookupData hld;
    heading_lookup_precompute(params, config.dt, hld);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    IpmSolverConfig ipm_cfg = ipm_solver_default_config();
    IpmWorkspace ws;
    ipm_workspace_init(ws);

    const int n_runs = 100;
    double total_ns = 0;
    for (int r = 0; r < n_runs; ++r) {
        QPSolution sol = ipm_solve(euler, hld, path, x0, config, sched, ipm_cfg, ws);
        total_ns += sol.solve_time_ns;
    }

    double avg_us = total_ns / n_runs / 1000.0;
    std::printf("avg=%.1f us over %d runs", avg_us, n_runs);
    std::printf(" PASS\n");  // informational, always passes
    return true;
}

// ---------------------------------------------------------------------------
int main()
{
    int n_pass = 0, n_fail = 0;
    auto run = [&](bool result) { result ? n_pass++ : n_fail++; };

    run(test_consistent_reference());
    run(test_neon_vs_scalar());
    run(test_ipm_convergence());
    run(test_ipm_vs_hpipm());
    run(test_warm_start());
    run(test_timing());

    std::printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
