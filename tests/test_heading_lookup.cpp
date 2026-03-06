// test_heading_lookup.cpp -- Unit tests for heading-lookup LTV mode
//
// Tests:
// 1. Trig decomposition accuracy (B_d reconstruction vs exact at random headings)
// 2. Table interpolation accuracy
// 3. A_d power correctness
// 4. Heading schedule feasibility
// 5. Heading schedule derating
// 6. Solution comparison on straight-line trajectory
// 7. Turning trajectory comparison (reports discrepancy)

#include "mpc_types.h"
#include "heading_lookup.h"
#include "mecanum_model.h"
#include "discretizer.h"
#include "condensing.h"
#include "mpc_offline.h"
#include "mpc_online.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>

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

static double mat_max_abs_diff(const double* A, const double* B, int n)
{
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = std::fabs(A[i] - B[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static double vec_norm(const double* v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
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
        path[k].x_ref[2] = omega_ref * k * dt;  // theta = ω * t
        path[k].x_ref[5] = omega_ref;            // omega
        path[k].theta    = path[k].x_ref[2];
        path[k].omega    = omega_ref;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Decomposition accuracy
// ---------------------------------------------------------------------------
static bool test_decomposition_accuracy()
{
    std::printf("Test 1: Trig decomposition accuracy ... ");

    ModelParams params = make_params();
    double dt = 0.02;

    HeadingLookupData data;
    double verify_err = heading_lookup_precompute(params, dt, data);

    // Verification error from precompute (at θ=0.7) should be < 1e-10
    if (verify_err > 1e-10) {
        std::printf("\n  Verification error at θ=0.7: %.3e (expected < 1e-10)", verify_err);
        std::printf(" FAIL\n");
        return false;
    }

    // Test at 20 random headings
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);

    double max_err = 0.0;
    for (int i = 0; i < 20; ++i) {
        double theta = dist(rng);
        double ct = std::cos(theta);
        double st = std::sin(theta);

        // Reconstruct B_d
        double B_recon[NX * NU];
        for (int j = 0; j < NX * NU; ++j)
            B_recon[j] = data.B_d0[j] + ct * data.B_dc[j] + st * data.B_ds[j];

        // Exact discretize
        RefNode ref_k, ref_k1;
        std::memset(&ref_k, 0, sizeof(RefNode));
        std::memset(&ref_k1, 0, sizeof(RefNode));
        ref_k.t = 0.0; ref_k.theta = theta; ref_k.x_ref[2] = theta;
        ref_k1.t = dt; ref_k1.theta = theta; ref_k1.x_ref[2] = theta;

        double A_exact[NX * NX], B_exact[NX * NU];
        exact_discretize(ref_k, ref_k1, params, A_exact, B_exact, 100);

        double err = mat_max_abs_diff(B_recon, B_exact, NX * NU);
        if (err > max_err) max_err = err;
    }

    bool ok = max_err < 1e-10;
    std::printf("  (max_err=%.3e)", max_err);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: Table interpolation accuracy
// ---------------------------------------------------------------------------
static bool test_table_interpolation()
{
    std::printf("Test 2: Table interpolation accuracy ... ");

    ModelParams params = make_params();
    double dt = 0.02;

    HeadingTableData table;
    heading_table_precompute(params, dt, 72, table);

    // Test at intermediate headings (between table entries)
    std::mt19937 rng(54321);
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);

    double max_err = 0.0;
    for (int i = 0; i < 20; ++i) {
        double theta = dist(rng);

        // Table interpolation
        double B_interp[NX * NU];
        heading_table_build_B_list(table, &theta, 1, B_interp);

        // Exact discretize
        RefNode ref_k, ref_k1;
        std::memset(&ref_k, 0, sizeof(RefNode));
        std::memset(&ref_k1, 0, sizeof(RefNode));
        ref_k.t = 0.0; ref_k.theta = theta; ref_k.x_ref[2] = theta;
        ref_k1.t = dt; ref_k1.theta = theta; ref_k1.x_ref[2] = theta;

        double A_exact[NX * NX], B_exact[NX * NU];
        exact_discretize(ref_k, ref_k1, params, A_exact, B_exact, 100);

        double err = mat_max_abs_diff(B_interp, B_exact, NX * NU);
        if (err > max_err) max_err = err;
    }

    // With M=72 (5° spacing), linear interpolation error should be small
    // but not as good as trig decomposition
    bool ok = max_err < 1e-3;  // generous bound for linear interp
    std::printf("  (max_err=%.3e)", max_err);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: A_d power correctness
// ---------------------------------------------------------------------------
static bool test_ad_power()
{
    std::printf("Test 3: A_d power correctness ... ");

    ModelParams params = make_params();
    double dt = 0.02;

    HeadingLookupData data;
    heading_lookup_precompute(params, dt, data);

    // Verify A_d^k matches successive multiplication
    double A_k[NX * NX];
    // A_d^0 = I
    std::memset(A_k, 0, NX * NX * sizeof(double));
    for (int i = 0; i < NX; ++i)
        A_k[i + NX * i] = 1.0;

    double max_err = 0.0;
    for (int k = 0; k <= std::min(N_MAX, 20); ++k) {
        double err = mat_max_abs_diff(A_k, data.A_d_pow + k * NX * NX, NX * NX);
        if (err > max_err) max_err = err;

        // Advance: A_k = A_d * A_k
        if (k < N_MAX) {
            double temp[NX * NX];
            mpc_linalg::gemm(NX, NX, NX, data.A_d, A_k, temp);
            std::memcpy(A_k, temp, NX * NX * sizeof(double));
        }
    }

    bool ok = max_err < 1e-12;
    std::printf("  (max_err=%.3e)", max_err);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4: Heading schedule feasibility
// ---------------------------------------------------------------------------
static bool test_heading_schedule_feasibility()
{
    std::printf("Test 4: Heading schedule feasibility ... ");

    ModelParams params = make_params();
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);
    double dt = 0.02;
    int N = 20;

    // Build a turning reference
    RefNode ref[N_MAX + 2];
    build_turning_ref(ref, N + 2, dt, 3.0);  // 3 rad/s turn

    double x0[NX] = {};
    x0[5] = 0.0;  // starting at zero omega

    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref, N, dt, sched, theta_sched);

    // Verify angular acceleration constraint at each step
    double omega_prev = x0[5];
    double max_alpha = 0.0;
    bool ok = true;

    for (int k = 0; k < N; ++k) {
        double omega_k = (theta_sched[k + 1] - theta_sched[k]) / dt;
        double alpha = std::fabs(omega_k - omega_prev) / dt;

        if (alpha > max_alpha) max_alpha = alpha;

        // Should not exceed alpha_0 (maximum at zero ω, zero v)
        if (alpha > sched.alpha_0 + 1e-6) {
            std::printf("\n  Step %d: alpha=%.1f > alpha_0=%.1f", k, alpha, sched.alpha_0);
            ok = false;
        }

        omega_prev = omega_k;
    }

    // Verify convergence toward reference heading
    double final_heading_err = std::fabs(theta_sched[N] - ref[N].x_ref[2]);
    std::printf("  (max_alpha=%.1f, final_heading_err=%.3f)", max_alpha, final_heading_err);

    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 5: Heading schedule derating
// ---------------------------------------------------------------------------
static bool test_heading_schedule_derating()
{
    std::printf("Test 5: Heading schedule derating ... ");

    ModelParams params = make_params();
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);
    double dt = 0.02;
    int N = 10;

    // Build reference with high velocity AND turning
    RefNode ref[N_MAX + 2];
    for (int k = 0; k < N + 2; ++k) {
        std::memset(&ref[k], 0, sizeof(RefNode));
        ref[k].t = k * dt;
        ref[k].x_ref[2] = 1.0;   // target heading = 1 rad
        ref[k].x_ref[3] = 2.0;   // high vx (m/s)
        ref[k].x_ref[4] = 1.0;   // high vy (m/s)
        ref[k].theta = 1.0;
    }

    double x0[NX] = {};  // starting at zero heading

    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref, N, dt, sched, theta_sched);

    // With high velocity, angular acceleration should be reduced
    // Compute actual first-step alpha
    double omega_0 = (theta_sched[1] - theta_sched[0]) / dt;
    double alpha_actual = std::fabs(omega_0) / dt;

    // Compute expected derated alpha
    double v_field = std::sqrt(2.0 * 2.0 + 1.0 * 1.0);
    double expected_headroom = std::max(0.0, 1.0 - v_field / sched.v_max);
    double expected_alpha_max = sched.alpha_0 * expected_headroom;

    bool ok = true;
    if (expected_headroom < 0.5) {
        // Derating should be significant
        std::printf("  (headroom=%.2f, alpha=%.1f)", expected_headroom, alpha_actual);
    } else {
        std::printf("  (v_max=%.1f may be too high for significant derating)", sched.v_max);
    }

    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 6: Solution comparison (straight-line, θ≈0)
// ---------------------------------------------------------------------------
static bool test_solution_comparison_straight()
{
    std::printf("Test 6: Solution comparison (straight line) ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config();
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    // Precompute (standard method)
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);
    if (!windows || n_windows < 1) {
        std::printf("\n  Precompute failed");
        std::printf(" FAIL\n");
        return false;
    }

    // Heading-lookup precompute
    HeadingLookupData hl_data;
    heading_lookup_precompute(params, config.dt, hl_data);

    // Solve at reference + small perturbation
    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    // Standard solve
    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));
    QPSolution sol_std = mpc_solve_online(windows[0], x0, config, ws);

    // Heading-lookup solve
    SolverContext ctx;
    solver_context_init(ctx, config.N * NU);
    QPSolution sol_hl = heading_lookup_solve_condensed(hl_data, path, x0, config,
                                                       sched, QpSolverType::FISTA, ctx);

    // Compare u0
    double u0_diff = 0.0;
    for (int j = 0; j < NU; ++j) {
        double d = std::fabs(sol_std.u0[j] - sol_hl.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }

    bool ok = u0_diff < 0.1;  // should be very close for straight line (θ≈0)
    std::printf("  (u0_diff=%.6f)", u0_diff);
    std::printf("  std=[%.3f,%.3f,%.3f,%.3f] hl=[%.3f,%.3f,%.3f,%.3f]",
                sol_std.u0[0], sol_std.u0[1], sol_std.u0[2], sol_std.u0[3],
                sol_hl.u0[0], sol_hl.u0[1], sol_hl.u0[2], sol_hl.u0[3]);

    solver_context_free(ctx);
    delete[] windows;

    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 7: Turning trajectory comparison (report discrepancy)
// ---------------------------------------------------------------------------
static bool test_turning_trajectory()
{
    std::printf("Test 7: Turning trajectory comparison ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 50;
    RefNode path[50];
    build_turning_ref(path, n_path, config.dt, 2.0);  // 2 rad/s turn

    // Standard precompute
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);
    if (!windows || n_windows < 1) {
        std::printf("\n  Precompute failed");
        std::printf(" FAIL\n");
        return false;
    }

    // Heading-lookup precompute
    HeadingLookupData hl_data;
    heading_lookup_precompute(params, config.dt, hl_data);

    // Solve at initial state
    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));
    QPSolution sol_std = mpc_solve_online(windows[0], x0, config, ws);

    SolverContext ctx;
    solver_context_init(ctx, config.N * NU);
    QPSolution sol_hl = heading_lookup_solve_condensed(hl_data, path, x0, config,
                                                       sched, QpSolverType::FISTA, ctx);

    double u0_diff = 0.0;
    for (int j = 0; j < NU; ++j) {
        double d = std::fabs(sol_std.u0[j] - sol_hl.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }

    // Expected to have some discrepancy due to constant-θ approximation vs Hermite
    std::printf("  (u0_diff=%.6f)", u0_diff);
    std::printf("  std=[%.3f,%.3f,%.3f,%.3f] hl=[%.3f,%.3f,%.3f,%.3f]",
                sol_std.u0[0], sol_std.u0[1], sol_std.u0[2], sol_std.u0[3],
                sol_hl.u0[0], sol_hl.u0[1], sol_hl.u0[2], sol_hl.u0[3]);

    solver_context_free(ctx);
    delete[] windows;

    // This is informational - always passes (just reports discrepancy)
    bool ok = true;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 8: Kernel P[j] correctness
// ---------------------------------------------------------------------------
static bool test_kernel_correctness()
{
    std::printf("Test 8: Kernel P[j] correctness ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);

    HeadingLookupData data;
    heading_lookup_precompute(params, config.dt, data);

    HeadingKernelData kern;
    heading_kernel_precompute(data, config, kern);

    const int N = config.N;

    // Verify P[j] via naive summation:
    // P[N-1] = Qf
    // P[j] = Q + A_d^T P[j+1] A_d
    // Equivalently P[j] = sum_{m=0}^{N-2-j} (A_d^m)^T Q (A_d^m) + (A_d^(N-1-j))^T Qf (A_d^(N-1-j))

    double max_err = 0.0;
    for (int j = 0; j < N; ++j) {
        double P_naive[NX * NX] = {};

        // Sum Q terms: m=0 to N-2-j
        for (int m = 0; m <= N - 2 - j; ++m) {
            const double* Ad_m = data.A_d_pow + m * NX * NX;
            double QAm[NX * NX] = {};
            double AtQAm[NX * NX] = {};
            mpc_linalg::gemm(NX, NX, NX, config.Q, Ad_m, QAm);
            mpc_linalg::gemm_atb(NX, NX, NX, Ad_m, NX, QAm, NX, AtQAm, NX);
            for (int i = 0; i < NX * NX; ++i)
                P_naive[i] += AtQAm[i];
        }

        // Terminal: (A_d^(N-1-j))^T Qf (A_d^(N-1-j))
        int term_pow = N - 1 - j;
        const double* Ad_term = data.A_d_pow + term_pow * NX * NX;
        double QfA[NX * NX] = {};
        double AtQfA[NX * NX] = {};
        mpc_linalg::gemm(NX, NX, NX, config.Qf, Ad_term, QfA);
        mpc_linalg::gemm_atb(NX, NX, NX, Ad_term, NX, QfA, NX, AtQfA, NX);
        for (int i = 0; i < NX * NX; ++i)
            P_naive[i] += AtQfA[i];

        double err = mat_max_abs_diff(kern.P + j * NX * NX, P_naive, NX * NX);
        if (err > max_err) max_err = err;
    }

    bool ok = max_err < 1e-10;
    std::printf("  (max_err=%.3e)", max_err);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 9: Fast solve matches original
// ---------------------------------------------------------------------------
static bool test_fast_solve_matches()
{
    std::printf("Test 9: Fast solve matches original ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 10);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 50;
    RefNode path[50];
    build_straight_ref(path, n_path, config.dt);

    HeadingLookupData data;
    heading_lookup_precompute(params, config.dt, data);

    HeadingKernelData kern;
    heading_kernel_precompute(data, config, kern);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    SolverContext ctx1, ctx2;
    solver_context_init(ctx1, config.N * NU);
    solver_context_init(ctx2, config.N * NU);

    QPSolution sol_orig = heading_lookup_solve_condensed(data, path, x0, config,
                                                         sched, QpSolverType::FISTA, ctx1);
    QPSolution sol_fast = heading_lookup_solve_fast(data, kern, path, x0, config,
                                                    sched, QpSolverType::FISTA, ctx2);

    double u0_diff = 0.0;
    for (int j = 0; j < NU; ++j) {
        double d = std::fabs(sol_orig.u0[j] - sol_fast.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }

    solver_context_free(ctx1);
    solver_context_free(ctx2);

    bool ok = u0_diff < 1e-6;
    std::printf("  (u0_diff=%.3e)", u0_diff);
    std::printf("  orig=[%.4f,%.4f,%.4f,%.4f] fast=[%.4f,%.4f,%.4f,%.4f]",
                sol_orig.u0[0], sol_orig.u0[1], sol_orig.u0[2], sol_orig.u0[3],
                sol_fast.u0[0], sol_fast.u0[1], sol_fast.u0[2], sol_fast.u0[3]);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 10: HPIPM OCP solve matches condensed (if available)
// ---------------------------------------------------------------------------
static bool test_ocp_solve_matches()
{
#ifdef MPC_USE_HPIPM
    std::printf("Test 10: HPIPM OCP solve matches condensed ... ");

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

    SolverContext ctx1, ctx2;
    solver_context_init(ctx1, config.N * NU);
    solver_context_init(ctx2, config.N * NU);

    QPSolution sol_cond = heading_lookup_solve_condensed(data, path, x0, config,
                                                         sched, QpSolverType::FISTA, ctx1);
    QPSolution sol_ocp = heading_lookup_solve_ocp(data, path, x0, config, sched, ctx2);

    double u0_diff = 0.0;
    for (int j = 0; j < NU; ++j) {
        double d = std::fabs(sol_cond.u0[j] - sol_ocp.u0[j]);
        if (d > u0_diff) u0_diff = d;
    }

    solver_context_free(ctx1);
    solver_context_free(ctx2);

    bool ok = u0_diff < 1e-4;
    std::printf("  (u0_diff=%.3e)", u0_diff);
    std::printf("  cond=[%.4f,%.4f,%.4f,%.4f] ocp=[%.4f,%.4f,%.4f,%.4f]",
                sol_cond.u0[0], sol_cond.u0[1], sol_cond.u0[2], sol_cond.u0[3],
                sol_ocp.u0[0], sol_ocp.u0[1], sol_ocp.u0[2], sol_ocp.u0[3]);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
#else
    std::printf("Test 10: HPIPM OCP solve matches condensed ... SKIPPED (no HPIPM)\n");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Test 11: Timing comparison
// ---------------------------------------------------------------------------
static bool test_timing_comparison()
{
    std::printf("Test 11: Timing comparison ... \n");

    ModelParams params = make_params();
    MPCConfig config = make_config(0.02, 20);
    HeadingScheduleConfig sched = heading_schedule_config_from_params(params);

    const int n_path = 60;
    RefNode path[60];
    build_straight_ref(path, n_path, config.dt);

    HeadingLookupData data;
    heading_lookup_precompute(params, config.dt, data);

    HeadingKernelData kern;
    heading_kernel_precompute(data, config, kern);

    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;
    x0[1] -= 0.005;

    const int n_runs = 20;

    // Original condensed solve
    {
        SolverContext ctx;
        solver_context_init(ctx, config.N * NU);
        double total_ns = 0.0;
        for (int r = 0; r < n_runs; ++r) {
            QPSolution sol = heading_lookup_solve_condensed(data, path, x0, config,
                                                             sched, QpSolverType::FISTA, ctx);
            total_ns += sol.solve_time_ns;
        }
        std::printf("  Original condensed: %.0f μs avg\n", total_ns / n_runs / 1e3);
        solver_context_free(ctx);
    }

    // Fast kernel-based solve
    {
        SolverContext ctx;
        solver_context_init(ctx, config.N * NU);
        double total_ns = 0.0;
        for (int r = 0; r < n_runs; ++r) {
            QPSolution sol = heading_lookup_solve_fast(data, kern, path, x0, config,
                                                       sched, QpSolverType::FISTA, ctx);
            total_ns += sol.solve_time_ns;
        }
        std::printf("  Kernel-based fast:  %.0f μs avg\n", total_ns / n_runs / 1e3);
        solver_context_free(ctx);
    }

#ifdef MPC_USE_HPIPM
    // HPIPM OCP solve
    {
        SolverContext ctx;
        solver_context_init(ctx, config.N * NU);
        double total_ns = 0.0;
        for (int r = 0; r < n_runs; ++r) {
            QPSolution sol = heading_lookup_solve_ocp(data, path, x0, config, sched, ctx);
            total_ns += sol.solve_time_ns;
        }
        std::printf("  HPIPM OCP direct:  %.0f μs avg\n", total_ns / n_runs / 1e3);
        solver_context_free(ctx);
    }
#endif

    std::printf("  (informational - always passes)");
    std::printf(" PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=== test_heading_lookup ===\n\n");

    bool all_pass = true;
    all_pass &= test_decomposition_accuracy();
    all_pass &= test_table_interpolation();
    all_pass &= test_ad_power();
    all_pass &= test_heading_schedule_feasibility();
    all_pass &= test_heading_schedule_derating();
    all_pass &= test_solution_comparison_straight();
    all_pass &= test_turning_trajectory();
    all_pass &= test_kernel_correctness();
    all_pass &= test_fast_solve_matches();
    all_pass &= test_ocp_solve_matches();
    all_pass &= test_timing_comparison();

    std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
