// test_mpc_roundtrip.cpp -- full integration test for offline-to-online MPC pipeline
//
// Tests: offline precomputation, online solve at reference, online solve with
// perturbation, closed-loop simulation, and serialization roundtrip.

#include "mpc_types.h"
#include "mpc_offline.h"
#include "mpc_online.h"
#include "mecanum_model.h"
#include "discretizer.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"

#include <cstdio>
#include <cmath>
#include <cstring>

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

static MPCConfig make_config()
{
    MPCConfig cfg{};
    cfg.N     = 10;
    cfg.dt    = 0.02;
    cfg.u_min = -1.0;
    cfg.u_max =  1.0;

    // Q = diag(10, 10, 5, 1, 1, 0.5), column-major NX*NX
    std::memset(cfg.Q, 0, sizeof(cfg.Q));
    cfg.Q[0 * NX + 0] = 10.0;
    cfg.Q[1 * NX + 1] = 10.0;
    cfg.Q[2 * NX + 2] =  5.0;
    cfg.Q[3 * NX + 3] =  1.0;
    cfg.Q[4 * NX + 4] =  1.0;
    cfg.Q[5 * NX + 5] =  0.5;

    // R = diag(0.1, 0.1, 0.1, 0.1), column-major NU*NU
    std::memset(cfg.R, 0, sizeof(cfg.R));
    cfg.R[0 * NU + 0] = 0.1;
    cfg.R[1 * NU + 1] = 0.1;
    cfg.R[2 * NU + 2] = 0.1;
    cfg.R[3 * NU + 3] = 0.1;

    // Qf = 2 * Q
    for (int i = 0; i < NX * NX; ++i)
        cfg.Qf[i] = 2.0 * cfg.Q[i];

    return cfg;
}

static double vec_norm(const double* v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
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

// Build reference path: straight line along x at 0.5 m/s, theta=0
static void build_ref_path(RefNode* path, int n_path, double dt)
{
    for (int k = 0; k < n_path; ++k) {
        std::memset(&path[k], 0, sizeof(RefNode));
        path[k].t        = k * dt;
        path[k].x_ref[0] = k * dt * 0.5;   // px
        path[k].x_ref[1] = 0.0;             // py
        path[k].x_ref[2] = 0.0;             // theta
        path[k].x_ref[3] = 0.5;             // vx
        path[k].x_ref[4] = 0.0;             // vy
        path[k].x_ref[5] = 0.0;             // omega
        path[k].theta    = 0.0;
        path[k].omega    = 0.0;
        // u_ref = 0 everywhere (MPC will find the right inputs)
        for (int j = 0; j < NU; ++j)
            path[k].u_ref[j] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Offline precomputation
// ---------------------------------------------------------------------------
static bool test_offline_precompute()
{
    std::printf("Test 1: Offline precomputation ... ");

    ModelParams params = make_params();
    MPCConfig   config = make_config();

    const int n_path = 50;
    RefNode path[50];
    build_ref_path(path, n_path, config.dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);

    bool ok = true;

    // Should produce n_path - N = 50 - 10 = 40 windows
    if (n_windows != n_path - config.N) {
        std::printf("\n  Expected %d windows, got %d", n_path - config.N, n_windows);
        ok = false;
    }

    // First window should have correct dimensions
    if (windows && n_windows > 0) {
        if (windows[0].N != config.N) {
            std::printf("\n  Window[0].N = %d, expected %d", windows[0].N, config.N);
            ok = false;
        }
        if (windows[0].n_vars != config.N * NU) {
            std::printf("\n  Window[0].n_vars = %d, expected %d",
                        windows[0].n_vars, config.N * NU);
            ok = false;
        }
    } else {
        std::printf("\n  windows is null or empty");
        ok = false;
    }

    delete[] windows;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: Online solve at reference (zero or small tracking error)
// ---------------------------------------------------------------------------
static bool test_online_at_reference()
{
    std::printf("Test 2: Online solve at reference ... ");

    ModelParams params = make_params();
    MPCConfig   config = make_config();

    const int n_path = 50;
    RefNode path[50];
    build_ref_path(path, n_path, config.dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);

    bool ok = true;

    if (!windows || n_windows < 1) {
        std::printf("\n  Precompute failed");
        delete[] windows;
        std::printf(" FAIL\n");
        return false;
    }

    BoxQPWorkspace workspace;
    std::memset(&workspace, 0, sizeof(workspace));

    // Solve at exact reference state
    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));

    QPSolution sol = mpc_solve_online(windows[0], x0, config, workspace);

    // At reference, error e0 = 0, so u0 should be near u_ref (which is 0)
    // The solution may not be exactly zero because of the affine offset / feedforward
    // but it should be reasonably small
    double u0_norm = vec_norm(sol.u0, NU);
    if (u0_norm > 5.0) {
        std::printf("\n  |u0| = %.4f at reference, expected < 5.0", u0_norm);
        ok = false;
    }

    if (sol.solve_time_ns <= 0.0) {
        std::printf("\n  solve_time_ns = %.1f, expected > 0", sol.solve_time_ns);
        ok = false;
    }

    // All controls within bounds
    for (int j = 0; j < NU; ++j) {
        if (sol.u0[j] < config.u_min - 1e-8 || sol.u0[j] > config.u_max + 1e-8) {
            std::printf("\n  u0[%d] = %.4f out of bounds [%.1f, %.1f]",
                        j, sol.u0[j], config.u_min, config.u_max);
            ok = false;
        }
    }

    std::printf("  (u0_norm=%.4f, time=%.0fns)", u0_norm, sol.solve_time_ns);

    delete[] windows;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: Online solve with perturbation (u0 should be nonzero, within bounds)
// ---------------------------------------------------------------------------
static bool test_online_with_perturbation()
{
    std::printf("Test 3: Online solve with perturbation ... ");

    ModelParams params = make_params();
    MPCConfig   config = make_config();

    const int n_path = 50;
    RefNode path[50];
    build_ref_path(path, n_path, config.dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);

    bool ok = true;

    if (!windows || n_windows < 1) {
        std::printf("\n  Precompute failed");
        delete[] windows;
        std::printf(" FAIL\n");
        return false;
    }

    BoxQPWorkspace workspace;
    std::memset(&workspace, 0, sizeof(workspace));

    // Perturbed initial state
    double x0[NX];
    std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;    // +1cm in px
    x0[1] -= 0.005;   // -5mm in py
    x0[2] += 0.02;    // +0.02 rad in theta

    QPSolution sol = mpc_solve_online(windows[0], x0, config, workspace);

    // u0 should be nonzero (correcting the error)
    double u0_norm = vec_norm(sol.u0, NU);
    if (u0_norm < 1e-6) {
        std::printf("\n  |u0| = %.6e, expected nonzero", u0_norm);
        ok = false;
    }

    // All controls within bounds
    for (int j = 0; j < NU; ++j) {
        if (sol.u0[j] < config.u_min - 1e-8 || sol.u0[j] > config.u_max + 1e-8) {
            std::printf("\n  u0[%d] = %.4f out of bounds [%.1f, %.1f]",
                        j, sol.u0[j], config.u_min, config.u_max);
            ok = false;
        }
    }

    std::printf("  (u0=[%.3f, %.3f, %.3f, %.3f])",
                sol.u0[0], sol.u0[1], sol.u0[2], sol.u0[3]);

    delete[] windows;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4: Closed-loop simulation
// ---------------------------------------------------------------------------
static bool test_closed_loop()
{
    std::printf("Test 4: Closed-loop simulation ...\n");

    ModelParams params = make_params();
    MPCConfig   config = make_config();

    const int n_path = 50;
    RefNode path[50];
    build_ref_path(path, n_path, config.dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);

    bool ok = true;

    if (!windows || n_windows < 30) {
        std::printf("  Precompute failed or not enough windows (%d)\n", n_windows);
        delete[] windows;
        std::printf("  FAIL\n");
        return false;
    }

    BoxQPWorkspace workspace;
    std::memset(&workspace, 0, sizeof(workspace));

    // Initial state: reference + perturbation
    double x_cur[NX];
    std::memcpy(x_cur, path[0].x_ref, NX * sizeof(double));
    x_cur[0] += 0.02;    // +2cm in px
    x_cur[1] -= 0.01;    // -1cm in py
    x_cur[2] += 0.01;    // +0.01 rad in theta

    const int n_sim = 30;
    double errors[30];
    bool bounds_ok = true;

    for (int k = 0; k < n_sim; ++k) {
        // Solve MPC using window[k]
        QPSolution sol = mpc_solve_online(windows[k], x_cur, config, workspace);

        // Check bounds on applied control
        for (int j = 0; j < NU; ++j) {
            if (sol.u0[j] < config.u_min - 1e-8 || sol.u0[j] > config.u_max + 1e-8) {
                bounds_ok = false;
            }
        }

        // Compute tracking error
        double err_vec[NX];
        for (int i = 0; i < NX; ++i)
            err_vec[i] = x_cur[i] - path[k].x_ref[i];
        errors[k] = vec_norm(err_vec, NX);

        std::printf("  step %2d: error=%.6f  u0=[%6.3f, %6.3f, %6.3f, %6.3f]\n",
                    k, errors[k], sol.u0[0], sol.u0[1], sol.u0[2], sol.u0[3]);

        // Simulate forward using Euler: x_next = x_cur + dt * (Ac*x_cur + Bc*u0)
        double Ac[NX * NX], Bc[NX * NU];
        continuous_dynamics(x_cur[2], params, Ac, Bc);

        double Ax[NX], Bu[NX];
        mpc_linalg::gemv(NX, NX, Ac, x_cur, Ax);
        mpc_linalg::gemv(NX, NU, Bc, sol.u0, Bu);

        for (int i = 0; i < NX; ++i)
            x_cur[i] += config.dt * (Ax[i] + Bu[i]);
    }

    // Check: tracking error should stay bounded
    if (errors[n_sim - 1] > 0.05) {
        std::printf("  Final tracking error %.6f > 0.05\n", errors[n_sim - 1]);
        ok = false;
    }

    if (!bounds_ok) {
        std::printf("  Some controls violated bounds\n");
        ok = false;
    }

    // Check error is decreasing on average (compare first quarter to last quarter)
    double avg_early = 0.0, avg_late = 0.0;
    int quarter = n_sim / 4;
    for (int k = 0; k < quarter; ++k)
        avg_early += errors[k];
    for (int k = n_sim - quarter; k < n_sim; ++k)
        avg_late += errors[k];
    avg_early /= quarter;
    avg_late  /= quarter;

    if (avg_late > avg_early + 0.01) {
        std::printf("  Error growing: early_avg=%.4f, late_avg=%.4f\n",
                    avg_early, avg_late);
        ok = false;
    }

    std::printf("  Final error: %.6f, early_avg: %.6f, late_avg: %.6f",
                errors[n_sim - 1], avg_early, avg_late);

    delete[] windows;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 5: Serialization roundtrip
// ---------------------------------------------------------------------------
static bool test_serialization_roundtrip()
{
    std::printf("Test 5: Serialization roundtrip ... ");

    ModelParams params = make_params();
    MPCConfig   config = make_config();

    const int n_path = 50;
    RefNode path[50];
    build_ref_path(path, n_path, config.dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);

    bool ok = true;

    if (!windows || n_windows < 1) {
        std::printf("\n  Precompute failed");
        delete[] windows;
        std::printf(" FAIL\n");
        return false;
    }

    const char* filename = "/tmp/test_mpc_windows.bin";

    // Save
    int save_ret = mpc_save_windows(filename, windows, n_windows, config);
    if (save_ret != 0) {
        std::printf("\n  mpc_save_windows returned %d", save_ret);
        ok = false;
    }

    // Load
    int n_loaded = 0;
    MPCConfig config_loaded{};
    PrecomputedWindow* loaded = mpc_load_windows(filename, n_loaded, config_loaded);

    if (!loaded) {
        std::printf("\n  mpc_load_windows returned null");
        ok = false;
    } else {
        // Check window count matches
        if (n_loaded != n_windows) {
            std::printf("\n  Loaded %d windows, expected %d", n_loaded, n_windows);
            ok = false;
        }

        // Check config roundtrip
        if (std::fabs(config_loaded.u_min - config.u_min) > 1e-14 ||
            std::fabs(config_loaded.u_max - config.u_max) > 1e-14) {
            std::printf("\n  Config mismatch: u_min=%.2f (expected %.2f), u_max=%.2f (expected %.2f)",
                        config_loaded.u_min, config.u_min,
                        config_loaded.u_max, config.u_max);
            ok = false;
        }

        // Compare first window's H matrix
        if (n_loaded > 0) {
            int nv = windows[0].n_vars;
            double h_diff = mat_max_abs_diff(windows[0].H, loaded[0].H, nv * nv);
            if (h_diff > 1e-14) {
                std::printf("\n  H matrix max diff = %.3e (expected < 1e-14)", h_diff);
                ok = false;
            }

            // Compare L matrix
            double l_diff = mat_max_abs_diff(windows[0].L, loaded[0].L, nv * nv);
            if (l_diff > 1e-14) {
                std::printf("\n  L matrix max diff = %.3e (expected < 1e-14)", l_diff);
                ok = false;
            }

            // Compare F matrix
            double f_diff = mat_max_abs_diff(windows[0].F, loaded[0].F, nv * NX);
            if (f_diff > 1e-14) {
                std::printf("\n  F matrix max diff = %.3e (expected < 1e-14)", f_diff);
                ok = false;
            }

            // Compare f_const
            double fc_diff = mat_max_abs_diff(windows[0].f_const, loaded[0].f_const, nv);
            if (fc_diff > 1e-14) {
                std::printf("\n  f_const max diff = %.3e (expected < 1e-14)", fc_diff);
                ok = false;
            }

            // Compare x_ref_0
            double xr_diff = mat_max_abs_diff(windows[0].x_ref_0, loaded[0].x_ref_0, NX);
            if (xr_diff > 1e-14) {
                std::printf("\n  x_ref_0 max diff = %.3e (expected < 1e-14)", xr_diff);
                ok = false;
            }
        }

        delete[] loaded;
    }

    // Clean up temp file
    std::remove(filename);

    delete[] windows;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 6: v2 serialization roundtrip (dt + lambda_max)
// ---------------------------------------------------------------------------
static bool test_v2_roundtrip()
{
    std::printf("Test 6: v2 roundtrip (dt + lambda_max) ... ");

    ModelParams params = make_params();
    MPCConfig   config = make_config();

    const int n_path = 50;
    RefNode path[50];
    build_ref_path(path, n_path, config.dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(path, n_path, params,
                                                     config, n_windows);

    bool ok = true;

    if (!windows || n_windows < 1) {
        std::printf("\n  Precompute failed");
        delete[] windows;
        std::printf(" FAIL\n");
        return false;
    }

    const char* filename = "/tmp/test_mpc_windows_v2.bin";

    // Save
    int save_ret = mpc_save_windows(filename, windows, n_windows, config);
    if (save_ret != 0) {
        std::printf("\n  mpc_save_windows returned %d", save_ret);
        ok = false;
    }

    // Load
    int n_loaded = 0;
    MPCConfig config_loaded{};
    PrecomputedWindow* loaded = mpc_load_windows(filename, n_loaded, config_loaded);

    if (!loaded) {
        std::printf("\n  mpc_load_windows returned null");
        ok = false;
    } else {
        // Check dt survives roundtrip
        if (std::fabs(config_loaded.dt - config.dt) > 1e-14) {
            std::printf("\n  dt mismatch: loaded=%.6f, expected=%.6f",
                        config_loaded.dt, config.dt);
            ok = false;
        }

        // Check lambda_max survives roundtrip for all windows
        for (int i = 0; i < n_loaded && i < n_windows; ++i) {
            if (std::fabs(loaded[i].lambda_max - windows[i].lambda_max) > 1e-14) {
                std::printf("\n  Window[%d] lambda_max mismatch: loaded=%.6e, expected=%.6e",
                            i, loaded[i].lambda_max, windows[i].lambda_max);
                ok = false;
                break;
            }
        }

        // Verify loaded windows produce identical solve results
        if (n_loaded > 0 && ok) {
            BoxQPWorkspace ws_orig{}, ws_loaded{};

            double x0[NX];
            std::memcpy(x0, path[0].x_ref, NX * sizeof(double));
            x0[0] += 0.01;
            x0[1] -= 0.005;

            QPSolution sol_orig = mpc_solve_online(windows[0], x0, config, ws_orig);
            QPSolution sol_loaded = mpc_solve_online(loaded[0], x0, config_loaded, ws_loaded);

            double u_diff = mat_max_abs_diff(sol_orig.U, sol_loaded.U, windows[0].n_vars);
            if (u_diff > 1e-12) {
                std::printf("\n  Solve output differs: max_diff=%.3e", u_diff);
                ok = false;
            }
        }

        delete[] loaded;
    }

    std::remove(filename);
    delete[] windows;
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=== test_mpc_roundtrip ===\n\n");

    bool all_pass = true;
    all_pass &= test_offline_precompute();
    all_pass &= test_online_at_reference();
    all_pass &= test_online_with_perturbation();
    all_pass &= test_closed_loop();
    all_pass &= test_serialization_roundtrip();
    all_pass &= test_v2_roundtrip();

    std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
