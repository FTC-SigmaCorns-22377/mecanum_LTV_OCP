// test_load_windows.cpp -- integration test for MecanumLTV::loadWindows()
//
// Tests: precompute via loadTrajectory → save → loadWindows → solve
//        and verifies results match the direct path.

#include "mecanum_ltv.h"
#include "mpc_offline.h"
#include "mecanum_model.h"

#include <cstdio>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers (same as test_mpc_roundtrip)
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
    return p;
}

static MPCConfig make_config()
{
    MPCConfig cfg{};
    cfg.N     = 10;
    cfg.dt    = 0.02;
    cfg.u_min = -1.0;
    cfg.u_max =  1.0;

    cfg.Q[0 * NX + 0] = 10.0;
    cfg.Q[1 * NX + 1] = 10.0;
    cfg.Q[2 * NX + 2] =  5.0;
    cfg.Q[3 * NX + 3] =  1.0;
    cfg.Q[4 * NX + 4] =  1.0;
    cfg.Q[5 * NX + 5] =  0.5;

    for (int i = 0; i < NU; ++i)
        cfg.R[i + NU * i] = 0.1;

    for (int i = 0; i < NX * NX; ++i)
        cfg.Qf[i] = 2.0 * cfg.Q[i];

    return cfg;
}

// Build raw trajectory samples in loadTrajectory format:
// [t, px, py, theta, vx, vy, omega] per row
static void build_samples(double* samples, int n_samples, double dt)
{
    for (int k = 0; k < n_samples; ++k) {
        double t = k * dt;
        samples[k * 7 + 0] = t;          // t
        samples[k * 7 + 1] = t * 0.5;    // px
        samples[k * 7 + 2] = 0.0;        // py
        samples[k * 7 + 3] = 0.0;        // theta
        samples[k * 7 + 4] = 0.5;        // vx
        samples[k * 7 + 5] = 0.0;        // vy
        samples[k * 7 + 6] = 0.0;        // omega
    }
}

// ---------------------------------------------------------------------------
// Test: loadWindows produces identical solve results to loadTrajectory
// ---------------------------------------------------------------------------
static bool test_load_windows_matches_direct()
{
    std::printf("Test: loadWindows matches loadTrajectory ... ");

    ModelParams params = make_params();
    MPCConfig config = make_config();

    const int n_samples = 50;
    const double dt = config.dt;
    double samples[50 * 7];
    build_samples(samples, n_samples, dt);

    // Path A: Direct (setModelParams + setConfig + loadTrajectory)
    MecanumLTV direct;
    direct.setModelParams(params);
    direct.setConfig(config);
    int n_direct = direct.loadTrajectory(samples, n_samples, dt);

    if (n_direct <= 0) {
        std::printf("\n  Direct loadTrajectory failed");
        std::printf(" FAIL\n");
        return false;
    }

    // Save the direct windows to a temp file
    // We need to use the low-level API to save since MecanumLTV doesn't expose windows_
    // Instead, we'll create a second controller that precomputes and saves, then load.

    // Actually, we can use loadTrajectory to precompute, then use the low-level
    // mpc_precompute_all + mpc_save_windows to create the file, then test loadWindows.

    // Build the same reference path that loadTrajectory would produce
    RefNode path[n_samples];
    for (int k = 0; k < n_samples; ++k) {
        std::memset(&path[k], 0, sizeof(RefNode));
        path[k].t        = samples[k * 7 + 0];
        path[k].x_ref[0] = samples[k * 7 + 1];
        path[k].x_ref[1] = samples[k * 7 + 2];
        path[k].x_ref[2] = samples[k * 7 + 3];
        path[k].x_ref[3] = samples[k * 7 + 4];
        path[k].x_ref[4] = samples[k * 7 + 5];
        path[k].x_ref[5] = samples[k * 7 + 6];
        path[k].theta    = samples[k * 7 + 3];
        path[k].omega    = samples[k * 7 + 6];
    }

    // Pad with N hold nodes (matching loadTrajectory behavior)
    const int N = config.N;
    const int n_padded = n_samples + N;
    RefNode padded_path[n_padded];
    std::memcpy(padded_path, path, n_samples * sizeof(RefNode));

    RefNode hold_node = path[n_samples - 1];
    hold_node.x_ref[3] = 0.0;
    hold_node.x_ref[4] = 0.0;
    hold_node.x_ref[5] = 0.0;
    hold_node.omega = 0.0;
    std::memset(hold_node.u_ref, 0, NU * sizeof(double));
    for (int i = 0; i < N; ++i) {
        hold_node.t = path[n_samples - 1].t + (i + 1) * dt;
        padded_path[n_samples + i] = hold_node;
    }

    // Precompute and save
    compute_mecanum_jacobian(params);
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(padded_path, n_padded, params, config, n_windows);

    if (!windows || n_windows <= 0) {
        std::printf("\n  mpc_precompute_all failed");
        std::printf(" FAIL\n");
        return false;
    }

    const char* filename = "/tmp/test_load_windows.bin";
    int save_ret = mpc_save_windows(filename, windows, n_windows, config);
    delete[] windows;

    if (save_ret != 0) {
        std::printf("\n  mpc_save_windows failed");
        std::printf(" FAIL\n");
        return false;
    }

    // Path B: loadWindows from file
    MecanumLTV loaded;
    int n_loaded = loaded.loadWindows(filename);

    if (n_loaded <= 0) {
        std::printf("\n  loadWindows failed");
        std::remove(filename);
        std::printf(" FAIL\n");
        return false;
    }

    bool ok = true;

    // Check window counts match
    if (n_loaded != n_direct) {
        std::printf("\n  Window count mismatch: loaded=%d, direct=%d", n_loaded, n_direct);
        ok = false;
    }

    // Check dt accessor
    if (std::fabs(loaded.dt() - dt) > 1e-14) {
        std::printf("\n  dt() mismatch: loaded=%.6f, expected=%.6f", loaded.dt(), dt);
        ok = false;
    }

    // Compare solve results at multiple windows
    if (ok) {
        double x0[NX] = {0.01, -0.005, 0.02, 0.5, 0.0, 0.0};  // perturbed
        double u_direct[N_MAX * NU];
        double u_loaded[N_MAX * NU];

        int test_indices[] = {0, n_direct / 2, n_direct - 1};
        for (int idx : test_indices) {
            int iters_direct = direct.solve(idx, x0, u_direct);
            int iters_loaded = loaded.solve(idx, x0, u_loaded);

            if (iters_direct < 0 || iters_loaded < 0) {
                std::printf("\n  Solve failed at window %d (direct=%d, loaded=%d)",
                            idx, iters_direct, iters_loaded);
                ok = false;
                break;
            }

            int n_vars = config.N * NU;
            double max_diff = 0.0;
            for (int j = 0; j < n_vars; ++j) {
                double d = std::fabs(u_direct[j] - u_loaded[j]);
                if (d > max_diff) max_diff = d;
            }

            if (max_diff > 1e-12) {
                std::printf("\n  Solve output differs at window %d: max_diff=%.3e",
                            idx, max_diff);
                ok = false;
                break;
            }
        }
    }

    std::remove(filename);
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test: loadWindows with invalid file returns 0
// ---------------------------------------------------------------------------
static bool test_load_windows_invalid_file()
{
    std::printf("Test: loadWindows with invalid file ... ");

    MecanumLTV ctrl;
    int n = ctrl.loadWindows("/tmp/nonexistent_file_xyz.bin");

    bool ok = (n == 0);
    if (!ok) {
        std::printf("\n  Expected 0, got %d", n);
    }

    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=== test_load_windows ===\n\n");

    bool all_pass = true;
    all_pass &= test_load_windows_matches_direct();
    all_pass &= test_load_windows_invalid_file();

    std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
