// test_timing.cpp -- benchmark for the online MPC solver
//
// Reports solve-time statistics for unconstrained, constrained, and
// varying-horizon cases.  No pass/fail -- purely informational.

#include "mpc_types.h"
#include "mpc_offline.h"
#include "mpc_online.h"
#include "mecanum_model.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <time.h>

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

static void set_diag(double* M, int n, const double* diag_vals)
{
    std::memset(M, 0, n * n * sizeof(double));
    for (int i = 0; i < n; ++i)
        M[i + i * n] = diag_vals[i];
}

static MPCConfig make_config(int N, double dt, double u_min, double u_max)
{
    MPCConfig cfg{};
    cfg.N     = N;
    cfg.dt    = dt;
    cfg.u_min = u_min;
    cfg.u_max = u_max;

    double q_diag[NX] = {10.0, 10.0, 5.0, 1.0, 1.0, 0.5};
    double r_diag[NU] = {0.1, 0.1, 0.1, 0.1};
    double qf_diag[NX];
    for (int i = 0; i < NX; ++i) qf_diag[i] = 2.0 * q_diag[i];

    set_diag(cfg.Q, NX, q_diag);
    set_diag(cfg.R, NU, r_diag);
    set_diag(cfg.Qf, NX, qf_diag);

    return cfg;
}

// Build a straight-line reference path along x at 0.5 m/s
static RefNode* make_ref_path(int n_path, double dt)
{
    RefNode* path = new RefNode[n_path];
    const double vx = 0.5;
    for (int k = 0; k < n_path; ++k) {
        std::memset(&path[k], 0, sizeof(RefNode));
        path[k].x_ref[0] = k * dt * vx;   // px
        path[k].x_ref[3] = vx;             // vx
        path[k].theta     = 0.0;
        path[k].omega     = 0.0;
        path[k].t         = k * dt;
    }
    return path;
}

struct TimingStats {
    double min_us;
    double mean_us;
    double median_us;
    double p95_us;
    double p99_us;
    double max_us;
};

static TimingStats compute_stats(double* times_ns, int n)
{
    std::sort(times_ns, times_ns + n);

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += times_ns[i];

    TimingStats s{};
    s.min_us    = times_ns[0] / 1000.0;
    s.mean_us   = (sum / n) / 1000.0;
    s.median_us = times_ns[n / 2] / 1000.0;
    s.p95_us    = times_ns[(int)(n * 0.95)] / 1000.0;
    s.p99_us    = times_ns[(int)(n * 0.99)] / 1000.0;
    s.max_us    = times_ns[n - 1] / 1000.0;
    return s;
}

static void print_stats(const char* label, const TimingStats& s)
{
    std::printf("  %-12s  min=%7.1f  mean=%7.1f  median=%7.1f  "
                "p95=%7.1f  p99=%7.1f  max=%7.1f  (us)\n",
                label, s.min_us, s.mean_us, s.median_us,
                s.p95_us, s.p99_us, s.max_us);
}

// ---------------------------------------------------------------------------
// Benchmark 1: Unconstrained solve timing
// ---------------------------------------------------------------------------
static void bench_unconstrained(const PrecomputedWindow* windows,
                                const MPCConfig& config,
                                const RefNode* ref_path)
{
    std::printf("\n--- Benchmark 1: Unconstrained solve (N=%d, 10000 iters) ---\n",
                config.N);

    const int N_RUNS = 10000;
    double times_ns[N_RUNS];
    BoxQPWorkspace ws{};

    // x0 = x_ref_0 + small perturbation
    double x0[NX];
    std::memcpy(x0, ref_path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.01;

    for (int i = 0; i < N_RUNS; ++i) {
        QPSolution sol = mpc_solve_online(windows[0], x0, config, ws);
        times_ns[i] = sol.solve_time_ns;
    }

    TimingStats s = compute_stats(times_ns, N_RUNS);
    print_stats("unconstr", s);
}

// ---------------------------------------------------------------------------
// Benchmark 2: Constrained solve timing
// ---------------------------------------------------------------------------
static void bench_constrained(const PrecomputedWindow* windows,
                              const MPCConfig& config_wide,
                              const RefNode* ref_path,
                              const ModelParams& params)
{
    // Use tight bounds to activate constraints
    MPCConfig config_tight = config_wide;
    config_tight.u_min = -1.0;
    config_tight.u_max =  1.0;

    std::printf("\n--- Benchmark 2: Constrained solve (N=%d, u=[%.1f,%.1f], 10000 iters) ---\n",
                config_tight.N, config_tight.u_min, config_tight.u_max);

    const int N_RUNS = 10000;
    double times_ns[N_RUNS];
    int iterations[N_RUNS];
    BoxQPWorkspace ws{};

    // x0 with larger perturbation to make constraints active
    double x0[NX];
    std::memcpy(x0, ref_path[0].x_ref, NX * sizeof(double));
    x0[0] += 0.1;
    x0[1] += 0.05;
    x0[2] += 0.1;
    x0[3] += 0.5;

    for (int i = 0; i < N_RUNS; ++i) {
        QPSolution sol = mpc_solve_online(windows[0], x0, config_tight, ws);
        times_ns[i]   = sol.solve_time_ns;
        iterations[i] = sol.n_iterations;
    }

    TimingStats s = compute_stats(times_ns, N_RUNS);
    print_stats("constr", s);

    // Mean iterations
    double sum_iter = 0.0;
    for (int i = 0; i < N_RUNS; ++i) sum_iter += iterations[i];
    std::printf("  mean fista iterations: %.1f\n", sum_iter / N_RUNS);
}

// ---------------------------------------------------------------------------
// Benchmark 3: Varying horizon
// ---------------------------------------------------------------------------
static void bench_varying_horizon(const ModelParams& params)
{
    std::printf("\n--- Benchmark 3: Varying horizon (1000 iters each) ---\n");
    std::printf("  %5s  %10s\n", "N", "mean (us)");
    std::printf("  %5s  %10s\n", "-----", "----------");

    const int horizons[] = {5, 10, 15, 20};
    const int n_horizons = 4;
    const int N_RUNS = 1000;
    const double dt = 0.02;

    for (int h = 0; h < n_horizons; ++h) {
        int N = horizons[h];
        int n_path = N + 10;

        MPCConfig cfg = make_config(N, dt, -1.0, 1.0);
        RefNode* ref = make_ref_path(n_path, dt);

        int n_win = 0;
        PrecomputedWindow* windows = mpc_precompute_all(ref, n_path, params, cfg, n_win);

        BoxQPWorkspace ws{};
        double x0[NX];
        std::memcpy(x0, ref[0].x_ref, NX * sizeof(double));
        x0[0] += 0.01;

        double times_ns[N_RUNS];
        for (int i = 0; i < N_RUNS; ++i) {
            QPSolution sol = mpc_solve_online(windows[0], x0, cfg, ws);
            times_ns[i] = sol.solve_time_ns;
        }

        double sum = 0.0;
        for (int i = 0; i < N_RUNS; ++i) sum += times_ns[i];
        double mean_us = (sum / N_RUNS) / 1000.0;

        std::printf("  %5d  %10.1f\n", N, mean_us);

        delete[] windows;
        delete[] ref;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=== test_timing: MPC online solver benchmark ===\n");

    ModelParams params = make_params();

    // Full-size setup for benchmarks 1 and 2
    const int N = 20;
    const double dt = 0.02;
    const int n_path = 50;

    MPCConfig config = make_config(N, dt, -1.0, 1.0);
    RefNode* ref_path = make_ref_path(n_path, dt);

    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(ref_path, n_path, params,
                                                     config, n_windows);

    std::printf("Precomputed %d windows (N=%d, n_path=%d)\n",
                n_windows, N, n_path);

    bench_unconstrained(windows, config, ref_path);
    bench_constrained(windows, config, ref_path, params);
    bench_varying_horizon(params);

    delete[] windows;
    delete[] ref_path;

    std::printf("\nBenchmark complete.\n");
    return 0;
}
