// test_solver_comparison_rerun.cpp -- Run all enabled QP solvers on the same
// trajectory and visualize solve times + trajectories side-by-side with rerun.
//
// Usage:  ./test_solver_comparison_rerun [project.json] [trajectory_index]
//         defaults: turntest.json, 0

#include "mpc_types.h"
#include "mpc_offline.h"
#include "mpc_online.h"
#include "mecanum_model.h"
#include "discretizer.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"
#include "qp_solvers.h"

#include <rerun.hpp>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Trajectory loading (shared with test_trajopt_rerun.cpp)
// ---------------------------------------------------------------------------
struct TrajoptTrajectory {
    std::vector<double> times;
    std::vector<std::array<double, 6>> states;   // [vx, vy, omega, px, py, theta]
    std::vector<std::array<double, 3>> controls;  // [drive, strafe, turn]
};

struct TrajoptParams {
    double mass, inertia, wheel_radius, lx, ly, stall_torque, free_speed;
};

static bool load_trajectory(const std::string& path, int traj_idx,
                            TrajoptTrajectory& traj, TrajoptParams& params)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }

    json doc = json::parse(f);
    auto& rp = doc["robotParams"];
    params.mass         = rp["mass"].get<double>();
    params.inertia      = rp["inertia"].get<double>();
    params.wheel_radius = rp["wheel_radius"].get<double>();
    params.lx           = rp["lx"].get<double>();
    params.ly           = rp["ly"].get<double>();
    params.stall_torque = rp["t_max"].get<double>();
    params.free_speed   = rp["w_max"].get<double>();

    auto& trajs = doc["trajectories"];
    if (traj_idx < 0 || traj_idx >= static_cast<int>(trajs.size())) {
        std::fprintf(stderr, "Trajectory index %d out of range\n", traj_idx);
        return false;
    }

    auto& tj = trajs[traj_idx];
    std::printf("Loading trajectory \"%s\"\n", tj.value("name", "unnamed").c_str());

    if (!tj.contains("trajectory") || tj["trajectory"].is_null()) {
        std::fprintf(stderr, "No solved trajectory data\n");
        return false;
    }

    auto& td = tj["trajectory"];
    int n_knots    = static_cast<int>(td["times"].size());
    int n_controls = static_cast<int>(td["controls"].size());

    traj.times.resize(n_knots);
    traj.states.resize(n_knots);
    traj.controls.resize(n_controls);

    for (int k = 0; k < n_knots; ++k) {
        traj.times[k] = td["times"][k].get<double>();
        for (int i = 0; i < 6; ++i)
            traj.states[k][i] = td["states"][k][i].get<double>();
    }
    for (int k = 0; k < n_controls; ++k) {
        for (int i = 0; i < 3; ++i)
            traj.controls[k][i] = td["controls"][k][i].get<double>();
    }

    std::printf("  %d knots, %d controls, %.3f s\n",
                n_knots, n_controls, traj.times.back());
    return true;
}

static void convert_to_refnodes(const TrajoptTrajectory& traj,
                                 std::vector<RefNode>& ref_path)
{
    int n = static_cast<int>(traj.times.size());
    ref_path.resize(n);

    for (int k = 0; k < n; ++k) {
        RefNode& r = ref_path[k];
        std::memset(&r, 0, sizeof(RefNode));

        const auto& s = traj.states[k];
        r.x_ref[0] = s[3]; r.x_ref[1] = s[4]; r.x_ref[2] = s[5];
        r.x_ref[3] = s[0]; r.x_ref[4] = s[1]; r.x_ref[5] = s[2];
        r.theta = s[5];
        r.omega = s[2];
        r.t     = traj.times[k];

        if (k < static_cast<int>(traj.controls.size())) {
            double d = traj.controls[k][0], s2 = traj.controls[k][1], t = traj.controls[k][2];
            r.u_ref[0] = d - s2 - t;
            r.u_ref[1] = d + s2 + t;
            r.u_ref[2] = d + s2 - t;
            r.u_ref[3] = d - s2 + t;
        }
    }
}

// ---------------------------------------------------------------------------
// RK4 simulation step
// ---------------------------------------------------------------------------
static void xdot(const double* x, const double* u, const ModelParams& params,
                 double* dx)
{
    double Ac[NX * NX], Bc[NX * NU];
    continuous_dynamics(x[2], params, Ac, Bc);

    double Bu[NX];
    mpc_linalg::gemv(NX, NX, Ac, x, dx);
    mpc_linalg::gemv(NX, NU, Bc, u, Bu);
    for (int i = 0; i < NX; ++i) dx[i] += Bu[i];
}

static void rk4_step(double* x, const double* u, double dt,
                     const ModelParams& params)
{
    double k1[NX], k2[NX], k3[NX], k4[NX], tmp[NX];

    xdot(x, u, params, k1);
    for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
    xdot(tmp, u, params, k2);
    for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
    xdot(tmp, u, params, k3);
    for (int i = 0; i < NX; ++i) tmp[i] = x[i] + dt * k3[i];
    xdot(tmp, u, params, k4);

    for (int i = 0; i < NX; ++i)
        x[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

// ---------------------------------------------------------------------------
// Per-solver run state
// ---------------------------------------------------------------------------
struct SolverRun {
    const char* name;
    QpSolverType type;
    SolverContext ctx;
    double x_cur[NX];
    std::vector<rerun::Position2D> actual_pts;
    std::vector<double> solve_times_us;
    rerun::Color color;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string project_file = "/home/dolphinpod/dev/mecanum_trajopt/projects/turntest.json";
    int traj_idx = 0;
    if (argc > 1) project_file = argv[1];
    if (argc > 2) traj_idx = std::atoi(argv[2]);

    // Load trajectory
    TrajoptTrajectory traj;
    TrajoptParams tp;
    if (!load_trajectory(project_file, traj_idx, traj, tp))
        return 1;

    // Model params
    ModelParams params{};
    params.mass            = tp.mass;
    params.inertia         = tp.inertia;
    params.damping_linear  = 0.0;
    params.damping_angular = 0.0;
    params.wheel_radius    = tp.wheel_radius;
    params.lx              = tp.lx;
    params.ly              = tp.ly;
    params.stall_torque    = tp.stall_torque;
    params.free_speed      = tp.free_speed;
    compute_mecanum_jacobian(params);

    // Convert trajectory
    std::vector<RefNode> ref_path;
    convert_to_refnodes(traj, ref_path);

    // MPC config
    MPCConfig config{};
    config.N     = 30;
    config.dt    = traj.times[1] - traj.times[0];
    config.u_min = -1.0;
    config.u_max =  1.0;

    std::memset(config.Q, 0, sizeof(config.Q));
    config.Q[0 + NX * 0] = 100.0;
    config.Q[1 + NX * 1] = 100.0;
    config.Q[2 + NX * 2] = 1000.0;
    config.Q[3 + NX * 3] = 100.0;
    config.Q[4 + NX * 4] = 100.0;
    config.Q[5 + NX * 5] = 10.0;

    std::memset(config.R, 0, sizeof(config.R));
    for (int i = 0; i < NU; ++i)
        config.R[i + NU * i] = 0.005;

    for (int i = 0; i < NX * NX; ++i)
        config.Qf[i] = 0.0 * config.Q[i];

    // Pad trajectory
    {
        const RefNode& last = ref_path.back();
        for (int i = 1; i <= 2 * config.N; ++i) {
            RefNode hold{};
            hold.x_ref[0] = last.x_ref[0];
            hold.x_ref[1] = last.x_ref[1];
            hold.x_ref[2] = last.x_ref[2];
            hold.theta = last.theta;
            hold.omega = 0.0;
            hold.t     = last.t + i * config.dt;
            ref_path.push_back(hold);
        }
    }
    int n_path = static_cast<int>(ref_path.size());

    // Precompute windows (shared across all solvers)
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(
        ref_path.data(), n_path, params, config, n_windows);

    if (!windows || n_windows < 1) {
        std::fprintf(stderr, "Precomputation failed\n");
        return 1;
    }
    std::printf("Precomputed %d windows\n", n_windows);

    // Build list of enabled solvers
    struct SolverDef {
        const char* name;
        QpSolverType type;
        rerun::Color color;
    };

    std::vector<SolverDef> solver_defs = {
        {"fista",      QpSolverType::FISTA,      rerun::Color(0, 200, 0)},       // green
        {"active_set", QpSolverType::ACTIVE_SET,  rerun::Color(0, 150, 255)},     // blue
    };

#ifdef MPC_USE_HPIPM
    solver_defs.push_back({"hpipm",   QpSolverType::HPIPM,   rerun::Color(255, 165, 0)});  // orange
#endif
#ifdef MPC_USE_QPOASES
    solver_defs.push_back({"qpoases", QpSolverType::QPOASES, rerun::Color(180, 0, 255)});  // purple
#endif

    std::printf("Enabled solvers:");
    for (auto& sd : solver_defs)
        std::printf(" %s", sd.name);
    std::printf("\n");

    // Initialize solver runs
    int n_vars = config.N * NU;
    std::vector<SolverRun> runs(solver_defs.size());
    for (size_t s = 0; s < solver_defs.size(); ++s) {
        runs[s].name  = solver_defs[s].name;
        runs[s].type  = solver_defs[s].type;
        runs[s].color = solver_defs[s].color;
        solver_context_init(runs[s].ctx, n_vars);
        std::memcpy(runs[s].x_cur, ref_path[0].x_ref, NX * sizeof(double));
        runs[s].actual_pts.reserve(n_windows + 1);
        runs[s].actual_pts.push_back({static_cast<float>(ref_path[0].x_ref[0]),
                                      static_cast<float>(ref_path[0].x_ref[1])});
    }

    // Initialize rerun
    auto rec = rerun::RecordingStream("mpc_solver_comparison");
    rec.spawn().exit_on_failure();

    // Log reference trajectory (static)
    {
        std::vector<rerun::Position2D> ref_pts;
        ref_pts.reserve(n_path);
        for (int k = 0; k < n_path; ++k)
            ref_pts.push_back({static_cast<float>(ref_path[k].x_ref[0]),
                               static_cast<float>(ref_path[k].x_ref[1])});
        rec.log_static("reference/trajectory",
                        rerun::LineStrips2D({ref_pts})
                            .with_colors({rerun::Color(100, 100, 255)}));
    }

    // Main simulation loop
    int n_sim = std::min(n_windows, n_path - 1);
    std::printf("\nRunning %d steps with %zu solvers...\n",
                n_sim, runs.size());

    for (int k = 0; k < n_sim; ++k) {
        double t = ref_path[k].t;
        rec.set_time_seconds("sim_time", t);

        for (size_t s = 0; s < runs.size(); ++s) {
            auto& run = runs[s];
            std::string prefix = run.name;

            // Solve
            QPSolution sol = mpc_solve_with_solver(
                windows[k], run.x_cur, config, run.type, run.ctx);

            double solve_us = sol.solve_time_ns / 1000.0;
            run.solve_times_us.push_back(solve_us);

            // Position error
            double err_pos = std::sqrt(
                (run.x_cur[0] - ref_path[k].x_ref[0]) * (run.x_cur[0] - ref_path[k].x_ref[0]) +
                (run.x_cur[1] - ref_path[k].x_ref[1]) * (run.x_cur[1] - ref_path[k].x_ref[1]));

            // Log metrics under shared entities so solvers overlay
            rec.log("metrics/solve_time_us/" + prefix,
                    rerun::Scalars(solve_us));
            rec.log("metrics/position_error/" + prefix,
                    rerun::Scalars(err_pos));
            rec.log("metrics/n_iterations/" + prefix,
                    rerun::Scalars(static_cast<double>(sol.n_iterations)));

            // Log solved control trajectory for first 4 inputs
            for (int j = 0; j < NU; ++j) {
                rec.log("control/u" + std::to_string(j) + "/" + prefix,
                        rerun::Scalars(sol.u0[j]));
            }

            // Log current robot position as a point
            rec.log("trajectory/" + prefix + "_position",
                    rerun::Points2D({{static_cast<float>(run.x_cur[0]),
                                      static_cast<float>(run.x_cur[1])}})
                        .with_colors({run.color})
                        .with_radii({0.02f}));

            // Simulate forward
            rk4_step(run.x_cur, sol.u0, config.dt, params);

            run.actual_pts.push_back({static_cast<float>(run.x_cur[0]),
                                      static_cast<float>(run.x_cur[1])});

            // Log growing actual trajectory
            rec.log("trajectory/" + prefix,
                    rerun::LineStrips2D({run.actual_pts})
                        .with_colors({run.color}));

            // Log per-state solved trajectory
            rec.log("state/px/" + prefix, rerun::Scalars(run.x_cur[0]));
            rec.log("state/py/" + prefix, rerun::Scalars(run.x_cur[1]));
            rec.log("state/theta/" + prefix, rerun::Scalars(run.x_cur[2]));
            rec.log("state/vx/" + prefix, rerun::Scalars(run.x_cur[3]));
            rec.log("state/vy/" + prefix, rerun::Scalars(run.x_cur[4]));
            rec.log("state/omega/" + prefix, rerun::Scalars(run.x_cur[5]));
        }

        if (k % 20 == 0 || k == n_sim - 1) {
            std::printf("  step %3d/%d", k, n_sim);
            for (auto& run : runs)
                std::printf("  %s=%.0fus", run.name, run.solve_times_us.back());
            std::printf("\n");
        }
    }

    // Print summary table
    std::printf("\n%-12s %8s %8s %8s %8s\n",
                "Solver", "Mean(us)", "Med(us)", "Max(us)", "Min(us)");
    std::printf("%-12s %8s %8s %8s %8s\n",
                "------", "--------", "-------", "-------", "-------");

    for (auto& run : runs) {
        auto& times = run.solve_times_us;
        if (times.empty()) continue;

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();

        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];
        double max_t = sorted.back();
        double min_t = sorted.front();

        std::printf("%-12s %8.1f %8.1f %8.1f %8.1f\n",
                    run.name, mean, median, max_t, min_t);
    }

    // Cleanup
    for (auto& run : runs)
        solver_context_free(run.ctx);
    delete[] windows;

    std::printf("\nDone. Check the rerun viewer for visualization.\n");
    return 0;
}
