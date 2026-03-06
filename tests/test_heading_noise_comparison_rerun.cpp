// test_heading_noise_comparison_rerun.cpp -- Compare offline LTV vs dynamic
// heading-lookup solvers under heading process noise.
//
// The offline (PRECOMPUTED) solver uses windows linearized at nominal headings.
// When heading deviates from the reference, the stored B_d matrices become
// stale.  The dynamic (HL_TRIG / HL_TABLE) solvers call generate_heading_schedule
// each step from the actual x_cur, so they always re-linearize around the
// true heading.
//
// Noise model: after each RK4 simulation step, zero-mean Gaussian noise is
// added directly to theta (x_cur[2]).  Control inputs are applied cleanly so
// any tracking difference is purely attributable to heading adaptation.
//
// Usage:  ./test_heading_noise_comparison_rerun [project.json] [trajectory_index]
//         defaults: turntest.json, 0

#include "mpc_types.h"
#include "mpc_offline.h"
#include "mpc_online.h"
#include "mecanum_model.h"
#include "discretizer.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"
#include "qp_solvers.h"
#include "heading_lookup.h"

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
#include <random>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Trajectory loading (same format as test_solver_comparison_rerun)
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

static void resample_trajectory(TrajoptTrajectory& traj, double target_dt)
{
    if (traj.times.size() < 2) return;

    double t_start = traj.times.front();
    double t_end   = traj.times.back();
    int n_new = static_cast<int>(std::ceil((t_end - t_start) / target_dt)) + 1;

    TrajoptTrajectory resampled;
    resampled.times.resize(n_new);
    resampled.states.resize(n_new);
    resampled.controls.resize(std::max(0, n_new - 1));

    int src = 0;
    int n_orig = static_cast<int>(traj.times.size());

    for (int k = 0; k < n_new; ++k) {
        double t = t_start + k * target_dt;
        if (t > t_end) t = t_end;
        resampled.times[k] = t;

        while (src < n_orig - 2 && traj.times[src + 1] < t)
            ++src;

        double t0 = traj.times[src];
        double t1 = traj.times[std::min(src + 1, n_orig - 1)];
        double alpha = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
        alpha = std::clamp(alpha, 0.0, 1.0);

        int s0 = src;
        int s1 = std::min(src + 1, n_orig - 1);

        for (int i = 0; i < 6; ++i)
            resampled.states[k][i] = (1.0 - alpha) * traj.states[s0][i]
                                   +        alpha  * traj.states[s1][i];

        if (k < n_new - 1) {
            int c0 = std::min(s0, static_cast<int>(traj.controls.size()) - 1);
            int c1 = std::min(s1, static_cast<int>(traj.controls.size()) - 1);
            for (int i = 0; i < 3; ++i)
                resampled.controls[k][i] = (1.0 - alpha) * traj.controls[c0][i]
                                         +        alpha  * traj.controls[c1][i];
        }
    }

    traj = std::move(resampled);
    std::printf("  Resampled to %d knots at dt=%.6f s\n", n_new, target_dt);
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
// RK4 plant simulation
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
enum class SolveMode { PRECOMPUTED, HL_TRIG, HL_TABLE, OFFLINE_OCP, HL_TRIG_OCP };

struct SolverRun {
    const char* name;
    QpSolverType type;
    SolveMode mode;
    SolverContext ctx;
    double x_cur[NX];
    std::vector<rerun::Vec2D> actual_pts;
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

    // Load and resample trajectory
    TrajoptTrajectory traj;
    TrajoptParams tp;
    if (!load_trajectory(project_file, traj_idx, traj, tp))
        return 1;

    {
        double min_dt = 1e9;
        for (size_t k = 0; k + 1 < traj.times.size(); ++k) {
            double dt = traj.times[k + 1] - traj.times[k];
            if (dt > 1e-12 && dt < min_dt) min_dt = dt;
        }
        resample_trajectory(traj, min_dt);
    }

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

    // Convert and pad trajectory
    std::vector<RefNode> ref_path;
    convert_to_refnodes(traj, ref_path);

    MPCConfig config{};
    config.N     = 30;
    config.dt    = traj.times[1] - traj.times[0];
    config.u_min = -1.0;
    config.u_max =  1.0;

    std::memset(config.Q, 0, sizeof(config.Q));
    config.Q[0 + NX * 0] = 100.0;
    config.Q[1 + NX * 1] = 100.0;
    config.Q[2 + NX * 2] = 1000.0;
    config.Q[3 + NX * 3] = 10.0;
    config.Q[4 + NX * 4] = 10.0;
    config.Q[5 + NX * 5] = 10.0;

    std::memset(config.R, 0, sizeof(config.R));
    for (int i = 0; i < NU; ++i)
        config.R[i + NU * i] = 0.005;

    for (int i = 0; i < NX * NX; ++i)
        config.Qf[i] = 0.0 * config.Q[i];

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

    // Precompute offline windows
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(
        ref_path.data(), n_path, params, config, n_windows);

    if (!windows || n_windows < 1) {
        std::fprintf(stderr, "Precomputation failed\n");
        return 1;
    }
    std::printf("Precomputed %d offline windows\n", n_windows);

    // Precompute heading-lookup data
    HeadingLookupData hl_data;
    heading_lookup_precompute(params, config.dt, hl_data);
    std::printf("Heading-lookup trig decomposition precomputed\n");

    HeadingTableData hl_table;
    heading_table_precompute(params, config.dt, HEADING_TABLE_M_DEFAULT, hl_table);
    std::printf("Heading table precomputed (M=%d)\n", HEADING_TABLE_M_DEFAULT);

    HeadingScheduleConfig sched_config = heading_schedule_config_from_params(params);

    // Build solver list
    struct SolverDef {
        const char* name;
        QpSolverType type;
        SolveMode mode;
        rerun::Color color;
    };

    std::vector<SolverDef> solver_defs = {
        // Offline LTV (precomputed windows, fixed heading linearization)
        {"offline_fista",      QpSolverType::FISTA,      SolveMode::PRECOMPUTED, rerun::Color(220,  80,  80)},

        // Dynamic heading — trig decomposition
        {"hl_trig_fista",      QpSolverType::FISTA,      SolveMode::HL_TRIG,    rerun::Color( 80, 220,  80)},

        // Dynamic heading — table interpolation
        {"hl_table_fista",     QpSolverType::FISTA,      SolveMode::HL_TABLE,   rerun::Color( 80, 180, 255)},
    };

#ifdef MPC_USE_HPIPM
    solver_defs.push_back({"offline_hpipm",     QpSolverType::HPIPM,     SolveMode::PRECOMPUTED,  rerun::Color(255, 100,  40)});
    solver_defs.push_back({"hl_trig_hpipm",     QpSolverType::HPIPM,     SolveMode::HL_TRIG,      rerun::Color(255, 200,  80)});
    solver_defs.push_back({"hl_table_hpipm",    QpSolverType::HPIPM,     SolveMode::HL_TABLE,     rerun::Color(180, 255, 200)});
    solver_defs.push_back({"offline_hpipm_ocp", QpSolverType::HPIPM_OCP, SolveMode::OFFLINE_OCP,  rerun::Color(255, 140, 100)});
    solver_defs.push_back({"hl_trig_hpipm_ocp", QpSolverType::HPIPM_OCP, SolveMode::HL_TRIG_OCP, rerun::Color(100, 255, 200)});
#endif
#ifdef MPC_USE_QPOASES
    solver_defs.push_back({"offline_qpoases", QpSolverType::QPOASES, SolveMode::PRECOMPUTED, rerun::Color(200,  80, 200)});
    solver_defs.push_back({"hl_trig_qpoases", QpSolverType::QPOASES, SolveMode::HL_TRIG,    rerun::Color(220, 140, 255)});
#endif

    std::printf("Enabled solvers:");
    for (auto& sd : solver_defs)
        std::printf(" %s", sd.name);
    std::printf("\n");

    // Heading process-noise table (shared across all solver runs so each sees
    // the same disturbance sequence and their x_cur trajectories stay coupled)
    constexpr double HEADING_NOISE_STDDEV = 0.5;  // rad (~2.9 deg per step)
    constexpr unsigned NOISE_SEED = 137;
    int n_sim = std::min(n_windows, n_path - 1);

    std::mt19937 rng(NOISE_SEED);
    std::normal_distribution<double> heading_noise_dist(0.0, HEADING_NOISE_STDDEV);
    std::vector<double> heading_noise_table(n_sim);
    for (int k = 0; k < n_sim; ++k)
        heading_noise_table[k] = heading_noise_dist(rng);

    std::printf("\nNoise model: heading θ += N(0, %.4f rad) per step\n",
                HEADING_NOISE_STDDEV);

    // Initialize solver runs
    int n_vars = config.N * NU;
    std::vector<SolverRun> runs(solver_defs.size());
    for (size_t s = 0; s < solver_defs.size(); ++s) {
        runs[s].name  = solver_defs[s].name;
        runs[s].type  = solver_defs[s].type;
        runs[s].mode  = solver_defs[s].mode;
        runs[s].color = solver_defs[s].color;
        solver_context_init(runs[s].ctx, n_vars);
        std::memcpy(runs[s].x_cur, ref_path[0].x_ref, NX * sizeof(double));
        runs[s].actual_pts.reserve(n_windows + 1);
        runs[s].actual_pts.push_back({static_cast<float>(ref_path[0].x_ref[0]),
                                      static_cast<float>(ref_path[0].x_ref[1])});
    }

    // Initialize Rerun
    auto rec = rerun::RecordingStream("mpc_heading_noise_comparison");
    rec.spawn().exit_on_failure();

    // Log static reference trajectory
    {
        std::vector<rerun::Vec2D> ref_pts;
        ref_pts.reserve(n_path);
        for (int k = 0; k < n_path; ++k)
            ref_pts.push_back({static_cast<float>(ref_path[k].x_ref[0]),
                               static_cast<float>(ref_path[k].x_ref[1])});
        rec.log_static("reference/trajectory",
                        rerun::LineStrips2D(rerun::LineStrip2D(ref_pts))
                            .with_colors({rerun::Color(100, 100, 255)}));

        // Also log reference heading over time
        std::vector<double> ref_thetas(n_path);
        for (int k = 0; k < n_path; ++k)
            ref_thetas[k] = ref_path[k].theta;
    }

    std::printf("Running %d steps with %zu solvers...\n", n_sim, runs.size());

    for (int k = 0; k < n_sim; ++k) {
        double t = ref_path[k].t;
        rec.set_time_seconds("sim_time", t);

        // Log shared heading noise for this step
        rec.log("noise/heading_rad", rerun::Scalars(heading_noise_table[k]));
        rec.log("reference/theta", rerun::Scalars(ref_path[k].theta));

        for (size_t s = 0; s < runs.size(); ++s) {
            auto& run = runs[s];
            std::string prefix = run.name;

            // Solve: dispatch to the appropriate solver
            QPSolution sol;
            switch (run.mode) {
                case SolveMode::PRECOMPUTED:
                    sol = mpc_solve_with_solver(
                        windows[k], run.x_cur, config, run.type, run.ctx);
                    break;
                case SolveMode::HL_TRIG:
                    sol = heading_lookup_solve_condensed(
                        hl_data, &ref_path[k], run.x_cur, config,
                        sched_config, run.type, run.ctx);
                    break;
                case SolveMode::HL_TABLE:
                    sol = heading_table_solve_condensed(
                        hl_table, &ref_path[k], run.x_cur, config,
                        sched_config, run.type, run.ctx);
                    break;
#ifdef MPC_USE_HPIPM
                case SolveMode::OFFLINE_OCP: {
                    // Re-discretize at reference headings, then solve via HPIPM OCP
                    struct timespec t0, t1;
                    clock_gettime(CLOCK_MONOTONIC, &t0);

                    double A_list_ocp[N_MAX * NX * NX];
                    double B_list_ocp[N_MAX * NX * NU];
                    double x_ref_con[(N_MAX + 1) * NX];
                    double u_ref_stk[N_MAX * NU];
                    double tAx[NX], tBu[NX];

                    for (int j = 0; j < config.N; ++j) {
                        int idx = k + j;
                        exact_discretize(ref_path[idx], ref_path[idx + 1], params,
                                         A_list_ocp + j * NX * NX,
                                         B_list_ocp + j * NX * NU);
                    }

                    std::memcpy(x_ref_con, ref_path[k].x_ref, NX * sizeof(double));
                    for (int j = 0; j < config.N; ++j) {
                        std::memcpy(u_ref_stk + j * NU, ref_path[k + j].u_ref, NU * sizeof(double));
                        mpc_linalg::gemv(NX, NX, A_list_ocp + j * NX * NX, x_ref_con + j * NX, tAx);
                        mpc_linalg::gemv(NX, NU, B_list_ocp + j * NX * NU, ref_path[k + j].u_ref, tBu);
                        for (int i = 0; i < NX; ++i)
                            x_ref_con[(j + 1) * NX + i] = tAx[i] + tBu[i];
                    }

                    std::memset(&sol, 0, sizeof(sol));
                    sol.n_iterations = hpipm_ocp_qp_solve(
                        A_list_ocp, B_list_ocp, config.Q, config.Qf, config.R,
                        x_ref_con, u_ref_stk, run.x_cur,
                        config.u_min, config.u_max, config.N,
                        sol.U, run.ctx.hpipm_ocp_ws);
                    std::memcpy(sol.u0, sol.U, NU * sizeof(double));

                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    sol.solve_time_ns = (t1.tv_sec - t0.tv_sec) * 1e9
                                      + (t1.tv_nsec - t0.tv_nsec);
                    break;
                }
                case SolveMode::HL_TRIG_OCP:
                    sol = heading_lookup_solve_ocp(
                        hl_data, &ref_path[k], run.x_cur, config,
                        sched_config, run.ctx);
                    break;
#else
                case SolveMode::OFFLINE_OCP:
                case SolveMode::HL_TRIG_OCP:
                    std::memset(&sol, 0, sizeof(sol));
                    break;
#endif
            }

            double solve_us = sol.solve_time_ns / 1000.0;
            run.solve_times_us.push_back(solve_us);

            // Compute errors against reference
            double err_pos = std::sqrt(
                (run.x_cur[0] - ref_path[k].x_ref[0]) * (run.x_cur[0] - ref_path[k].x_ref[0]) +
                (run.x_cur[1] - ref_path[k].x_ref[1]) * (run.x_cur[1] - ref_path[k].x_ref[1]));
            double err_heading = run.x_cur[2] - ref_path[k].x_ref[2];

            // Log per-solver metrics
            rec.log("metrics/solve_time_us/"  + prefix, rerun::Scalars(solve_us));
            rec.log("metrics/position_error/" + prefix, rerun::Scalars(err_pos));
            rec.log("metrics/heading_error/"  + prefix, rerun::Scalars(err_heading));
            rec.log("metrics/n_iterations/"   + prefix,
                    rerun::Scalars(static_cast<double>(sol.n_iterations)));

            for (int j = 0; j < NU; ++j)
                rec.log("control/u" + std::to_string(j) + "/" + prefix,
                        rerun::Scalars(sol.u0[j]));

            rec.log("trajectory/" + prefix + "_position",
                    rerun::Points2D({{static_cast<float>(run.x_cur[0]),
                                      static_cast<float>(run.x_cur[1])}})
                        .with_colors({run.color})
                        .with_radii({0.02f}));

            // Simulate forward with clean control (no actuator noise so that
            // any trajectory divergence is solely due to heading adaptation)
            rk4_step(run.x_cur, sol.u0, config.dt, params);

            // Inject heading process noise (same value for all solvers)
            run.x_cur[2] += heading_noise_table[k];

            // Log state
            rec.log("state/px/"    + prefix, rerun::Scalars(run.x_cur[0]));
            rec.log("state/py/"    + prefix, rerun::Scalars(run.x_cur[1]));
            rec.log("state/theta/" + prefix, rerun::Scalars(run.x_cur[2]));
            rec.log("state/vx/"    + prefix, rerun::Scalars(run.x_cur[3]));
            rec.log("state/vy/"    + prefix, rerun::Scalars(run.x_cur[4]));
            rec.log("state/omega/" + prefix, rerun::Scalars(run.x_cur[5]));

            run.actual_pts.push_back({static_cast<float>(run.x_cur[0]),
                                      static_cast<float>(run.x_cur[1])});
            rec.log("trajectory/" + prefix,
                    rerun::LineStrips2D(rerun::LineStrip2D(run.actual_pts))
                        .with_colors({run.color}));
        }

        if (k % 20 == 0 || k == n_sim - 1) {
            std::printf("  step %3d/%d", k, n_sim);
            for (auto& run : runs)
                std::printf("  %s=%.0fus", run.name, run.solve_times_us.back());
            std::printf("\n");
        }
    }

    // Summary table: solve times + final heading/position tracking
    std::printf("\n%-22s %8s %8s %8s %8s  %s\n",
                "Solver", "Mean(us)", "Med(us)", "Max(us)", "Min(us)", "Mode");
    std::printf("%-22s %8s %8s %8s %8s  %s\n",
                "----------------------", "--------", "-------", "-------", "-------", "----");

    auto mode_str = [](SolveMode m) -> const char* {
        switch (m) {
            case SolveMode::PRECOMPUTED:  return "offline";
            case SolveMode::HL_TRIG:      return "hl_trig";
            case SolveMode::HL_TABLE:     return "hl_table";
            case SolveMode::OFFLINE_OCP:  return "offline_ocp";
            case SolveMode::HL_TRIG_OCP:  return "hl_trig_ocp";
        }
        return "?";
    };

    for (auto& run : runs) {
        auto& times = run.solve_times_us;
        if (times.empty()) continue;

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / static_cast<double>(times.size());

        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];
        double max_t = sorted.back();
        double min_t = sorted.front();

        std::printf("%-22s %8.1f %8.1f %8.1f %8.1f  %s\n",
                    run.name, mean, median, max_t, min_t, mode_str(run.mode));
    }

    // Final position and heading errors
    std::printf("\n%-22s %12s %12s\n", "Solver", "Final|pos|(m)", "Final|θ|(rad)");
    std::printf("%-22s %12s %12s\n", "----------------------", "-------------", "-------------");
    for (size_t s = 0; s < runs.size(); ++s) {
        const auto& run = runs[s];
        double ep = std::sqrt(
            (run.x_cur[0] - ref_path[n_sim].x_ref[0]) * (run.x_cur[0] - ref_path[n_sim].x_ref[0]) +
            (run.x_cur[1] - ref_path[n_sim].x_ref[1]) * (run.x_cur[1] - ref_path[n_sim].x_ref[1]));
        double eth = std::fabs(run.x_cur[2] - ref_path[n_sim].x_ref[2]);
        std::printf("%-22s %12.4f %12.4f\n", run.name, ep, eth);
    }

    // Cleanup
    for (auto& run : runs)
        solver_context_free(run.ctx);
    delete[] windows;

    std::printf("\nDone. View results in the Rerun viewer.\n");
    return 0;
}
