// test_trajopt_rerun.cpp -- MPC tracking of mecanum_trajopt trajectories,
// visualized with rerun.
//
// Reads a trajectory JSON from ~/dev/mecanum_trajopt/projects/, converts
// the state/control format, runs closed-loop MPC tracking, and logs
// reference vs actual trajectories + metrics to the rerun viewer.
//
// Usage:  ./test_trajopt_rerun [project.json] [trajectory_index]
//         defaults: base.json, 0

#include "mpc_types.h"
#include "mpc_offline.h"
#include "mpc_online.h"
#include "mecanum_model.h"
#include "discretizer.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"
#include "qp_solvers.h"
#include "heading_lookup.h"
#include "ipm_solver.h"

#include <rerun.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Trajectory loading from mecanum_trajopt JSON
// ---------------------------------------------------------------------------
struct TrajoptTrajectory {
    std::vector<double> times;
    // states[k] = [vx, vy, omega, px, py, theta]  (trajopt ordering)
    std::vector<std::array<double, 6>> states;
    // controls[k] = [drive, strafe, turn]
    std::vector<std::array<double, 3>> controls;
};

struct TrajoptParams {
    double mass;
    double inertia;
    double wheel_radius;
    double lx, ly;
    double stall_torque;   // t_max
    double free_speed;     // w_max
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

    // Robot parameters
    auto& rp = doc["robotParams"];
    params.mass         = rp["mass"].get<double>();
    params.inertia      = rp["inertia"].get<double>();
    params.wheel_radius = rp["wheel_radius"].get<double>();
    params.lx           = rp["lx"].get<double>();
    params.ly           = rp["ly"].get<double>();
    params.stall_torque = rp["t_max"].get<double>();
    params.free_speed   = rp["w_max"].get<double>();

    // Find the requested trajectory
    auto& trajs = doc["trajectories"];
    if (traj_idx < 0 || traj_idx >= static_cast<int>(trajs.size())) {
        std::fprintf(stderr, "Trajectory index %d out of range (0..%d)\n",
                     traj_idx, static_cast<int>(trajs.size()) - 1);
        return false;
    }

    auto& tj = trajs[traj_idx];
    std::string name = tj.value("name", "unnamed");
    std::printf("Loading trajectory \"%s\" from %s\n", name.c_str(), path.c_str());

    if (!tj.contains("trajectory") || tj["trajectory"].is_null()) {
        std::fprintf(stderr, "Trajectory \"%s\" has no solved data\n", name.c_str());
        return false;
    }

    auto& td = tj["trajectory"];
    auto& times_j   = td["times"];
    auto& states_j   = td["states"];
    auto& controls_j = td["controls"];

    int n_knots    = static_cast<int>(times_j.size());
    int n_controls = static_cast<int>(controls_j.size());

    traj.times.resize(n_knots);
    traj.states.resize(n_knots);
    traj.controls.resize(n_controls);

    for (int k = 0; k < n_knots; ++k) {
        traj.times[k] = times_j[k].get<double>();
        for (int i = 0; i < 6; ++i)
            traj.states[k][i] = states_j[k][i].get<double>();
    }
    for (int k = 0; k < n_controls; ++k) {
        for (int i = 0; i < 3; ++i)
            traj.controls[k][i] = controls_j[k][i].get<double>();
    }

    std::printf("  %d knots, %d controls, total_time=%.3f s\n",
                n_knots, n_controls, traj.times.back());
    return true;
}

// ---------------------------------------------------------------------------
// Resample trajectory to uniform dt via linear interpolation
// ---------------------------------------------------------------------------
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

    int src = 0;  // current index in original trajectory
    int n_orig = static_cast<int>(traj.times.size());

    for (int k = 0; k < n_new; ++k) {
        double t = t_start + k * target_dt;
        if (t > t_end) t = t_end;
        resampled.times[k] = t;

        // Advance src so traj.times[src] <= t < traj.times[src+1]
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

// ---------------------------------------------------------------------------
// Convert trajopt format to MPC RefNode array
// ---------------------------------------------------------------------------
// trajopt state: [vx, vy, omega, px, py, theta]  (field-frame velocities)
// MPC state:     [px, py, theta, vx, vy, omega]
//
// trajopt control: [drive, strafe, turn]
// MPC control:     [d_FL, d_FR, d_RL, d_RR]
//   FL = drive - strafe - turn
//   FR = drive + strafe + turn
//   RL = drive + strafe - turn
//   RR = drive - strafe + turn

static void convert_to_refnodes(const TrajoptTrajectory& traj,
                                 std::vector<RefNode>& ref_path)
{
    int n = static_cast<int>(traj.times.size());
    ref_path.resize(n);

    for (int k = 0; k < n; ++k) {
        RefNode& r = ref_path[k];
        std::memset(&r, 0, sizeof(RefNode));

        const auto& s = traj.states[k];
        // Reorder: [vx,vy,omega,px,py,theta] -> [px,py,theta,vx,vy,omega]
        r.x_ref[0] = s[3];  // px
        r.x_ref[1] = s[4];  // py
        r.x_ref[2] = s[5];  // theta
        r.x_ref[3] = s[0];  // vx (field frame)
        r.x_ref[4] = s[1];  // vy (field frame)
        r.x_ref[5] = s[2];  // omega

        r.theta = s[5];
        r.omega = s[2];
        r.t     = traj.times[k];

        // Convert controls (last knot has no control, use zeros)
        if (k < static_cast<int>(traj.controls.size())) {
            double drive  = traj.controls[k][0];
            double strafe = traj.controls[k][1];
            double turn   = traj.controls[k][2];
            r.u_ref[0] = drive - strafe - turn;   // FL
            r.u_ref[1] = drive + strafe + turn;    // FR
            r.u_ref[2] = drive + strafe - turn;    // RL
            r.u_ref[3] = drive - strafe + turn;    // RR
        }
    }
}

// ---------------------------------------------------------------------------
// RK4 simulation step (using continuous dynamics)
// ---------------------------------------------------------------------------
static void xdot(const double* x, const double* u, const ModelParams& params,
                 double* dx)
{
    double Ac[NX * NX], Bc[NX * NU];
    continuous_dynamics(x[2], params, Ac, Bc);

    double Bu[NX];
    mpc_linalg::gemv(NX, NX, Ac, x, dx);
    mpc_linalg::gemv(NX, NU, Bc, u, Bu);
    for (int i = 0; i < NX; ++i)
        dx[i] += Bu[i];
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
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Parse arguments
    std::string project_file = "/home/dolphinpod/dev/mecanum_trajopt/projects/turntest.json";
    int traj_idx = 0;
    if (argc > 1) project_file = argv[1];
    if (argc > 2) traj_idx = std::atoi(argv[2]);

    // Load trajectory
    TrajoptTrajectory traj;
    TrajoptParams tp;
    if (!load_trajectory(project_file, traj_idx, traj, tp))
        return 1;

    // Resample to uniform dt (use the minimum segment dt)
    {
        double min_dt = 1e9;
        for (size_t k = 0; k + 1 < traj.times.size(); ++k) {
            double dt = traj.times[k + 1] - traj.times[k];
            if (dt > 1e-12 && dt < min_dt) min_dt = dt;
        }
        resample_trajectory(traj, min_dt);
    }

    // Set up MPC model params (matching trajopt)
    ModelParams params{};
    params.mass            = tp.mass;
    params.inertia         = tp.inertia;
    params.damping_linear  = 0.0;   // trajopt doesn't model viscous drag in base solver
    params.damping_angular = 0.0;
    params.wheel_radius    = tp.wheel_radius;
    params.lx              = tp.lx;
    params.ly              = tp.ly;
    params.stall_torque    = tp.stall_torque;
    params.free_speed      = tp.free_speed;
    compute_mecanum_jacobian(params);

    std::printf("Robot: mass=%.1f  inertia=%.2f  r=%.3f  lx=%.2f  ly=%.2f  "
                "stall=%.2f  free=%.1f\n",
                params.mass, params.inertia, params.wheel_radius,
                params.lx, params.ly, params.stall_torque, params.free_speed);

    // Convert trajectory
    std::vector<RefNode> ref_path;
    convert_to_refnodes(traj, ref_path);
    int n_path_orig = static_cast<int>(ref_path.size());

    // MPC config
    MPCConfig config{};
    config.N     = 30;
    config.dt    = traj.times[1] - traj.times[0];  // use trajectory dt
    config.u_min = -1.0;
    config.u_max =  1.0;

    // Q = diag(10, 10, 5, 1, 1, 0.5)
    std::memset(config.Q, 0, sizeof(config.Q));
    config.Q[0 + NX * 0] = 300.0;  // px
    config.Q[1 + NX * 1] = 300.0;  // py
    config.Q[2 + NX * 2] =  300.0;  // theta
    config.Q[3 + NX * 3] =  10.0;  // vx
    config.Q[4 + NX * 4] =  10.0;  // vy
    config.Q[5 + NX * 5] =  10.0;  // omega

    // R = diag(0.01, 0.01, 0.01, 0.01)
    std::memset(config.R, 0, sizeof(config.R));
    for (int i = 0; i < NU; ++i)
        config.R[i + NU * i] = 0.005;

    // Qf = 2*Q
    for (int i = 0; i < NX * NX; ++i)
        config.Qf[i] = 0.0 * config.Q[i];

    std::printf("MPC: N=%d  dt=%.4f  u_bounds=[%.1f, %.1f]\n",
                config.N, config.dt, config.u_min, config.u_max);

    // Pad with 2*N hold-at-final-position nodes so the sim can run a full
    // horizon past the last trajectory knot.  Final position is held,
    // velocities and controls are zero.
    {
        const RefNode& last = ref_path.back();
        double dt_pad = config.dt;
        for (int i = 1; i <= 2 * config.N; ++i) {
            RefNode hold{};
            hold.x_ref[0] = last.x_ref[0];  // px
            hold.x_ref[1] = last.x_ref[1];  // py
            hold.x_ref[2] = last.x_ref[2];  // theta
            // vx, vy, omega = 0 (already zero-initialized)
            hold.theta = last.theta;
            hold.omega = 0.0;
            hold.t     = last.t + i * dt_pad;
            // u_ref all zeros (already zero-initialized)
            ref_path.push_back(hold);
        }
    }
    int n_path = static_cast<int>(ref_path.size());

    // Offline precomputation
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(
        ref_path.data(), n_path, params, config, n_windows);

    if (!windows || n_windows < 1) {
        std::fprintf(stderr, "Precomputation failed (n_windows=%d)\n", n_windows);
        return 1;
    }
    std::printf("Precomputed %d windows\n", n_windows);

    // Precompute heading-lookup data
    HeadingLookupData hl_data;
    heading_lookup_precompute(params, config.dt, hl_data);
    HeadingScheduleConfig sched_config = heading_schedule_config_from_params(params);

    // Precompute Euler dynamics for IPM solver
    EulerDynamicsData euler_data;
    euler_dynamics_precompute(params, config.dt, euler_data);
    IpmSolverConfig ipm_config{};

    // Per-solver run state
    enum class SolveMode { PRECOMPUTED, HL_NEON_IPM };

    struct SolverRun {
        const char* name;
        SolveMode mode;
        SolverContext ctx;
        double x_cur[NX];
        std::vector<rerun::Vec2D> actual_pts;
        std::vector<double> solve_times_us;
        rerun::Color color;
    };

    struct SolverDef {
        const char* name;
        SolveMode mode;
        rerun::Color color;
    };

    std::vector<SolverDef> solver_defs = {
        {"fista",       SolveMode::PRECOMPUTED, rerun::Color(0, 200, 0)},
        {"hl_neon_ipm", SolveMode::HL_NEON_IPM, rerun::Color(255, 180, 0)},
    };

    int n_vars = config.N * NU;
    std::vector<SolverRun> runs(solver_defs.size());
    for (size_t s = 0; s < solver_defs.size(); ++s) {
        runs[s].name  = solver_defs[s].name;
        runs[s].mode  = solver_defs[s].mode;
        runs[s].color = solver_defs[s].color;
        solver_context_init(runs[s].ctx, n_vars);
        std::memcpy(runs[s].x_cur, ref_path[0].x_ref, NX * sizeof(double));
        runs[s].actual_pts.reserve(n_windows + 1);
        runs[s].actual_pts.push_back({static_cast<float>(ref_path[0].x_ref[0]),
                                      static_cast<float>(ref_path[0].x_ref[1])});
    }

    std::printf("Enabled solvers:");
    for (auto& sd : solver_defs)
        std::printf(" %s", sd.name);
    std::printf("\n");

    // Initialize rerun
    auto rec = rerun::RecordingStream("mecanum_mpc_tracking");
    rec.spawn().exit_on_failure();

    // Log reference trajectory as a static line strip
    {
        std::vector<rerun::Vec2D> ref_pts;
        ref_pts.reserve(n_path);
        for (int k = 0; k < n_path; ++k)
            ref_pts.push_back({static_cast<float>(ref_path[k].x_ref[0]),
                               static_cast<float>(ref_path[k].x_ref[1])});
        rec.log_static("trajectory/reference",
                        rerun::LineStrips2D(rerun::LineStrip2D(ref_pts))
                            .with_colors({rerun::Color(100, 100, 255)}));
    }

    // Log reference controls as static time series
    for (int k = 0; k < n_path - 1; ++k) {
        rec.set_time_seconds("sim_time", ref_path[k].t);
        for (int j = 0; j < NU; ++j) {
            rec.log("reference/u" + std::to_string(j),
                    rerun::Scalars(ref_path[k].u_ref[j]));
        }
        rec.log("reference/vx", rerun::Scalars(ref_path[k].x_ref[3]));
        rec.log("reference/vy", rerun::Scalars(ref_path[k].x_ref[4]));
        rec.log("reference/omega", rerun::Scalars(ref_path[k].x_ref[5]));
    }

    // Closed-loop simulation
    int n_sim = std::min(n_windows, n_path - 1);

    std::printf("\nRunning closed-loop simulation (%d steps, %zu solvers)...\n",
                n_sim, runs.size());

    for (int k = 0; k < n_sim; ++k) {
        double t = ref_path[k].t;
        rec.set_time_seconds("sim_time", t);

        // Log reference position (shared)
        rec.log("trajectory/ref_position",
                rerun::Points2D({{static_cast<float>(ref_path[k].x_ref[0]),
                                  static_cast<float>(ref_path[k].x_ref[1])}})
                    .with_colors({rerun::Color(255, 100, 100)})
                    .with_radii({0.02f}));

        for (size_t s = 0; s < runs.size(); ++s) {
            auto& run = runs[s];
            std::string prefix = run.name;

            // Solve MPC (dispatch by mode)
            QPSolution sol;
            switch (run.mode) {
                case SolveMode::PRECOMPUTED:
                    sol = mpc_solve_online(
                        windows[k], run.x_cur, config, run.ctx.box_ws);
                    break;
                case SolveMode::HL_NEON_IPM:
                    sol = heading_lookup_solve_ipm(
                        euler_data, hl_data, &ref_path[k], run.x_cur, config,
                        sched_config, ipm_config, *run.ctx.ipm_ws);
                    break;
            }

            double solve_us = sol.solve_time_ns / 1000.0;
            run.solve_times_us.push_back(solve_us);

            // Tracking error
            double err_pos = std::sqrt(
                (run.x_cur[0] - ref_path[k].x_ref[0]) * (run.x_cur[0] - ref_path[k].x_ref[0]) +
                (run.x_cur[1] - ref_path[k].x_ref[1]) * (run.x_cur[1] - ref_path[k].x_ref[1]));
            double err_heading = std::fabs(run.x_cur[2] - ref_path[k].x_ref[2]);

            // Log metrics
            rec.log("metrics/position_error/" + prefix, rerun::Scalars(err_pos));
            rec.log("metrics/heading_error/" + prefix, rerun::Scalars(err_heading));
            rec.log("metrics/solve_time_us/" + prefix, rerun::Scalars(solve_us));
            rec.log("metrics/n_iterations/" + prefix,
                    rerun::Scalars(static_cast<double>(sol.n_iterations)));

            // Log control outputs
            for (int j = 0; j < NU; ++j)
                rec.log("control/u" + std::to_string(j) + "/" + prefix,
                        rerun::Scalars(sol.u0[j]));

            // Log actual state
            rec.log("state/px/" + prefix, rerun::Scalars(run.x_cur[0]));
            rec.log("state/py/" + prefix, rerun::Scalars(run.x_cur[1]));
            rec.log("state/theta/" + prefix, rerun::Scalars(run.x_cur[2]));
            rec.log("state/vx/" + prefix, rerun::Scalars(run.x_cur[3]));
            rec.log("state/vy/" + prefix, rerun::Scalars(run.x_cur[4]));
            rec.log("state/omega/" + prefix, rerun::Scalars(run.x_cur[5]));

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
                    rerun::LineStrips2D(rerun::LineStrip2D(run.actual_pts))
                        .with_colors({run.color}));
        }

        if (k % 10 == 0 || k == n_sim - 1) {
            std::printf("  step %3d/%d", k, n_sim);
            for (auto& run : runs)
                std::printf("  %s=%.0fus", run.name, run.solve_times_us.back());
            std::printf("\n");
        }
    }

    // Summary table
    std::printf("\n%-20s %8s %8s %8s %8s  %12s %12s\n",
                "Solver", "Mean(us)", "Med(us)", "Max(us)", "Min(us)",
                "Final|pos|", "Final|θ|");
    std::printf("%-20s %8s %8s %8s %8s  %12s %12s\n",
                "--------------------", "--------", "-------", "-------", "-------",
                "------------", "------------");

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

        double ep = std::sqrt(
            (run.x_cur[0] - ref_path[n_path_orig - 1].x_ref[0]) *
            (run.x_cur[0] - ref_path[n_path_orig - 1].x_ref[0]) +
            (run.x_cur[1] - ref_path[n_path_orig - 1].x_ref[1]) *
            (run.x_cur[1] - ref_path[n_path_orig - 1].x_ref[1]));
        double eth = std::fabs(run.x_cur[2] - ref_path[n_path_orig - 1].x_ref[2]);

        std::printf("%-20s %8.1f %8.1f %8.1f %8.1f  %12.4f %12.4f\n",
                    run.name, mean, median, max_t, min_t, ep, eth);
    }

    // Cleanup
    for (auto& run : runs)
        solver_context_free(run.ctx);
    delete[] windows;

    std::printf("\nDone. Check the rerun viewer for visualization.\n");
    return 0;
}
