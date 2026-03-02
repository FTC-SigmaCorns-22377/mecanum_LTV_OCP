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

#include <rerun.hpp>
#include <nlohmann/json.hpp>

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
    config.Q[0 + NX * 0] = 100.0;  // px
    config.Q[1 + NX * 1] = 100.0;  // py
    config.Q[2 + NX * 2] =  1000.0;  // theta
    config.Q[3 + NX * 3] =  100.0;  // vx
    config.Q[4 + NX * 4] =  100.0;  // vy
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

    // Initialize rerun
    auto rec = rerun::RecordingStream("mecanum_mpc_tracking");
    rec.spawn().exit_on_failure();

    // Log reference trajectory as a static line strip
    {
        std::vector<rerun::Position2D> ref_pts;
        ref_pts.reserve(n_path);
        for (int k = 0; k < n_path; ++k)
            ref_pts.push_back({static_cast<float>(ref_path[k].x_ref[0]),
                               static_cast<float>(ref_path[k].x_ref[1])});
        rec.log_static("trajectory/reference",
                        rerun::LineStrips2D({ref_pts})
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
    BoxQPWorkspace workspace;
    std::memset(&workspace, 0, sizeof(workspace));

    double x_cur[NX];
    std::memcpy(x_cur, ref_path[0].x_ref, NX * sizeof(double));

    int n_sim = std::min(n_windows, n_path - 1);
    std::vector<rerun::Position2D> actual_pts;
    actual_pts.reserve(n_sim + 1);
    actual_pts.push_back({static_cast<float>(x_cur[0]),
                          static_cast<float>(x_cur[1])});

    std::printf("\nRunning closed-loop simulation (%d steps)...\n", n_sim);

    for (int k = 0; k < n_sim; ++k) {
        double t = ref_path[k].t;
        rec.set_time_seconds("sim_time", t);

        // Solve MPC
        QPSolution sol = mpc_solve_online(windows[k], x_cur, config, workspace);

        // Tracking error
        double err_pos = std::sqrt(
            (x_cur[0] - ref_path[k].x_ref[0]) * (x_cur[0] - ref_path[k].x_ref[0]) +
            (x_cur[1] - ref_path[k].x_ref[1]) * (x_cur[1] - ref_path[k].x_ref[1]));
        double err_heading = std::fabs(x_cur[2] - ref_path[k].x_ref[2]);  // unsigned magnitude

        // Signed per-state errors
        double err_px    = x_cur[0] - ref_path[k].x_ref[0];
        double err_py    = x_cur[1] - ref_path[k].x_ref[1];
        double err_theta = x_cur[2] - ref_path[k].x_ref[2];
        double err_vx    = x_cur[3] - ref_path[k].x_ref[3];
        double err_vy    = x_cur[4] - ref_path[k].x_ref[4];
        double err_omega = x_cur[5] - ref_path[k].x_ref[5];

        // Log metrics
        rec.log("metrics/position_error", rerun::Scalars(err_pos));
        rec.log("metrics/heading_error", rerun::Scalars(err_heading));
        rec.log("errors/px",    rerun::Scalars(err_px));
        rec.log("errors/py",    rerun::Scalars(err_py));
        rec.log("errors/theta", rerun::Scalars(err_theta));
        rec.log("errors/vx",    rerun::Scalars(err_vx));
        rec.log("errors/vy",    rerun::Scalars(err_vy));
        rec.log("errors/omega", rerun::Scalars(err_omega));
        rec.log("metrics/solve_time_us", rerun::Scalars(sol.solve_time_ns / 1000.0));
        rec.log("metrics/n_active", rerun::Scalars(static_cast<double>(sol.n_active)));

        // Log control outputs
        for (int j = 0; j < NU; ++j) {
            rec.log("control/u" + std::to_string(j), rerun::Scalars(sol.u0[j]));
        }

        // Log actual state
        rec.log("state/px", rerun::Scalars(x_cur[0]));
        rec.log("state/py", rerun::Scalars(x_cur[1]));
        rec.log("state/theta", rerun::Scalars(x_cur[2]));
        rec.log("state/vx", rerun::Scalars(x_cur[3]));
        rec.log("state/vy", rerun::Scalars(x_cur[4]));
        rec.log("state/omega", rerun::Scalars(x_cur[5]));

        // Log current robot position as a point
        rec.log("trajectory/actual_position",
                rerun::Points2D({{static_cast<float>(x_cur[0]),
                                  static_cast<float>(x_cur[1])}})
                    .with_colors({rerun::Color(0, 255, 0)})
                    .with_radii({0.02f}));

        // Log reference position as a point
        rec.log("trajectory/ref_position",
                rerun::Points2D({{static_cast<float>(ref_path[k].x_ref[0]),
                                  static_cast<float>(ref_path[k].x_ref[1])}})
                    .with_colors({rerun::Color(255, 100, 100)})
                    .with_radii({0.02f}));

        // Simulate forward
        rk4_step(x_cur, sol.u0, config.dt, params);

        actual_pts.push_back({static_cast<float>(x_cur[0]),
                              static_cast<float>(x_cur[1])});

        // Log growing actual trajectory
        rec.log("trajectory/actual",
                rerun::LineStrips2D({actual_pts})
                    .with_colors({rerun::Color(0, 200, 0)}));

        if (k % 10 == 0 || k == n_sim - 1) {
            std::printf("  step %3d/%d  pos_err=%.4f  heading_err=%.4f  "
                        "u0=[%6.3f %6.3f %6.3f %6.3f]  solve=%.0fus\n",
                        k, n_sim, err_pos, err_heading,
                        sol.u0[0], sol.u0[1], sol.u0[2], sol.u0[3],
                        sol.solve_time_ns / 1000.0);
        }
    }

    // Final error (relative to the original trajectory endpoint)
    double final_err = std::sqrt(
        (x_cur[0] - ref_path[n_path_orig - 1].x_ref[0]) * (x_cur[0] - ref_path[n_path_orig - 1].x_ref[0]) +
        (x_cur[1] - ref_path[n_path_orig - 1].x_ref[1]) * (x_cur[1] - ref_path[n_path_orig - 1].x_ref[1]));
    std::printf("\nFinal position error: %.4f m  (vs original trajectory endpoint)\n", final_err);

    delete[] windows;
    std::printf("Done. Check the rerun viewer for visualization.\n");
    return 0;
}
