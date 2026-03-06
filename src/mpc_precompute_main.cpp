// mpc_precompute -- CLI tool to precompute MPC windows offline.
//
// Reads a trajopt JSON and a config JSON, precomputes all MPC windows,
// and writes them to a .bin file for fast loading on the robot.
//
// Usage:
//   mpc_precompute <trajopt.json> <config.json> <output.bin> [--traj-index N]

#include "mpc_types.h"
#include "mpc_offline.h"
#include "mecanum_model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// ---------------------------------------------------------------------------
// Convert trajopt format to MPC RefNode array
// ---------------------------------------------------------------------------
// trajopt state: [vx, vy, omega, px, py, theta]
// MPC state:     [px, py, theta, vx, vy, omega]
//
// trajopt control: [drive, strafe, turn]
// MPC control:     [d_FL, d_FR, d_RL, d_RR]

static void convert_to_refnodes(const TrajoptTrajectory& traj,
                                 std::vector<RefNode>& ref_path)
{
    int n = static_cast<int>(traj.times.size());
    ref_path.resize(n);

    for (int k = 0; k < n; ++k) {
        RefNode& r = ref_path[k];
        std::memset(&r, 0, sizeof(RefNode));

        const auto& s = traj.states[k];
        r.x_ref[0] = s[3];  // px
        r.x_ref[1] = s[4];  // py
        r.x_ref[2] = s[5];  // theta
        r.x_ref[3] = s[0];  // vx
        r.x_ref[4] = s[1];  // vy
        r.x_ref[5] = s[2];  // omega

        r.theta = s[5];
        r.omega = s[2];
        r.t     = traj.times[k];

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
// Load MPC config from JSON
// ---------------------------------------------------------------------------
static bool load_config(const std::string& path, MPCConfig& config)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Cannot open config %s\n", path.c_str());
        return false;
    }

    json doc = json::parse(f);

    config = MPCConfig{};
    config.N     = doc["N"].get<int>();
    config.dt    = doc["dt"].get<double>();
    config.u_min = doc["u_min"].get<double>();
    config.u_max = doc["u_max"].get<double>();

    auto& Q_diag = doc["Q_diag"];
    auto& R_diag = doc["R_diag"];
    auto& Qf_diag = doc["Qf_diag"];

    for (int i = 0; i < NX; ++i) {
        config.Q[i + NX * i]  = Q_diag[i].get<double>();
        config.Qf[i + NX * i] = Qf_diag[i].get<double>();
    }
    for (int i = 0; i < NU; ++i) {
        config.R[i + NU * i] = R_diag[i].get<double>();
    }

    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "Usage: %s <trajopt.json> <config.json> <output.bin> [--traj-index N]\n"
        "\n"
        "  trajopt.json   Trajectory optimization project file\n"
        "  config.json    MPC configuration (N, dt, Q_diag, R_diag, etc.)\n"
        "  output.bin     Output binary file for precomputed windows\n"
        "  --traj-index N Trajectory index within the project (default: 0)\n",
        argv0);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string trajopt_path = argv[1];
    std::string config_path  = argv[2];
    std::string output_path  = argv[3];
    int traj_idx = 0;

    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--traj-index" && i + 1 < argc) {
            traj_idx = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Load MPC config
    MPCConfig config{};
    if (!load_config(config_path, config))
        return 1;

    std::printf("MPC config: N=%d  dt=%.4f  u_bounds=[%.2f, %.2f]\n",
                config.N, config.dt, config.u_min, config.u_max);

    // Load trajectory
    TrajoptTrajectory traj;
    TrajoptParams tp;
    if (!load_trajectory(trajopt_path, traj_idx, traj, tp))
        return 1;

    // Load optional damping from config (default 0)
    double damping_linear  = 0.0;
    double damping_angular = 0.0;
    {
        std::ifstream f(config_path);
        json doc = json::parse(f);
        if (doc.contains("damping_linear"))
            damping_linear = doc["damping_linear"].get<double>();
        if (doc.contains("damping_angular"))
            damping_angular = doc["damping_angular"].get<double>();
    }

    // Set up model params from trajopt JSON
    ModelParams params{};
    params.mass            = tp.mass;
    params.inertia         = tp.inertia;
    params.damping_linear  = damping_linear;
    params.damping_angular = damping_angular;
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

    // Resample to config dt
    resample_trajectory(traj, config.dt);

    // Convert to RefNodes
    std::vector<RefNode> ref_path;
    convert_to_refnodes(traj, ref_path);

    // Pad with N hold-at-final-position nodes
    {
        const RefNode& last = ref_path.back();
        for (int i = 1; i <= config.N; ++i) {
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
    std::printf("Reference path: %d nodes (including %d padding)\n",
                n_path, config.N);

    // Precompute all windows
    int n_windows = 0;
    PrecomputedWindow* windows = mpc_precompute_all(
        ref_path.data(), n_path, params, config, n_windows);

    if (!windows || n_windows < 1) {
        std::fprintf(stderr, "Precomputation failed (n_windows=%d)\n", n_windows);
        return 1;
    }
    std::printf("Precomputed %d windows\n", n_windows);

    // Save to binary
    int ret = mpc_save_windows(output_path.c_str(), windows, n_windows, config);
    delete[] windows;

    if (ret != 0) {
        std::fprintf(stderr, "Failed to write %s\n", output_path.c_str());
        return 1;
    }

    std::printf("Wrote %s (%d windows, v%d format)\n",
                output_path.c_str(), n_windows, MPC_FILE_VERSION);
    return 0;
}
