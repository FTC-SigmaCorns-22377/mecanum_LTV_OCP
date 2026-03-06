#include "mecanum_ltv.h"
#include "mpc_offline.h"
#include "mpc_online.h"

#include <cmath>
#include <cstring>

MecanumLTV::MecanumLTV()
    : params_{}
    , config_{}
    , params_set_(false)
    , config_set_(false)
    , windows_(nullptr)
    , n_windows_(0)
    , n_traj_windows_(0)
    , workspace_{}
{
}

MecanumLTV::~MecanumLTV()
{
    delete[] windows_;
}

void MecanumLTV::setModelParams(const ModelParams& params)
{
    params_ = params;
    compute_mecanum_jacobian(params_);
    params_set_ = true;
}

void MecanumLTV::setConfig(const MPCConfig& config)
{
    config_ = config;
    config_set_ = true;
}

// ---------------------------------------------------------------------------
// Linear interpolation helper for resampling
// ---------------------------------------------------------------------------
static void lerp_sample(const double* a, const double* b, double frac, double* out)
{
    // a, b are [t, px, py, theta, vx, vy, omega] (7 doubles)
    for (int i = 0; i < 7; ++i)
        out[i] = a[i] + frac * (b[i] - a[i]);

    // Wrap-safe theta interpolation: handle angle discontinuities
    double dtheta = b[3] - a[3];
    if (dtheta > M_PI) dtheta -= 2.0 * M_PI;
    else if (dtheta < -M_PI) dtheta += 2.0 * M_PI;
    out[3] = a[3] + frac * dtheta;
}

int MecanumLTV::loadWindows(const char* filepath)
{
    // Free previous windows
    delete[] windows_;
    windows_ = nullptr;
    n_windows_ = 0;
    n_traj_windows_ = 0;
    std::memset(&workspace_, 0, sizeof(workspace_));

    MPCConfig loaded_config{};
    int n_loaded = 0;
    windows_ = mpc_load_windows(filepath, n_loaded, loaded_config);
    if (!windows_ || n_loaded <= 0) {
        windows_ = nullptr;
        return 0;
    }

    n_windows_ = n_loaded;
    n_traj_windows_ = n_loaded;
    config_ = loaded_config;
    params_set_ = true;   // not needed for solve, but mark as ready
    config_set_ = true;

    return n_windows_;
}

int MecanumLTV::loadTrajectory(const double* samples, int n_samples, double dt)
{
    if (!params_set_ || !config_set_)
        return 0;
    if (n_samples < 2 || dt <= 0.0)
        return 0;

    // Free previous windows
    delete[] windows_;
    windows_ = nullptr;
    n_windows_ = 0;
    n_traj_windows_ = 0;
    std::memset(&workspace_, 0, sizeof(workspace_));

    // Override config dt with the requested uniform dt
    config_.dt = dt;

    // Determine time range
    const double t_start = samples[0];  // first sample's t
    const double t_end = samples[(n_samples - 1) * 7];  // last sample's t
    const double duration = t_end - t_start;
    if (duration <= 0.0)
        return 0;

    const int n_resampled = static_cast<int>(std::floor(duration / dt)) + 1;
    if (n_resampled < config_.N + 1)
        return 0;

    // Resample to uniform dt
    RefNode* path = new RefNode[n_resampled];

    int src_idx = 0;
    for (int i = 0; i < n_resampled; ++i) {
        double t_target = t_start + i * dt;

        // Clamp to end
        if (t_target >= t_end) {
            t_target = t_end;
        }

        // Advance source index
        while (src_idx < n_samples - 2 && samples[(src_idx + 1) * 7] < t_target)
            ++src_idx;

        const double* sa = samples + src_idx * 7;
        const double* sb = samples + (src_idx + 1) * 7;
        double seg_dt = sb[0] - sa[0];

        double interp[7];
        if (seg_dt > 1e-12) {
            double frac = (t_target - sa[0]) / seg_dt;
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            lerp_sample(sa, sb, frac, interp);
        } else {
            std::memcpy(interp, sa, 7 * sizeof(double));
        }

        path[i].t = t_target;
        path[i].x_ref[0] = interp[1]; // px
        path[i].x_ref[1] = interp[2]; // py
        path[i].x_ref[2] = interp[3]; // theta
        path[i].x_ref[3] = interp[4]; // vx
        path[i].x_ref[4] = interp[5]; // vy
        path[i].x_ref[5] = interp[6]; // omega
        path[i].theta = interp[3];
        path[i].omega = interp[6];
        std::memset(path[i].u_ref, 0, NU * sizeof(double));
    }

    // Pad path with N extra nodes at the final position with zero velocity,
    // so the last resampled point still has a full horizon window ahead of it.
    const int N = config_.N;
    const int n_padded = n_resampled + N;
    RefNode* padded_path = new RefNode[n_padded];
    std::memcpy(padded_path, path, n_resampled * sizeof(RefNode));

    RefNode hold_node = path[n_resampled - 1];
    hold_node.x_ref[3] = 0.0;  // vx = 0
    hold_node.x_ref[4] = 0.0;  // vy = 0
    hold_node.x_ref[5] = 0.0;  // omega = 0
    hold_node.omega = 0.0;
    std::memset(hold_node.u_ref, 0, NU * sizeof(double));
    for (int i = 0; i < N; ++i) {
        hold_node.t = path[n_resampled - 1].t + (i + 1) * dt;
        padded_path[n_resampled + i] = hold_node;
    }

    delete[] path;

    // Precompute all windows (now n_padded - N = n_resampled windows)
    windows_ = mpc_precompute_all(padded_path, n_padded, params_, config_, n_windows_);
    delete[] padded_path;

    // Store the index of the last real trajectory window for clamping
    n_traj_windows_ = n_resampled;

    return n_windows_;
}

int MecanumLTV::solve(int window_idx, const double x0[NX], double* u_out)
{
    if (!windows_ || window_idx < 0)
        return -1;

    // Clamp to the last window once past the end of the trajectory.
    // When clamped, freeze the warm-start so it doesn't shift.
    bool clamped = (window_idx >= n_windows_);
    int idx = clamped ? n_windows_ - 1 : window_idx;

    if (clamped) {
        // Invalidate warm-start so the solver does a cold start on the
        // same final window every time, rather than shifting U_prev.
        workspace_.warm_valid = false;
    }

    QPSolution sol = mpc_solve_online(windows_[idx], x0, config_, workspace_);

    const int n_vars = windows_[idx].n_vars;
    std::memcpy(u_out, sol.U, n_vars * sizeof(double));

    return sol.n_iterations;
}
