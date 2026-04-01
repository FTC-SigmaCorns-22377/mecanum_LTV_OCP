#include "mecanum_ltv.h"
#include "ipm_solver.h"
#include "mpc_offline.h"
#include "mpc_online.h"

#include <cmath>
#include <cstring>
#include <algorithm>

MecanumLTV::MecanumLTV()
    : params_{}
    , config_{}
    , params_set_(false)
    , config_set_(false)
    , windows_(nullptr)
    , n_windows_(0)
    , n_traj_windows_(0)
    , ref_nodes_(nullptr)
    , n_ref_nodes_(0)
    , hld_{}
    , sched_config_{}
    , hld_valid_(false)
    , euler_data_{}
    , euler_valid_(false)
    , solver_ctx_{}
    , solver_type_(QpSolverType::FISTA)
    , win_sel_config_{}
    , prev_idx_(0)
    , elapsed_total_(0.0)
    , was_holding_(false)
{
}

MecanumLTV::~MecanumLTV()
{
    delete[] windows_;
    delete[] ref_nodes_;
#ifdef MPC_USE_HPIPM
    solver_context_free(solver_ctx_);
#endif
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

bool MecanumLTV::getWindowRef(int window_idx, double x_ref_out[NX]) const
{
    if (!windows_ || window_idx < 0 || window_idx >= n_windows_)
        return false;
    std::memcpy(x_ref_out, windows_[window_idx].x_ref_0, NX * sizeof(double));
    return true;
}

int MecanumLTV::saveWindows(const char* filepath) const
{
    if (!windows_ || n_windows_ <= 0)
        return -1;
    return mpc_save_windows(filepath, windows_, n_windows_, config_);
}

int MecanumLTV::loadTrajectory(const double* samples, int n_samples, double dt)
{
    if (!params_set_ || !config_set_)
        return 0;
    if (n_samples < 2 || dt <= 0.0)
        return 0;

    // Free previous windows and ref nodes
    delete[] windows_;
    windows_ = nullptr;
    n_windows_ = 0;
    n_traj_windows_ = 0;
    delete[] ref_nodes_;
    ref_nodes_ = nullptr;
    n_ref_nodes_ = 0;
    hld_valid_ = false;
    std::memset(&solver_ctx_, 0, sizeof(solver_ctx_));

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

    // Transfer ownership of padded_path to ref_nodes_ for the HPIPM OCP path
    ref_nodes_ = padded_path;
    n_ref_nodes_ = n_padded;

    // Store the index of the last real trajectory window for clamping
    n_traj_windows_ = n_resampled;
    prev_idx_ = 0;
    elapsed_total_ = 0.0;
    was_holding_ = false;

    // Precompute heading-lookup LTV data for the HPIPM path
    heading_lookup_precompute(params_, config_.dt, hld_);
    sched_config_ = heading_schedule_config_from_params(params_);
    hld_valid_ = true;

    return n_windows_;
}

// ---------------------------------------------------------------------------
// Angle wrapping helper
// ---------------------------------------------------------------------------
static double angle_wrap(double a)
{
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

int MecanumLTV::solve(const double x0[NX], double dt_since_last, double* u_out)
{
    if (!windows_ || n_windows_ <= 0)
        return -1;

    const double dt_nominal = config_.dt;

    // ---- Hold check: freeze window when robot is too far off-path ----
    // Compute XY distance from robot to the current reference window.
    const double* cur_ref = windows_[prev_idx_].x_ref_0;
    const double dx_hold  = x0[0] - cur_ref[0];
    const double dy_hold  = x0[1] - cur_ref[1];
    const double pos_dist = std::sqrt(dx_hold*dx_hold + dy_hold*dy_hold);

    int window_idx;

    if (pos_dist > win_sel_config_.hold_radius) {
        // Off-path: hold at current window, do not accumulate elapsed time.
        window_idx = prev_idx_;
        was_holding_ = true;
    } else {
        // On-path: if we just transitioned in from holding, reset elapsed_total_
        // so time_idx_float picks up cleanly from prev_idx_ with no accumulated debt.
        if (was_holding_) {
            elapsed_total_ = prev_idx_ * dt_nominal;
            was_holding_ = false;
        }
        elapsed_total_ += dt_since_last;
        const double time_idx_float = elapsed_total_ / dt_nominal;

        // ---- Cost-based window selection ----
        const int search_start = prev_idx_;
        const int search_end   = std::min(prev_idx_ + win_sel_config_.search_radius,
                                          n_windows_ - 1);

        int cost_idx = search_start;
        double best_cost = 1e18;
        const double hw = win_sel_config_.heading_weight;

        for (int i = search_start; i <= search_end; ++i) {
            const double* ref = windows_[i].x_ref_0;
            double dx     = x0[0] - ref[0];
            double dy     = x0[1] - ref[1];
            double dtheta = angle_wrap(x0[2] - ref[2]);
            double dist2  = dx*dx + dy*dy + hw*hw*dtheta*dtheta;
            double td     = static_cast<double>(i) - time_idx_float;
            double cost   = win_sel_config_.pos_weight  * dist2
                          + win_sel_config_.time_weight * td * td;
            if (cost < best_cost) {
                best_cost = cost;
                cost_idx  = i;
            }
        }

        // Snap to time if on/ahead of schedule, otherwise use position winner.
        double time_delta = time_idx_float - static_cast<double>(cost_idx);
        if (time_delta < 0.5) {
            window_idx = static_cast<int>(std::round(time_idx_float));
        } else {
            window_idx = cost_idx;
        }

        // Clamp: monotone and max-jump cap
        window_idx = std::max(prev_idx_,
                              std::min(window_idx, prev_idx_ + win_sel_config_.max_jump));
        window_idx = std::max(0, std::min(window_idx, n_windows_ - 1));
    }

    const int delta = window_idx - prev_idx_;
    prev_idx_ = window_idx;

    // ---- Clamping at end: freeze warm-start ----
    if (window_idx >= n_windows_ - 1) {
        solver_ctx_.box_ws.warm_valid = false;
    }

    if (solver_type_ == QpSolverType::NEON_IPM
            && ref_nodes_
            && window_idx + config_.N + 1 <= n_ref_nodes_) {
        if (!euler_valid_) {
            euler_dynamics_precompute(params_, config_.dt, euler_data_);
            euler_valid_ = true;
        }
        if (window_idx >= n_windows_ - 1)
            solver_ctx_.ipm_ws->warm_valid = false;
        QPSolution sol = heading_lookup_solve_ipm(
            euler_data_, hld_, ref_nodes_ + window_idx, x0, config_,
            sched_config_, *solver_ctx_.ipm_config, *solver_ctx_.ipm_ws);
        std::memcpy(u_out, sol.U,
                    static_cast<std::size_t>(config_.N * NU) * sizeof(double));
        return window_idx;
    }

#ifdef MPC_USE_HPIPM
    if (solver_type_ == QpSolverType::HPIPM_OCP
            && hld_valid_ && ref_nodes_
            && window_idx + config_.N + 1 <= n_ref_nodes_) {
        if (window_idx >= n_windows_ - 1)
            solver_ctx_.hpipm_ocp_ws.warm_valid = false;
        QPSolution sol = heading_lookup_solve_ocp(
            hld_, ref_nodes_ + window_idx, x0, config_, sched_config_, solver_ctx_);
        std::memcpy(u_out, sol.U,
                    static_cast<std::size_t>(config_.N * NU) * sizeof(double));
        return window_idx;
    }
#endif

    QPSolution sol = mpc_solve_online(windows_[window_idx], x0, config_,
                                      solver_ctx_.box_ws, delta);
    const int n_vars = windows_[window_idx].n_vars;
    std::memcpy(u_out, sol.U, n_vars * sizeof(double));
    return window_idx;
}
