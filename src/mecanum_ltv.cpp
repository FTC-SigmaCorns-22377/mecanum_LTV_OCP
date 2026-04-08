#include "mecanum_ltv.h"
#include "ipm_solver.h"
#include "waypoint_solve.h"
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
    , solver_ctx_valid_(false)
    , solver_type_(QpSolverType::FISTA)
    , win_sel_config_{}
    , prev_idx_(0)
    , prev_waypoint_n_(-1)
    , prev_waypoint_eta_(0.0)
    , elapsed_total_(0.0)
    , was_holding_(false)
{
}

MecanumLTV::~MecanumLTV()
{
    delete[] windows_;
    delete[] ref_nodes_;
    solver_context_free(solver_ctx_);
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
    euler_valid_ = false;
    solver_context_free(solver_ctx_);
    solver_context_init(solver_ctx_, config_.N * NU);
    solver_ctx_valid_ = true;

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

    // Precompute heading-lookup LTV data and Euler dynamics
    ensure_hld_ready();
    ensure_euler_ready();

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

// ---------------------------------------------------------------------------
// Lazy-init helpers
// ---------------------------------------------------------------------------
void MecanumLTV::ensure_hld_ready()
{
    if (hld_valid_) return;
    heading_lookup_precompute(params_, config_.dt, hld_);
    sched_config_ = heading_schedule_config_from_params(params_);
    hld_valid_ = true;
}

void MecanumLTV::ensure_euler_ready()
{
    if (euler_valid_) return;
    euler_dynamics_precompute(params_, config_.dt, euler_data_);
    euler_valid_ = true;
}

void MecanumLTV::ensure_solver_ctx_ready()
{
    if (solver_ctx_valid_) return;
    solver_context_init(solver_ctx_, config_.N * NU);
    solver_ctx_valid_ = true;
}

// ---------------------------------------------------------------------------
// Waypoint solve — online, no precomputed trajectory required
// ---------------------------------------------------------------------------
int MecanumLTV::solve_waypoint(const double x0[NX],
                                const double x_target[NX],
                                double t_remaining,
                                double dt_hint,
                                bool lqr_ref,
                                const double q_diag[NX],
                                double r_scalar,
                                double* u_out)
{
    if (!params_set_ || !config_set_) return -1;

    // config_.dt is normally set by loadTrajectory(); use dt_hint if not yet set
    if (config_.dt <= 0.0) {
        if (dt_hint <= 0.0) return -1;
        config_.dt = dt_hint;
    }

    ensure_hld_ready();
    ensure_euler_ready();
    ensure_solver_ctx_ready();

    const double dt = config_.dt;

    // Constants that don't change across the tRemaining refinement loop.
    const double dtheta = angle_wrap(x_target[2] - x0[2]);
    const double theta1 = x0[2] + dtheta;
    const double omega0 = x0[5];
    const double omega1 = x_target[5];

    const double q2_blended = q_diag[2];

    // ---------------------------------------------------------------------------
    // Self-consistent tRemaining loop.
    //
    // The fixed point is ETA(t*) = t*: the horizon exactly matches how long the
    // solver predicts it will take to reach the target.  Because ETA is monotone
    // in t_remaining (larger horizon → gentler controls → later simulated arrival),
    // the iteration contracts quickly — typically 1-2 steps from a large seed.
    // We cap at 3 iterations; each re-uses the warm-start from the previous one
    // so marginal cost is small.
    // ---------------------------------------------------------------------------
    double t_curr = t_remaining;
    QPSolution sol;

    for (int outer = 0; outer < 3; ++outer) {
        const int N_eff = (t_curr > 0.0)
            ? std::max(1, std::min((int)std::ceil(t_curr / dt), config_.N))
            : 1;

        // Invalidate warm-start when horizon length changes
        if (N_eff != prev_waypoint_n_) {
            solver_ctx_.ipm_ws->warm_valid = false;
            prev_waypoint_n_ = N_eff;
        }

        // Build per-iteration config
        MPCConfig eff_cfg = config_;
        eff_cfg.N = N_eff;
        for (int i = 0; i < NX; ++i) eff_cfg.Q[i + NX * i] = q_diag[i];
        for (int i = 0; i < NU; ++i) eff_cfg.R[i + NU * i] = r_scalar;
        if (lqr_ref) eff_cfg.Q[2 + NX * 2] = q2_blended;

        // Avoid division by zero for Hermite
        const double T = t_curr > 1e-6 ? t_curr : 1e-6;

        // Build reference window with Hermite heading schedule
        RefNode ref_window[N_MAX + 1];
        double  hermite_theta_sched[N_MAX + 1];

        for (int k = 0; k <= N_eff; ++k) {
            const double tau = k * dt;
            double t = tau / T;
            if (t > 1.0) t = 1.0;

            const double t2 = t * t, t3 = t2 * t;
            const double h00 =  2.0*t3 - 3.0*t2 + 1.0;
            const double h10 =      t3 - 2.0*t2 + t;
            const double h01 = -2.0*t3 + 3.0*t2;
            const double h11 =      t3 -      t2;
            const double dh00 = ( 6.0*t2 - 6.0*t) / T;
            const double dh10 =   3.0*t2 - 4.0*t + 1.0;
            const double dh01 = (-6.0*t2 + 6.0*t) / T;
            const double dh11 =   3.0*t2 - 2.0*t;

            const double theta = angle_wrap(h00*x0[2] + h10*T*omega0
                                          + h01*theta1 + h11*T*omega1);
            const double omega = dh00*x0[2] + dh10*omega0
                               + dh01*theta1 + dh11*omega1;
            hermite_theta_sched[k] = theta;
            ref_window[k].theta    = theta;
            ref_window[k].omega    = omega;

            // Heading cost reference: always constant target so the Riccati sweep
            // penalises the full heading error in both lqr_ref and Hermite modes.
            // The Hermite arc is passed as theta_sched_override so B-matrices use
            // the physically-correct rotation path regardless.
            ref_window[k].x_ref[2] = theta1;
            ref_window[k].x_ref[5] = x_target[5];

            // Position cost reference differs by mode
            if (lqr_ref) {
                ref_window[k].x_ref[0] = x_target[0];
                ref_window[k].x_ref[1] = x_target[1];
                ref_window[k].x_ref[3] = x_target[3];
                ref_window[k].x_ref[4] = x_target[4];
            } else {
                ref_window[k].x_ref[0] = h00*x0[0]       + h10*T*x0[3]
                                        + h01*x_target[0] + h11*T*x_target[3];
                ref_window[k].x_ref[3] = dh00*x0[0]      + dh10*x0[3]
                                        + dh01*x_target[0]+ dh11*x_target[3];
                ref_window[k].x_ref[1] = h00*x0[1]       + h10*T*x0[4]
                                        + h01*x_target[1] + h11*T*x_target[4];
                ref_window[k].x_ref[4] = dh00*x0[1]      + dh10*x0[4]
                                        + dh01*x_target[1]+ dh11*x_target[4];
            }
            ref_window[k].t = tau;
            std::memset(ref_window[k].u_ref, 0, NU * sizeof(double));
        }

        sol = ipm_solve_terminal(
            euler_data_, hld_, ref_window, x0, x_target, eff_cfg,
            sched_config_, *solver_ctx_.ipm_config, *solver_ctx_.ipm_ws,
            hermite_theta_sched);

        // ETA: forward-simulate sol.U, find step closest to target in XY
        double xs[3] = { x0[0], x0[1], x0[2] };
        double vs[3] = { x0[3], x0[4], x0[5] };
        int    k_min    = 0;
        double dist_min = (xs[0]-x_target[0])*(xs[0]-x_target[0])
                        + (xs[1]-x_target[1])*(xs[1]-x_target[1]);

        for (int k = 0; k < N_eff; ++k) {
            const double ct = std::cos(xs[2]), st = std::sin(xs[2]);
            double Bl[12];
            for (int j = 0; j < 4; ++j) {
                Bl[0*4+j] =  ct*(double)euler_data_.B_body[0*4+j] - st*(double)euler_data_.B_body[1*4+j];
                Bl[1*4+j] =  st*(double)euler_data_.B_body[0*4+j] + ct*(double)euler_data_.B_body[1*4+j];
                Bl[2*4+j] = (double)euler_data_.B_body[2*4+j];
            }
            const double* uk = sol.U + k * NU;
            double Bu[3] = {};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 4; ++j)
                    Bu[i] += Bl[i*4+j] * uk[j];
            for (int i = 0; i < 3; ++i) {
                xs[i] += dt * vs[i];
                vs[i]  = (double)euler_data_.D_diag[i] * vs[i] + Bu[i];
            }
            const double dist = (xs[0]-x_target[0])*(xs[0]-x_target[0])
                              + (xs[1]-x_target[1])*(xs[1]-x_target[1]);
            if (dist < dist_min) { dist_min = dist; k_min = k + 1; }
        }

        const double eta = k_min * dt;
        prev_waypoint_eta_ = eta;

        // Converged when ETA matches the current horizon to within one step
        if (std::fabs(eta - t_curr) <= dt) break;
        t_curr = std::max(eta, dt);
    }

    std::memcpy(u_out, sol.U, NU * sizeof(double));
    return 0;
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
        ensure_euler_ready();
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
