#pragma once

#include "mpc_types.h"
#include "mecanum_model.h"
#include "qp_solvers.h"
#include "heading_lookup.h"
#include "ipm_solver.h"

// High-level MPC controller wrapping offline precomputation + online solve.
// Owns all allocated memory; no raw pointers exposed.
class MecanumLTV {
public:
    MecanumLTV();
    ~MecanumLTV();

    // ---- Configuration (call before loadTrajectory) ----

    // Set robot model parameters. Must be called before loadTrajectory.
    void setModelParams(const ModelParams& params);

    // Set MPC tuning. Must be called before loadTrajectory.
    void setConfig(const MPCConfig& config);

    // Set the QP solver used in solve(). Default: FISTA.
    // HPIPM_OCP requires loadTrajectory (not loadWindows) and MPC_USE_HPIPM build.
    void setSolverType(QpSolverType type) { solver_type_ = type; }

    // Set cost-based window selection tuning. Optional; defaults are reasonable.
    void setWindowSelConfig(const WindowSelConfig& cfg) { win_sel_config_ = cfg; }

    // ---- Trajectory loading ----

    // Load a reference trajectory from raw state samples.
    //   samples: flat array of (n_samples) rows, each row is
    //            [t, px, py, theta, vx, vy, omega]  (7 doubles)
    //   n_samples: number of rows
    //   dt:        desired uniform timestep for resampling (seconds)
    //
    // The trajectory is linearly resampled to uniform dt, converted to
    // RefNodes with zero feedforward, and then precomputed.
    // Returns the number of MPC windows produced (0 on failure).
    int loadTrajectory(const double* samples, int n_samples, double dt);

    // Save precomputed windows to a .bin file (v2 format).
    // Returns 0 on success, non-zero on failure.
    int saveWindows(const char* filepath) const;

    // ---- Online solve ----

    // Solve MPC given current state x0[6] and time elapsed since the last solve call.
    // On the first call after loadTrajectory/loadWindows, dt_since_last should be
    // the total elapsed time since trajectory start.
    // Writes the full control horizon to u_out (N*4 doubles).
    // Returns the selected window index (>=0). Returns -1 on error.
    // Internally maintains prev_idx_ (monotone, never decreases).
    int solve(const double x0[NX], double dt_since_last, double* u_out);

    // Solve to a target waypoint without any preloaded trajectory.
    // Generates a Hermite-interpolated reference from x0 to x_target over
    // t_remaining seconds, then solves via NEON_IPM.
    //
    // Call every control loop with the current state and remaining time.
    // The target can change each call with zero precomputation overhead.
    // Requires setModelParams() and setConfig() but NOT loadTrajectory().
    //
    // x0:          current state [px, py, theta, vx, vy, omega]
    // x_target:    desired state at t_remaining seconds from now
    // t_remaining: seconds until waypoint must be reached (clamped to >= 0)
    // u_out:       output — first 4 controls [V1,V2,V3,V4]
    // Returns 0 on success, -1 if not configured.
    // dt_hint:  control timestep to use if loadTrajectory has not been called
    //           (ignored if loadTrajectory has already set config_.dt)
    // lqr_ref:  if false (default), use a Hermite-interpolated reference — good for
    //           nonzero arrival velocities but sub-optimal in general.
    //           if true, use x_target as a constant reference — the Riccati backward
    //           pass then solves the exact discrete LQR problem and finds the
    //           dynamically-optimal trajectory. Best for zero-velocity arrival.
    //           The effective horizon is also shortened to ceil(t_remaining/dt) so
    //           the terminal Qf cost lands exactly on the deadline in both modes.
    int solve_waypoint(const double x0[NX],
                       const double x_target[NX],
                       double t_remaining,
                       double dt_hint,
                       bool lqr_ref,
                       double* u_out);

    // Index selected by the most recent solve() call. Useful for logging.
    int prevIdx() const { return prev_idx_; }

    // Copy x_ref_0[NX] for window_idx into x_ref_out. Returns false on bad index.
    bool getWindowRef(int window_idx, double x_ref_out[NX]) const;

    // ---- Accessors ----

    int numWindows() const { return n_windows_; }
    int numTrajectoryWindows() const { return n_traj_windows_; }
    int horizonLength() const { return config_.N; }
    int numVars() const { return config_.N * NU; }

private:
    MecanumLTV(const MecanumLTV&) = delete;
    MecanumLTV& operator=(const MecanumLTV&) = delete;

    void ensure_hld_ready();
    void ensure_euler_ready();
    void ensure_solver_ctx_ready();

    ModelParams params_;
    MPCConfig config_;
    bool params_set_;
    bool config_set_;

    PrecomputedWindow* windows_;
    int n_windows_;
    int n_traj_windows_;  // number of original trajectory points (before padding)

    // Padded reference trajectory (needed for HPIPM OCP path)
    RefNode* ref_nodes_;
    int n_ref_nodes_;

    // Heading-lookup LTV data (precomputed from model params + dt)
    HeadingLookupData hld_;
    HeadingScheduleConfig sched_config_;
    bool hld_valid_;

    // Euler dynamics for IPM solver
    EulerDynamicsData euler_data_;
    bool euler_valid_;

    // Solver context and type
    SolverContext solver_ctx_;
    bool solver_ctx_valid_;
    QpSolverType solver_type_;

    // Cost-based window selection state
    WindowSelConfig win_sel_config_;
    int    prev_idx_;
    int    prev_waypoint_n_;   // N_eff from previous solve_waypoint call; -1 = never called
    double elapsed_total_;  // seconds accumulated while on-path; drives time_idx_float
    bool   was_holding_;    // true if the previous solve was in hold (off-path) mode
};
