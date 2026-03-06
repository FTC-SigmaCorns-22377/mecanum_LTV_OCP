#pragma once

#include "mpc_types.h"
#include "mecanum_model.h"

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

    // Load precomputed windows from a .bin file (v2 format).
    // No setModelParams/setConfig/loadTrajectory needed — all config
    // is reconstructed from the file header.
    // Returns the number of windows loaded (0 on failure).
    int loadWindows(const char* filepath);

    // ---- Online solve ----

    // Solve MPC at window index `window_idx` given current state x0[6].
    // Writes the full control horizon to u_out (N*4 doubles).
    // Returns the number of QP iterations (0 = warm-start hit).
    // Returns -1 on error (bad index, not precomputed).
    int solve(int window_idx, const double x0[NX], double* u_out);

    // ---- Accessors ----

    int numWindows() const { return n_windows_; }
    int numTrajectoryWindows() const { return n_traj_windows_; }
    int horizonLength() const { return config_.N; }
    int numVars() const { return config_.N * NU; }
    double dt() const { return config_.dt; }

private:
    MecanumLTV(const MecanumLTV&) = delete;
    MecanumLTV& operator=(const MecanumLTV&) = delete;

    ModelParams params_;
    MPCConfig config_;
    bool params_set_;
    bool config_set_;

    PrecomputedWindow* windows_;
    int n_windows_;
    int n_traj_windows_;  // number of original trajectory points (before padding)

    BoxQPWorkspace workspace_;
};
