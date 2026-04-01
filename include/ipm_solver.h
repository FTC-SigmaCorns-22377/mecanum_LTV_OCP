#pragma once

#include "mpc_types.h"
#include "heading_lookup.h"

#include <array>

// ---------------------------------------------------------------------------
// IPM solver configuration
// ---------------------------------------------------------------------------
struct IpmSolverConfig {
    int   max_outer_iters = 4;       // number of barrier parameter reductions
    float mu_init         = 1.0f;    // initial barrier parameter
    float mu_factor       = 0.1f;    // geometric reduction: mu *= mu_factor
    float mu_min          = 1e-4f;   // stop below this
    float interior_margin = 0.01f;   // strict interior clamp: [-1+margin, 1-margin]
};

// Default config
inline IpmSolverConfig ipm_solver_default_config() { return IpmSolverConfig{}; }

// ---------------------------------------------------------------------------
// IPM workspace (all stack-allocated, no heap)
// ---------------------------------------------------------------------------
struct IpmWorkspace {
    // Per-stage Riccati backward pass storage:
    //   Kp(3×4=12) + Kv(3×4=12) + v(4) + Bc_rot(3×4=12) = 40 floats
    float stage_data[N_MAX * 40];

    // IPM iterate
    float u_bar[N_MAX * NU];

    // Per-stage effective R and ur (barrier-modified)
    float R_eff[N_MAX * NU];
    float ur_eff[N_MAX * NU];

    // Precomputed sin/cos table
    float sincos[N_MAX * 2];

    // Warm-start from previous MPC call
    float u_prev[N_MAX * NU];
    bool  warm_valid;
    int   prev_N;
};

inline void ipm_workspace_init(IpmWorkspace& ws) {
    ws.warm_valid = false;
    ws.prev_N     = 0;
}

// ---------------------------------------------------------------------------
// Precomputed Euler dynamics data (extracted from ModelParams + dt)
// ---------------------------------------------------------------------------
struct EulerDynamicsData {
    float D_diag[3];    // velocity damping: [1+dt*Ac[3,3], 1+dt*Ac[4,4], 1+dt*Ac[5,5]]
    float B_body[12];   // dt * Bc_lower(theta=0), 3×4 row-major
    float dt;
};

// Extract Euler dynamics from model params.
void euler_dynamics_precompute(const ModelParams& params, double dt,
                               EulerDynamicsData& data);

// ---------------------------------------------------------------------------
// Scalar 6-state block-sparse Riccati (double precision, always available)
// ---------------------------------------------------------------------------

// Solves the 6-state tracking LQR with block-sparse Euler dynamics.
// A = [I, dt·I; 0, D], B = [0; Bl(θ)], per-stage R_eff and ur_eff.
// Returns u_star[N*4].
void riccati_6state_scalar(
    const double Q_diag[6],          // state cost diagonal [Qp0..2, Qv0..2]
    const double Qf_diag[6],         // terminal cost diagonal
    const double R_eff[/*N*4*/],     // per-stage input cost diagonal
    const double D_diag[3],          // velocity damping diagonal
    double dt,
    const double B_body[12],         // 3×4 row-major body-frame Bc_lower * dt
    int N,
    const double* theta,             // [N] heading schedule
    const double* xr,                // [(N+1)*6] full-state reference
    const double* ur_eff,            // [N*4] effective reference input
    const double x0[NX],
    double* u_out);                  // [N*4] output controls

// ---------------------------------------------------------------------------
// NEON 6-state block-sparse Riccati (float32)
// ---------------------------------------------------------------------------
#ifdef MPC_USE_NEON

int ipm_riccati_workspace_sz(int N);

void riccati_6state_neon(
    float* workspace,                // [N * 40] per-stage storage
    std::array<float, 6> Q,          // state cost diagonal
    std::array<float, 6> Qf,         // terminal cost diagonal
    const float* R_eff,              // [N*4] per-stage input cost diagonal
    std::array<float, 3> D,          // velocity damping diagonal
    float dt,
    std::array<float, 12> B_body,    // 3×4 body-frame Bc_lower * dt
    int N,
    const float* theta,              // [N] heading schedule
    const float* xr,                 // [(N+1)*6] full-state reference
    const float* ur_eff,             // [N*4] effective input reference
    const float* x0,                 // [6] initial state
    float* u_star);                  // [N*4] output controls

#endif // MPC_USE_NEON

// ---------------------------------------------------------------------------
// Full IPM solve: heading-lookup LTV + log-barrier method
// ---------------------------------------------------------------------------

// High-level entry point: generates heading schedule, builds consistent
// Euler reference, runs barrier iterations with 6-state Riccati.
QPSolution ipm_solve(const EulerDynamicsData& euler,
                     const HeadingLookupData& hld,
                     const RefNode* ref_window,
                     const double x0[NX],
                     const MPCConfig& config,
                     const HeadingScheduleConfig& sched_config,
                     const IpmSolverConfig& ipm_config,
                     IpmWorkspace& ws);
