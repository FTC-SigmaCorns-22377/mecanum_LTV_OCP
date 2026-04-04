#pragma once

#include "mpc_types.h"
#include "ipm_solver.h"
#include "heading_lookup.h"

// ---------------------------------------------------------------------------
// Terminal-targeting IPM solve for waypoint tracking
// ---------------------------------------------------------------------------
//
// Identical to ipm_solve / heading_lookup_solve_ipm, with one critical fix:
// the Qf terminal cost is computed against an *explicit* target x_f rather
// than against the last node of the consistent-Euler forward simulation.
//
// Why this matters:
//   ipm_solve builds a "consistent" reference by forward-simulating from
//   ref_window[0].x_ref with u_ref=0. For any target with nonzero velocity
//   (including lqr_ref=true with nonzero v_target), this simulated endpoint
//   drifts away from x_f. The Riccati terminal costate p_N = -Qf·xr[N]
//   then points at the wrong state, so the Qf cost does not penalise
//   |x_N - x_f|^2 as intended.
//
//   This variant patches x_ref[N] = x_f after the forward sim so both the
//   scalar and NEON Riccati kernels correctly initialise p_N = -Qf·x_f.
//
// Parameters:
//   euler, hld, ref_window, x0, config, sched_config, ipm_config, ws —
//       identical to ipm_solve.
//   x_f — desired terminal state [px, py, theta, vx, vy, omega].
//          Decoupled from ref_window[N].x_ref.
//
// Use for all solve_waypoint calls. Safe to use even when x_f has zero
// velocity (identical to ipm_solve in that degenerate case since the Euler
// sim ends at the same point).
QPSolution ipm_solve_terminal(
    const EulerDynamicsData& euler,
    const HeadingLookupData& hld,
    const RefNode* ref_window,
    const double x0[NX],
    const double x_f[NX],
    const MPCConfig& config,
    const HeadingScheduleConfig& sched_config,
    const IpmSolverConfig& ipm_config,
    IpmWorkspace& ws);
