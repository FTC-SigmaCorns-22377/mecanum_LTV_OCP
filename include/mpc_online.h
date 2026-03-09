#pragma once

#include "mpc_types.h"

// Main online MPC solve function
// Given a precomputed window and current state x0, solve for optimal control.
// delta: number of windows advanced since last solve (used for warm-start shifting).
//   delta=1 is normal, delta=0 means same window (re-use), delta>MAX_WARM_SHIFT -> cold start.
QPSolution mpc_solve_online(const PrecomputedWindow& window, const double x0[NX],
                            const MPCConfig& config, BoxQPWorkspace& workspace, int delta = 1);
