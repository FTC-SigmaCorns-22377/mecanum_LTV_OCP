#pragma once

#include "mpc_types.h"

// FISTA (accelerated projected gradient) solver for box-constrained QP
// min 0.5 U' H U + g' U  s.t.  u_min <= U <= u_max
// O(n^2) per-iteration cost (one gemv), O(1/k^2) convergence rate
// step_size should be 1.0 / lambda_max(H)
int fista_box_qp_solve(const double* H, const double* g,
                       double u_min, double u_max, int n, int max_iter,
                       double step_size, BoxQPWorkspace& workspace);

// Check box-constrained KKT conditions at U.
// Computes gradient grad = H*U + g into grad_out (must be size n).
// Returns true if KKT conditions are satisfied.
bool check_box_kkt(const double* H, const double* g, const double* U,
                   double u_min, double u_max, int n, double* grad_out);

// Unconstrained solve: U = -H^{-1} g via Cholesky
void unconstrained_solve(const double* L, const double* g, int n, double* U);

// Check if U is feasible (all elements in [u_min, u_max])
bool is_feasible(const double* U, int n, double u_min, double u_max);

// Clip U to [u_min, u_max], return number of clipped elements
int clip_to_bounds(double* U, int n, double u_min, double u_max);
