#include "box_qp_solver.h"
#include "blas_dispatch.h"
#include "cholesky.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// unconstrained_solve:  U = -H^{-1} g  via Cholesky factor L (H = L L')
//   Forward solve:  L y = -g
//   Backward solve: L' U = y
// ---------------------------------------------------------------------------
void unconstrained_solve(const double* L, const double* g, int n, double* U)
{
    double neg_g[N_MAX * NU];
    for (int i = 0; i < n; ++i)
        neg_g[i] = -g[i];

    double temp[N_MAX * NU];
    mpc_linalg::trsv_lower(n, L, neg_g, temp);
    mpc_linalg::trsv_upper_trans(n, L, temp, U);
}

// ---------------------------------------------------------------------------
// is_feasible: check all elements in [u_min, u_max]
// ---------------------------------------------------------------------------
bool is_feasible(const double* U, int n, double u_min, double u_max)
{
    for (int i = 0; i < n; ++i) {
        if (U[i] < u_min || U[i] > u_max)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// clip_to_bounds: clamp each element, return count of clipped elements
// ---------------------------------------------------------------------------
int clip_to_bounds(double* U, int n, double u_min, double u_max)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (U[i] < u_min) {
            U[i] = u_min;
            ++count;
        } else if (U[i] > u_max) {
            U[i] = u_max;
            ++count;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// box_qp_solve: active-set method for box-constrained QP
//   min 0.5 U' H U + g' U   s.t.  u_min <= U_i <= u_max
// ---------------------------------------------------------------------------
int box_qp_solve(const double* H, const double* L, const double* g,
                 double u_min, double u_max, int n, int max_iter,
                 BoxQPWorkspace& ws, bool skip_unconstrained)
{
    constexpr double BOUND_TOL = 1.0e-12;
    constexpr double KKT_TOL   = 1.0e-10;

    if (!skip_unconstrained) {
        // Step 1: unconstrained solution
        unconstrained_solve(L, g, n, ws.U);

        if (is_feasible(ws.U, n, u_min, u_max))
            return 0;
    }

    // Clip to bounds before entering active-set loop
    clip_to_bounds(ws.U, n, u_min, u_max);

    // Active-set iterations
    int iter = 0;
    for (; iter < max_iter; ++iter) {

        // (a) Compute gradient: grad = H * U + g
        mpc_linalg::gemv(n, n, H, ws.U, ws.grad);
        mpc_linalg::axpy(n, 1.0, g, ws.grad);

        // (b) Classify variables as free or clamped
        int n_free    = 0;
        int n_clamped = 0;

        for (int i = 0; i < n; ++i) {
            bool at_lower = (ws.U[i] <= u_min + BOUND_TOL);
            bool at_upper = (ws.U[i] >= u_max - BOUND_TOL);
            bool clamped  = (at_lower && ws.grad[i] >= 0.0) ||
                            (at_upper && ws.grad[i] <= 0.0);
            if (clamped) {
                ws.clamped_idx[n_clamped++] = i;
            } else {
                ws.free_idx[n_free++] = i;
            }
        }

        // (c) All clamped -- nothing to improve
        if (n_free == 0)
            break;

        // (d) Extract reduced Hessian H_ff (n_free x n_free)
        for (int b = 0; b < n_free; ++b) {
            for (int a = 0; a < n_free; ++a) {
                ws.H_ff[a + n_free * b] = H[ws.free_idx[a] + n * ws.free_idx[b]];
            }
        }

        // (e) Compute reduced RHS
        //     rhs[a] = -g[free_idx[a]]
        //              - sum_j H[free_idx[a], clamped_idx[j]] * U[clamped_idx[j]]
        for (int a = 0; a < n_free; ++a) {
            double r = -g[ws.free_idx[a]];
            for (int j = 0; j < n_clamped; ++j) {
                r -= H[ws.free_idx[a] + n * ws.clamped_idx[j]] * ws.U[ws.clamped_idx[j]];
            }
            ws.rhs[a] = r;
        }

        // (f) Factorize H_ff and solve for free variables
        std::memcpy(ws.L_ff, ws.H_ff,
                    static_cast<std::size_t>(n_free) * static_cast<std::size_t>(n_free) * sizeof(double));
        cholesky_factor_inplace(n_free, ws.L_ff);
        cholesky_solve_inplace(n_free, ws.L_ff, ws.rhs);
        // ws.rhs[0..n_free-1] now holds the candidate free-variable values

        // (g) Line search with bound clamping
        double alpha = 1.0;
        for (int a = 0; a < n_free; ++a) {
            int idx = ws.free_idx[a];
            double d = ws.rhs[a] - ws.U[idx];
            if (d > 0.0 && ws.U[idx] + d > u_max) {
                double a_cand = (u_max - ws.U[idx]) / d;
                if (a_cand < alpha) alpha = a_cand;
            } else if (d < 0.0 && ws.U[idx] + d < u_min) {
                double a_cand = (u_min - ws.U[idx]) / d;
                if (a_cand < alpha) alpha = a_cand;
            }
        }

        // Apply step
        for (int a = 0; a < n_free; ++a) {
            int idx = ws.free_idx[a];
            ws.U[idx] += alpha * (ws.rhs[a] - ws.U[idx]);
        }

        // Snap to bounds to avoid floating-point drift
        clip_to_bounds(ws.U, n, u_min, u_max);

        // (h) KKT check (only meaningful when full step was taken)
        if (alpha >= 1.0 - 1.0e-14) {
            // Recompute gradient at the new point
            mpc_linalg::gemv(n, n, H, ws.U, ws.grad);
            mpc_linalg::axpy(n, 1.0, g, ws.grad);

            bool kkt_ok = true;
            for (int j = 0; j < n_clamped; ++j) {
                int ci = ws.clamped_idx[j];
                bool at_lower = (ws.U[ci] <= u_min + BOUND_TOL);
                bool at_upper = (ws.U[ci] >= u_max - BOUND_TOL);
                if (at_lower && ws.grad[ci] < -KKT_TOL) { kkt_ok = false; break; }
                if (at_upper && ws.grad[ci] >  KKT_TOL) { kkt_ok = false; break; }
            }
            if (kkt_ok)
                break;
        }
    }

    return iter + 1;  // number of iterations consumed (1-based)
}

// ---------------------------------------------------------------------------
// fista_box_qp_solve: FISTA (accelerated projected gradient) for box QP
//   min 0.5 U' H U + g' U   s.t.  u_min <= U_i <= u_max
//   O(n^2) per iteration (one gemv), O(1/k^2) convergence rate
// ---------------------------------------------------------------------------
int fista_box_qp_solve(const double* H, const double* g,
                       double u_min, double u_max, int n, int max_iter,
                       double step_size, BoxQPWorkspace& ws)
{
    constexpr double CONV_TOL = 1.0e-10;

    // V = ws.U (initial extrapolation point)
    std::memcpy(ws.V, ws.U, static_cast<std::size_t>(n) * sizeof(double));

    double t = 1.0;

    for (int k = 0; k < max_iter; ++k) {
        // grad = H * V + g  (into ws.U_old to avoid aliasing — g may be ws.grad)
        mpc_linalg::gemv(n, n, H, ws.V, ws.U_old);
        mpc_linalg::axpy(n, 1.0, g, ws.U_old);

        // U_new = clip(V - step * grad)  (into ws.temp)
        for (int i = 0; i < n; ++i) {
            double val = ws.V[i] - step_size * ws.U_old[i];
            if (val < u_min) val = u_min;
            else if (val > u_max) val = u_max;
            ws.temp[i] = val;
        }

        // Convergence check: max|U_new - U_old| < tol
        double max_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = std::fabs(ws.temp[i] - ws.U[i]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff < CONV_TOL) {
            std::memcpy(ws.U, ws.temp, static_cast<std::size_t>(n) * sizeof(double));
            return k + 1;
        }

        // Gradient restart: if momentum caused overshoot, reset t
        // Check: dot(grad, U_new - U_old) > 0  means we overshot
        double restart_dot = 0.0;
        for (int i = 0; i < n; ++i)
            restart_dot += ws.U_old[i] * (ws.temp[i] - ws.U[i]);
        if (restart_dot > 0.0)
            t = 1.0;

        // FISTA momentum update
        double t_new = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
        double beta = (t - 1.0) / t_new;

        for (int i = 0; i < n; ++i)
            ws.V[i] = ws.temp[i] + beta * (ws.temp[i] - ws.U[i]);

        // U = U_new
        std::memcpy(ws.U, ws.temp, static_cast<std::size_t>(n) * sizeof(double));
        t = t_new;
    }

    return max_iter;
}

// ---------------------------------------------------------------------------
// check_box_kkt: verify KKT conditions for box-constrained QP at point U
//   Computes grad = H*U + g, then checks:
//     - free variables: |grad_i| <= tol
//     - at lower bound: grad_i >= -tol
//     - at upper bound: grad_i <= tol
// ---------------------------------------------------------------------------
bool check_box_kkt(const double* H, const double* g, const double* U,
                   double u_min, double u_max, int n, double* grad_out)
{
    constexpr double BOUND_TOL = 1.0e-12;
    constexpr double KKT_TOL   = 1.0e-10;

    // grad = H * U + g
    mpc_linalg::gemv(n, n, H, U, grad_out);
    mpc_linalg::axpy(n, 1.0, g, grad_out);

    for (int i = 0; i < n; ++i) {
        bool at_lower = (U[i] <= u_min + BOUND_TOL);
        bool at_upper = (U[i] >= u_max - BOUND_TOL);

        if (at_lower && grad_out[i] < -KKT_TOL)
            return false;
        if (at_upper && grad_out[i] > KKT_TOL)
            return false;
        if (!at_lower && !at_upper && std::fabs(grad_out[i]) > KKT_TOL)
            return false;
    }
    return true;
}
