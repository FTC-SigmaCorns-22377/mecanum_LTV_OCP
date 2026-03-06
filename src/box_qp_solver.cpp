#include "box_qp_solver.h"
#include "blas_dispatch.h"

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
// fista_box_qp_solve: FISTA (accelerated projected gradient) for box QP
//   min 0.5 U' H U + g' U   s.t.  u_min <= U_i <= u_max
//   O(n^2) per iteration (one gemv), O(1/k^2) convergence rate
// ---------------------------------------------------------------------------
int fista_box_qp_solve(const double* H, const double* g,
                       double u_min, double u_max, int n, int max_iter,
                       double step_size, BoxQPWorkspace& ws)
{
    constexpr double CONV_TOL = 1.0e-6;

    // V = ws.U (initial extrapolation point)
    std::memcpy(ws.V, ws.U, static_cast<std::size_t>(n) * sizeof(double));

    // Precompute step_size * g (loop-invariant)
    double g_scaled[N_MAX * NU];
    for (int i = 0; i < n; ++i)
        g_scaled[i] = step_size * g[i];

    double t = 1.0;

    for (int k = 0; k < max_iter; ++k) {
        // Compute step_size * (H * V + g) using dsymv:
        //   load g_scaled into U_old, then dsymv with alpha=step_size, beta=1.0
        //   gives step_size * H * V + step_size * g = step_size * grad
        std::memcpy(ws.U_old, g_scaled, static_cast<std::size_t>(n) * sizeof(double));
        mpc_linalg::symv_full(n, step_size, H, ws.V, 1.0, ws.U_old);
        // ws.U_old now = step_size * (H * V + g)

        // Fused: projected step + convergence check + restart dot
        double max_diff = 0.0;
        double restart_dot = 0.0;
        for (int i = 0; i < n; ++i) {
            double val = ws.V[i] - ws.U_old[i];
            if (val < u_min) val = u_min;
            else if (val > u_max) val = u_max;
            double diff = val - ws.U[i];
            double ad = (diff >= 0) ? diff : -diff;
            if (ad > max_diff) max_diff = ad;
            restart_dot += ws.U_old[i] * diff;
            ws.temp[i] = val;
        }

        if (max_diff < CONV_TOL) {
            std::memcpy(ws.U, ws.temp, static_cast<std::size_t>(n) * sizeof(double));
            return k + 1;
        }

        // Gradient restart: if momentum caused overshoot, reset t
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

    // grad = H * U + g  (H is symmetric, use dsymv)
    std::memcpy(grad_out, g, static_cast<std::size_t>(n) * sizeof(double));
    mpc_linalg::symv_full(n, 1.0, H, U, 1.0, grad_out);

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
