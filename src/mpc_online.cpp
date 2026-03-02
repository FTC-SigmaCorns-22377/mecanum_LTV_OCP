#include "mpc_online.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"
#include <cstring>
#include <time.h>

QPSolution mpc_solve_online(const PrecomputedWindow& window, const double x0[NX],
                            const MPCConfig& config, BoxQPWorkspace& workspace)
{
    const int n_vars = window.n_vars;

    // ---- start timing ----
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Step 1: error e0 = x0 - x_ref_0
    double e0[NX];
    for (int i = 0; i < NX; ++i) {
        e0[i] = x0[i] - window.x_ref_0[i];
    }

    // Step 2: form gradient  g = F * e0 + f_const
    mpc_linalg::gemv(n_vars, NX, window.F, e0, workspace.grad);
    mpc_linalg::axpy(n_vars, 1.0, window.f_const, workspace.grad);

    int n_iter = 0;
    bool warm_hit = false;
    double step_size = 1.0 / window.lambda_max;

    // ---- Path A: try shifted warm-start ----
    if (workspace.warm_valid && workspace.prev_n_vars == n_vars) {
        // Shift previous solution by one timestep
        for (int i = 0; i < n_vars - NU; ++i)
            workspace.U[i] = workspace.U_prev[i + NU];
        for (int i = n_vars - NU; i < n_vars; ++i)
            workspace.U[i] = 0.0;

        // Accept if shifted solution satisfies KKT for the new QP
        if (is_feasible(workspace.U, n_vars, config.u_min, config.u_max) &&
            check_box_kkt(window.H, workspace.grad, workspace.U,
                          config.u_min, config.u_max, n_vars, workspace.temp)) {
            warm_hit = true;  // 0 iterations — use shifted solution directly
        } else {
            // FISTA from shifted warm-start
            n_iter = fista_box_qp_solve(window.H, workspace.grad,
                                        config.u_min, config.u_max, n_vars, 50,
                                        step_size, workspace);
        }
    }

    // ---- Path B: cold start (first call, n_vars mismatch) ----
    if (!warm_hit && n_iter == 0) {
        // Unconstrained solve via precomputed Cholesky
        for (int i = 0; i < n_vars; ++i)
            workspace.temp[i] = -workspace.grad[i];
        mpc_linalg::trsv_lower(n_vars, window.L, workspace.temp, workspace.rhs);
        mpc_linalg::trsv_upper_trans(n_vars, window.L, workspace.rhs, workspace.U);

        if (!is_feasible(workspace.U, n_vars, config.u_min, config.u_max)) {
            // FISTA from clipped unconstrained solution
            clip_to_bounds(workspace.U, n_vars, config.u_min, config.u_max);
            n_iter = fista_box_qp_solve(window.H, workspace.grad,
                                        config.u_min, config.u_max, n_vars, 50,
                                        step_size, workspace);
        }
    }

    // Store solution for next call's warm-start
    std::memcpy(workspace.U_prev, workspace.U, static_cast<std::size_t>(n_vars) * sizeof(double));
    workspace.prev_n_vars = n_vars;
    workspace.warm_valid = true;

    // ---- stop timing ----
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);

    // Step 5: package solution
    QPSolution sol;

    // First NU elements -> u0
    for (int i = 0; i < NU; ++i) {
        sol.u0[i] = workspace.U[i];
    }

    // Full input sequence
    for (int i = 0; i < n_vars; ++i) {
        sol.U[i] = workspace.U[i];
    }

    sol.n_iterations = n_iter;

    // Count active constraints (elements at bounds)
    int n_active = 0;
    for (int i = 0; i < n_vars; ++i) {
        if (workspace.U[i] <= config.u_min || workspace.U[i] >= config.u_max) {
            ++n_active;
        }
    }
    sol.n_active = n_active;
    sol.solve_time_ns = elapsed_ns;

    return sol;
}
