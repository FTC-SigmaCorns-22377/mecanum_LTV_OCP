#include "qp_solvers.h"
#include "ipm_solver.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"
#include <cstring>
#include <time.h>

// ---------------------------------------------------------------------------
// Solver context init/free
// ---------------------------------------------------------------------------

void solver_context_init(SolverContext& ctx, int n)
{
    std::memset(&ctx.box_ws, 0, sizeof(ctx.box_ws));

#ifdef MPC_USE_HPIPM
    std::memset(&ctx.hpipm_ocp_ws, 0, sizeof(ctx.hpipm_ocp_ws));
    // OCP workspace is lazily initialized on first use
#endif

    ctx.ipm_ws = new IpmWorkspace();
    ipm_workspace_init(*ctx.ipm_ws);
    ctx.ipm_config = new IpmSolverConfig(ipm_solver_default_config());

    (void)n;  // suppress unused warning when no optional solvers
}

void solver_context_free(SolverContext& ctx)
{
#ifdef MPC_USE_HPIPM
    hpipm_ocp_workspace_free(ctx.hpipm_ocp_ws);
#endif

    delete ctx.ipm_ws;
    ctx.ipm_ws = nullptr;
    delete ctx.ipm_config;
    ctx.ipm_config = nullptr;
}

// ---------------------------------------------------------------------------
// Unified MPC solve
// ---------------------------------------------------------------------------

QPSolution mpc_solve_with_solver(const PrecomputedWindow& window,
                                 const double x0[NX],
                                 const MPCConfig& config,
                                 QpSolverType solver_type,
                                 SolverContext& ctx)
{
    BoxQPWorkspace& ws = ctx.box_ws;
    const int n_vars = window.n_vars;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Step 1: error e0 = x0 - x_ref_0
    double e0[NX];
    for (int i = 0; i < NX; ++i)
        e0[i] = x0[i] - window.x_ref_0[i];

    // Step 2: form gradient  g = F * e0 + f_const
    mpc_linalg::gemv(n_vars, NX, window.F, e0, ws.grad);
    mpc_linalg::axpy(n_vars, 1.0, window.f_const, ws.grad);

    // Step 3: warm-start shift
    bool have_warm = false;
    if (ws.warm_valid && ws.prev_n_vars == n_vars) {
        for (int i = 0; i < n_vars - NU; ++i)
            ws.U[i] = ws.U_prev[i + NU];
        for (int i = n_vars - NU; i < n_vars; ++i)
            ws.U[i] = 0.0;
        have_warm = true;
    }

    // Step 4: dispatch to solver
    int n_iter = 0;

    switch (solver_type) {

    case QpSolverType::FISTA: {
        double step_size = 1.0 / window.lambda_max;
        if (have_warm) {
            // Check if warm-start already satisfies KKT
            if (is_feasible(ws.U, n_vars, config.u_min, config.u_max) &&
                check_box_kkt(window.H, ws.grad, ws.U,
                              config.u_min, config.u_max, n_vars, ws.temp)) {
                n_iter = 0;  // accept directly
            } else {
                n_iter = fista_box_qp_solve(window.H, ws.grad,
                                            config.u_min, config.u_max,
                                            n_vars, 50, step_size, ws);
            }
        } else {
            // Cold start: unconstrained solve
            for (int i = 0; i < n_vars; ++i) ws.temp[i] = -ws.grad[i];
            mpc_linalg::trsv_lower(n_vars, window.L, ws.temp, ws.rhs);
            mpc_linalg::trsv_upper_trans(n_vars, window.L, ws.rhs, ws.U);
            if (!is_feasible(ws.U, n_vars, config.u_min, config.u_max)) {
                clip_to_bounds(ws.U, n_vars, config.u_min, config.u_max);
                n_iter = fista_box_qp_solve(window.H, ws.grad,
                                            config.u_min, config.u_max,
                                            n_vars, 50, step_size, ws);
            }
        }
        break;
    }

    case QpSolverType::HPIPM_OCP:
    case QpSolverType::NEON_IPM:
        // HPIPM OCP and NEON IPM are called directly from heading_lookup_online,
        // not through this condensed-QP dispatch. Fall through to default.
    default:
        // Solver not available in this build — zero output
        std::memset(ws.U, 0, n_vars * sizeof(double));
        break;
    }

    // Store for next warm-start
    std::memcpy(ws.U_prev, ws.U, static_cast<std::size_t>(n_vars) * sizeof(double));
    ws.prev_n_vars = n_vars;
    ws.warm_valid = true;

    // Stop timing
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);

    // Package solution
    QPSolution sol;
    for (int i = 0; i < NU; ++i)
        sol.u0[i] = ws.U[i];
    for (int i = 0; i < n_vars; ++i)
        sol.U[i] = ws.U[i];

    sol.n_iterations = n_iter;

    int n_active = 0;
    for (int i = 0; i < n_vars; ++i) {
        if (ws.U[i] <= config.u_min || ws.U[i] >= config.u_max)
            ++n_active;
    }
    sol.n_active = n_active;
    sol.solve_time_ns = elapsed_ns;

    return sol;
}
