#ifdef MPC_USE_QPOASES

#include "qp_solvers.h"

#include <qpOASES.hpp>
#include <cstring>

// ---------------------------------------------------------------------------
// Workspace lifecycle
// ---------------------------------------------------------------------------

void qpoases_workspace_init(QpoasesWorkspace& ws, int n)
{
    auto* qp = new qpOASES::QProblemB(n);

    // Configure for MPC speed
    qpOASES::Options opts;
    opts.setToMPC();
    opts.printLevel = qpOASES::PL_NONE;
    qp->setOptions(opts);

    ws.qp = static_cast<void*>(qp);
    ws.n_alloc = n;
}

void qpoases_workspace_free(QpoasesWorkspace& ws)
{
    if (ws.qp) {
        delete static_cast<qpOASES::QProblemB*>(ws.qp);
        ws.qp = nullptr;
    }
    ws.n_alloc = 0;
}

// ---------------------------------------------------------------------------
// Dense QP solve
// ---------------------------------------------------------------------------
int qpoases_box_qp_solve(const double* H, const double* g,
                          double u_min, double u_max, int n,
                          const double* U_warm, double* U_out,
                          QpoasesWorkspace& ws)
{
    // Reallocate if dimension changed
    if (ws.n_alloc != n) {
        qpoases_workspace_free(ws);
        qpoases_workspace_init(ws, n);
    }

    auto* qp = static_cast<qpOASES::QProblemB*>(ws.qp);

    // Build bound vectors
    // Stack-allocate for typical sizes, heap for very large
    double lb_buf[N_MAX * NU];
    double ub_buf[N_MAX * NU];
    double* lb = (n <= N_MAX * NU) ? lb_buf : new double[n];
    double* ub = (n <= N_MAX * NU) ? ub_buf : new double[n];

    for (int i = 0; i < n; ++i) {
        lb[i] = u_min;
        ub[i] = u_max;
    }

    // qpOASES uses row-major H, but our H is symmetric so it doesn't matter.
    // We use init() each time since H changes per window.
    qpOASES::int_t nWSR = 200;  // max working set recalculations

    // Reset the problem for a fresh init
    qp->reset();
    qpOASES::Options opts;
    opts.setToMPC();
    opts.printLevel = qpOASES::PL_NONE;
    qp->setOptions(opts);

    qpOASES::returnValue ret = qp->init(H, g, lb, ub, nWSR);

    if (ret != qpOASES::SUCCESSFUL_RETURN) {
        // Fallback: zero solution
        std::memset(U_out, 0, n * sizeof(double));
        if (n > N_MAX * NU) { delete[] lb; delete[] ub; }
        return static_cast<int>(nWSR);
    }

    // Extract solution
    qp->getPrimalSolution(U_out);

    if (n > N_MAX * NU) {
        delete[] lb;
        delete[] ub;
    }

    return static_cast<int>(nWSR);
}

#endif // MPC_USE_QPOASES
