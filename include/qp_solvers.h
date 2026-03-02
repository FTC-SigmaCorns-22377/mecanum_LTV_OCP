#pragma once

#include "mpc_types.h"

// ---------------------------------------------------------------------------
// Solver type enum
// ---------------------------------------------------------------------------
enum class QpSolverType {
    FISTA,
    ACTIVE_SET,
    HPIPM,
    QPOASES
};

// ---------------------------------------------------------------------------
// Optional solver workspaces (conditionally compiled)
// ---------------------------------------------------------------------------
#ifdef MPC_USE_HPIPM

struct HpipmWorkspace {
    void* memory;       // single aligned allocation for all HPIPM/BLASFEO structs
    int memory_size;
    int n_alloc;        // dimension this workspace was allocated for
};

void hpipm_workspace_init(HpipmWorkspace& ws, int n);
void hpipm_workspace_free(HpipmWorkspace& ws);

// Solve box-constrained QP using HPIPM dense interior-point method.
// H: n x n Hessian (column-major, symmetric), g: gradient, lb/ub: bounds
// U_warm: warm-start primal (may be null), U_out: solution output
// Returns IPM iteration count.
int hpipm_box_qp_solve(const double* H, const double* g,
                       double u_min, double u_max, int n,
                       const double* U_warm, double* U_out,
                       HpipmWorkspace& ws);

#endif // MPC_USE_HPIPM

#ifdef MPC_USE_QPOASES

struct QpoasesWorkspace {
    void* qp;           // opaque QProblemB pointer
    int n_alloc;        // dimension this workspace was allocated for
};

void qpoases_workspace_init(QpoasesWorkspace& ws, int n);
void qpoases_workspace_free(QpoasesWorkspace& ws);

// Solve box-constrained QP using qpOASES active-set method.
// Returns number of working set recalculations.
int qpoases_box_qp_solve(const double* H, const double* g,
                          double u_min, double u_max, int n,
                          const double* U_warm, double* U_out,
                          QpoasesWorkspace& ws);

#endif // MPC_USE_QPOASES

// ---------------------------------------------------------------------------
// Unified solver context — holds workspaces for all enabled solvers
// ---------------------------------------------------------------------------
struct SolverContext {
    BoxQPWorkspace box_ws;     // always available (FISTA / active-set)

#ifdef MPC_USE_HPIPM
    HpipmWorkspace hpipm_ws;
#endif
#ifdef MPC_USE_QPOASES
    QpoasesWorkspace qpoases_ws;
#endif
};

// Initialize all workspaces in the context for problems of dimension n.
void solver_context_init(SolverContext& ctx, int n);

// Free dynamically allocated workspaces.
void solver_context_free(SolverContext& ctx);

// ---------------------------------------------------------------------------
// Unified MPC solve — dispatches to the requested solver
// ---------------------------------------------------------------------------
QPSolution mpc_solve_with_solver(const PrecomputedWindow& window,
                                 const double x0[NX],
                                 const MPCConfig& config,
                                 QpSolverType solver_type,
                                 SolverContext& ctx);

// Check if a solver type is available in this build.
inline bool solver_available(QpSolverType type) {
    switch (type) {
        case QpSolverType::FISTA:
        case QpSolverType::ACTIVE_SET:
            return true;
        case QpSolverType::HPIPM:
#ifdef MPC_USE_HPIPM
            return true;
#else
            return false;
#endif
        case QpSolverType::QPOASES:
#ifdef MPC_USE_QPOASES
            return true;
#else
            return false;
#endif
    }
    return false;
}

inline const char* solver_name(QpSolverType type) {
    switch (type) {
        case QpSolverType::FISTA:      return "fista";
        case QpSolverType::ACTIVE_SET: return "active_set";
        case QpSolverType::HPIPM:      return "hpipm";
        case QpSolverType::QPOASES:    return "qpoases";
    }
    return "unknown";
}
