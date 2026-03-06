#pragma once

#include "mpc_types.h"

// ---------------------------------------------------------------------------
// Solver type enum
// ---------------------------------------------------------------------------
enum class QpSolverType {
    FISTA,
    HPIPM_OCP
};

// ---------------------------------------------------------------------------
// Optional solver workspaces (conditionally compiled)
// ---------------------------------------------------------------------------
#ifdef MPC_USE_HPIPM

struct HpipmOcpWorkspace {
    void* memory;       // single aligned allocation for all HPIPM OCP structs
    int memory_size;
    int N_alloc;        // horizon length this workspace was allocated for

    // Persistent HPIPM struct pointers (cast to proper types in .cpp)
    void* dim;
    void* qp;
    void* sol;
    void* arg;
    void* ipm_ws;
    bool structures_created;  // true after _create calls done
    bool static_data_set;     // true after Q/R/bounds set
    double cached_u_min, cached_u_max;

    // Warm-start cache: primal solution from previous solve, shifted by 1 step
    bool   warm_valid;
    double ws_u[(N_MAX) * NU];          // controls  u[0..N-1]
    double ws_x[(N_MAX + 1) * NX];     // states    x[0..N]
};

void hpipm_ocp_workspace_init(HpipmOcpWorkspace& ws, int N);
void hpipm_ocp_workspace_free(HpipmOcpWorkspace& ws);

// Solve OCP-structured QP using HPIPM Riccati-based IPM.
// A_d: single NX*NX matrix (same for all stages), B_list: N x (NX*NU)
// Returns IPM iteration count.
int hpipm_ocp_qp_solve(const double* A_d, const double* B_list,
                        const double* Q, const double* Qf, const double* R,
                        const double* x_ref_consistent, const double* u_ref,
                        const double x0[NX],
                        double u_min, double u_max, int N,
                        double* U_out,
                        HpipmOcpWorkspace& ws);

#endif // MPC_USE_HPIPM

// ---------------------------------------------------------------------------
// Unified solver context — holds workspaces for all enabled solvers
// ---------------------------------------------------------------------------
struct SolverContext {
    BoxQPWorkspace box_ws;     // always available (FISTA)

#ifdef MPC_USE_HPIPM
    HpipmOcpWorkspace hpipm_ocp_ws;
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
            return true;
        case QpSolverType::HPIPM_OCP:
#ifdef MPC_USE_HPIPM
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
        case QpSolverType::HPIPM_OCP:  return "hpipm_ocp";
    }
    return "unknown";
}
