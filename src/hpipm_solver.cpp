#ifdef MPC_USE_HPIPM

#include "qp_solvers.h"

#include <cstdlib>
#include <cstring>

// BLASFEO headers
extern "C" {
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>
}

// HPIPM dense QP headers
extern "C" {
#include <hpipm_d_dense_qp_dim.h>
#include <hpipm_d_dense_qp.h>
#include <hpipm_d_dense_qp_sol.h>
#include <hpipm_d_dense_qp_ipm.h>
}

// ---------------------------------------------------------------------------
// Workspace lifecycle
// ---------------------------------------------------------------------------

void hpipm_workspace_init(HpipmWorkspace& ws, int n)
{
    ws.n_alloc = n;

    // Compute memory sizes by building temporary dim and arg
    hpipm_size_t dim_size = d_dense_qp_dim_memsize();
    void* dim_mem = std::malloc(dim_size);
    struct d_dense_qp_dim dim;
    d_dense_qp_dim_create(&dim, dim_mem);
    d_dense_qp_dim_set_all(n, 0, n, 0, 0, &dim);  // nv=n, ne=0, nb=n, ng=0, ns=0

    hpipm_size_t qp_size  = d_dense_qp_memsize(&dim);
    hpipm_size_t sol_size = d_dense_qp_sol_memsize(&dim);
    hpipm_size_t arg_size = d_dense_qp_ipm_arg_memsize(&dim);

    // Need arg to compute ipm_ws size
    void* arg_mem = std::malloc(arg_size);
    struct d_dense_qp_ipm_arg arg;
    d_dense_qp_ipm_arg_create(&dim, &arg, arg_mem);
    d_dense_qp_ipm_arg_set_default(SPEED, &arg);

    hpipm_size_t ipm_size = d_dense_qp_ipm_ws_memsize(&dim, &arg);

    std::free(arg_mem);
    std::free(dim_mem);

    // Allocate single block with generous alignment padding
    // Need space for: dim structs (5) + dim_mem + qp_mem + sol_mem + arg_mem + ipm_mem
    hpipm_size_t total = sizeof(struct d_dense_qp_dim)
                       + sizeof(struct d_dense_qp)
                       + sizeof(struct d_dense_qp_sol)
                       + sizeof(struct d_dense_qp_ipm_arg)
                       + sizeof(struct d_dense_qp_ipm_ws)
                       + dim_size + qp_size + sol_size + arg_size + ipm_size
                       + 6 * 64;  // alignment padding

    ws.memory = std::malloc(total);
    ws.memory_size = static_cast<int>(total);
    std::memset(ws.memory, 0, total);
}

void hpipm_workspace_free(HpipmWorkspace& ws)
{
    if (ws.memory) {
        std::free(ws.memory);
        ws.memory = nullptr;
    }
    ws.memory_size = 0;
    ws.n_alloc = 0;
}

// ---------------------------------------------------------------------------
// Dense QP solve
// ---------------------------------------------------------------------------
int hpipm_box_qp_solve(const double* H, const double* g,
                       double u_min, double u_max, int n,
                       const double* U_warm, double* U_out,
                       HpipmWorkspace& ws)
{
    if (ws.n_alloc != n) {
        hpipm_workspace_free(ws);
        hpipm_workspace_init(ws, n);
    }

    char* ptr = static_cast<char*>(ws.memory);
    auto align64 = [](char*& p) {
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 63) & ~63ULL);
    };

    // Place structs at the beginning
    auto* dim    = reinterpret_cast<struct d_dense_qp_dim*>(ptr);     ptr += sizeof(*dim);
    auto* qp     = reinterpret_cast<struct d_dense_qp*>(ptr);        ptr += sizeof(*qp);
    auto* sol    = reinterpret_cast<struct d_dense_qp_sol*>(ptr);    ptr += sizeof(*sol);
    auto* arg    = reinterpret_cast<struct d_dense_qp_ipm_arg*>(ptr); ptr += sizeof(*arg);
    auto* ipm_ws = reinterpret_cast<struct d_dense_qp_ipm_ws*>(ptr); ptr += sizeof(*ipm_ws);

    // Dim memory
    align64(ptr);
    d_dense_qp_dim_create(dim, ptr);
    d_dense_qp_dim_set_all(n, 0, n, 0, 0, dim);
    ptr += d_dense_qp_dim_memsize();

    // QP memory
    align64(ptr);
    d_dense_qp_create(dim, qp, ptr);
    ptr += d_dense_qp_memsize(dim);

    // Sol memory
    align64(ptr);
    d_dense_qp_sol_create(dim, sol, ptr);
    ptr += d_dense_qp_sol_memsize(dim);

    // Arg memory
    align64(ptr);
    d_dense_qp_ipm_arg_create(dim, arg, ptr);
    d_dense_qp_ipm_arg_set_default(SPEED, arg);
    ptr += d_dense_qp_ipm_arg_memsize(dim);

    // IPM workspace memory
    align64(ptr);
    d_dense_qp_ipm_ws_create(dim, arg, ipm_ws, ptr);

    // --- Pack problem data using typed setters ---
    d_dense_qp_set_H(const_cast<double*>(H), qp);
    d_dense_qp_set_g(const_cast<double*>(g), qp);

    // Box constraint indices: [0, 1, ..., n-1]
    int* idxb = static_cast<int*>(std::malloc(n * sizeof(int)));
    for (int i = 0; i < n; ++i) idxb[i] = i;
    d_dense_qp_set_idxb(idxb, qp);

    // Bounds
    double* lb = static_cast<double*>(std::malloc(n * sizeof(double)));
    double* ub = static_cast<double*>(std::malloc(n * sizeof(double)));
    for (int i = 0; i < n; ++i) {
        lb[i] = u_min;
        ub[i] = u_max;
    }
    d_dense_qp_set_lb(lb, qp);
    d_dense_qp_set_ub(ub, qp);

    // Warm-start
    if (U_warm) {
        d_dense_qp_sol_set_v(const_cast<double*>(U_warm), sol);
        int warm = 1;
        d_dense_qp_ipm_arg_set_warm_start(&warm, arg);
    }

    // Solve
    d_dense_qp_ipm_solve(qp, sol, arg, ipm_ws);

    // Extract solution
    d_dense_qp_sol_get_v(sol, U_out);

    int iter_count = 0;
    d_dense_qp_ipm_get_iter(ipm_ws, &iter_count);

    std::free(idxb);
    std::free(lb);
    std::free(ub);

    return iter_count;
}

#endif // MPC_USE_HPIPM
