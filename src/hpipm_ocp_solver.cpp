#ifdef MPC_USE_HPIPM

#include "qp_solvers.h"

#include <cstdlib>
#include <cstring>

// BLASFEO headers
extern "C" {
#include <blasfeo_d_aux.h>
}

// HPIPM OCP QP headers
extern "C" {
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>
}

// ---------------------------------------------------------------------------
// Workspace lifecycle
// ---------------------------------------------------------------------------

void hpipm_ocp_workspace_init(HpipmOcpWorkspace& ws, int N)
{
    ws.N_alloc = N;
    ws.structures_created = false;
    ws.static_data_set = false;
    ws.warm_valid = false;
    ws.dim = nullptr;
    ws.qp = nullptr;
    ws.sol = nullptr;
    ws.arg = nullptr;
    ws.ipm_ws = nullptr;

    // Compute memory sizes using temporary structs
    hpipm_size_t dim_size = d_ocp_qp_dim_memsize(N);
    void* dim_mem = std::malloc(dim_size);
    struct d_ocp_qp_dim tmp_dim;
    d_ocp_qp_dim_create(N, &tmp_dim, dim_mem);

    int nx[N_MAX + 1], nu_arr[N_MAX + 1], nbx[N_MAX + 1], nbu[N_MAX + 1];
    int ng[N_MAX + 1], ns[N_MAX + 1];

    for (int k = 0; k <= N; ++k) {
        nx[k] = NX;
        nu_arr[k] = (k < N) ? NU : 0;
        nbx[k] = (k == 0) ? NX : 0;
        nbu[k] = (k < N) ? NU : 0;
        ng[k] = 0;
        ns[k] = 0;
    }

    d_ocp_qp_dim_set_all(nx, nu_arr, nbx, nbu, ng, ns, &tmp_dim);

    hpipm_size_t qp_size  = d_ocp_qp_memsize(&tmp_dim);
    hpipm_size_t sol_size = d_ocp_qp_sol_memsize(&tmp_dim);
    hpipm_size_t arg_size = d_ocp_qp_ipm_arg_memsize(&tmp_dim);

    void* arg_mem = std::malloc(arg_size);
    struct d_ocp_qp_ipm_arg tmp_arg;
    d_ocp_qp_ipm_arg_create(&tmp_dim, &tmp_arg, arg_mem);
    d_ocp_qp_ipm_arg_set_default(SPEED, &tmp_arg);

    hpipm_size_t ipm_size = d_ocp_qp_ipm_ws_memsize(&tmp_dim, &tmp_arg);

    std::free(arg_mem);
    std::free(dim_mem);

    // Single allocation
    hpipm_size_t total = sizeof(struct d_ocp_qp_dim)
                       + sizeof(struct d_ocp_qp)
                       + sizeof(struct d_ocp_qp_sol)
                       + sizeof(struct d_ocp_qp_ipm_arg)
                       + sizeof(struct d_ocp_qp_ipm_ws)
                       + dim_size + qp_size + sol_size + arg_size + ipm_size
                       + 6 * 64;  // alignment padding

    ws.memory = std::malloc(total);
    ws.memory_size = static_cast<int>(total);
    std::memset(ws.memory, 0, total);

    // Create all HPIPM structures once
    char* ptr = static_cast<char*>(ws.memory);
    auto align64 = [](char*& p) {
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 63) & ~63ULL);
    };

    auto* dim    = reinterpret_cast<struct d_ocp_qp_dim*>(ptr);     ptr += sizeof(*dim);
    auto* qp     = reinterpret_cast<struct d_ocp_qp*>(ptr);        ptr += sizeof(*qp);
    auto* sol    = reinterpret_cast<struct d_ocp_qp_sol*>(ptr);    ptr += sizeof(*sol);
    auto* arg    = reinterpret_cast<struct d_ocp_qp_ipm_arg*>(ptr); ptr += sizeof(*arg);
    auto* ipm_ws_ptr = reinterpret_cast<struct d_ocp_qp_ipm_ws*>(ptr); ptr += sizeof(*ipm_ws_ptr);

    align64(ptr);
    d_ocp_qp_dim_create(N, dim, ptr);
    ptr += dim_size;

    d_ocp_qp_dim_set_all(nx, nu_arr, nbx, nbu, ng, ns, dim);

    align64(ptr);
    d_ocp_qp_create(dim, qp, ptr);
    ptr += qp_size;

    align64(ptr);
    d_ocp_qp_sol_create(dim, sol, ptr);
    ptr += sol_size;

    align64(ptr);
    d_ocp_qp_ipm_arg_create(dim, arg, ptr);
    d_ocp_qp_ipm_arg_set_default(SPEED, arg);
    int warm_start = 1;  // primal-only warm start (dual warm-start biases convergence near path end)
    d_ocp_qp_ipm_arg_set_warm_start(&warm_start, arg);
    ptr += arg_size;

    align64(ptr);
    d_ocp_qp_ipm_ws_create(dim, arg, ipm_ws_ptr, ptr);

    // Set b_k = 0 for all stages (never changes)
    double b_zero[NX] = {};
    for (int k = 0; k < N; ++k)
        d_ocp_qp_set_b(k, b_zero, qp);

    // Set idxbu indices (never change)
    int idxbu[NU];
    for (int j = 0; j < NU; ++j)
        idxbu[j] = j;
    for (int k = 0; k < N; ++k)
        d_ocp_qp_set_idxbu(k, idxbu, qp);

    // Set idxbx for stage 0 (never changes)
    int idxbx[NX];
    for (int i = 0; i < NX; ++i)
        idxbx[i] = i;
    d_ocp_qp_set_idxbx(0, idxbx, qp);

    ws.dim = dim;
    ws.qp = qp;
    ws.sol = sol;
    ws.arg = arg;
    ws.ipm_ws = ipm_ws_ptr;
    ws.structures_created = true;
}

void hpipm_ocp_workspace_free(HpipmOcpWorkspace& ws)
{
    if (ws.memory) {
        std::free(ws.memory);
        ws.memory = nullptr;
    }
    ws.memory_size = 0;
    ws.N_alloc = 0;
    ws.structures_created = false;
    ws.static_data_set = false;
    ws.dim = nullptr;
    ws.qp = nullptr;
    ws.sol = nullptr;
    ws.arg = nullptr;
    ws.ipm_ws = nullptr;
}

// ---------------------------------------------------------------------------
// OCP QP solve
// ---------------------------------------------------------------------------
int hpipm_ocp_qp_solve(const double* A_d, const double* B_list,
                        const double* Q, const double* Qf, const double* R,
                        const double* x_ref_consistent, const double* u_ref,
                        const double x0[NX],
                        double u_min, double u_max, int N,
                        double* U_out,
                        HpipmOcpWorkspace& ws)
{
    if (ws.N_alloc != N || !ws.structures_created) {
        hpipm_ocp_workspace_free(ws);
        hpipm_ocp_workspace_init(ws, N);
    }

    auto* qp     = static_cast<struct d_ocp_qp*>(ws.qp);
    auto* sol    = static_cast<struct d_ocp_qp_sol*>(ws.sol);
    auto* arg    = static_cast<struct d_ocp_qp_ipm_arg*>(ws.arg);
    auto* ipm_ws = static_cast<struct d_ocp_qp_ipm_ws*>(ws.ipm_ws);

    // Set static data once (Q, R, Qf, bounds) — only on first solve or bound change
    if (!ws.static_data_set || ws.cached_u_min != u_min || ws.cached_u_max != u_max) {
        double lb[NU], ub[NU];
        for (int j = 0; j < NU; ++j) {
            lb[j] = u_min;
            ub[j] = u_max;
        }

        for (int k = 0; k < N; ++k) {
            d_ocp_qp_set_Q(k, const_cast<double*>(Q), qp);
            d_ocp_qp_set_R(k, const_cast<double*>(R), qp);
            d_ocp_qp_set_lbu(k, lb, qp);
            d_ocp_qp_set_ubu(k, ub, qp);
        }
        d_ocp_qp_set_Q(N, const_cast<double*>(Qf), qp);

        ws.cached_u_min = u_min;
        ws.cached_u_max = u_max;
        ws.static_data_set = true;
    }

    // --- Set per-solve data ---

    // Dynamics: same A_d for all stages, per-stage B_k
    for (int k = 0; k < N; ++k) {
        d_ocp_qp_set_A(k, const_cast<double*>(A_d), qp);
        d_ocp_qp_set_B(k, const_cast<double*>(B_list + k * NX * NU), qp);
    }

    // Linear cost terms (exploit diagonal Q and R)
    double q_k[NX], r_k[NU];
    for (int k = 0; k < N; ++k) {
        const double* x_ref_k = x_ref_consistent + k * NX;
        const double* u_ref_k = u_ref + k * NU;

        for (int i = 0; i < NX; ++i)
            q_k[i] = -Q[i + NX * i] * x_ref_k[i];
        d_ocp_qp_set_q(k, q_k, qp);

        for (int i = 0; i < NU; ++i)
            r_k[i] = -R[i + NU * i] * u_ref_k[i];
        d_ocp_qp_set_r(k, r_k, qp);
    }

    // Terminal linear cost (diagonal Qf)
    const double* x_ref_N = x_ref_consistent + N * NX;
    double qf_k[NX];
    for (int i = 0; i < NX; ++i)
        qf_k[i] = -Qf[i + NX * i] * x_ref_N[i];
    d_ocp_qp_set_q(N, qf_k, qp);

    // Initial state constraint
    d_ocp_qp_set_lbx(0, const_cast<double*>(x0), qp);
    d_ocp_qp_set_ubx(0, const_cast<double*>(x0), qp);

    // Populate warm-start initial point
    if (ws.warm_valid) {
        // Controls: shift left, repeat last stage
        for (int k = 0; k < N - 1; ++k)
            d_ocp_qp_sol_set_u(k, ws.ws_u + (k + 1) * NU, sol);
        d_ocp_qp_sol_set_u(N - 1, ws.ws_u + (N - 1) * NU, sol);

        // States: shift left, repeat last stage (skip k=0: x0 is fixed by constraint)
        for (int k = 1; k <= N - 1; ++k)
            d_ocp_qp_sol_set_x(k, ws.ws_x + (k + 1) * NX, sol);
        d_ocp_qp_sol_set_x(N, ws.ws_x + N * NX, sol);

        // Dual variables (pi, lam) not warm-started: stale duals near path end
        // cause IPM to converge to biased critical points. Primal-only is sufficient.
    }

    // Solve
    d_ocp_qp_ipm_solve(qp, sol, arg, ipm_ws);

    // Extract solution
    for (int k = 0; k < N; ++k) {
        double u_k[NU];
        d_ocp_qp_sol_get_u(k, sol, u_k);
        std::memcpy(U_out + k * NU, u_k, NU * sizeof(double));
    }

    // Cache primal solution for next warm start
    for (int k = 0; k < N; ++k) {
        d_ocp_qp_sol_get_u(k, sol, ws.ws_u + k * NU);
        d_ocp_qp_sol_get_x(k, sol, ws.ws_x + k * NX);
    }
    d_ocp_qp_sol_get_x(N, sol, ws.ws_x + N * NX);
    ws.warm_valid = true;

    int iter_count = 0;
    d_ocp_qp_ipm_get_iter(ipm_ws, &iter_count);

    return iter_count;
}

#endif // MPC_USE_HPIPM
