#include "condensing.h"
#include "cholesky.h"
#include "blas_dispatch.h"

#include <cstdio>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// power_iteration_lambda_max: estimate largest eigenvalue of H via power method
// ---------------------------------------------------------------------------
static double power_iteration_lambda_max(const double* H, int n, int n_iters = 30)
{
    double v[N_MAX * NU];
    double Hv[N_MAX * NU];

    // Initialize v = [1, 1, ..., 1] / sqrt(n)
    double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
    for (int i = 0; i < n; ++i)
        v[i] = inv_sqrt_n;

    double lambda = 0.0;
    for (int iter = 0; iter < n_iters; ++iter) {
        // Hv = H * v
        mpc_linalg::gemv(n, n, H, v, Hv);

        // lambda = v' * Hv  (Rayleigh quotient)
        lambda = mpc_linalg::dot(n, v, Hv);

        // v = Hv / ||Hv||
        double norm = std::sqrt(mpc_linalg::dot(n, Hv, Hv));
        if (norm < 1e-15) break;
        double inv_norm = 1.0 / norm;
        mpc_linalg::scal(n, inv_norm, Hv);
        std::memcpy(v, Hv, static_cast<std::size_t>(n) * sizeof(double));
    }

    return lambda;
}

// ---------------------------------------------------------------------------
// build_prediction_matrices
// ---------------------------------------------------------------------------
void build_prediction_matrices(const double* A_list, const double* B_list, int N,
                               double* Phi_blocks, double* Gamma)
{
    const int nx2 = NX * NX;
    const int nxnu = NX * NU;
    const int gamma_rows = (N + 1) * NX;          // total rows of Gamma
    const int gamma_cols = N * NU;                 // total cols of Gamma

    // Phi[0] = I_6
    std::memset(Phi_blocks, 0, nx2 * sizeof(double));
    for (int i = 0; i < NX; ++i)
        Phi_blocks[i + NX * i] = 1.0;

    // Phi[k] = A_{k-1} * Phi[k-1],  k = 1..N
    for (int k = 1; k <= N; ++k) {
        const double* A_km1 = A_list + (k - 1) * nx2;
        const double* Phi_prev = Phi_blocks + (k - 1) * nx2;
        double* Phi_cur = Phi_blocks + k * nx2;

        mpc_linalg::gemm_full(NX, NX, NX, 1.0, A_km1, NX, Phi_prev, NX, 0.0, Phi_cur, NX);
    }

    // Zero-initialize Gamma
    std::memset(Gamma, 0, (size_t)gamma_rows * gamma_cols * sizeof(double));

    // Recurrence for Gamma blocks
    // Gamma[k, k-1] = B_{k-1}   (copy)
    // Gamma[k, j]   = A_{k-1} * Gamma[k-1, j]   for j = 0..k-2
    for (int k = 1; k <= N; ++k) {
        const double* A_km1 = A_list + (k - 1) * nx2;
        const double* B_km1 = B_list + (k - 1) * nxnu;

        // Copy B_{k-1} into block [k, k-1]
        // Block [k, k-1] starts at Gamma + k*NX + gamma_rows * (k-1)*NU
        double* dst = Gamma + k * NX + (size_t)gamma_rows * (k - 1) * NU;
        for (int l = 0; l < NU; ++l) {
            std::memcpy(dst + (size_t)gamma_rows * l,
                        B_km1 + NX * l,
                        NX * sizeof(double));
        }

        // For j = 0..k-2: Gamma[k,j] = A_{k-1} * Gamma[k-1,j]
        for (int j = 0; j <= k - 2; ++j) {
            const double* Gkm1_j = Gamma + (k - 1) * NX + (size_t)gamma_rows * j * NU;
            double* Gk_j = Gamma + k * NX + (size_t)gamma_rows * j * NU;

            // A_{k-1} (NX x NX) * Gamma[k-1,j] (NX x NU) -> Gamma[k,j] (NX x NU)
            mpc_linalg::gemm_full(NX, NU, NX, 1.0, A_km1, NX, Gkm1_j, gamma_rows, 0.0, Gk_j, gamma_rows);
        }
    }
}

// ---------------------------------------------------------------------------
// form_hessian
// ---------------------------------------------------------------------------
void form_hessian(const double* Gamma, const double* Q, const double* Qf,
                  const double* R, int N, double* H)
{
    const int n_vars = N * NU;
    const int gamma_rows = (N + 1) * NX;

    // Zero-initialize H
    std::memset(H, 0, (size_t)n_vars * n_vars * sizeof(double));

    // Temporary buffer for Q * Gamma_block (NX x NU)
    double temp[NX * NU];

    // For each block pair (i, j) with 0 <= i <= j < N
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            double* H_ij = H + i * NU + (size_t)n_vars * j * NU;

            // Sum over k = max(i,j)+1 .. N-1 using Q, plus k = N using Qf
            int k_start = j + 1;  // since j >= i, max(i,j)+1 = j+1

            for (int k = k_start; k <= N; ++k) {
                const double* Gk_i = Gamma + k * NX + (size_t)gamma_rows * i * NU;
                const double* Gk_j = Gamma + k * NX + (size_t)gamma_rows * j * NU;
                const double* Qk = (k < N) ? Q : Qf;

                // Step 1: temp (NX x NU) = Qk (NX x NX) * Gamma[k,j] (NX x NU)
                mpc_linalg::gemm_full(NX, NU, NX, 1.0, Qk, NX, Gk_j, gamma_rows, 0.0, temp, NX);

                // Step 2: H_ij (NU x NU) += Gamma[k,i]^T (NU x NX) * temp (NX x NU)
                mpc_linalg::gemm_atb_full(NU, NU, NX, 1.0, Gk_i, gamma_rows, temp, NX, 1.0, H_ij, n_vars);
            }

            // Add R on the diagonal blocks
            if (i == j) {
                for (int c = 0; c < NU; ++c) {
                    for (int r = 0; r < NU; ++r) {
                        H_ij[r + (size_t)n_vars * c] += R[r + NU * c];
                    }
                }
            }

            // Copy upper triangle block to lower triangle: H_block[j,i] = H_block[i,j]^T
            if (i != j) {
                double* H_ji = H + j * NU + (size_t)n_vars * i * NU;
                for (int c = 0; c < NU; ++c) {
                    for (int r = 0; r < NU; ++r) {
                        H_ji[r + (size_t)n_vars * c] = H_ij[c + (size_t)n_vars * r];
                    }
                }
            }
        }
    }

    // Final symmetrization pass: H = 0.5 * (H + H^T) element-wise
    for (int c = 0; c < n_vars; ++c) {
        for (int r = c + 1; r < n_vars; ++r) {
            double avg = 0.5 * (H[r + (size_t)n_vars * c] + H[c + (size_t)n_vars * r]);
            H[r + (size_t)n_vars * c] = avg;
            H[c + (size_t)n_vars * r] = avg;
        }
    }
}

// ---------------------------------------------------------------------------
// form_gradient_matrices
// ---------------------------------------------------------------------------
void form_gradient_matrices(const double* Gamma, const double* Phi_blocks,
                            const double* Q, const double* Qf,
                            const double* x_ref_consistent, int N,
                            double* F, double* f_const)
{
    const int n_vars = N * NU;
    const int gamma_rows = (N + 1) * NX;
    const int nx2 = NX * NX;

    // Zero-initialize F and f_const
    std::memset(F, 0, (size_t)n_vars * NX * sizeof(double));
    std::memset(f_const, 0, (size_t)n_vars * sizeof(double));

    // Temporary buffers
    double temp_qphi[NX * NX];   // Qk * Phi[k]  -> NX x NX
    double temp_block[NU * NX];   // Gamma[k,j]^T * temp_qphi -> NU x NX

    // F = Gamma^T * Q_bar * Phi
    // Block row j of F (NU x NX):
    //   F_j = sum_{k=j+1}^{N-1} Gamma[k,j]^T * Q * Phi[k]  +  Gamma[N,j]^T * Qf * Phi[N]
    static bool debug_once = true;
    for (int j = 0; j < N; ++j) {
        double* F_j = F + j * NU;   // F element (j*NU + l, s) = F[(j*NU + l) + n_vars * s]

        for (int k = j + 1; k <= N; ++k) {
            const double* Gk_j = Gamma + k * NX + (size_t)gamma_rows * j * NU;
            const double* Phi_k = Phi_blocks + k * nx2;
            const double* Qk = (k < N) ? Q : Qf;

            // Step 1: temp_qphi (NX x NX) = Qk * Phi[k]
            // Use explicit loop to avoid BLASFEO dgemm_ panel-read bug on small matrices
            for (int col = 0; col < NX; ++col) {
                for (int row = 0; row < NX; ++row) {
                    double sum = 0.0;
                    for (int p = 0; p < NX; ++p)
                        sum += Qk[row + NX * p] * Phi_k[p + NX * col];
                    temp_qphi[row + NX * col] = sum;
                }
            }

            // Step 2: F_j (NU x NX) += Gamma[k,j]^T (NU x NX) * temp_qphi (NX x NX)
            // F_j is stored with leading dimension n_vars (rows of F)
            mpc_linalg::gemm_atb_full(NU, NX, NX, 1.0, Gk_j, gamma_rows, temp_qphi, NX, 1.0, F_j, n_vars);

            if (debug_once) {
                bool tq_nan = false, fj_nan = false;
                for (int i = 0; i < NX * NX; ++i)
                    if (std::isnan(temp_qphi[i])) { tq_nan = true; break; }
                for (int c = 0; c < NX; ++c)
                    for (int r = 0; r < NU; ++r)
                        if (std::isnan(F_j[r + n_vars * c])) { fj_nan = true; break; }
                if (tq_nan || fj_nan) {
                    std::printf("  [form_grad] NaN at j=%d k=%d: temp_qphi=%s F_j=%s\n",
                                j, k, tq_nan ? "NaN" : "ok", fj_nan ? "NaN" : "ok");
                    debug_once = false;
                }
            }
        }
    }
    debug_once = true;  // reset for next window

    // f_const = -H * u_ref_stacked
    // But we don't have H here directly. The caller (condense_window) will handle this.
    // Instead, we leave f_const = 0 here. condense_window will compute f_const = -H * u_ref.
}

// ---------------------------------------------------------------------------
// condense_window
// ---------------------------------------------------------------------------
void condense_window(const double* A_list, const double* B_list,
                     const double* x_ref_consistent, const double* u_ref,
                     const MPCConfig& config, PrecomputedWindow& window)
{
    const int N = config.N;
    const int n_vars = N * NU;

    // Store metadata
    window.N = N;
    window.n_vars = n_vars;
    std::memcpy(window.x_ref_0, x_ref_consistent, NX * sizeof(double));

    // Temporary buffers for prediction matrices
    double Phi_blocks[(N_MAX + 1) * NX * NX];
    double Gamma[(N_MAX + 1) * NX * N_MAX * NU];

    // Build prediction matrices
    build_prediction_matrices(A_list, B_list, N, Phi_blocks, Gamma);

    // DEBUG: check Phi and Gamma for NaN
    {
        static bool first_check = true;
        if (first_check) {
            first_check = false;
            for (int k = 0; k <= N; ++k) {
                const double* Phi_k = Phi_blocks + k * NX * NX;
                for (int i = 0; i < NX * NX; ++i) {
                    if (std::isnan(Phi_k[i])) {
                        std::printf("  [condensing] Phi[%d] has NaN at element %d\n", k, i);
                        break;
                    }
                }
            }
            int gamma_rows = (N + 1) * NX;
            for (int k = 0; k <= N; ++k) {
                for (int j = 0; j < k; ++j) {
                    const double* Gk_j = Gamma + k * NX + (size_t)gamma_rows * j * NU;
                    for (int i = 0; i < NX * NU; ++i) {
                        int r = i % NX;
                        int c = i / NX;
                        double val = Gk_j[r + gamma_rows * c];
                        if (std::isnan(val)) {
                            std::printf("  [condensing] Gamma[%d,%d] has NaN\n", k, j);
                            goto gamma_done;
                        }
                    }
                }
            }
            gamma_done:;
        }
    }

    // Form Hessian: H = Gamma^T Q_bar Gamma + R_bar
    form_hessian(Gamma, config.Q, config.Qf, config.R, N, window.H);

    // Compute largest eigenvalue for FISTA step size
    window.lambda_max = power_iteration_lambda_max(window.H, n_vars);

    // Cholesky factorization: H = L * L^T
    cholesky_factor(n_vars, window.H, window.L);

    // Form gradient matrices: F and f_const (f_const left as 0 by form_gradient_matrices)
    form_gradient_matrices(Gamma, Phi_blocks, config.Q, config.Qf,
                           x_ref_consistent, N, window.F, window.f_const);

    // DEBUG: check F for NaN right after form_gradient_matrices
    {
        static bool first = true;
        if (first) {
            first = false;
            int nan_count = 0;
            for (int i = 0; i < n_vars * NX; ++i)
                if (std::isnan(window.F[i])) nan_count++;
            if (nan_count > 0)
                std::printf("  [condensing] F has %d NaN after form_gradient_matrices\n", nan_count);

            // Also check: is Phi still clean after form_hessian ran?
            for (int k = 0; k <= N; ++k) {
                const double* Phi_k = Phi_blocks + k * NX * NX;
                for (int i = 0; i < NX * NX; ++i) {
                    if (std::isnan(Phi_k[i])) {
                        std::printf("  [condensing] Phi[%d] NOW has NaN at element %d (corrupted!)\n", k, i);
                        goto phi_recheck_done;
                    }
                }
            }
            std::printf("  [condensing] Phi still clean after form_hessian+form_gradient\n");
            phi_recheck_done:;
        }
    }

    // f_const = -H * u_ref_stacked
    // Stack u_ref into a contiguous vector of length n_vars
    double u_ref_stacked[N_MAX * NU];
    for (int k = 0; k < N; ++k) {
        std::memcpy(u_ref_stacked + k * NU, u_ref + k * NU, NU * sizeof(double));
    }

    // f_const = -H * u_ref_stacked  via  f_const = H * u_ref_stacked, then negate
    mpc_linalg::gemv(n_vars, n_vars, window.H, u_ref_stacked, window.f_const);
    mpc_linalg::scal(n_vars, -1.0, window.f_const);
}
