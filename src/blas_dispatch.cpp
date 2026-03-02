#include "blas_dispatch.h"

#include <cblas.h>
#include <cstring>

#ifdef MPC_USE_NEON
#include "neon_kernels.h"
#endif

namespace mpc_linalg {

void gemv(int m, int n, const double* A, const double* x, double* y) {
#ifdef MPC_USE_NEON
    if (n <= 120) {
        neon_gemv_colmajor(m, n, A, x, y);
        return;
    }
#endif
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1.0, A, m, x, 1, 0.0, y, 1);
}

void gemv_trans(int m, int n, const double* A, const double* x, double* y) {
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, 1.0, A, m, x, 1, 0.0, y, 1);
}

void trsv_lower(int n, const double* L, const double* b, double* y) {
#ifdef MPC_USE_NEON
    if (n <= 120) {
        neon_trsv_lower_colmajor(n, L, b, y);
        return;
    }
#endif
    std::memcpy(y, b, static_cast<std::size_t>(n) * sizeof(double));
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L, n, y, 1);
}

void trsv_upper_trans(int n, const double* L, const double* y, double* x) {
#ifdef MPC_USE_NEON
    if (n <= 120) {
        neon_trsv_upper_trans_colmajor(n, L, y, x);
        return;
    }
#endif
    std::memcpy(x, y, static_cast<std::size_t>(n) * sizeof(double));
    cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, n, L, n, x, 1);
}

void axpy(int n, double alpha, const double* x, double* y) {
    cblas_daxpy(n, alpha, x, 1, y, 1);
}

double dot(int n, const double* x, const double* y) {
    return cblas_ddot(n, x, 1, y, 1);
}

void copy(int n, const double* x, double* y) {
    cblas_dcopy(n, x, 1, y, 1);
}

void gemm(int m, int n, int k, const double* A, const double* B, double* C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, A, m, B, k, 0.0, C, m);
}

void gemm_full(int m, int n, int k, double alpha, const double* A, int lda,
               const double* B, int ldb, double beta, double* C, int ldc) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_atb(int m, int n, int k, const double* A, int lda,
              const double* B, int ldb, double* C, int ldc) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k,
                1.0, A, lda, B, ldb, 0.0, C, ldc);
}

void symv(int n, const double* A, const double* x, double* y) {
    cblas_dsymv(CblasColMajor, CblasUpper, n, 1.0, A, n, x, 1, 0.0, y, 1);
}

void scal(int n, double alpha, double* x) {
    cblas_dscal(n, alpha, x, 1);
}

}  // namespace mpc_linalg
