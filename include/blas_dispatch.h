#pragma once

namespace mpc_linalg {

// y = A * x,  A is m x n column-major
void gemv(int m, int n, const double* A, const double* x, double* y);

// Solve L * y = b,  L is n x n lower triangular column-major
void trsv_lower(int n, const double* L, const double* b, double* y);

// Solve L' * x = y,  L is n x n lower triangular column-major
void trsv_upper_trans(int n, const double* L, const double* y, double* x);

// y = alpha * x + y
void axpy(int n, double alpha, const double* x, double* y);

// dot product
double dot(int n, const double* x, const double* y);

// y = x (vector copy)
void copy(int n, const double* x, double* y);

// y = A' * x,  A is m x n column-major (transpose multiply)
void gemv_trans(int m, int n, const double* A, const double* x, double* y);

// C = A * B,  A is m x k, B is k x n, all column-major
void gemm(int m, int n, int k, const double* A, const double* B, double* C);

// C = alpha * A * B + beta * C
void gemm_full(int m, int n, int k, double alpha, const double* A, int lda,
               const double* B, int ldb, double beta, double* C, int ldc);

// C = A' * B,  A is k x m (transposed to m x k), B is k x n
void gemm_atb(int m, int n, int k, const double* A, int lda,
              const double* B, int ldb, double* C, int ldc);

// y = A * x,  A is n x n symmetric (upper triangle stored), column-major
void symv(int n, const double* A, const double* x, double* y);

// Scale vector: x = alpha * x
void scal(int n, double alpha, double* x);

}  // namespace mpc_linalg
