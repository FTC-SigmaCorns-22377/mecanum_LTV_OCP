#pragma once

#include <arm_neon.h>
#include <array>
#ifdef MPC_USE_NEON

// y = A * x,  A is m x n column-major
void neon_gemv_colmajor(int m, int n, const double* A, const double* x, double* y);

// Solve L*y = b, L is n x n lower triangular column-major
void neon_trsv_lower_colmajor(int n, const double* L, const double* b, double* y);

// Solve L'*x = y, L is n x n lower triangular column-major
void neon_trsv_upper_trans_colmajor(int n, const double* L, const double* y, double* x);

// y = clip(x, lo, hi), returns number of clipped elements
int neon_clip_and_count(int n, const double* x, double lo, double hi, double* y);

inline float32x4x4_t cholesky4_from_rows(float32x4_t c0, float32x4_t c1, float32x4_t c2, float32x4_t c3, std::array<float, 4> r);

inline float32x4x3_t A33B34(
    float32x4_t a0, float32x4_t a1, float32x4_t a2,
    float32x4_t b0, float32x4_t b1, float32x4_t b2
);

inline float32x4x4_t AT34B34(float32x4x3_t A, float32x4x3_t B);

inline float32x4x4_t transpose34(
    float32x4_t r0, float32x4_t r1, float32x4_t r2
);

inline float32x4x3_t DSD(
    float32x4_t s0, float32x4_t s1, float32x4_t s2,
    float32x4_t d
);

inline float32x4x3_t ATSdinvA(
    float32x4_t s0, float32x4_t s1, float32x4_t s2, float32x4_t s3,
    float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3,
    std::array<float, 4> d);

inline float32x4x3_t recurse(float32x4x3_t P, float32x4_t A, float32x4x3_t B, float32x4_t q, std::array<float, 6> Q, std::array<float, 4> R);
void neon_full(std::array<float, 6> Q,std::array<float, 4> R, std::array<float, 3> A, std::array<float, 12> B, int N, const float* theta);


#endif
