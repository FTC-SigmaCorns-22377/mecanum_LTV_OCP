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

// Tracking LQR via affine Riccati recursion (backward + forward pass).
// Produces the optimal control sequence u* for the affine system:
//   x_{k+1} = A x_k + B_k u_k + c_k
// minimizing:
//   sum (x_k - xr_k)^T Q (x_k - xr_k) + (u_k - ur_k)^T R (u_k - ur_k)
//
// B_k = M(theta_k) * B0 where M is a 3x3 rotation (heading-dependent).
// Only the upper 3 states (positions) are tracked through the recursion;
// lower 3 states contribute constant Q[3:5] cost via the P structure.
//
// xr_upper: [N+1][3] reference positions (row-major, includes terminal)
// ur:       [N][4]   reference inputs (row-major)
// c_upper:  [N][3]   affine offsets for upper 3 states (row-major)
// x0_upper: [3]      initial upper state
// u_star:   [N][4]   output optimal controls (row-major)
void riccati_tracking(
    std::array<float, 6> Q,
    std::array<float, 4> R,
    std::array<float, 3> A,
    std::array<float, 12> B0,
    int N,
    const float* theta,
    const float* xr_upper,
    const float* ur,
    const float* c_upper,
    const float* x0_upper,
    float* u_star);

#endif
