#pragma once

#ifdef __aarch64__
#include <arm_neon.h>
#else
#include "neon_sim.h"
#endif

#include <array>
#ifdef MPC_USE_NEON

// y = A * x,  A is m x n column-major
void neon_gemv_colmajor(int m, int n, const double* A, const double* x, double* y);

// Solve L*y = b, L is n x n lower triangular column-major
void neon_trsv_lower_colmajor(int n, const double* L, const double* b, double* y);

// Solve L'*x = y, L is n x n lower triangular column-major
void neon_trsv_upper_trans_colmajor(int n, const double* L, const double* y, double* x);

// Returns workspace size in floats for riccati_tracking.
const int riccati_workspace_sz(int N);

// Tracking LQR via affine Riccati recursion (backward + forward pass).
void riccati_tracking(
    float* workspace,
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
