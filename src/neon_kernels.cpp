#include <vector>
#ifdef MPC_USE_NEON

#include <array>
#include "neon_kernels.h"
#include <cmath>

// 1/sqrt(x) via rsqrte + 2 Newton steps (full float32 precision).
// On A53: ~22cy latency from 7 FP instructions vs 51cy from fsqrt+fdiv.
// Critically, the 7 instructions don't block the FP pipeline like fsqrt/fdiv do.
static inline float rsqrt_nr(float x) {
    float32_t e = vrsqrtes_f32(x);
    e *= vrsqrtss_f32(x * e, e);
    e *= vrsqrtss_f32(x * e, e);
    return e;
}

// Cholesky factorize S = C + diag(r), compute explicit inverse S^{-1}.
// c0..c3: rows of C (4x4 symmetric), r: diagonal of R.
// Returns columns of S^{-1} (4x4 symmetric, column-major).
static inline float32x4x4_t cholesky4_inv(
    float32x4_t c0, float32x4_t c1, float32x4_t c2, float32x4_t c3,
    std::array<float, 4> r)
{
    float s00 = vgetq_lane_f32(c0, 0) + r[0];
    float s10 = vgetq_lane_f32(c1, 0);
    float s20 = vgetq_lane_f32(c2, 0);
    float s30 = vgetq_lane_f32(c3, 0);

    float inv0 = rsqrt_nr(s00);
    float l10 = s10 * inv0;
    float l20 = s20 * inv0;
    float l30 = s30 * inv0;

    float s11 = vgetq_lane_f32(c1, 1) + r[1];
    float s21 = vgetq_lane_f32(c2, 1);
    float s31 = vgetq_lane_f32(c3, 1);

    float inv1 = rsqrt_nr(s11 - l10 * l10);
    float l21 = (s21 - l20 * l10) * inv1;
    float l31 = (s31 - l30 * l10) * inv1;

    float s22 = vgetq_lane_f32(c2, 2) + r[2];
    float s32 = vgetq_lane_f32(c3, 2);

    float inv2 = rsqrt_nr(s22 - l20 * l20 - l21 * l21);
    float l32 = (s32 - l30 * l20 - l31 * l21) * inv2;

    float s33 = vgetq_lane_f32(c3, 3) + r[3];

    float inv3 = rsqrt_nr(s33 - l30 * l30 - l31 * l31 - l32 * l32);

    // L^{-1} by forward substitution, building column vectors directly.
    // Column 0: solve L*x = e0
    float li00 = inv0;
    float li10 = -l10 * li00 * inv1;
    float li20 = (-l20 * li00 - l21 * li10) * inv2;
    float li30 = (-l30 * li00 - l31 * li10 - l32 * li20) * inv3;

    // Column 1: solve L*x = e1
    float li11 = inv1;
    float li21 = -l21 * li11 * inv2;
    float li31 = (-l31 * li11 - l32 * li21) * inv3;

    // Column 2: solve L*x = e2
    float li22 = inv2;
    float li32 = -l32 * li22 * inv3;

    // Build L^{-1} rows as NEON vectors (row layout makes Li^T*Li natural)
    float32x4_t li0 = { li00, 0.0f, 0.0f, 0.0f };
    float32x4_t li1 = { li10, li11, 0.0f, 0.0f };
    float32x4_t li2 = { li20, li21, li22, 0.0f };
    float32x4_t li3 = { li30, li31, li32, inv3 };

    // S^{-1} = Li^T * Li (symmetric, column-major output)
    float32x4_t i0 = vmulq_laneq_f32(li0, li0, 0);
    i0 = vfmaq_laneq_f32(i0, li1, li1, 0);
    i0 = vfmaq_laneq_f32(i0, li2, li2, 0);
    i0 = vfmaq_laneq_f32(i0, li3, li3, 0);

    float32x4_t i1 = vmulq_laneq_f32(li1, li1, 1);
    i1 = vfmaq_laneq_f32(i1, li2, li2, 1);
    i1 = vfmaq_laneq_f32(i1, li3, li3, 1);

    float32x4_t i2 = vmulq_laneq_f32(li2, li2, 2);
    i2 = vfmaq_laneq_f32(i2, li3, li3, 2);

    float32x4_t i3 = vmulq_laneq_f32(li3, li3, 3);

    return float32x4x4_t { i0, i1, i2, i3 };
}

const int riccati_workspace_sz(int N) {
    // Per-step: K(3*4) + v(4) + rotated B(3*4) = 28 floats
    return N * 28;
}

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
    float* u_star)
{
    // ---- Constants ----
    float32x4_t a = { A[0], A[1], A[2], 1.0f };
    float32x4_t a_upper = { A[0], A[1], A[2], 0.0f };

    float32x4_t b0_body = vld1q_f32(B0.data());
    float32x4_t b1_body = vld1q_f32(B0.data() + 4);
    float32x4_t b2_body = vld1q_f32(B0.data() + 8);

    float32x4_t R_vec = { R[0], R[1], R[2], R[3] };
    float32x4_t Qsum = { Q[0] + Q[3], Q[1] + Q[4], Q[2] + Q[5], 0.0f };

    // Precompute Q diagonal row vectors (constant across all iterations)
    float32x4_t Qdiag0 = { Q[0] + Q[3], 0.0f, 0.0f, 0.0f };
    float32x4_t Qdiag1 = { 0.0f, Q[1] + Q[4], 0.0f, 0.0f };
    float32x4_t Qdiag2 = { 0.0f, 0.0f, Q[2] + Q[5], 0.0f };

    // Precompute sin/cos table to avoid bl sincosf inside the loop.
    // This eliminates the function call, its 100+cy overhead, and 6 spill/reload pairs.
    float sincos_table[N * 2];
    for (int k = 0; k < N; k++) {
        sincos_table[k * 2 + 0] = cosf(theta[k]);
        sincos_table[k * 2 + 1] = sinf(theta[k]);
    }

    // ---- Terminal condition ----
    // P_N = Q, p_N = -Q * xr_N  (lane 3 holds p_upper_i)
    const float* xr_N = xr_upper + N * 3;
    float32x4_t p0 = { Q[0], 0.0f, 0.0f, -Q[0] * xr_N[0] };
    float32x4_t p1 = { 0.0f, Q[1], 0.0f, -Q[1] * xr_N[1] };
    float32x4_t p2 = { 0.0f, 0.0f, Q[2], -Q[2] * xr_N[2] };

    // ---- Backward pass ----
    for (int k = N - 1; k >= 0; k--) {
        const float* xr_k = xr_upper + k * 3;
        const float* ur_k = ur + k * 4;
        const float* c_k  = c_upper + k * 3;
        float* sd = workspace + k * 28;

        // 1. Rotate B from precomputed sin/cos (no function call)
        float ct = sincos_table[k * 2 + 0];
        float st = sincos_table[k * 2 + 1];
        float32x4_t ct_v = vdupq_n_f32(ct);
        float32x4_t st_v = vdupq_n_f32(st);

        float32x4_t b0 = vfmsq_f32(vmulq_f32(ct_v, b0_body), st_v, b1_body);
        float32x4_t b1 = vfmaq_f32(vmulq_f32(st_v, b0_body), ct_v, b1_body);
        float32x4_t b2 = b2_body;

        // Store rotated B for forward pass
        vst1q_f32(sd + 16, b0);
        vst1q_f32(sd + 20, b1);
        vst1q_f32(sd + 24, b2);

        // 2. PB = P_uu * B_u  (only uses lanes 0-2 of P rows)
        float32x4_t pb0 = vmulq_laneq_f32(b0, p0, 0);
        pb0 = vfmaq_laneq_f32(pb0, b1, p0, 1);
        pb0 = vfmaq_laneq_f32(pb0, b2, p0, 2);

        float32x4_t pb1 = vmulq_laneq_f32(b0, p1, 0);
        pb1 = vfmaq_laneq_f32(pb1, b1, p1, 1);
        pb1 = vfmaq_laneq_f32(pb1, b2, p1, 2);

        float32x4_t pb2 = vmulq_laneq_f32(b0, p2, 0);
        pb2 = vfmaq_laneq_f32(pb2, b1, p2, 1);
        pb2 = vfmaq_laneq_f32(pb2, b2, p2, 2);

        // 3. Lambda = P_uu c + p_upper via 4th-lane trick
        float32x4_t c_ext = { c_k[0], c_k[1], c_k[2], 1.0f };
        float L0 = vaddvq_f32(vmulq_f32(p0, c_ext));
        float L1 = vaddvq_f32(vmulq_f32(p1, c_ext));
        float L2 = vaddvq_f32(vmulq_f32(p2, c_ext));

        // 4. B^T PB  (4x4 symmetric)
        float32x4_t btpb0 = vmulq_laneq_f32(pb0, b0, 0);
        btpb0 = vfmaq_laneq_f32(btpb0, pb1, b1, 0);
        btpb0 = vfmaq_laneq_f32(btpb0, pb2, b2, 0);

        float32x4_t btpb1 = vmulq_laneq_f32(pb0, b0, 1);
        btpb1 = vfmaq_laneq_f32(btpb1, pb1, b1, 1);
        btpb1 = vfmaq_laneq_f32(btpb1, pb2, b2, 1);

        float32x4_t btpb2 = vmulq_laneq_f32(pb0, b0, 2);
        btpb2 = vfmaq_laneq_f32(btpb2, pb1, b1, 2);
        btpb2 = vfmaq_laneq_f32(btpb2, pb2, b2, 2);

        float32x4_t btpb3 = vmulq_laneq_f32(pb0, b0, 3);
        btpb3 = vfmaq_laneq_f32(btpb3, pb1, b1, 3);
        btpb3 = vfmaq_laneq_f32(btpb3, pb2, b2, 3);

        // 5. Cholesky factorize S = BTPB + R, get explicit inverse
        //    Uses rsqrte+Newton instead of sqrtf+fdiv (saves ~120cy on A53)
        float32x4x4_t Cinv = cholesky4_inv(btpb0, btpb1, btpb2, btpb3, R);

        // 6. invBt = S^{-1} B^T
        float32x4_t invBt0 = vmulq_laneq_f32(Cinv.val[0], b0, 0);
        invBt0 = vfmaq_laneq_f32(invBt0, Cinv.val[1], b0, 1);
        invBt0 = vfmaq_laneq_f32(invBt0, Cinv.val[2], b0, 2);
        invBt0 = vfmaq_laneq_f32(invBt0, Cinv.val[3], b0, 3);

        float32x4_t invBt1 = vmulq_laneq_f32(Cinv.val[0], b1, 0);
        invBt1 = vfmaq_laneq_f32(invBt1, Cinv.val[1], b1, 1);
        invBt1 = vfmaq_laneq_f32(invBt1, Cinv.val[2], b1, 2);
        invBt1 = vfmaq_laneq_f32(invBt1, Cinv.val[3], b1, 3);

        float32x4_t invBt2 = vmulq_laneq_f32(Cinv.val[0], b2, 0);
        invBt2 = vfmaq_laneq_f32(invBt2, Cinv.val[1], b2, 1);
        invBt2 = vfmaq_laneq_f32(invBt2, Cinv.val[2], b2, 2);
        invBt2 = vfmaq_laneq_f32(invBt2, Cinv.val[3], b2, 3);

        // 7. PA = P * a, then K columns = invBt * PA
        float32x4_t pa0 = vmulq_f32(p0, a);
        float32x4_t pa1 = vmulq_f32(p1, a);
        float32x4_t pa2 = vmulq_f32(p2, a);

        float32x4_t K_col0 = vmulq_laneq_f32(invBt0, pa0, 0);
        K_col0 = vfmaq_laneq_f32(K_col0, invBt1, pa1, 0);
        K_col0 = vfmaq_laneq_f32(K_col0, invBt2, pa2, 0);

        float32x4_t K_col1 = vmulq_laneq_f32(invBt0, pa0, 1);
        K_col1 = vfmaq_laneq_f32(K_col1, invBt1, pa1, 1);
        K_col1 = vfmaq_laneq_f32(K_col1, invBt2, pa2, 1);

        float32x4_t K_col2 = vmulq_laneq_f32(invBt0, pa0, 2);
        K_col2 = vfmaq_laneq_f32(K_col2, invBt1, pa1, 2);
        K_col2 = vfmaq_laneq_f32(K_col2, invBt2, pa2, 2);

        // 8. z = S^{-1} B^T lambda = invBt * lambda
        float32x4_t z = vmulq_n_f32(invBt0, L0);
        z = vfmaq_n_f32(z, invBt1, L1);
        z = vfmaq_n_f32(z, invBt2, L2);

        // 9. v = z - S^{-1} R ur
        float32x4_t ur_v = vld1q_f32(ur_k);
        float32x4_t Rur = vmulq_f32(R_vec, ur_v);
        float32x4_t Cinv_Rur = vmulq_laneq_f32(Cinv.val[0], Rur, 0);
        Cinv_Rur = vfmaq_laneq_f32(Cinv_Rur, Cinv.val[1], Rur, 1);
        Cinv_Rur = vfmaq_laneq_f32(Cinv_Rur, Cinv.val[2], Rur, 2);
        Cinv_Rur = vfmaq_laneq_f32(Cinv_Rur, Cinv.val[3], Rur, 3);
        float32x4_t v = vsubq_f32(z, Cinv_Rur);

        // Store K and v for forward pass
        vst1q_f32(sd + 0,  K_col0);
        vst1q_f32(sd + 4,  K_col1);
        vst1q_f32(sd + 8,  K_col2);
        vst1q_f32(sd + 12, v);

        // 10. ATPB = a_i * PB_row_i
        float32x4_t atpb0 = vmulq_laneq_f32(pb0, a, 0);
        float32x4_t atpb1 = vmulq_laneq_f32(pb1, a, 1);
        float32x4_t atpb2 = vmulq_laneq_f32(pb2, a, 2);

        // 11. Schur + affine dot products via batched FADDP
        float32x4_t m00 = vmulq_f32(atpb0, K_col0);
        float32x4_t m01 = vmulq_f32(atpb0, K_col1);
        float32x4_t m02 = vmulq_f32(atpb0, K_col2);
        float32x4_t m0z = vmulq_f32(atpb0, z);
        float32x4_t r00_01 = vpaddq_f32(m00, m01);
        float32x4_t r02_0z = vpaddq_f32(m02, m0z);
        float32x4_t row0_dots = vpaddq_f32(r00_01, r02_0z); // {sc00, sc01, sc02, d0}

        float32x4_t m11 = vmulq_f32(atpb1, K_col1);
        float32x4_t m12 = vmulq_f32(atpb1, K_col2);
        float32x4_t m1z = vmulq_f32(atpb1, z);
        float32x4_t r11_12 = vpaddq_f32(m11, m12);
        float32x4_t r1z_xx = vpaddq_f32(m1z, m1z);
        float32x4_t row1_dots = vpaddq_f32(r11_12, r1z_xx); // {sc11, sc12, d1, d1}

        float32x4_t m22 = vmulq_f32(atpb2, K_col2);
        float32x4_t m2z = vmulq_f32(atpb2, z);
        float32x4_t r22_2z = vpaddq_f32(m22, m2z);
        float32x4_t row2_dots = vpaddq_f32(r22_2z, r22_2z); // {sc22, d2, sc22, d2}

        float sc00 = vgetq_lane_f32(row0_dots, 0);
        float sc01 = vgetq_lane_f32(row0_dots, 1);
        float sc02 = vgetq_lane_f32(row0_dots, 2);
        float d0   = vgetq_lane_f32(row0_dots, 3);
        float sc11 = vgetq_lane_f32(row1_dots, 0);
        float sc12 = vgetq_lane_f32(row1_dots, 1);
        float d1   = vgetq_lane_f32(row1_dots, 2);
        float sc22 = vgetq_lane_f32(row2_dots, 0);
        float d2   = vgetq_lane_f32(row2_dots, 1);

        // 12. ATPA = PA * a
        float32x4_t ATPA0 = vmulq_f32(pa0, a);
        float32x4_t ATPA1 = vmulq_f32(pa1, a);
        float32x4_t ATPA2 = vmulq_f32(pa2, a);

        // 13. Affine correction for lane 3 (vectorized)
        //     affine_i = -Qsum_i*xr_i + a_i*(lambda_i - p_upper_i)
        float32x4_t lambda_v = { L0, L1, L2, 0.0f };
        float32x4_t p_upper_v = { vgetq_lane_f32(p0, 3),
                                   vgetq_lane_f32(p1, 3),
                                   vgetq_lane_f32(p2, 3), 0.0f };
        float32x4_t xr_v = { xr_k[0], xr_k[1], xr_k[2], 0.0f };
        float32x4_t d_v = { d0, d1, d2, 0.0f };

        float32x4_t f3_v = vfmsq_f32(
            vfmaq_f32(vnegq_f32(d_v), a_upper, vsubq_f32(lambda_v, p_upper_v)),
            Qsum, xr_v);

        // 14. Build f correction rows and compute P_new = ATPA + f
        float32x4_t f0 = { Q[0] + Q[3] - sc00, -sc01,              -sc02,              vgetq_lane_f32(f3_v, 0) };
        float32x4_t f1 = { -sc01,               Q[1] + Q[4] - sc11, -sc12,              vgetq_lane_f32(f3_v, 1) };
        float32x4_t f2 = { -sc02,               -sc12,               Q[2] + Q[5] - sc22, vgetq_lane_f32(f3_v, 2) };

        p0 = vaddq_f32(ATPA0, f0);
        p1 = vaddq_f32(ATPA1, f1);
        p2 = vaddq_f32(ATPA2, f2);
    }

    // ---- Forward pass ----
    float x0 = x0_upper[0];
    float x1 = x0_upper[1];
    float x2 = x0_upper[2];

    float32x4_t clamp_lo = vdupq_n_f32(-1.0f);
    float32x4_t clamp_hi = vdupq_n_f32(1.0f);

    for (int k = 0; k < N; k++) {
        const float* sd = workspace + k * 28;
        float32x4_t K_col0 = vld1q_f32(sd + 0);
        float32x4_t K_col1 = vld1q_f32(sd + 4);
        float32x4_t K_col2 = vld1q_f32(sd + 8);
        float32x4_t v      = vld1q_f32(sd + 12);

        // u = -(K_col0*x0 + K_col1*x1 + K_col2*x2) - v
        float32x4_t u_k = vnegq_f32(vfmaq_n_f32(
            vfmaq_n_f32(vmulq_n_f32(K_col0, x0), K_col1, x1),
            K_col2, x2));
        u_k = vsubq_f32(u_k, v);

        u_k = vmaxq_f32(vminq_f32(u_k, clamp_hi), clamp_lo);
        vst1q_f32(u_star + k * 4, u_k);

        // Propagate: x_upper_{k+1} = D_u * x_upper + B_u * u + c_upper
        float32x4_t b0_k = vld1q_f32(sd + 16);
        float32x4_t b1_k = vld1q_f32(sd + 20);
        float32x4_t b2_k = vld1q_f32(sd + 24);

        float32x4_t bu0 = vmulq_f32(b0_k, u_k);
        float32x4_t bu1 = vmulq_f32(b1_k, u_k);
        float32x4_t bu2 = vmulq_f32(b2_k, u_k);
        float32x4_t p01 = vpaddq_f32(bu0, bu1);
        float32x4_t p2x = vpaddq_f32(bu2, bu2);
        float32x4_t sums = vpaddq_f32(p01, p2x);

        const float* c_k = c_upper + k * 3;
        x0 = A[0] * x0 + vgetq_lane_f32(sums, 0) + c_k[0];
        x1 = A[1] * x1 + vgetq_lane_f32(sums, 1) + c_k[1];
        x2 = A[2] * x2 + vgetq_lane_f32(sums, 2) + c_k[2];
    }
}

// ---------------------------------------------------------------------------
// y = A * x,  A is m x n column-major
// ---------------------------------------------------------------------------
void neon_gemv_colmajor(int m, int n, const double* A, const double* x, double* y)
{
    {
        int i = 0;
        for (; i + 1 < m; i += 2) {
            vst1q_f64(y + i, vdupq_n_f64(0.0));
        }
        for (; i < m; ++i) {
            y[i] = 0.0;
        }
    }

    for (int j = 0; j < n; ++j) {
        const double* col = A + (long)m * j;
        float64x2_t xj = vdupq_n_f64(x[j]);
        int i = 0;
        for (; i + 1 < m; i += 2) {
            float64x2_t yi = vld1q_f64(y + i);
            float64x2_t ai = vld1q_f64(col + i);
            yi = vfmaq_f64(yi, ai, xj);
            vst1q_f64(y + i, yi);
        }
        for (; i < m; ++i) {
            y[i] += col[i] * x[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Solve L*y = b  (forward substitution)
// L is n x n lower triangular, column-major.
// ---------------------------------------------------------------------------
void neon_trsv_lower_colmajor(int n, const double* L, const double* b, double* y)
{
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
    }

    for (int j = 0; j < n; ++j) {
        const double* Lj = L + (long)n * j;
        y[j] /= Lj[j];

        double yj_val = y[j];
        float64x2_t yj = vdupq_n_f64(yj_val);

        int i = j + 1;
        if ((i & 1) && i < n) {
            y[i] -= Lj[i] * yj_val;
            ++i;
        }
        for (; i + 1 < n; i += 2) {
            float64x2_t yi  = vld1q_f64(y + i);
            float64x2_t lij = vld1q_f64(Lj + i);
            yi = vfmsq_f64(yi, lij, yj);
            vst1q_f64(y + i, yi);
        }
        for (; i < n; ++i) {
            y[i] -= Lj[i] * yj_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Solve L^T * x = y  (backward substitution with transposed lower factor)
// L is n x n lower triangular, column-major.
// ---------------------------------------------------------------------------
void neon_trsv_upper_trans_colmajor(int n, const double* L, const double* y, double* x)
{
    for (int i = 0; i < n; ++i) {
        x[i] = y[i];
    }

    for (int i = n - 1; i >= 0; --i) {
        const double* Li = L + (long)n * i;

        float64x2_t acc = vdupq_n_f64(0.0);
        double sum = 0.0;

        int j = i + 1;
        if ((j & 1) && j < n) {
            sum += Li[j] * x[j];
            ++j;
        }
        for (; j + 1 < n; j += 2) {
            float64x2_t lj = vld1q_f64(Li + j);
            float64x2_t xj = vld1q_f64(x + j);
            acc = vfmaq_f64(acc, lj, xj);
        }
        for (; j < n; ++j) {
            sum += Li[j] * x[j];
        }

        sum += vaddvq_f64(acc);
        x[i] = (x[i] - sum) / Li[i];
    }
}

#endif // MPC_USE_NEON
