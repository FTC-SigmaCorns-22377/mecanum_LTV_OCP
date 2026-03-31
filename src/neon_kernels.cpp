#include <vector>
#ifdef MPC_USE_NEON

#include <array>
#include "neon_kernels.h"
#include <arm_neon.h>
#include <cmath>

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

    float l00 = sqrtf(s00);
    float inv0 = 1.0f / l00;
    float l10 = s10 * inv0;
    float l20 = s20 * inv0;
    float l30 = s30 * inv0;

    float s11 = vgetq_lane_f32(c1, 1) + r[1];
    float s21 = vgetq_lane_f32(c2, 1);
    float s31 = vgetq_lane_f32(c3, 1);

    float l11 = sqrtf(s11 - l10 * l10);
    float inv1 = 1.0f / l11;
    float l21 = (s21 - l20 * l10) * inv1;
    float l31 = (s31 - l30 * l10) * inv1;

    float s22 = vgetq_lane_f32(c2, 2) + r[2];
    float s32 = vgetq_lane_f32(c3, 2);

    float l22 = sqrtf(s22 - l20 * l20 - l21 * l21);
    float inv2 = 1.0f / l22;
    float l32 = (s32 - l30 * l20 - l31 * l21) * inv2;

    float s33 = vgetq_lane_f32(c3, 3) + r[3];

    float l33 = sqrtf(s33 - l30 * l30 - l31 * l31 - l32 * l32);
    float inv3 = 1.0f / l33;

    // L^{-1} by forward substitution on identity columns (column-major)
    float Li[16] = {};

    Li[0]  = inv0;
    Li[1]  = -l10 * Li[0] * inv1;
    Li[2]  = (-l20 * Li[0] - l21 * Li[1]) * inv2;
    Li[3]  = (-l30 * Li[0] - l31 * Li[1] - l32 * Li[2]) * inv3;

    Li[5]  = inv1;
    Li[6]  = -l21 * Li[5] * inv2;
    Li[7]  = (-l31 * Li[5] - l32 * Li[6]) * inv3;

    Li[10] = inv2;
    Li[11] = -l32 * Li[10] * inv3;

    Li[15] = inv3;

    // S^{-1} = Li^T * Li (symmetric, column-major output)
    float32x4_t li0 = vld1q_f32(Li);
    float32x4_t li1 = vld1q_f32(Li + 4);
    float32x4_t li2 = vld1q_f32(Li + 8);
    float32x4_t li3 = vld1q_f32(Li + 12);

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
    // Per-step storage: K (3 columns, each 4-wide) + v (4-wide) = 16 floats
    // + rotated B (3 rows, each 4-wide) = 12 floats = 28 floats/step
    // For N=30: 28*30*4 = 3360 bytes (fits comfortably in L1)
    return N*28;
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

    float32x4_t b0_body = vld1q_f32(B0.data());
    float32x4_t b1_body = vld1q_f32(B0.data() + 4);
    float32x4_t b2_body = vld1q_f32(B0.data() + 8);

    float32x4_t R_vec = { R[0], R[1], R[2], R[3] };
    float32x4_t Qsum = { Q[0] + Q[3], Q[1] + Q[4], Q[2] + Q[5], 0.0f };

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

        // 1. Rotate B: B_u(theta) = [R(theta), 0; 0, 1] * B_0
        float ct = cosf(theta[k]);
        float st = sinf(theta[k]);
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
        //    dot(p_row_i, {c0,c1,c2,1}) = P_uu_row_i . c + p_upper_i
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
        float32x4x4_t Cinv = cholesky4_inv(btpb0, btpb1, btpb2, btpb3, R);

        // 6. invBt = S^{-1} B^T  (Cinv * b_row_i = column i of S^{-1}B^T)
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

        // 7. PA = P * a (element-wise), then K columns = invBt * PA
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

        // 10. ATPB = a_i * PB_row_i (scale each row by diagonal of A)
        float32x4_t atpb0 = vmulq_laneq_f32(pb0, a, 0);
        float32x4_t atpb1 = vmulq_laneq_f32(pb1, a, 1);
        float32x4_t atpb2 = vmulq_laneq_f32(pb2, a, 2);

        // 11. Schur complement: ATPB * K  (only upper triangle needed)
        //     sc_ij = dot(atpb_i, K_col_j)
        //     Also: d_i = dot(atpb_i, z) for affine p update
        //
        //     Vectorize: compute atpb_i * [K_col0 | K_col1 | K_col2 | z] as
        //     paired multiplies, then use FADDP to reduce pairs across all 4
        //     results simultaneously instead of 4 separate vaddvq calls.

        // Row 0 products: atpb0 * K_col0, atpb0 * K_col1, atpb0 * K_col2, atpb0 * z
        float32x4_t m00 = vmulq_f32(atpb0, K_col0);
        float32x4_t m01 = vmulq_f32(atpb0, K_col1);
        float32x4_t m02 = vmulq_f32(atpb0, K_col2);
        float32x4_t m0z = vmulq_f32(atpb0, z);
        // Pairwise add: reduce 4 lanes to 2 for each pair
        float32x4_t r00_01 = vpaddq_f32(m00, m01);  // {m00[0]+m00[1], m00[2]+m00[3], m01[0]+m01[1], m01[2]+m01[3]}
        float32x4_t r02_0z = vpaddq_f32(m02, m0z);
        // Second pairwise add to get final sums
        float32x4_t row0_dots = vpaddq_f32(r00_01, r02_0z);  // {sc00, sc01, sc02, d0}

        // Row 1 products: atpb1 * K_col1, atpb1 * K_col2, atpb1 * z, (unused)
        float32x4_t m11 = vmulq_f32(atpb1, K_col1);
        float32x4_t m12 = vmulq_f32(atpb1, K_col2);
        float32x4_t m1z = vmulq_f32(atpb1, z);
        float32x4_t r11_12 = vpaddq_f32(m11, m12);
        float32x4_t r1z_xx = vpaddq_f32(m1z, m1z);  // d1 in lane 0, duplicate in lane 2
        float32x4_t row1_dots = vpaddq_f32(r11_12, r1z_xx);  // {sc11, sc12, d1, d1}

        // Row 2 products: atpb2 * K_col2, atpb2 * z
        float32x4_t m22 = vmulq_f32(atpb2, K_col2);
        float32x4_t m2z = vmulq_f32(atpb2, z);
        float32x4_t r22_2z = vpaddq_f32(m22, m2z);
        float32x4_t row2_dots = vpaddq_f32(r22_2z, r22_2z);  // {sc22, d2, sc22, d2}

        // Extract scalars
        float sc00 = vgetq_lane_f32(row0_dots, 0);
        float sc01 = vgetq_lane_f32(row0_dots, 1);
        float sc02 = vgetq_lane_f32(row0_dots, 2);
        float d0   = vgetq_lane_f32(row0_dots, 3);
        float sc11 = vgetq_lane_f32(row1_dots, 0);
        float sc12 = vgetq_lane_f32(row1_dots, 1);
        float d1   = vgetq_lane_f32(row1_dots, 2);
        float sc22 = vgetq_lane_f32(row2_dots, 0);
        float d2   = vgetq_lane_f32(row2_dots, 1);

        // 12. ATPA = PA * a  (= D_u P_uu D_u, lane 3 = a_i * p_upper_i)
        float32x4_t ATPA0 = vmulq_f32(pa0, a);
        float32x4_t ATPA1 = vmulq_f32(pa1, a);
        float32x4_t ATPA2 = vmulq_f32(pa2, a);

        // 13. Construct correction f
        //     Lanes 0-2: Qsum_diag - Schur entries
        //     Lane 3:    -(Q_i+Q_{3+i})*xr_i + a_i*(lambda_i - p_upper_i) - d_i
        //
        //     Note: a_i * p_upper_i is already in ATPA lane 3 (from DSD with a[3]=1).
        //     So f3_i = p_upper_new_i - ATPA_lane3_i.
        //
        //     Vectorize f3: use {L0,L1,L2} and {p_upper} as vectors.
        float32x4_t lambda_v = { L0, L1, L2, 0.0f };
        float32x4_t p_upper_v = { vgetq_lane_f32(p0, 3),
                                   vgetq_lane_f32(p1, 3),
                                   vgetq_lane_f32(p2, 3), 0.0f };
        float32x4_t xr_v = { xr_k[0], xr_k[1], xr_k[2], 0.0f };
        float32x4_t d_v = { d0, d1, d2, 0.0f };

        // p_new = -Qsum*xr + a_upper*(lambda - p_upper) + a_upper*p_upper - d
        //       = -Qsum*xr + a_upper*lambda - d
        // But ATPA lane3 already has a_i*p_upper_i, so:
        // f3 = (-Qsum*xr + a_upper*(lambda - p_upper) - d)
        float32x4_t a_upper = { A[0], A[1], A[2], 0.0f };
        float32x4_t f3_v = vfmsq_f32(
            vfmaq_f32(vnegq_f32(d_v), a_upper, vsubq_f32(lambda_v, p_upper_v)),
            Qsum, xr_v);

        float32x4_t f0 = { Q[0] + Q[3] - sc00, -sc01,              -sc02,              vgetq_lane_f32(f3_v, 0) };
        float32x4_t f1 = { -sc01,               Q[1] + Q[4] - sc11, -sc12,              vgetq_lane_f32(f3_v, 1) };
        float32x4_t f2 = { -sc02,               -sc12,               Q[2] + Q[5] - sc22, vgetq_lane_f32(f3_v, 2) };

        // 14. P_new = ATPA + f
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

        // Propagate state: x_upper_{k+1} = D_u * x_upper + B_u * u + c_upper
        // Compute B_u * u via pairwise add to avoid 3 separate vaddvq
        float32x4_t b0_k = vld1q_f32(sd + 16);
        float32x4_t b1_k = vld1q_f32(sd + 20);
        float32x4_t b2_k = vld1q_f32(sd + 24);

        float32x4_t bu0 = vmulq_f32(b0_k, u_k);
        float32x4_t bu1 = vmulq_f32(b1_k, u_k);
        float32x4_t bu2 = vmulq_f32(b2_k, u_k);
        // Reduce 3 dot products: pairwise add pairs, then finish
        float32x4_t p01 = vpaddq_f32(bu0, bu1);   // {b0u[0]+b0u[1], b0u[2]+b0u[3], b1u[0]+b1u[1], b1u[2]+b1u[3]}
        float32x4_t p2x = vpaddq_f32(bu2, bu2);   // {b2u[0]+b2u[1], b2u[2]+b2u[3], ...}
        float32x4_t sums = vpaddq_f32(p01, p2x);   // {Bu0, Bu1, Bu2, ...}

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
