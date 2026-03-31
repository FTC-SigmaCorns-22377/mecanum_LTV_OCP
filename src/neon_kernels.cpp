#include <vector>
#ifdef MPC_USE_NEON

#include <array>
#include "neon_kernels.h"
#include <arm_neon.h>

void neon_full(std::array<float, 6> Q,std::array<float, 4> R, std::array<float, 3> A, std::array<float, 12> B, int N, const float* theta) {
    float32x4_t a = { A[0],A[1],A[2], 1.0 };

    // symmetric (last element represents lower 3x3 diagonal)
    float32x4_t p0 = { Q[0], 0.0, 0.0, Q[3] };
    float32x4_t p1 = { 0.0, Q[1], 0.0, Q[4] };
    float32x4_t p2 = { 0.0, 0.0, Q[2], Q[5] };

    // row major
    float32x4_t b0 = vld1q_f32(B.begin());
    float32x4_t b1 = vld1q_f32(B.begin() + 4);
    float32x4_t b2 = vld1q_f32(B.begin() + 8);

    // M=P*B, row by row
    // M[0] = p00*B[0] + p01*B[1] + p02*B[2]
    float32x4_t pb0 = vmulq_laneq_f32(b0, p0, 0);
    pb0 = vfmaq_laneq_f32(pb0, b1, p0, 1);
    pb0 = vfmaq_laneq_f32(pb0, b2, p0, 2);

    // M[1] = p10*B[0] + p11*B[1] + p12*B[2]
    float32x4_t pb1 = vmulq_laneq_f32(b0, p1, 0);
    pb1 = vfmaq_laneq_f32(pb1, b1, p1, 1);
    pb1 = vfmaq_laneq_f32(pb1, b2, p1, 2);

    // M[2] = p20*B[0] + p21*B[1] + p22*B[2]
    float32x4_t pb2 = vmulq_laneq_f32(b0, p2, 0);
    pb2 = vfmaq_laneq_f32(pb2, b1, p2, 1);
    pb2 = vfmaq_laneq_f32(pb2, b2, p2, 2);

    // C = A^T * M, row i = A[0][i]*M[0] + A[1][i]*M[1] + A[2][i]*M[2]
    float32x4_t c0 = vmulq_laneq_f32(pb0, p0, 0);
    c0 = vfmaq_laneq_f32(c0, pb1, p1, 0);
    c0 = vfmaq_laneq_f32(c0, pb2, p2, 0);

    float32x4_t c1 = vmulq_laneq_f32(pb0, b0, 1);
    c1 = vfmaq_laneq_f32(c1, pb1, b1, 1);
    c1 = vfmaq_laneq_f32(c1, pb2, b2, 1);

    float32x4_t c2 = vmulq_laneq_f32(pb0, b0, 2);
    c2 = vfmaq_laneq_f32(c2, pb1, b1, 2);
    c2 = vfmaq_laneq_f32(c2, pb2, b2, 2);

    float32x4_t c3 = vmulq_laneq_f32(pb0, b0, 3);
    c3 = vfmaq_laneq_f32(c3, pb1, b1, 3);
    c3 = vfmaq_laneq_f32(c3, pb2, b2, 3);

    // invert

    // column-major
    float32x4x4_t inv = cholesky4_from_rows(c0,c1,c2,c3,R);

    float32x4_t invBt0 = vmulq_laneq_f32(inv.val[0], b0, 0);
    invBt0 = vfmaq_laneq_f32(invBt0, inv.val[1], b0, 1);
    invBt0 = vfmaq_laneq_f32(invBt0, inv.val[2], b0, 2);
    invBt0 = vfmaq_laneq_f32(invBt0, inv.val[3], b0, 3);

    float32x4_t invBt1 = vmulq_laneq_f32(inv.val[0], b1, 0);
    invBt1 = vfmaq_laneq_f32(invBt1, inv.val[1], b1, 1);
    invBt1 = vfmaq_laneq_f32(invBt1, inv.val[2], b1, 2);
    invBt1 = vfmaq_laneq_f32(invBt1, inv.val[3], b1, 3);

    float32x4_t invBt2 = vmulq_laneq_f32(inv.val[0], b2, 0);
    invBt2 = vfmaq_laneq_f32(invBt2, inv.val[1], b2, 1);
    invBt2 = vfmaq_laneq_f32(invBt2, inv.val[2], b2, 2);
    invBt2 = vfmaq_laneq_f32(invBt2, inv.val[3], b2, 3);

    // pa is column-major, and symmetric in the upper 3x3
    auto pa0 = vmulq_f32(p0, a);
    auto pa1 = vmulq_f32(p1, a);
    auto pa2 = vmulq_f32(p2, a);

    float32x4_t invBtPa0 = vmulq_laneq_f32(invBt0, pa0, 0);
    invBtPa0 = vfmaq_laneq_f32(invBtPa0, invBt1, pa0, 1);
    invBtPa0 = vfmaq_laneq_f32(invBtPa0, invBt2, pa0, 2);

    float32x4_t invBtPa1 = vmulq_laneq_f32(invBt0, pa1, 0);
    invBtPa1 = vfmaq_laneq_f32(invBtPa1, invBt1, pa1, 1);
    invBtPa1 = vfmaq_laneq_f32(invBtPa1, invBt2, pa1, 2);

    float32x4_t invBtPa2 = vmulq_laneq_f32(invBt0, pa2, 0);
    invBtPa2 = vfmaq_laneq_f32(invBtPa2, invBt1, pa2, 1);
    invBtPa2 = vfmaq_laneq_f32(invBtPa2, invBt2, pa2, 2);

    // a^T p a (column-major): this is just a multiplication by a + the off diagonal part that ends up coming from the lower 3x3 diagonal of p
    // this is just diag(Q[3],Q[4],Q[5])
    // Also adding in Q here because its already extracted. The following is Q + A^T P A

    float32x4_t P0 = vmulq_f32(pa0, a);
    float32x4_t P1 = vmulq_f32(pa1, a);
    float32x4_t P2 = vmulq_f32(pa2, a);

    float32x4_t atPB0 = vmulq_f32(pb0, a);
    float32x4_t atPB1 = vmulq_f32(pb1, a);
    float32x4_t atPB2 = vmulq_f32(pb2, a);

    float32x4_t p00 = vmulq_f32(atPB0, invBtPa0);
    float32x4_t p01 = vmulq_f32(atPB0, invBtPa1);
    float32x4_t p02 = vmulq_f32(atPB0, invBtPa2);
    float32x4_t p11 = vmulq_f32(atPB1, invBtPa1);
    float32x4_t p12 = vmulq_f32(atPB1, invBtPa2);
    float32x4_t p22 = vmulq_f32(atPB2, invBtPa2);

    float c00 = vaddvq_f32(p00);
    float c01 = vaddvq_f32(p01);
    float c02 = vaddvq_f32(p02);
    float c11 = vaddvq_f32(p11);
    float c12 = vaddvq_f32(p12);
    float c22 = vaddvq_f32(p22);

    float32x4_t f0 = { Q[0] + Q[3] - c00, -c01, -c02 };
    float32x4_t f1 = { -c01, Q[1] + Q[4] - c11, -c12 };
    float32x4_t f2 = { -c02, -c12, Q[2] + Q[5] - c22 };

    p0 = vaddq_f32(P0, f0);
    p1 = vaddq_f32(P1, f1);
    p2 = vaddq_f32(P2, f2);
}

void riccati(std::array<float, 6> Q,std::array<float, 4> R, std::array<float, 3> A, std::array<float, 12> B, int N, std::vector<float> Ps, const float* theta) {
    float32x4_t q = { Q[0], Q[1], Q[2], 0.0 };

    float32x4_t a = { A[0],A[1],A[2], 0.0 };

    // symmetric (last element represents lower 3x3 diagonal)
    float32x4_t p0 = { Q[0], 0.0, 0.0, Q[3] };
    float32x4_t p1 = { 0.0, Q[1], 0.0, Q[4] };
    float32x4_t p2 = { 0.0, 0.0, Q[2], Q[5] };

    float32x4x3_t P = { p0,p1,p2 };

    // row major
    float32x4x3_t b = {
        vld1q_f32(B.begin()),
        vld1q_f32(B.begin() + 4),
        vld1q_f32(B.begin() + 8)
    };

    for (int i=0; i<N; i++) {
        vst1q_f32(Ps.data() + i*12 + 0, P.val[0]);
        vst1q_f32(Ps.data() + i*12 + 4, P.val[1]);
        vst1q_f32(Ps.data() + i*12 + 8, P.val[2]);
        recurse(P,a,b,q,Q,R);
    }
}

inline float32x4x3_t recurse(float32x4x3_t P, float32x4_t A, float32x4x3_t B, float32x4_t q, std::array<float, 6> Q, std::array<float, 4> R) {
    //armv8-clang -O3 -mcpu=cortex-a53 -mtune=cortex-a53 ~140 instructions, branchless and 0 stores
    auto PB = A33B34(P.val[0],P.val[1],P.val[2],B.val[0],B.val[1],B.val[2]);

    float32x4x3_t PA = {
        vmulq_f32(P.val[0], A),
        vmulq_f32(P.val[1], A),
        vmulq_f32(P.val[2], A),
    };

    auto BT = transpose34(B.val[0],B.val[1],B.val[2]);

    auto BTPB = AT34B34(B,PB);

    auto x = ATSdinvA(BTPB.val[0], BTPB.val[1], BTPB.val[2], BTPB.val[3], APB_col.val[0], APB_col.val[1], APB_col.val[2], APB_col.val[3], R);

    auto ATPA = DSD(P.val[0], P.val[1], P.val[2], q);

    float32x4_t q0 = { Q[0]+Q[3], 0, 0, 0 };
    float32x4_t q1 = { 0, Q[1]+Q[4], 0, 0 };
    float32x4_t q2 = { 0, 0, Q[2]+Q[5], 0 };

    float32x4x3_t P_prev = {
        vaddq_f32(q0, vsubq_f32(ATPA.val[0], x.val[0])),
        vaddq_f32(q1, vsubq_f32(ATPA.val[1], x.val[1])),
        vaddq_f32(q2, vsubq_f32(ATPA.val[2], x.val[2])),
    };

    return P_prev;
}

// returns the product AB when A is a 3x3 matrix and B is a 3x4 matrix
// inputs and outputs in row-major order
inline float32x4x3_t A33B34(
    float32x4_t a0, float32x4_t a1, float32x4_t a2,
    float32x4_t b0, float32x4_t b1, float32x4_t b2
) {
    float32x4_t r0 = vmulq_laneq_f32(b0, a0, 0);
    r0 = vfmaq_laneq_f32(r0, b1, a0, 1);
    r0 = vfmaq_laneq_f32(r0, b2, a0, 2);

    float32x4_t r1 = vmulq_laneq_f32(b0, a1, 0);
    r1 = vfmaq_laneq_f32(r1, b1, a1, 1);
    r1 = vfmaq_laneq_f32(r1, b2, a1, 2);

    float32x4_t r2 = vmulq_laneq_f32(b0, a2, 0);
    r2 = vfmaq_laneq_f32(r2, b1, a2, 1);
    r2 = vfmaq_laneq_f32(r2, b2, a2, 2);

    return { r0, r1, r2 };
}

// returns the product A^TB when A is a 3x4 matrix and B is a 3x4 matrix
// inputs and outputs in row-major order
inline float32x4x4_t AT34B34(float32x4x3_t A, float32x4x3_t B) {
    float32x4_t r0 = vmulq_laneq_f32(B.val[0], A.val[0], 0);
    r0 = vfmaq_laneq_f32(r0, B.val[1], A.val[1], 0);
    r0 = vfmaq_laneq_f32(r0, B.val[2], A.val[2], 0);

    float32x4_t r1 = vmulq_laneq_f32(B.val[0], A.val[0], 1);
    r1 = vfmaq_laneq_f32(r1, B.val[1], A.val[1], 1);
    r1 = vfmaq_laneq_f32(r1, B.val[2], A.val[2], 1);

    float32x4_t r2 = vmulq_laneq_f32(B.val[0], A.val[0], 2);
    r2 = vfmaq_laneq_f32(r2, B.val[1], A.val[1], 2);
    r2 = vfmaq_laneq_f32(r2, B.val[2], A.val[2], 2);

    float32x4_t r3 = vmulq_laneq_f32(B.val[0], A.val[0], 3);
    r3 = vfmaq_laneq_f32(r3, B.val[1], A.val[1], 3);
    r3 = vfmaq_laneq_f32(r3, B.val[2], A.val[2], 3);

    return { r0, r1, r2, r3 };
}


// transposes 3x4 matrix with rows r0..r3
inline float32x4x4_t transpose34(
    float32x4_t r0, float32x4_t r1, float32x4_t r2
) {
    float32x4_t r3 = vdupq_n_f32(0.0f);

    // 4x4 transpose via TRN1/TRN2 pairs
    // Step 1: interleave pairs of rows
    float32x4_t t01lo = vtrn1q_f32(r0, r1); // (d00, d10, d02, d12)
    float32x4_t t01hi = vtrn2q_f32(r0, r1); // (d01, d11, d03, d13)
    float32x4_t t23lo = vtrn1q_f32(r2, r3); // (d20,  0,  d22,  0 )
    float32x4_t t23hi = vtrn2q_f32(r2, r3); // (d21,  0,  d23,  0 )

    // Step 2: combine halves (reinterpret as f64 for 64-bit element zip)
    // col_j = (r[0][j], r[1][j], r[2][j], 0)
    float32x4_t col0 = vcombine_f32(vget_low_f32(t01lo),  vget_low_f32(t23lo));
    float32x4_t col1 = vcombine_f32(vget_low_f32(t01hi),  vget_low_f32(t23hi));
    float32x4_t col2 = vcombine_f32(vget_high_f32(t01lo), vget_high_f32(t23lo));
    float32x4_t col3 = vcombine_f32(vget_high_f32(t01hi), vget_high_f32(t23hi));

    return { col0, col1, col2, col3 };
}

inline float32x4x3_t DSD(
    float32x4_t s0, float32x4_t s1, float32x4_t s2,
    float32x4_t d
) {
    // Step 1: scale each row by d[i]  (left multiply: D*S)
    float32x4_t r0 = vmulq_laneq_f32(s0, d, 0);  // row0 *= d0
    float32x4_t r1 = vmulq_laneq_f32(s1, d, 1);  // row1 *= d1
    float32x4_t r2 = vmulq_laneq_f32(s2, d, 2);  // row2 *= d2

    // Step 2: scale each column by d[j]  (right multiply: (D*S)*D)
    r0 = vmulq_f32(r0, d);  // [d0*s00*d0, d0*s01*d1, d0*s02*d2, _]
    r1 = vmulq_f32(r1, d);  // [d1*s10*d0, d1*s11*d1, d1*s12*d2, _]
    r2 = vmulq_f32(r2, d);  // [d2*s20*d0, d2*s21*d1, d2*s22*d2, _]

    return {r0,r1,r2};
}

// A is 4x3, a0..a3 are the rows of A (zero padded)
// S is 4x4 symmetric matrix, s0..s3 are the rows/columns of S
// D is a 4x4 diagonal matrix, d=[D00, D11, D22, D33]
// Computes (S+D)^-1 A
// (L L^T)-1 A where L comes from cholesky factorization of (S+D)
// L X = A -> X = L^-1 A
// L^T Y = X -> Y = L^T^-1 X -> Y = L^T^-1 L^-1 A = (L L^T)^-1 A = (S+D)^-1 A
// armv8-a clang -O3 -mcpu=cortex-a53 -mtune=cortex-a53 = ~80 instructions no branching
inline float32x4x3_t SdinvA(
    float32x4_t s0, float32x4_t s1, float32x4_t s2, float32x4_t s3,
    float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3,
    std::array<float, 4> d)
{

    float s00 = vgetq_lane_f32(s0, 0) + d[0];
    float inv0sq = 1.0f / s00;

    float s11 = vgetq_lane_f32(s1, 1) + d[1];
    float s10 = vgetq_lane_f32(s1, 0);  // == c0[1] by symmetry, but c1[0] is what we computed
    float l11sq = s11 - s10*s10*inv0sq;
    float inv1sq = 1.0f/l11sq;

    float s22 = vgetq_lane_f32(s2, 2) + d[2];
    float s20 = vgetq_lane_f32(s2, 0);  // == c0[2] by symmetry
    float s21 = vgetq_lane_f32(s2, 1);
    float t1 = s21 - s20 * s10 * inv0sq; // = s21 - l20*l10 = l21/inv0
    float l21sq =t1*t1*inv1sq;
    float l22sq = s22 - s20*s20*inv0sq - l21sq;
    float inv2sq = 1.0f/l22sq;

    float s33 = vgetq_lane_f32(s3, 3) + d[3];
    float s30 = vgetq_lane_f32(s3, 0);  // == c0[3] by symmetry
    float s31 = vgetq_lane_f32(s3, 1);
    float s32 = vgetq_lane_f32(s3, 2);
    float t2 = s31 - s30 * s10 * inv0sq; // = s31 - l30 * l10 = l31/inv1
    float l31sq = t2*t2*inv1sq;
    float t3 = s32 - s30 * s20 * inv0sq - (s31 - s30 * s10 * inv0sq) * (s21 - s20 * s10 * inv0sq) * inv1sq; // = s32 - l30 * l20 - l31 * l21 = l32/inv2
    float l32sq = t3*t3 * inv2sq;
    float l33sq = s33 - s30*s30*inv0sq - l31sq - l32sq;

    float32x4_t sq = { s00, l11sq, l22sq, l33sq };
    float32x4_t inv = vrsqrteq_f32(sq);
    inv = vmulq_f32(vrsqrtsq_f32(vmulq_f32(sq, inv), inv), inv); // newton step

    float inv1 = vgetq_lane_f32(inv, 1);
    float inv2 = vgetq_lane_f32(inv, 2);

    float32x4_t sx0 = { 0.0f, s10, s20, s30 };

    float32x4_t Lcol0 = vmulq_laneq_f32(sx0, inv, 0);  // {0, l10, l20, l30}

    float32x4_t Lcol12 = {
        0.0f,       // padding
        t1 * inv1,  // l21  -- lane 1
        t2 * inv1,  // l31  -- lane 2
        t3 * inv2   // l32  -- lane 3
    };

    float32x4_t tvec = { 0.0f, t1, t2, t3 };

    // Forward solve LX=A using laneq indexing:

    // Row 0
    float32x4_t x0 = vmulq_laneq_f32(a0, inv, 0);       // a0 * inv[0]

    // Row 1:  (a1 - l10*x0) * inv[1]
    float32x4_t x1 = vfmsq_laneq_f32(a1, x0, Lcol0, 1);  // a1 - l10*x0
    x1 = vmulq_laneq_f32(x1, inv, 1);

    // Row 2:  (a2 - l20*x0 - l21*x1) * inv[2]
    float32x4_t x2 = vfmsq_laneq_f32(a2, x0, Lcol0, 2);   // a2 - l20*x0
    x2 = vfmsq_laneq_f32(x2, x1, Lcol12, 1);               // ... - l21*x1
    x2 = vmulq_laneq_f32(x2, inv, 2);

    // Row 3:  (a3 - l30*x0 - l31*x1 - l32*x2) * inv[3]
    float32x4_t x3 = vfmsq_laneq_f32(a3, x0, Lcol0, 3);   // a3 - l30*x0
    x3 = vfmsq_laneq_f32(x3, x1, Lcol12, 2);               // ... - l31*x1
    x3 = vfmsq_laneq_f32(x3, x2, Lcol12, 3);               // ... - l32*x2
    x3 = vmulq_laneq_f32(x3, inv, 3);

    // L^T Y = X
    // Row 3: y_row3 = x_row3 * r33
    float32x4_t y3 = vmulq_laneq_f32(x3, inv, 3);

    // Row 2: y_row2 = (x_row2 - l32 * y_row3) * r22
    float32x4_t y2 = vfmsq_laneq_f32(x2, y3, Lcol12, 3);
    y2 = vmulq_laneq_f32(y2, inv, 2);

    // Row 1: x_row1 = (x_row1 - l21*y_row2 - l31*y_row3) * r11
    float32x4_t y1 = vfmsq_laneq_f32(x1, y2, Lcol12, 1);
    y1 = vfmsq_laneq_f32(y1, x3, Lcol12, 2);
    y1 = vmulq_laneq_f32(y1, inv, 1);

    // Row 0: x_row0 = (x0 - l10*y_row1 - l20*y_row2 - l30*y_row3) * r00
    float32x4_t y0 = vfmsq_laneq_f32(x0, y1, Lcol0, 1);
    y0 = vfmsq_laneq_f32(y0, y2, Lcol0, 2);
    y0 = vfmsq_laneq_f32(y0, y3, Lcol0, 3);
    y0 = vmulq_laneq_f32(y0, inv, 0);

    return { y0, y1, y2 };
}

// c0..c3: rows of C (the 4x4 symmetric result from A^T B A)
// r: float32x4_t diagonal of R = {r0, r1, r2, r3}
// Returns L column-major in the Chol4 struct, or inline everything
inline float32x4x4_t cholesky4_from_rows(
    float32x4_t c0, float32x4_t c1, float32x4_t c2, float32x4_t c3,
    std::array<float, 4> r)
{
    // Extract the elements we need via vgetq_lane_f32
    // Diagonal gets R added
    float s00 = vgetq_lane_f32(c0, 0) + r[0];
    float s10 = vgetq_lane_f32(c1, 0);  // == c0[1] by symmetry, but c1[0] is what we computed
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

    // now for inverse

    // L^{-1} by forward substitution on identity columns
    float Li[16] = {};

    // Column 0 of L^{-1}: solve L*x = e0

    Li[0]  = inv0;
    Li[1]  = -l10 * Li[0] * inv1;
    Li[2]  = (-l20 * Li[0] - l21 * Li[1]) * inv2;
    Li[3]  = (-l30 * Li[0] - l31 * Li[1] - l32 * Li[2]) * inv3;

    // Column 1: solve L*x = e1
    Li[5]  = inv1;
    Li[6]  = -l21 * Li[5] * inv2;
    Li[7]  = (-l31 * Li[5] - l32 * Li[6]) * inv3;

    // Column 2: solve L*x = e2
    Li[10] = inv2;
    Li[11] = -l32 * Li[10] * inv3;

    // Column 3
    Li[15] = inv3;

    // Now Cinv = Li^T * Li (symmetric)
    // Li is lower triangular column-major
    // Can use NEON: columns of Li are float32x4_t
    float32x4_t li0 = vld1q_f32(Li);
    float32x4_t li1 = vld1q_f32(Li + 4);
    float32x4_t li2 = vld1q_f32(Li + 8);
    float32x4_t li3 = vld1q_f32(Li + 12);

    // Cinv[i][j] = dot(Li[:,i], Li[:,j]) = sum_k Li[k][i]*Li[k][j]
    // Column j of Cinv: Cinv[:,j] = Li^T * Li[:,j]
    //   = li0 * Li[0][j] + li1 * Li[1][j] + li2 * Li[2][j] + li3 * Li[3][j]
    // But Li is lower triangular so Li[k][j] = 0 for k < j

    // Col 0: all columns contribute
    float32x4_t i0 = vmulq_laneq_f32(li0, li0, 0);
    i0 = vfmaq_laneq_f32(i0, li1, li1, 0);
    i0 = vfmaq_laneq_f32(i0, li2, li2, 0);
    i0 = vfmaq_laneq_f32(i0, li3, li3, 0);

    // Col 1: only li1, li2, li3 (Li[0][1] = 0)
    float32x4_t i1 = vmulq_laneq_f32(li1, li1, 1);
    i1 = vfmaq_laneq_f32(i1, li2, li2, 1);
    i1 = vfmaq_laneq_f32(i1, li3, li3, 1);

    // Col 2: only li2, li3
    float32x4_t i2 = vmulq_laneq_f32(li2, li2, 2);
    i2 = vfmaq_laneq_f32(i2, li3, li3, 2);

    // Col 3: only li3
    float32x4_t i3 = vmulq_laneq_f32(li3, li3, 3);

    return float32x4x4_t { i0,i1,i2,i3 };
}

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
    float* u_star)
{
    // ---- Constants ----
    float32x4_t a = { A[0], A[1], A[2], 1.0f };

    float32x4_t b0_body = vld1q_f32(B0.data());
    float32x4_t b1_body = vld1q_f32(B0.data() + 4);
    float32x4_t b2_body = vld1q_f32(B0.data() + 8);

    float32x4_t R_vec = { R[0], R[1], R[2], R[3] };

    // Per-step storage: K (3 columns, each 4-wide) + v (4-wide) = 16 floats
    // Layout: [K_col0(4) | K_col1(4) | K_col2(4) | v(4)] per step
    std::vector<float> gains(N * 16);
    // Also store rotated B per step for forward pass
    std::vector<float> B_rotated(N * 12);

    // ---- Terminal condition ----
    // P_N = Q, p_N = -Q * xr_N
    // Lane 3 holds p_upper_i = -Q[i] * xr_N[i]
    const float* xr_N = xr_upper + N * 3;
    float32x4_t p0 = { Q[0], 0.0f, 0.0f, -Q[0] * xr_N[0] };
    float32x4_t p1 = { 0.0f, Q[1], 0.0f, -Q[1] * xr_N[1] };
    float32x4_t p2 = { 0.0f, 0.0f, Q[2], -Q[2] * xr_N[2] };

    // ---- Backward pass ----
    for (int k = N - 1; k >= 0; k--) {
        const float* xr_k = xr_upper + k * 3;
        const float* ur_k = ur + k * 4;
        const float* c_k  = c_upper + k * 3;

        // 1. Rotate B
        float ct = cosf(theta[k]);
        float st = sinf(theta[k]);
        float32x4_t ct_v = vdupq_n_f32(ct);
        float32x4_t st_v = vdupq_n_f32(st);

        float32x4_t b0 = vfmsq_f32(vmulq_f32(ct_v, b0_body), st_v, b1_body);
        float32x4_t b1 = vfmaq_f32(vmulq_f32(st_v, b0_body), ct_v, b1_body);
        float32x4_t b2 = b2_body;

        // Store rotated B for forward pass
        vst1q_f32(B_rotated.data() + k * 12 + 0, b0);
        vst1q_f32(B_rotated.data() + k * 12 + 4, b1);
        vst1q_f32(B_rotated.data() + k * 12 + 8, b2);

        // 2. PB = P_uu * B_u  (uses lanes 0-2 of p rows)
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
        float32x4x4_t Cinv = cholesky4_from_rows(btpb0, btpb1, btpb2, btpb3, R);

        // 6. invBt = S^{-1} B^T  (column by column: invBt_i = Cinv * b_i_transposed)
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
        float* g = gains.data() + k * 16;
        vst1q_f32(g + 0,  K_col0);
        vst1q_f32(g + 4,  K_col1);
        vst1q_f32(g + 8,  K_col2);
        vst1q_f32(g + 12, v);

        // 10. Schur complement entries (upper triangle)
        float32x4_t atpb0 = vmulq_laneq_f32(pb0, a, 0);
        float32x4_t atpb1 = vmulq_laneq_f32(pb1, a, 1);
        float32x4_t atpb2 = vmulq_laneq_f32(pb2, a, 2);

        float sc00 = vaddvq_f32(vmulq_f32(atpb0, K_col0));
        float sc01 = vaddvq_f32(vmulq_f32(atpb0, K_col1));
        float sc02 = vaddvq_f32(vmulq_f32(atpb0, K_col2));
        float sc11 = vaddvq_f32(vmulq_f32(atpb1, K_col1));
        float sc12 = vaddvq_f32(vmulq_f32(atpb1, K_col2));
        float sc22 = vaddvq_f32(vmulq_f32(atpb2, K_col2));

        // 11. d = ATPB * z (for affine p update)
        float d0 = vaddvq_f32(vmulq_f32(atpb0, z));
        float d1 = vaddvq_f32(vmulq_f32(atpb1, z));
        float d2 = vaddvq_f32(vmulq_f32(atpb2, z));

        // 12. ATPA = DSD(P, a)  — lane 3 gives a_i * p_upper_i
        float32x4_t ATPA0 = vmulq_f32(pa0, a);
        float32x4_t ATPA1 = vmulq_f32(pa1, a);
        float32x4_t ATPA2 = vmulq_f32(pa2, a);

        // 13. Construct correction f
        // Lanes 0-2: Q_combined - Schur
        // Lane 3: f3_i = -(Q[i]+Q[3+i])*xr[i] + a[i]*(L_i - p_upper_i) - d_i
        //   where (L_i - p_upper_i) = (P_uu * c)_i
        float p_upper_0 = vgetq_lane_f32(p0, 3);
        float p_upper_1 = vgetq_lane_f32(p1, 3);
        float p_upper_2 = vgetq_lane_f32(p2, 3);

        float f3_0 = -(Q[0] + Q[3]) * xr_k[0] + A[0] * (L0 - p_upper_0) - d0;
        float f3_1 = -(Q[1] + Q[4]) * xr_k[1] + A[1] * (L1 - p_upper_1) - d1;
        float f3_2 = -(Q[2] + Q[5]) * xr_k[2] + A[2] * (L2 - p_upper_2) - d2;

        float32x4_t f0 = { Q[0] + Q[3] - sc00, -sc01,              -sc02,              f3_0 };
        float32x4_t f1 = { -sc01,               Q[1] + Q[4] - sc11, -sc12,              f3_1 };
        float32x4_t f2 = { -sc02,               -sc12,               Q[2] + Q[5] - sc22, f3_2 };

        // 14. P_new = ATPA + f
        p0 = vaddq_f32(ATPA0, f0);
        p1 = vaddq_f32(ATPA1, f1);
        p2 = vaddq_f32(ATPA2, f2);
    }

    // ---- Forward pass ----
    float x0 = x0_upper[0];
    float x1 = x0_upper[1];
    float x2 = x0_upper[2];

    for (int k = 0; k < N; k++) {
        const float* g = gains.data() + k * 16;
        float32x4_t K_col0 = vld1q_f32(g + 0);
        float32x4_t K_col1 = vld1q_f32(g + 4);
        float32x4_t K_col2 = vld1q_f32(g + 8);
        float32x4_t v      = vld1q_f32(g + 12);

        // u = -(K_col0*x0 + K_col1*x1 + K_col2*x2) - v
        float32x4_t u_k = vnegq_f32(vfmaq_n_f32(
            vfmaq_n_f32(vmulq_n_f32(K_col0, x0), K_col1, x1),
            K_col2, x2));
        u_k = vsubq_f32(u_k, v);

        // Clamp to [-1, 1]
        u_k = vmaxq_f32(vminq_f32(u_k, vdupq_n_f32(1.0f)), vdupq_n_f32(-1.0f));

        vst1q_f32(u_star + k * 4, u_k);

        // Propagate state: x_upper_{k+1} = D_u * x_upper + B_u * u + c_upper
        float32x4_t b0_k = vld1q_f32(B_rotated.data() + k * 12 + 0);
        float32x4_t b1_k = vld1q_f32(B_rotated.data() + k * 12 + 4);
        float32x4_t b2_k = vld1q_f32(B_rotated.data() + k * 12 + 8);

        float Bu0 = vaddvq_f32(vmulq_f32(b0_k, u_k));
        float Bu1 = vaddvq_f32(vmulq_f32(b1_k, u_k));
        float Bu2 = vaddvq_f32(vmulq_f32(b2_k, u_k));

        const float* c_k = c_upper + k * 3;
        x0 = A[0] * x0 + Bu0 + c_k[0];
        x1 = A[1] * x1 + Bu1 + c_k[1];
        x2 = A[2] * x2 + Bu2 + c_k[2];
    }
}

// ---------------------------------------------------------------------------
// y = A * x,  A is m x n column-major
// A[i + m*j] is element (i,j).
// y[i] = sum_j A[i + m*j] * x[j]
//
// We use a column-oriented approach: for each column j, scatter-add
// A[:,j] * x[j] into y.  This gives unit-stride access to A.
// ---------------------------------------------------------------------------
void neon_gemv_colmajor(int m, int n, const double* A, const double* x, double* y)
{
    // Zero out y
    {
        int i = 0;
        for (; i + 1 < m; i += 2) {
            vst1q_f64(y + i, vdupq_n_f64(0.0));
        }
        for (; i < m; ++i) {
            y[i] = 0.0;
        }
    }

    // Accumulate column by column
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
// L is n x n lower triangular, column-major: L[i,j] = L[i + n*j], j <= i.
//
// Column-oriented forward substitution:
//   Copy b into y.
//   For j = 0 .. n-1:
//     y[j] /= L[j + n*j]
//     y[i] -= L[i + n*j] * y[j]   for i = j+1 .. n-1
//
// The inner update is a unit-stride AXPY on column j of L, which is
// SIMD-friendly.
// ---------------------------------------------------------------------------
void neon_trsv_lower_colmajor(int n, const double* L, const double* b, double* y)
{
    // Copy b into y
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
    }

    for (int j = 0; j < n; ++j) {
        const double* Lj = L + (long)n * j;  // column j of L
        y[j] /= Lj[j];                       // L[j,j]

        double yj_val = y[j];
        float64x2_t yj = vdupq_n_f64(yj_val);

        int i = j + 1;
        // Align i to even index if needed for clean SIMD
        if ((i & 1) && i < n) {
            y[i] -= Lj[i] * yj_val;
            ++i;
        }
        for (; i + 1 < n; i += 2) {
            float64x2_t yi  = vld1q_f64(y + i);
            float64x2_t lij = vld1q_f64(Lj + i);
            yi = vfmsq_f64(yi, lij, yj);   // yi -= lij * yj
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
// L^T[i,j] = L[j,i] = L[j + n*i].  L^T is upper triangular.
//
// Column-oriented backward substitution on L^T:
//   Copy y into x.
//   For j = n-1 .. 0:
//     x[j] /= L[j + n*j]          (diagonal of L^T is same as L)
//     x[i] -= L[j + n*i] * x[j]   for i = 0 .. j-1
//
// But L[j + n*i] for varying i with fixed j is stride-n (bad for SIMD).
//
// Alternative: row-oriented backward substitution (standard upper trsv):
//   Copy y into x.
//   For i = n-1 .. 0:
//     sum = 0
//     for j = i+1 .. n-1: sum += L^T[i,j] * x[j] = L[j + n*i] * x[j]
//     x[i] = (y[i] - sum) / L[i + n*i]
//
// Here L[j + n*i] for varying j with fixed i is column i of L starting at
// row j -- this IS unit-stride in memory!  So we vectorize over j.
// ---------------------------------------------------------------------------
void neon_trsv_upper_trans_colmajor(int n, const double* L, const double* y, double* x)
{
    // Copy y into x
    for (int i = 0; i < n; ++i) {
        x[i] = y[i];
    }

    for (int i = n - 1; i >= 0; --i) {
        // Accumulate sum = sum_j L[j + n*i] * x[j] for j = i+1..n-1
        // L column i starts at L + n*i.  We read L[j + n*i] for j = i+1..n-1.
        const double* Li = L + (long)n * i;  // column i of L

        float64x2_t acc = vdupq_n_f64(0.0);
        double sum = 0.0;

        int j = i + 1;
        // Align j to even if needed
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

        // Horizontal add the NEON accumulator
        sum += vaddvq_f64(acc);

        x[i] = (x[i] - sum) / Li[i];
    }
}

// ---------------------------------------------------------------------------
// y = clip(x, lo, hi), returns number of elements that were clipped.
// ---------------------------------------------------------------------------
int neon_clip_and_count(int n, const double* x, double lo, double hi, double* y)
{
    float64x2_t vlo = vdupq_n_f64(lo);
    float64x2_t vhi = vdupq_n_f64(hi);
    int count = 0;

    int i = 0;
    for (; i + 1 < n; i += 2) {
        float64x2_t xi = vld1q_f64(x + i);
        float64x2_t clamped = vmaxq_f64(vminq_f64(xi, vhi), vlo);
        vst1q_f64(y + i, clamped);

        // Compare to detect clipping: if clamped != xi, element was clipped
        uint64x2_t eq = vceqq_f64(clamped, xi);
        // eq lanes are all-ones if equal, all-zeros if not
        // Count not-equal lanes
        uint64_t lane0 = vgetq_lane_u64(eq, 0);
        uint64_t lane1 = vgetq_lane_u64(eq, 1);
        if (lane0 == 0) ++count;
        if (lane1 == 0) ++count;
    }
    for (; i < n; ++i) {
        double v = x[i];
        if (v < lo) { y[i] = lo; ++count; }
        else if (v > hi) { y[i] = hi; ++count; }
        else { y[i] = v; }
    }

    return count;
}

#endif // MPC_USE_NEON
