// ipm_solver.cpp -- Custom interior point solver with 6-state block-sparse
//                   Riccati recursion for Euler-discretized mecanum dynamics.
//
// The Euler discretization gives:
//   A = [I₃  dt·I₃]    B(θ) = [0₃ₓ₄       ]
//       [0₃  D₃   ]            [Bl(θ)      ]
//
// where D₃ = velocity damping diagonal, Bl = dt·Rot(θ)·Bc_body.
//
// The log-barrier IPM solves box-constrained OCP by iterating Riccati
// solves with per-stage R_eff = R + barrier_hessian.

#include "ipm_solver.h"
#include "mecanum_model.h"
#include "heading_lookup.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <time.h>

// =========================================================================
// Section 1: Euler dynamics precomputation
// =========================================================================

void euler_dynamics_precompute(const ModelParams& params, double dt,
                               EulerDynamicsData& data)
{
    data.dt = (float)dt;

    // Get continuous-time matrices at theta=0
    double Ac[NX * NX], Bc[NX * NU];
    continuous_dynamics(0.0, params, Ac, Bc);

    // D_diag: velocity damping from Euler discretization
    // A_d = I + dt*Ac, so diagonal of lower-right block = 1 + dt*Ac[3+i, 3+i]
    for (int i = 0; i < 3; ++i)
        data.D_diag[i] = (float)(1.0 + dt * Ac[(3 + i) + NX * (3 + i)]);

    // B_body: dt * lower 3 rows of Bc at theta=0, stored as 3×4 row-major
    // Bc is column-major: Bc[row + NX*col]
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < NU; ++j)
            data.B_body[i * 4 + j] = (float)(dt * Bc[(3 + i) + NX * j]);
}

// =========================================================================
// Section 2: Scalar 6-state block-sparse Riccati (double precision)
// =========================================================================

// Symmetric 3×3 stored as [P00, P01, P02, P11, P12, P22] (upper triangle, row-major)
// Index helpers:
//   sym3(i,j) with i<=j: idx = i*3 - i*(i-1)/2 + (j-i)
//   or just use flat: [0]=00, [1]=01, [2]=02, [3]=11, [4]=12, [5]=22

static inline int sym3_idx(int i, int j) {
    if (i > j) { int t = i; i = j; j = t; }
    return i * 3 - i * (i - 1) / 2 + (j - i);
}

static inline double sym3_get(const double* S, int i, int j) {
    return S[sym3_idx(i, j)];
}

static inline void sym3_set(double* S, int i, int j, double v) {
    S[sym3_idx(i, j)] = v;
}

void riccati_6state_scalar(
    const double Q_diag[6],
    const double Qf_diag[6],
    const double R_eff[],
    const double D_diag[3],
    double dt,
    const double B_body[12],
    int N,
    const double* theta,
    const double* xr,
    const double* ur_eff,
    const double x0[NX],
    double* u_out)
{
    // P blocks: Pp(6), Ppv(9), Pv(6) — symmetric stored as upper triangle
    // Ppv is general 3×3 stored row-major [Ppv00, Ppv01, Ppv02, Ppv10, ...]
    double Pp[6], Ppv[9], Pv[6];
    double pp[3], pv[3];  // affine costates

    // Terminal condition: P_N = Qf, p_N = -Qf * xr_N
    const double* xr_N = xr + N * NX;
    Pp[0] = Qf_diag[0]; Pp[1] = 0; Pp[2] = 0;
    Pp[3] = Qf_diag[1]; Pp[4] = 0; Pp[5] = Qf_diag[2];

    std::memset(Ppv, 0, 9 * sizeof(double));

    Pv[0] = Qf_diag[3]; Pv[1] = 0; Pv[2] = 0;
    Pv[3] = Qf_diag[4]; Pv[4] = 0; Pv[5] = Qf_diag[5];

    for (int i = 0; i < 3; ++i) {
        pp[i] = -Qf_diag[i] * xr_N[i];
        pv[i] = -Qf_diag[3 + i] * xr_N[3 + i];
    }

    // Per-stage backward pass storage
    double Kp_store[N_MAX * 12];  // 4×3 per stage, column-major (col j = 4 values)
    double Kv_store[N_MAX * 12];
    double v_store[N_MAX * 4];
    double Bl_store[N_MAX * 12];  // 3×4 row-major

    for (int k = N - 1; k >= 0; --k) {
        const double* xr_k = xr + k * NX;
        const double* ur_k = ur_eff + k * NU;
        const double* R_k  = R_eff + k * NU;

        // 1. Rotate B_body by theta[k]
        double ct = std::cos(theta[k]);
        double st = std::sin(theta[k]);
        double Bl[12]; // 3×4 row-major: Bl[row][col] = Bl[row*4+col]
        for (int j = 0; j < 4; ++j) {
            Bl[0 * 4 + j] =  ct * B_body[0 * 4 + j] - st * B_body[1 * 4 + j];
            Bl[1 * 4 + j] =  st * B_body[0 * 4 + j] + ct * B_body[1 * 4 + j];
            Bl[2 * 4 + j] = B_body[2 * 4 + j];
        }
        std::memcpy(Bl_store + k * 12, Bl, sizeof(Bl));

        // 2. PvBl = Pv · Bl (3×4, using symmetric Pv)
        double PvBl[12];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int m = 0; m < 3; ++m)
                    s += sym3_get(Pv, i, m) * Bl[m * 4 + j];
                PvBl[i * 4 + j] = s;
            }

        // 3. S = Bl' · PvBl + diag(R_k) (4×4 symmetric)
        double S[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int m = 0; m < 3; ++m)
                    s += Bl[m * 4 + i] * PvBl[m * 4 + j];
                S[i * 4 + j] = s + (i == j ? R_k[i] : 0.0);
            }

        // 4. Cholesky S = LL', then S_inv via forward/back substitution
        double Lch[16] = {};
        for (int i = 0; i < 4; ++i) {
            double s = S[i * 4 + i];
            for (int m = 0; m < i; ++m) s -= Lch[i * 4 + m] * Lch[i * 4 + m];
            Lch[i * 4 + i] = std::sqrt(s);
            double inv = 1.0 / Lch[i * 4 + i];
            for (int j = i + 1; j < 4; ++j) {
                s = S[j * 4 + i];
                for (int m = 0; m < i; ++m) s -= Lch[j * 4 + m] * Lch[i * 4 + m];
                Lch[j * 4 + i] = s * inv;
            }
        }

        double Sinv[16] = {};
        for (int col = 0; col < 4; ++col) {
            double y[4] = {};
            for (int i = 0; i < 4; ++i) {
                double s = (i == col) ? 1.0 : 0.0;
                for (int j = 0; j < i; ++j) s -= Lch[i * 4 + j] * y[j];
                y[i] = s / Lch[i * 4 + i];
            }
            for (int i = 3; i >= 0; --i) {
                double s = y[i];
                for (int j = i + 1; j < 4; ++j) s -= Lch[j * 4 + i] * Sinv[j * 4 + col];
                Sinv[i * 4 + col] = s / Lch[i * 4 + i];
            }
        }

        // 5. invBt = S_inv · Bl' (4×3)
        //    invBt[a*3+b] = sum_m S_inv[a*4+m] * Bl[b*4+m]
        double invBt[12];
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 3; ++b) {
                double s = 0;
                for (int m = 0; m < 4; ++m)
                    s += Sinv[a * 4 + m] * Bl[b * 4 + m];
                invBt[a * 3 + b] = s;
            }

        // 6. PpvBl = Ppv · Bl (3×4)
        double PpvBl[12];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int m = 0; m < 3; ++m)
                    s += Ppv[i * 3 + m] * Bl[m * 4 + j];
                PpvBl[i * 4 + j] = s;
            }

        // 7. DTBl = dt·PpvBl + D·PvBl (3×4)
        //    = (dt·Ppv + D·Pv) · Bl
        double DTBl[12];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                DTBl[i * 4 + j] = dt * PpvBl[i * 4 + j] + D_diag[i] * PvBl[i * 4 + j];

        // 8. Kp = invBt · Ppv' (4×3)
        //    Kp[a][b] = sum_m invBt[a][m] * Ppv[b][m]  (Ppv' col b = Ppv row b)
        double Kp[12]; // 4×3, Kp[a*3+b]
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 3; ++b) {
                double s = 0;
                for (int m = 0; m < 3; ++m)
                    s += invBt[a * 3 + m] * Ppv[b * 3 + m];
                Kp[a * 3 + b] = s;
            }

        // 9. Form M = dt·Ppv' + Pv·D (3×3)
        //    M[i][j] = dt*Ppv[j][i] + Pv[i][j]*D[j]
        double M_kv[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                M_kv[i * 3 + j] = dt * Ppv[j * 3 + i] + sym3_get(Pv, i, j) * D_diag[j];

        // 10. Kv = invBt · M_kv (4×3)
        double Kv[12];
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 3; ++b) {
                double s = 0;
                for (int m = 0; m < 3; ++m)
                    s += invBt[a * 3 + m] * M_kv[m * 3 + b];
                Kv[a * 3 + b] = s;
            }

        // Store gains for forward pass (column-major: col j has 4 entries)
        // We store as Kp[a*3+b] → rearrange to col-major for forward pass
        std::memcpy(Kp_store + k * 12, Kp, sizeof(Kp));
        std::memcpy(Kv_store + k * 12, Kv, sizeof(Kv));

        // 11. Affine feedforward: z = invBt · pv, then v = z - Sinv·R_k·ur_k
        double z[4];
        for (int a = 0; a < 4; ++a) {
            z[a] = 0;
            for (int m = 0; m < 3; ++m)
                z[a] += invBt[a * 3 + m] * pv[m];
        }
        double v_aff[4];
        for (int a = 0; a < 4; ++a) {
            double sinv_rur = 0;
            for (int m = 0; m < 4; ++m)
                sinv_rur += Sinv[a * 4 + m] * R_k[m] * ur_k[m];
            v_aff[a] = z[a] - sinv_rur;
        }
        std::memcpy(v_store + k * 4, v_aff, sizeof(v_aff));

        // 12. Schur corrections
        // Schur_pp = PpvBl · Kp (3×3 sym)
        double Sch_pp[6] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) {
                double s = 0;
                for (int m = 0; m < 4; ++m)
                    s += PpvBl[i * 4 + m] * Kp[m * 3 + j];
                sym3_set(Sch_pp, i, j, s);
            }

        // Schur_pv = PpvBl · Kv (3×3 full)
        double Sch_pv[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int m = 0; m < 4; ++m)
                    s += PpvBl[i * 4 + m] * Kv[m * 3 + j];
                Sch_pv[i * 3 + j] = s;
            }

        // Schur_vv = DTBl · Kv (3×3 sym)
        double Sch_vv[6] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) {
                double s = 0;
                for (int m = 0; m < 4; ++m)
                    s += DTBl[i * 4 + m] * Kv[m * 3 + j];
                sym3_set(Sch_vv, i, j, s);
            }

        // 13. Compute d_p and d_v for affine costate (dot of PpvBl/DTBl with v_aff)
        //     The costate update uses v (full feedforward including R·ur term),
        //     NOT z (which is just S^{-1}·B'·p without the ur correction).
        double d_p[3], d_v[3];
        for (int i = 0; i < 3; ++i) {
            d_p[i] = 0;
            d_v[i] = 0;
            for (int m = 0; m < 4; ++m) {
                d_p[i] += PpvBl[i * 4 + m] * v_aff[m];
                d_v[i] += DTBl[i * 4 + m] * v_aff[m];
            }
        }

        // 14. P updates

        // Pp_new = Pp + Qp - Schur_pp  (A'PA_pp = Pp, identity block!)
        double Pp_new[6];
        for (int idx = 0; idx < 6; ++idx)
            Pp_new[idx] = Pp[idx] + 0.0 - Sch_pp[idx]; // Qp added below per diagonal
        // Add Qp to diagonal
        Pp_new[0] += Q_diag[0]; // (0,0)
        Pp_new[3] += Q_diag[1]; // (1,1)
        Pp_new[5] += Q_diag[2]; // (2,2)

        // Ppv_new = dt·Pp + Ppv·D - Schur_pv
        double Ppv_new[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double dtPp_ij = dt * sym3_get(Pp, i, j);
                double PpvD_ij = Ppv[i * 3 + j] * D_diag[j];
                Ppv_new[i * 3 + j] = dtPp_ij + PpvD_ij - Sch_pv[i * 3 + j];
            }

        // Pv_new = dt²·Pp + dt·(Ppv·D + D·Ppv') + D·Pv·D + Qv - Schur_vv
        double Pv_new[6];
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) {
                double dt2Pp = dt * dt * sym3_get(Pp, i, j);
                double cross = dt * (Ppv[i * 3 + j] * D_diag[j] + D_diag[i] * Ppv[j * 3 + i]);
                double DPvD = D_diag[i] * sym3_get(Pv, i, j) * D_diag[j];
                double Qv_ij = (i == j) ? Q_diag[3 + i] : 0.0;
                sym3_set(Pv_new, i, j, dt2Pp + cross + DPvD + Qv_ij - sym3_get(Sch_vv, i, j));
            }

        // 15. Affine costate update
        //   pp_new = pp - d_p - Qp·xr_p    (A' maps pp → pp, pv → dt·pp + D·pv)
        //   pv_new = dt·pp + D·pv - d_v - Qv·xr_v
        double pp_new[3], pv_new[3];
        for (int i = 0; i < 3; ++i) {
            pp_new[i] = pp[i] - d_p[i] - Q_diag[i] * xr_k[i];
            pv_new[i] = dt * pp[i] + D_diag[i] * pv[i] - d_v[i] - Q_diag[3 + i] * xr_k[3 + i];
        }

        std::memcpy(Pp, Pp_new, sizeof(Pp));
        std::memcpy(Ppv, Ppv_new, sizeof(Ppv));
        std::memcpy(Pv, Pv_new, sizeof(Pv));
        std::memcpy(pp, pp_new, sizeof(pp));
        std::memcpy(pv, pv_new, sizeof(pv));
    }

    // ---- Forward pass ----
    double xp[3] = { x0[0], x0[1], x0[2] };
    double xv[3] = { x0[3], x0[4], x0[5] };

    for (int k = 0; k < N; ++k) {
        const double* Kp = Kp_store + k * 12;
        const double* Kv = Kv_store + k * 12;
        const double* v  = v_store  + k * 4;
        const double* Bl = Bl_store + k * 12;

        // u_k = -(Kp·x_p + Kv·x_v) - v
        for (int a = 0; a < 4; ++a) {
            double u = -v[a];
            for (int b = 0; b < 3; ++b) {
                u -= Kp[a * 3 + b] * xp[b];
                u -= Kv[a * 3 + b] * xv[b];
            }
            u_out[k * 4 + a] = u;
        }

        // Propagate: x_p_next = x_p + dt·x_v
        //            x_v_next = D·x_v + Bl·u
        double Bu[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                Bu[i] += Bl[i * 4 + j] * u_out[k * 4 + j];

        double xp_next[3], xv_next[3];
        for (int i = 0; i < 3; ++i) {
            xp_next[i] = xp[i] + dt * xv[i];
            xv_next[i] = D_diag[i] * xv[i] + Bu[i];
        }
        std::memcpy(xp, xp_next, sizeof(xp));
        std::memcpy(xv, xv_next, sizeof(xv));
    }
}

// =========================================================================
// Section 3: NEON 6-state block-sparse Riccati (float32)
// =========================================================================
#ifdef MPC_USE_NEON

#ifdef __aarch64__
#include <arm_neon.h>
#else
#include "neon_sim.h"
#endif

// Reuse rsqrt_nr and cholesky4_inv from neon_kernels.cpp
// (declared static there, so we duplicate the small functions here)

static inline float rsqrt_nr_ipm(float x) {
    float32_t e = vrsqrtes_f32(x);
    e *= vrsqrtss_f32(x * e, e);
    e *= vrsqrtss_f32(x * e, e);
    return e;
}

static inline float32x4x4_t cholesky4_inv_ipm(
    float32x4_t c0, float32x4_t c1, float32x4_t c2, float32x4_t c3,
    float32x4_t r_diag)  // per-stage R loaded as a vector
{
    float s00 = vgetq_lane_f32(c0, 0) + vgetq_lane_f32(r_diag, 0);
    float s10 = vgetq_lane_f32(c1, 0);
    float s20 = vgetq_lane_f32(c2, 0);
    float s30 = vgetq_lane_f32(c3, 0);

    float inv0 = rsqrt_nr_ipm(s00);
    float l10 = s10 * inv0;
    float l20 = s20 * inv0;
    float l30 = s30 * inv0;

    float s11 = vgetq_lane_f32(c1, 1) + vgetq_lane_f32(r_diag, 1);
    float s21 = vgetq_lane_f32(c2, 1);
    float s31 = vgetq_lane_f32(c3, 1);

    float inv1 = rsqrt_nr_ipm(s11 - l10 * l10);
    float l21 = (s21 - l20 * l10) * inv1;
    float l31 = (s31 - l30 * l10) * inv1;

    float s22 = vgetq_lane_f32(c2, 2) + vgetq_lane_f32(r_diag, 2);
    float s32 = vgetq_lane_f32(c3, 2);

    float inv2 = rsqrt_nr_ipm(s22 - l20 * l20 - l21 * l21);
    float l32 = (s32 - l30 * l20 - l31 * l21) * inv2;

    float s33 = vgetq_lane_f32(c3, 3) + vgetq_lane_f32(r_diag, 3);

    float inv3 = rsqrt_nr_ipm(s33 - l30 * l30 - l31 * l31 - l32 * l32);

    // L^{-1} by forward substitution
    float li00 = inv0;
    float li10 = -l10 * li00 * inv1;
    float li20 = (-l20 * li00 - l21 * li10) * inv2;
    float li30 = (-l30 * li00 - l31 * li10 - l32 * li20) * inv3;

    float li11 = inv1;
    float li21 = -l21 * li11 * inv2;
    float li31 = (-l31 * li11 - l32 * li21) * inv3;

    float li22 = inv2;
    float li32 = -l32 * li22 * inv3;

    // Pack as rows of L^{-1} so sum_k row_k * row_k[j] = Li^T * Li = S^{-1}
    float32x4_t li0 = { li00, 0.0f, 0.0f, 0.0f };
    float32x4_t li1 = { li10, li11, 0.0f, 0.0f };
    float32x4_t li2 = { li20, li21, li22, 0.0f };
    float32x4_t li3 = { li30, li31, li32, inv3 };

    // S^{-1} = Li^T * Li
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

    return float32x4x4_t{ i0, i1, i2, i3 };
}

int ipm_riccati_workspace_sz(int N) { return N * 40; }

void riccati_6state_neon(
    float* workspace,
    std::array<float, 6> Q,
    std::array<float, 6> Qf,
    const float* R_eff,
    std::array<float, 3> D,
    float dt,
    std::array<float, 12> B_body,
    int N,
    const float* theta,
    const float* xr,
    const float* ur_eff,
    const float* x0,
    float* u_star)
{
    // Body-frame B rows (each 4-wide for NEON)
    float32x4_t b0_body = vld1q_f32(B_body.data());
    float32x4_t b1_body = vld1q_f32(B_body.data() + 4);
    float32x4_t b2_body = vld1q_f32(B_body.data() + 8);

    float32x4_t D_vec = { D[0], D[1], D[2], 0.0f };

    // Precompute sin/cos table
    float sincos_table[N_MAX * 2];
    for (int k = 0; k < N; ++k) {
        sincos_table[k * 2 + 0] = cosf(theta[k]);
        sincos_table[k * 2 + 1] = sinf(theta[k]);
    }

    // ---- Terminal condition ----
    const float* xr_N = xr + N * 6;

    // Pp block rows (lane 3 = position affine costate)
    float32x4_t pp0 = { Qf[0], 0.0f,  0.0f,  -Qf[0] * xr_N[0] };
    float32x4_t pp1 = { 0.0f,  Qf[1], 0.0f,  -Qf[1] * xr_N[1] };
    float32x4_t pp2 = { 0.0f,  0.0f,  Qf[2], -Qf[2] * xr_N[2] };

    // Ppv block rows (lane 3 = velocity affine costate)
    float32x4_t ppv0 = { 0.0f, 0.0f, 0.0f, -Qf[3] * xr_N[3] };
    float32x4_t ppv1 = { 0.0f, 0.0f, 0.0f, -Qf[4] * xr_N[4] };
    float32x4_t ppv2 = { 0.0f, 0.0f, 0.0f, -Qf[5] * xr_N[5] };

    // Pv block rows (lane 3 unused)
    float32x4_t pvv0 = { Qf[3], 0.0f,  0.0f,  0.0f };
    float32x4_t pvv1 = { 0.0f,  Qf[4], 0.0f,  0.0f };
    float32x4_t pvv2 = { 0.0f,  0.0f,  Qf[5], 0.0f };

    // ---- Backward pass ----
    for (int k = N - 1; k >= 0; --k) {
        const float* xr_k = xr + k * 6;
        const float* ur_k = ur_eff + k * 4;
        float* sd = workspace + k * 40;

        // 1. Rotate B
        float ct = sincos_table[k * 2 + 0];
        float st = sincos_table[k * 2 + 1];
        float32x4_t ct_v = vdupq_n_f32(ct);
        float32x4_t st_v = vdupq_n_f32(st);

        float32x4_t b0 = vfmsq_f32(vmulq_f32(ct_v, b0_body), st_v, b1_body);
        float32x4_t b1 = vfmaq_f32(vmulq_f32(st_v, b0_body), ct_v, b1_body);
        float32x4_t b2 = b2_body;

        // Store rotated B for forward pass
        vst1q_f32(sd + 28, b0);
        vst1q_f32(sd + 32, b1);
        vst1q_f32(sd + 36, b2);

        // 2. PvBl = Pv · Bl (3 rows, each 4-wide)
        float32x4_t pvbl0 = vmulq_laneq_f32(b0, pvv0, 0);
        pvbl0 = vfmaq_laneq_f32(pvbl0, b1, pvv0, 1);
        pvbl0 = vfmaq_laneq_f32(pvbl0, b2, pvv0, 2);

        float32x4_t pvbl1 = vmulq_laneq_f32(b0, pvv1, 0);
        pvbl1 = vfmaq_laneq_f32(pvbl1, b1, pvv1, 1);
        pvbl1 = vfmaq_laneq_f32(pvbl1, b2, pvv1, 2);

        float32x4_t pvbl2 = vmulq_laneq_f32(b0, pvv2, 0);
        pvbl2 = vfmaq_laneq_f32(pvbl2, b1, pvv2, 1);
        pvbl2 = vfmaq_laneq_f32(pvbl2, b2, pvv2, 2);

        // 3. S = Bl' · PvBl (4×4 symmetric)
        float32x4_t btpb0 = vmulq_laneq_f32(pvbl0, b0, 0);
        btpb0 = vfmaq_laneq_f32(btpb0, pvbl1, b1, 0);
        btpb0 = vfmaq_laneq_f32(btpb0, pvbl2, b2, 0);

        float32x4_t btpb1 = vmulq_laneq_f32(pvbl0, b0, 1);
        btpb1 = vfmaq_laneq_f32(btpb1, pvbl1, b1, 1);
        btpb1 = vfmaq_laneq_f32(btpb1, pvbl2, b2, 1);

        float32x4_t btpb2 = vmulq_laneq_f32(pvbl0, b0, 2);
        btpb2 = vfmaq_laneq_f32(btpb2, pvbl1, b1, 2);
        btpb2 = vfmaq_laneq_f32(btpb2, pvbl2, b2, 2);

        float32x4_t btpb3 = vmulq_laneq_f32(pvbl0, b0, 3);
        btpb3 = vfmaq_laneq_f32(btpb3, pvbl1, b1, 3);
        btpb3 = vfmaq_laneq_f32(btpb3, pvbl2, b2, 3);

        // 4. Cholesky + invert S = BTPB + R_eff_k
        float32x4_t R_k = vld1q_f32(R_eff + k * 4);
        float32x4x4_t Cinv = cholesky4_inv_ipm(btpb0, btpb1, btpb2, btpb3, R_k);

        // 5. invBt = S_inv · Bl' (4×3): invBt[:,j] = S_inv * Bl[j,:]
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

        // 6. PpvBl = Ppv · Bl (3×4)
        float32x4_t ppvbl0 = vmulq_laneq_f32(b0, ppv0, 0);
        ppvbl0 = vfmaq_laneq_f32(ppvbl0, b1, ppv0, 1);
        ppvbl0 = vfmaq_laneq_f32(ppvbl0, b2, ppv0, 2);

        float32x4_t ppvbl1 = vmulq_laneq_f32(b0, ppv1, 0);
        ppvbl1 = vfmaq_laneq_f32(ppvbl1, b1, ppv1, 1);
        ppvbl1 = vfmaq_laneq_f32(ppvbl1, b2, ppv1, 2);

        float32x4_t ppvbl2 = vmulq_laneq_f32(b0, ppv2, 0);
        ppvbl2 = vfmaq_laneq_f32(ppvbl2, b1, ppv2, 1);
        ppvbl2 = vfmaq_laneq_f32(ppvbl2, b2, ppv2, 2);

        // 7. DTBl = dt·PpvBl + D·PvBl (3×4)
        float32x4_t dt_v = vdupq_n_f32(dt);
        float32x4_t dtbl0 = vfmaq_laneq_f32(vmulq_f32(dt_v, ppvbl0), pvbl0, D_vec, 0);
        float32x4_t dtbl1 = vfmaq_laneq_f32(vmulq_f32(dt_v, ppvbl1), pvbl1, D_vec, 1);
        float32x4_t dtbl2 = vfmaq_laneq_f32(vmulq_f32(dt_v, ppvbl2), pvbl2, D_vec, 2);

        // 8. Kp = invBt · Ppv' (4×3)
        //    Kp[:,j] = invBt0*Ppv[j,0] + invBt1*Ppv[j,1] + invBt2*Ppv[j,2]
        //    Ppv[j,0] = lane 0 of ppvJ register
        float32x4_t Kp_col0 = vmulq_laneq_f32(invBt0, ppv0, 0);
        Kp_col0 = vfmaq_laneq_f32(Kp_col0, invBt1, ppv0, 1);
        Kp_col0 = vfmaq_laneq_f32(Kp_col0, invBt2, ppv0, 2);

        float32x4_t Kp_col1 = vmulq_laneq_f32(invBt0, ppv1, 0);
        Kp_col1 = vfmaq_laneq_f32(Kp_col1, invBt1, ppv1, 1);
        Kp_col1 = vfmaq_laneq_f32(Kp_col1, invBt2, ppv1, 2);

        float32x4_t Kp_col2 = vmulq_laneq_f32(invBt0, ppv2, 0);
        Kp_col2 = vfmaq_laneq_f32(Kp_col2, invBt1, ppv2, 1);
        Kp_col2 = vfmaq_laneq_f32(Kp_col2, invBt2, ppv2, 2);

        // 9. Form M = dt·Ppv' + Pv·D (3×3) row by row
        //    M_row_i[j] = dt*Ppv[j][i] + Pv[i][j]*D[j]
        //    M_row_0[j] = dt*{ppv0[0],ppv1[0],ppv2[0]} + pvv0[j]*D[j]
        // Extract Ppv transpose columns:
        float ppv_t00 = vgetq_lane_f32(ppv0, 0), ppv_t10 = vgetq_lane_f32(ppv1, 0), ppv_t20 = vgetq_lane_f32(ppv2, 0);
        float ppv_t01 = vgetq_lane_f32(ppv0, 1), ppv_t11 = vgetq_lane_f32(ppv1, 1), ppv_t21 = vgetq_lane_f32(ppv2, 1);
        float ppv_t02 = vgetq_lane_f32(ppv0, 2), ppv_t12 = vgetq_lane_f32(ppv1, 2), ppv_t22 = vgetq_lane_f32(ppv2, 2);

        float32x4_t M_row0 = vfmaq_f32(
            vmulq_f32(pvv0, D_vec),
            dt_v,
            (float32x4_t){ ppv_t00, ppv_t10, ppv_t20, 0.0f });
        float32x4_t M_row1 = vfmaq_f32(
            vmulq_f32(pvv1, D_vec),
            dt_v,
            (float32x4_t){ ppv_t01, ppv_t11, ppv_t21, 0.0f });
        float32x4_t M_row2 = vfmaq_f32(
            vmulq_f32(pvv2, D_vec),
            dt_v,
            (float32x4_t){ ppv_t02, ppv_t12, ppv_t22, 0.0f });

        // 10. Kv = invBt · M (4×3): Kv[:,j] = invBt · M[:,j]
        //     M stored as rows; column j = lane j of each row register
        float32x4_t Kv_col0 = vmulq_laneq_f32(invBt0, M_row0, 0);
        Kv_col0 = vfmaq_laneq_f32(Kv_col0, invBt1, M_row1, 0);
        Kv_col0 = vfmaq_laneq_f32(Kv_col0, invBt2, M_row2, 0);

        float32x4_t Kv_col1 = vmulq_laneq_f32(invBt0, M_row0, 1);
        Kv_col1 = vfmaq_laneq_f32(Kv_col1, invBt1, M_row1, 1);
        Kv_col1 = vfmaq_laneq_f32(Kv_col1, invBt2, M_row2, 1);

        float32x4_t Kv_col2 = vmulq_laneq_f32(invBt0, M_row0, 2);
        Kv_col2 = vfmaq_laneq_f32(Kv_col2, invBt1, M_row1, 2);
        Kv_col2 = vfmaq_laneq_f32(Kv_col2, invBt2, M_row2, 2);

        // 11. Affine: z = invBt · pv (pv is in lane 3 of ppv0,ppv1,ppv2)
        float32x4_t z = vmulq_n_f32(invBt0, vgetq_lane_f32(ppv0, 3));
        z = vfmaq_n_f32(z, invBt1, vgetq_lane_f32(ppv1, 3));
        z = vfmaq_n_f32(z, invBt2, vgetq_lane_f32(ppv2, 3));

        // v = z - Sinv · R_k · ur_k
        float32x4_t ur_v = vld1q_f32(ur_k);
        float32x4_t Rur = vmulq_f32(R_k, ur_v);
        float32x4_t Cinv_Rur = vmulq_laneq_f32(Cinv.val[0], Rur, 0);
        Cinv_Rur = vfmaq_laneq_f32(Cinv_Rur, Cinv.val[1], Rur, 1);
        Cinv_Rur = vfmaq_laneq_f32(Cinv_Rur, Cinv.val[2], Rur, 2);
        Cinv_Rur = vfmaq_laneq_f32(Cinv_Rur, Cinv.val[3], Rur, 3);
        float32x4_t v_aff = vsubq_f32(z, Cinv_Rur);

        // Store Kp, Kv, v
        vst1q_f32(sd + 0,  Kp_col0);
        vst1q_f32(sd + 4,  Kp_col1);
        vst1q_f32(sd + 8,  Kp_col2);
        vst1q_f32(sd + 12, Kv_col0);
        vst1q_f32(sd + 16, Kv_col1);
        vst1q_f32(sd + 20, Kv_col2);
        vst1q_f32(sd + 24, v_aff);

        // 12. Schur corrections via dot products
        // Schur_pp[i,j] = dot(PpvBl_row_i, Kp_col_j)
        // Schur_pv[i,j] = dot(PpvBl_row_i, Kv_col_j)
        // Schur_vv[i,j] = dot(DTBl_row_i, Kv_col_j)
        // d_p[i] = dot(PpvBl_row_i, z), d_v[i] = dot(DTBl_row_i, z)

        // Use horizontal adds to compute all dots
        // Schur_pp: 3×3 sym (6 unique)
        float sc_pp00 = vaddvq_f32(vmulq_f32(ppvbl0, Kp_col0));
        float sc_pp01 = vaddvq_f32(vmulq_f32(ppvbl0, Kp_col1));
        float sc_pp02 = vaddvq_f32(vmulq_f32(ppvbl0, Kp_col2));
        float sc_pp11 = vaddvq_f32(vmulq_f32(ppvbl1, Kp_col1));
        float sc_pp12 = vaddvq_f32(vmulq_f32(ppvbl1, Kp_col2));
        float sc_pp22 = vaddvq_f32(vmulq_f32(ppvbl2, Kp_col2));

        // Schur_pv: 3×3 full
        float sc_pv00 = vaddvq_f32(vmulq_f32(ppvbl0, Kv_col0));
        float sc_pv01 = vaddvq_f32(vmulq_f32(ppvbl0, Kv_col1));
        float sc_pv02 = vaddvq_f32(vmulq_f32(ppvbl0, Kv_col2));
        float sc_pv10 = vaddvq_f32(vmulq_f32(ppvbl1, Kv_col0));
        float sc_pv11 = vaddvq_f32(vmulq_f32(ppvbl1, Kv_col1));
        float sc_pv12 = vaddvq_f32(vmulq_f32(ppvbl1, Kv_col2));
        float sc_pv20 = vaddvq_f32(vmulq_f32(ppvbl2, Kv_col0));
        float sc_pv21 = vaddvq_f32(vmulq_f32(ppvbl2, Kv_col1));
        float sc_pv22 = vaddvq_f32(vmulq_f32(ppvbl2, Kv_col2));

        // Schur_vv: 3×3 sym (6 unique)
        float sc_vv00 = vaddvq_f32(vmulq_f32(dtbl0, Kv_col0));
        float sc_vv01 = vaddvq_f32(vmulq_f32(dtbl0, Kv_col1));
        float sc_vv02 = vaddvq_f32(vmulq_f32(dtbl0, Kv_col2));
        float sc_vv11 = vaddvq_f32(vmulq_f32(dtbl1, Kv_col1));
        float sc_vv12 = vaddvq_f32(vmulq_f32(dtbl1, Kv_col2));
        float sc_vv22 = vaddvq_f32(vmulq_f32(dtbl2, Kv_col2));

        // Affine dots: d_p[i] = dot(PpvBl_i, v_aff), d_v[i] = dot(DTBl_i, v_aff)
        // Use v_aff (full feedforward) not z (missing R·ur correction)
        float dp0 = vaddvq_f32(vmulq_f32(ppvbl0, v_aff));
        float dp1 = vaddvq_f32(vmulq_f32(ppvbl1, v_aff));
        float dp2 = vaddvq_f32(vmulq_f32(ppvbl2, v_aff));
        float dv0 = vaddvq_f32(vmulq_f32(dtbl0, v_aff));
        float dv1 = vaddvq_f32(vmulq_f32(dtbl1, v_aff));
        float dv2 = vaddvq_f32(vmulq_f32(dtbl2, v_aff));

        // 13. P block updates

        // Pp_new = Pp + Qp - Schur_pp  (A'PA_pp = Pp, free!)
        float32x4_t pp0_new = {
            vgetq_lane_f32(pp0, 0) + Q[0] - sc_pp00,
            vgetq_lane_f32(pp0, 1) - sc_pp01,
            vgetq_lane_f32(pp0, 2) - sc_pp02,
            vgetq_lane_f32(pp0, 3) - dp0 - Q[0] * xr_k[0] };  // affine pp0
        float32x4_t pp1_new = {
            vgetq_lane_f32(pp1, 0) - sc_pp01,
            vgetq_lane_f32(pp1, 1) + Q[1] - sc_pp11,
            vgetq_lane_f32(pp1, 2) - sc_pp12,
            vgetq_lane_f32(pp1, 3) - dp1 - Q[1] * xr_k[1] };
        float32x4_t pp2_new = {
            vgetq_lane_f32(pp2, 0) - sc_pp02,
            vgetq_lane_f32(pp2, 1) - sc_pp12,
            vgetq_lane_f32(pp2, 2) + Q[2] - sc_pp22,
            vgetq_lane_f32(pp2, 3) - dp2 - Q[2] * xr_k[2] };

        // Ppv_new = dt·Pp + Ppv·D - Schur_pv
        // Ppv_new[i][j] = dt*Pp[i][j] + Ppv[i][j]*D[j] - Schur_pv[i][j]
        // lane 3 = velocity affine costate: dt*pp[i] + D[i]*pv[i] - d_v[i] - Qv[i]*xr_v[i]
        float pp_aff0 = vgetq_lane_f32(pp0, 3);
        float pp_aff1 = vgetq_lane_f32(pp1, 3);
        float pp_aff2 = vgetq_lane_f32(pp2, 3);
        float pv_aff0 = vgetq_lane_f32(ppv0, 3);
        float pv_aff1 = vgetq_lane_f32(ppv1, 3);
        float pv_aff2 = vgetq_lane_f32(ppv2, 3);

        float32x4_t ppv0_new = {
            dt * vgetq_lane_f32(pp0, 0) + vgetq_lane_f32(ppv0, 0) * D[0] - sc_pv00,
            dt * vgetq_lane_f32(pp0, 1) + vgetq_lane_f32(ppv0, 1) * D[1] - sc_pv01,
            dt * vgetq_lane_f32(pp0, 2) + vgetq_lane_f32(ppv0, 2) * D[2] - sc_pv02,
            dt * pp_aff0 + D[0] * pv_aff0 - dv0 - Q[3] * xr_k[3] };
        float32x4_t ppv1_new = {
            dt * vgetq_lane_f32(pp1, 0) + vgetq_lane_f32(ppv1, 0) * D[0] - sc_pv10,
            dt * vgetq_lane_f32(pp1, 1) + vgetq_lane_f32(ppv1, 1) * D[1] - sc_pv11,
            dt * vgetq_lane_f32(pp1, 2) + vgetq_lane_f32(ppv1, 2) * D[2] - sc_pv12,
            dt * pp_aff1 + D[1] * pv_aff1 - dv1 - Q[4] * xr_k[4] };
        float32x4_t ppv2_new = {
            dt * vgetq_lane_f32(pp2, 0) + vgetq_lane_f32(ppv2, 0) * D[0] - sc_pv20,
            dt * vgetq_lane_f32(pp2, 1) + vgetq_lane_f32(ppv2, 1) * D[1] - sc_pv21,
            dt * vgetq_lane_f32(pp2, 2) + vgetq_lane_f32(ppv2, 2) * D[2] - sc_pv22,
            dt * pp_aff2 + D[2] * pv_aff2 - dv2 - Q[5] * xr_k[5] };

        // Pv_new = dt²·Pp + dt·(Ppv·D + D·Ppv') + D·Pv·D + Qv - Schur_vv
        // Element [i,j] for sym:
        //   dt²*Pp[i,j] + dt*(Ppv[i,j]*D[j] + D[i]*Ppv[j,i]) + D[i]*Pv[i,j]*D[j] + Qv_ij - Sch_vv[i,j]
        float dt2 = dt * dt;

        // Helper macro for Ppv access: ppv_ij = lane j of ppv_i register
        #define PPV(i,j) vgetq_lane_f32(ppv##i, j)
        #define PP(i,j)  vgetq_lane_f32(pp##i, j)
        #define PVV(i,j) vgetq_lane_f32(pvv##i, j)

        float pvv_00 = dt2*PP(0,0) + dt*(PPV(0,0)*D[0] + D[0]*PPV(0,0)) + D[0]*PVV(0,0)*D[0] + Q[3] - sc_vv00;
        float pvv_01 = dt2*PP(0,1) + dt*(PPV(0,1)*D[1] + D[0]*PPV(1,0)) + D[0]*PVV(0,1)*D[1]        - sc_vv01;
        float pvv_02 = dt2*PP(0,2) + dt*(PPV(0,2)*D[2] + D[0]*PPV(2,0)) + D[0]*PVV(0,2)*D[2]        - sc_vv02;
        float pvv_11 = dt2*PP(1,1) + dt*(PPV(1,1)*D[1] + D[1]*PPV(1,1)) + D[1]*PVV(1,1)*D[1] + Q[4] - sc_vv11;
        float pvv_12 = dt2*PP(1,2) + dt*(PPV(1,2)*D[2] + D[1]*PPV(2,1)) + D[1]*PVV(1,2)*D[2]        - sc_vv12;
        float pvv_22 = dt2*PP(2,2) + dt*(PPV(2,2)*D[2] + D[2]*PPV(2,2)) + D[2]*PVV(2,2)*D[2] + Q[5] - sc_vv22;

        #undef PPV
        #undef PP
        #undef PVV

        float32x4_t pvv0_new = { pvv_00, pvv_01, pvv_02, 0.0f };
        float32x4_t pvv1_new = { pvv_01, pvv_11, pvv_12, 0.0f };
        float32x4_t pvv2_new = { pvv_02, pvv_12, pvv_22, 0.0f };

        pp0 = pp0_new; pp1 = pp1_new; pp2 = pp2_new;
        ppv0 = ppv0_new; ppv1 = ppv1_new; ppv2 = ppv2_new;
        pvv0 = pvv0_new; pvv1 = pvv1_new; pvv2 = pvv2_new;
    }

    // ---- Forward pass ----
    float xp0 = x0[0], xp1 = x0[1], xp2 = x0[2];
    float xv0 = x0[3], xv1 = x0[4], xv2 = x0[5];

    for (int k = 0; k < N; ++k) {
        const float* sd = workspace + k * 40;
        float32x4_t Kp_c0 = vld1q_f32(sd + 0);
        float32x4_t Kp_c1 = vld1q_f32(sd + 4);
        float32x4_t Kp_c2 = vld1q_f32(sd + 8);
        float32x4_t Kv_c0 = vld1q_f32(sd + 12);
        float32x4_t Kv_c1 = vld1q_f32(sd + 16);
        float32x4_t Kv_c2 = vld1q_f32(sd + 20);
        float32x4_t v_k   = vld1q_f32(sd + 24);

        // u = -(Kp·xp + Kv·xv) - v
        float32x4_t u_k = vnegq_f32(vfmaq_n_f32(
            vfmaq_n_f32(vmulq_n_f32(Kp_c0, xp0), Kp_c1, xp1),
            Kp_c2, xp2));
        u_k = vfmsq_n_f32(u_k, Kv_c0, xv0);
        u_k = vfmsq_n_f32(u_k, Kv_c1, xv1);
        u_k = vfmsq_n_f32(u_k, Kv_c2, xv2);
        u_k = vsubq_f32(u_k, v_k);

        vst1q_f32(u_star + k * 4, u_k);

        // Propagate
        float32x4_t b0_k = vld1q_f32(sd + 28);
        float32x4_t b1_k = vld1q_f32(sd + 32);
        float32x4_t b2_k = vld1q_f32(sd + 36);

        float32x4_t bu0 = vmulq_f32(b0_k, u_k);
        float32x4_t bu1 = vmulq_f32(b1_k, u_k);
        float32x4_t bu2 = vmulq_f32(b2_k, u_k);
        float32x4_t p01 = vpaddq_f32(bu0, bu1);
        float32x4_t p2x = vpaddq_f32(bu2, bu2);
        float32x4_t Bu = vpaddq_f32(p01, p2x); // {Bu0, Bu1, Bu2, Bu2}

        float xp0_n = xp0 + dt * xv0;
        float xp1_n = xp1 + dt * xv1;
        float xp2_n = xp2 + dt * xv2;
        float xv0_n = D[0] * xv0 + vgetq_lane_f32(Bu, 0);
        float xv1_n = D[1] * xv1 + vgetq_lane_f32(Bu, 1);
        float xv2_n = D[2] * xv2 + vgetq_lane_f32(Bu, 2);

        xp0 = xp0_n; xp1 = xp1_n; xp2 = xp2_n;
        xv0 = xv0_n; xv1 = xv1_n; xv2 = xv2_n;
    }
}

#endif // MPC_USE_NEON

// =========================================================================
// Section 4: IPM outer loop
// =========================================================================

QPSolution ipm_solve(const EulerDynamicsData& euler,
                     const HeadingLookupData& hld,
                     const RefNode* ref_window,
                     const double x0[NX],
                     const MPCConfig& config,
                     const HeadingScheduleConfig& sched_config,
                     const IpmSolverConfig& ipm_config,
                     IpmWorkspace& ws)
{
    const int N = config.N;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // 1. Generate heading schedule (reuse existing function)
    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref_window, N, config.dt, sched_config, theta_sched);

    // 2. Build consistent 6-state reference using Euler dynamics
    //    x_ref[k+1] = A_euler · x_ref[k] + B_d(θ_k) · u_ref[k]
    double x_ref[(N_MAX + 1) * NX];
    double u_ref_stacked[N_MAX * NU];
    std::memcpy(x_ref, ref_window[0].x_ref, NX * sizeof(double));

    for (int k = 0; k < N; ++k) {
        std::memcpy(u_ref_stacked + k * NU, ref_window[k].u_ref, NU * sizeof(double));

        const double* xk = x_ref + k * NX;
        double* xn = x_ref + (k + 1) * NX;

        // Rotate B_body by heading
        double ct = std::cos(theta_sched[k]);
        double st = std::sin(theta_sched[k]);
        double Bl[12];
        for (int j = 0; j < 4; ++j) {
            Bl[0 * 4 + j] =  ct * euler.B_body[0 * 4 + j] - st * euler.B_body[1 * 4 + j];
            Bl[1 * 4 + j] =  st * euler.B_body[0 * 4 + j] + ct * euler.B_body[1 * 4 + j];
            Bl[2 * 4 + j] = euler.B_body[2 * 4 + j];
        }

        // Bu = Bl · u_ref[k]
        double Bu[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                Bu[i] += Bl[i * 4 + j] * ref_window[k].u_ref[j];

        // Euler propagation: p_{k+1} = p_k + dt·v_k, v_{k+1} = D·v_k + Bl·u_k
        for (int i = 0; i < 3; ++i) {
            xn[i]     = xk[i] + config.dt * xk[3 + i];  // position
            xn[3 + i] = euler.D_diag[i] * xk[3 + i] + Bu[i];  // velocity
        }
    }

    // 3. Extract Q/Qf diagonals
    double Q_diag[6], Qf_diag[6], R_diag[4];
    for (int i = 0; i < NX; ++i) Q_diag[i] = config.Q[i + NX * i];
    for (int i = 0; i < NX; ++i) Qf_diag[i] = config.Qf[i + NX * i];
    for (int i = 0; i < NU; ++i) R_diag[i] = config.R[i + NU * i];

    double D_diag_d[3] = { euler.D_diag[0], euler.D_diag[1], euler.D_diag[2] };
    double B_body_d[12];
    for (int i = 0; i < 12; ++i) B_body_d[i] = euler.B_body[i];

    // 4. Initialize u_bar
    float* u_bar = ws.u_bar;
    if (ws.warm_valid && ws.prev_N == N) {
        // Shift previous solution by 1 step
        std::memcpy(u_bar, ws.u_prev + NU, (N - 1) * NU * sizeof(float));
        std::memcpy(u_bar + (N - 1) * NU, ws.u_prev + (N - 1) * NU, NU * sizeof(float));
    } else {
        for (int i = 0; i < N * NU; ++i)
            u_bar[i] = (float)std::clamp(u_ref_stacked[i], (double)(config.u_min), (double)(config.u_max));
    }

    // Clip to strict interior
    float margin = ipm_config.interior_margin;
    float lo = -1.0f + margin, hi = 1.0f - margin;
    for (int i = 0; i < N * NU; ++i)
        u_bar[i] = std::clamp(u_bar[i], lo, hi);

    // 5. IPM barrier iterations
    int total_iters = 0;
    float mu = ipm_config.mu_init;

    for (int outer = 0; outer < ipm_config.max_outer_iters && mu >= ipm_config.mu_min; ++outer) {
        // Compute barrier terms and form R_eff, ur_eff
        for (int i = 0; i < N * NU; ++i) {
            float u = u_bar[i];
            float slack_lo = u + 1.0f;
            float slack_hi = 1.0f - u;
            float W = mu / (slack_lo * slack_lo) + mu / (slack_hi * slack_hi);
            float g = -mu / slack_lo + mu / slack_hi;
            int j = i % NU;
            float R_j = (float)R_diag[j];
            ws.R_eff[i] = R_j + W;
            ws.ur_eff[i] = (float)u_ref_stacked[i] - g / (R_j + W);
        }

        // Convert to float arrays for Riccati
        float theta_f[N_MAX];
        for (int i = 0; i < N; ++i) theta_f[i] = (float)theta_sched[i];

        float xr_f[(N_MAX + 1) * 6];
        for (int i = 0; i < (N + 1) * NX; ++i) xr_f[i] = (float)x_ref[i];

        float x0_f[6];
        for (int i = 0; i < NX; ++i) x0_f[i] = (float)x0[i];

        // Run Riccati
        float u_new[N_MAX * NU];

#ifdef MPC_USE_NEON
        std::array<float, 6> Q_f, Qf_f;
        std::array<float, 3> D_f;
        std::array<float, 12> B_f;
        for (int i = 0; i < 6; ++i) { Q_f[i] = (float)Q_diag[i]; Qf_f[i] = (float)Qf_diag[i]; }
        for (int i = 0; i < 3; ++i) D_f[i] = euler.D_diag[i];
        for (int i = 0; i < 12; ++i) B_f[i] = euler.B_body[i];

        riccati_6state_neon(ws.stage_data, Q_f, Qf_f, ws.R_eff, D_f,
                            euler.dt, B_f, N, theta_f, xr_f, ws.ur_eff,
                            x0_f, u_new);
#else
        // Convert R_eff and ur_eff to double for scalar path
        double R_eff_d[N_MAX * NU], ur_eff_d[N_MAX * NU];
        for (int i = 0; i < N * NU; ++i) {
            R_eff_d[i] = ws.R_eff[i];
            ur_eff_d[i] = ws.ur_eff[i];
        }

        double u_new_d[N_MAX * NU];
        riccati_6state_scalar(Q_diag, Qf_diag, R_eff_d, D_diag_d,
                              config.dt, B_body_d, N,
                              theta_sched, x_ref, ur_eff_d, x0, u_new_d);
        for (int i = 0; i < N * NU; ++i) u_new[i] = (float)u_new_d[i];
#endif

        // Clip to strict interior and update u_bar
        for (int i = 0; i < N * NU; ++i)
            u_bar[i] = std::clamp(u_new[i], lo, hi);

        total_iters++;
        mu *= ipm_config.mu_factor;
    }

    // 6. Final clip to exact bounds
    for (int i = 0; i < N * NU; ++i)
        u_bar[i] = std::clamp(u_bar[i], (float)config.u_min, (float)config.u_max);

    // 7. Store for warm-start
    std::memcpy(ws.u_prev, u_bar, N * NU * sizeof(float));
    ws.warm_valid = true;
    ws.prev_N = N;

    // 8. Build QPSolution (convert float32 → double)
    QPSolution sol;
    std::memset(&sol, 0, sizeof(sol));
    for (int i = 0; i < N * NU; ++i)
        sol.U[i] = (double)u_bar[i];
    std::memcpy(sol.u0, sol.U, NU * sizeof(double));
    sol.n_iterations = total_iters;

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    sol.solve_time_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);
    return sol;
}
