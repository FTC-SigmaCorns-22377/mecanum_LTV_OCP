#include "heading_lookup.h"
#include "discretizer.h"
#include "mecanum_model.h"
#include "blas_dispatch.h"

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Helper: create a RefNode pair for exact_discretize at constant heading
// ---------------------------------------------------------------------------
static void make_const_heading_refs(double theta, double dt,
                                     RefNode& ref_k, RefNode& ref_k1)
{
    std::memset(&ref_k, 0, sizeof(RefNode));
    std::memset(&ref_k1, 0, sizeof(RefNode));

    ref_k.t = 0.0;
    ref_k.theta = theta;
    ref_k.omega = 0.0;
    ref_k.x_ref[2] = theta;

    ref_k1.t = dt;
    ref_k1.theta = theta;
    ref_k1.omega = 0.0;
    ref_k1.x_ref[2] = theta;
}

// ---------------------------------------------------------------------------
// Trig decomposition precomputation
// ---------------------------------------------------------------------------
double heading_lookup_precompute(const ModelParams& params, double dt,
                                 HeadingLookupData& data)
{
    data.dt = dt;

    // Discretize at θ=0, θ=π/2, θ=π
    double A_0[NX * NX], B_0[NX * NU];
    double A_half[NX * NX], B_half[NX * NU];
    double A_pi[NX * NX], B_pi[NX * NU];

    RefNode ref_k, ref_k1;

    make_const_heading_refs(0.0, dt, ref_k, ref_k1);
    exact_discretize(ref_k, ref_k1, params, A_0, B_0, 100);

    make_const_heading_refs(M_PI / 2.0, dt, ref_k, ref_k1);
    exact_discretize(ref_k, ref_k1, params, A_half, B_half, 100);

    make_const_heading_refs(M_PI, dt, ref_k, ref_k1);
    exact_discretize(ref_k, ref_k1, params, A_pi, B_pi, 100);

    // A_d is the same for all headings (heading-independent)
    std::memcpy(data.A_d, A_0, NX * NX * sizeof(double));

    // Extract trig components:
    // B_d(0) = B_d0 + B_dc      (cos(0)=1, sin(0)=0)
    // B_d(π) = B_d0 - B_dc      (cos(π)=-1, sin(π)=0)
    // B_d(π/2) = B_d0 + B_ds    (cos(π/2)=0, sin(π/2)=1)
    //
    // B_d0 = (B_d(0) + B_d(π)) / 2
    // B_dc = (B_d(0) - B_d(π)) / 2
    // B_ds = B_d(π/2) - B_d0
    for (int i = 0; i < NX * NU; ++i) {
        data.B_d0[i] = 0.5 * (B_0[i] + B_pi[i]);
        data.B_dc[i] = 0.5 * (B_0[i] - B_pi[i]);
        data.B_ds[i] = B_half[i] - data.B_d0[i];
    }

    // Compute A_d^k for k=0..N_MAX
    // A_d^0 = I
    std::memset(data.A_d_pow, 0, (N_MAX + 1) * NX * NX * sizeof(double));
    double* pow0 = data.A_d_pow;
    for (int i = 0; i < NX; ++i)
        pow0[i + NX * i] = 1.0;

    for (int k = 1; k <= N_MAX; ++k) {
        double* prev = data.A_d_pow + (k - 1) * NX * NX;
        double* cur  = data.A_d_pow + k * NX * NX;
        mpc_linalg::gemm(NX, NX, NX, data.A_d, prev, cur);
    }

    // Verify at θ=0.7
    double theta_test = 0.7;
    make_const_heading_refs(theta_test, dt, ref_k, ref_k1);
    double A_test[NX * NX], B_test[NX * NU];
    exact_discretize(ref_k, ref_k1, params, A_test, B_test, 100);

    double ct = std::cos(theta_test);
    double st = std::sin(theta_test);
    double max_err = 0.0;
    for (int i = 0; i < NX * NU; ++i) {
        double B_recon = data.B_d0[i] + ct * data.B_dc[i] + st * data.B_ds[i];
        double err = std::fabs(B_recon - B_test[i]);
        if (err > max_err) max_err = err;
    }

    return max_err;
}

// ---------------------------------------------------------------------------
// Kernel precomputation: P[j] cost-to-go and G[j] gradient kernels
// ---------------------------------------------------------------------------
void heading_kernel_precompute(const HeadingLookupData& data,
                                const MPCConfig& config,
                                HeadingKernelData& kern)
{
    const int N = config.N;
    kern.N = N;

    const int nx2 = NX * NX;

    // Backward Riccati for P[j]:
    //   P[N-1] = Qf
    //   P[j]   = Q + A_d^T · P[j+1] · A_d   for j = N-2 down to 0

    // P[N-1] = Qf
    std::memcpy(kern.P + (N - 1) * nx2, config.Qf, nx2 * sizeof(double));

    double temp[NX * NX] = {};
    for (int j = N - 2; j >= 0; --j) {
        const double* Pjp1 = kern.P + (j + 1) * nx2;
        double* Pj = kern.P + j * nx2;

        // temp = P[j+1] · A_d
        mpc_linalg::gemm(NX, NX, NX, Pjp1, data.A_d, temp);

        // Pj = A_d^T · temp  (+ Q added below)
        mpc_linalg::gemm_atb(NX, NX, NX, data.A_d, NX, temp, NX, Pj, NX);

        // Pj += Q (diagonal add for diagonal Q, but handle general case)
        for (int i = 0; i < nx2; ++i)
            Pj[i] += config.Q[i];
    }

    // Forward pass for G[j] = P[j] · A_d^(j+1)
    for (int j = 0; j < N; ++j) {
        const double* Pj = kern.P + j * nx2;
        const double* Ad_pow_jp1 = data.A_d_pow + (j + 1) * nx2;
        double* Gj = kern.G + j * nx2;

        mpc_linalg::gemm(NX, NX, NX, Pj, Ad_pow_jp1, Gj);
    }
}

// ---------------------------------------------------------------------------
// Heading table precomputation
// ---------------------------------------------------------------------------
void heading_table_precompute(const ModelParams& params, double dt, int M,
                              HeadingTableData& table)
{
    table.dt = dt;
    table.M = M;

    RefNode ref_k, ref_k1;
    double A_tmp[NX * NX];

    for (int i = 0; i < M; ++i) {
        double theta_i = 2.0 * M_PI * i / M;
        make_const_heading_refs(theta_i, dt, ref_k, ref_k1);
        exact_discretize(ref_k, ref_k1, params, A_tmp,
                         table.B_d_table + i * NX * NU, 100);

        // Store A_d from first evaluation
        if (i == 0) {
            std::memcpy(table.A_d, A_tmp, NX * NX * sizeof(double));
        }
    }

    // Compute A_d^k for k=0..N_MAX
    std::memset(table.A_d_pow, 0, (N_MAX + 1) * NX * NX * sizeof(double));
    double* pow0 = table.A_d_pow;
    for (int i = 0; i < NX; ++i)
        pow0[i + NX * i] = 1.0;

    for (int k = 1; k <= N_MAX; ++k) {
        double* prev = table.A_d_pow + (k - 1) * NX * NX;
        double* cur  = table.A_d_pow + k * NX * NX;
        mpc_linalg::gemm(NX, NX, NX, table.A_d, prev, cur);
    }
}
