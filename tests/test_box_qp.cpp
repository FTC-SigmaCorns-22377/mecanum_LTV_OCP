#include <cstdio>
#include <cmath>
#include <cstring>

#include "mpc_types.h"
#include "box_qp_solver.h"
#include "cholesky.h"
#include "blas_dispatch.h"

// ---------------------------------------------------------------------------
// Simple deterministic LCG pseudo-random number generator
// ---------------------------------------------------------------------------
static uint64_t lcg_state = 0;

static void lcg_seed(uint64_t seed) { lcg_state = seed; }

// Returns a double in (-1, 1)
static double lcg_rand() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    int32_t s = static_cast<int32_t>(lcg_state >> 33);
    return static_cast<double>(s) / static_cast<double>(1LL << 31);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int g_tests_run    = 0;
static int g_tests_passed = 0;

static void check(bool cond, const char* name) {
    g_tests_run++;
    if (cond) {
        g_tests_passed++;
        std::printf("  [PASS] %s\n", name);
    } else {
        std::printf("  [FAIL] %s\n", name);
    }
}

// Max absolute element of a vector
static double max_abs(int len, const double* v) {
    double m = 0.0;
    for (int i = 0; i < len; i++) {
        double a = std::fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

// Build an n x n SPD matrix: A = B' * B + eps * I
// B is n x n with pseudo-random entries.  A and B must be pre-allocated.
static void make_spd(int n, double* A, double* B, double eps) {
    for (int i = 0; i < n * n; i++) B[i] = lcg_rand();
    mpc_linalg::gemm_atb(n, n, n, B, n, B, n, A, n);
    for (int i = 0; i < n; i++) A[i + i * n] += eps;
}

// ---------------------------------------------------------------------------
// Test 1: Unconstrained case (8x8)
// ---------------------------------------------------------------------------
static void test_unconstrained() {
    std::printf("--- Test 1: Unconstrained case (8x8) ---\n");

    const int n = 8;
    double H[n * n], B[n * n], L[n * n];
    double g[n];

    // Build SPD Hessian
    lcg_seed(100);
    make_spd(n, H, B, 1.0);

    // Cholesky factor
    int ret = cholesky_factor(n, H, L);
    check(ret == 0, "Cholesky factor succeeds");

    // Set gradient vector
    for (int i = 0; i < n; i++) g[i] = lcg_rand() * 5.0;

    // Wide bounds -> effectively unconstrained
    double u_min = -1000.0;
    double u_max =  1000.0;

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = box_qp_solve(H, L, g, u_min, u_max, n, 50, ws);

    check(iters == 0, "Unconstrained returns 0 iterations");

    // Verify: H*U + g should be approximately zero
    double HU[n];
    mpc_linalg::gemv(n, n, H, ws.U, HU);
    // residual = H*U + g
    double residual[n];
    for (int i = 0; i < n; i++) residual[i] = HU[i] + g[i];

    double err = max_abs(n, residual);
    char label[128];
    std::snprintf(label, sizeof(label),
                  "H*U + g ~ 0, max_err=%.3e", err);
    check(err < 1e-10, label);
}

// ---------------------------------------------------------------------------
// Test 2: Simple 2D bound-constrained case
// ---------------------------------------------------------------------------
static void test_2d_clamped() {
    std::printf("\n--- Test 2: Simple 2D bound-constrained ---\n");

    const int n = 2;
    // H = [[2, 0], [0, 2]], column-major
    double H[4] = {2.0, 0.0, 0.0, 2.0};
    double L[4];
    double g[2] = {-6.0, -6.0};
    // Unconstrained optimum: U = H^{-1}*(-g) = [3, 3]

    int ret = cholesky_factor(n, H, L);
    check(ret == 0, "Cholesky factor succeeds (2x2)");

    double u_min = -1.0;
    double u_max =  2.0;

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = box_qp_solve(H, L, g, u_min, u_max, n, 50, ws);

    char label[128];
    std::snprintf(label, sizeof(label),
                  "Converged in %d iterations", iters);
    check(iters >= 0, label);

    // Solution should be [2, 2] (both clamped at upper bound)
    double err0 = std::fabs(ws.U[0] - 2.0);
    double err1 = std::fabs(ws.U[1] - 2.0);

    std::snprintf(label, sizeof(label),
                  "U[0] = 2.0, err=%.3e", err0);
    check(err0 < 1e-10, label);

    std::snprintf(label, sizeof(label),
                  "U[1] = 2.0, err=%.3e", err1);
    check(err1 < 1e-10, label);
}

// ---------------------------------------------------------------------------
// Test 3: Mixed free and clamped
// ---------------------------------------------------------------------------
static void test_mixed_free_clamped() {
    std::printf("\n--- Test 3: Mixed free and clamped ---\n");

    const int n = 2;
    // H = [[4, 1], [1, 4]], column-major
    double H[4] = {4.0, 1.0, 1.0, 4.0};
    double L[4];
    double g[2] = {-10.0, -2.0};

    int ret = cholesky_factor(n, H, L);
    check(ret == 0, "Cholesky factor succeeds (mixed test)");

    // Part A: wide bounds, unconstrained optimum should be within bounds
    {
        double u_min = -5.0;
        double u_max =  5.0;

        BoxQPWorkspace ws;
        std::memset(&ws, 0, sizeof(ws));

        int iters = box_qp_solve(H, L, g, u_min, u_max, n, 50, ws);
        check(iters == 0, "Wide bounds: 0 iterations (unconstrained feasible)");

        // Unconstrained optimum: solve [[4,1],[1,4]]*x = [10,2]
        // det = 16-1 = 15, x = [1/15*(4*10-1*2), 1/15*(4*2-1*10)] = [38/15, -2/15]
        // Wait: x = H^{-1}*(-g) = H^{-1}*[10,2]
        // H^{-1} = 1/15 * [[4,-1],[-1,4]]
        // x = [1/15*(40-2), 1/15*(-10+8)] = [38/15, -2/15]
        double expected0 = 38.0 / 15.0;
        double expected1 = -2.0 / 15.0;

        char label[128];
        double err0 = std::fabs(ws.U[0] - expected0);
        double err1 = std::fabs(ws.U[1] - expected1);
        std::snprintf(label, sizeof(label),
                      "U = [%.6f, %.6f], expected [%.6f, %.6f], err=[%.3e, %.3e]",
                      ws.U[0], ws.U[1], expected0, expected1, err0, err1);
        check(err0 < 1e-10 && err1 < 1e-10, label);
    }

    // Part B: tight upper bound u_max = 2, so U[0] is clamped
    {
        double u_min = -5.0;
        double u_max =  2.0;

        BoxQPWorkspace ws;
        std::memset(&ws, 0, sizeof(ws));

        int iters = box_qp_solve(H, L, g, u_min, u_max, n, 50, ws);

        char label[128];
        std::snprintf(label, sizeof(label),
                      "Tight u_max: converged in %d iterations", iters);
        check(iters >= 0, label);

        // Verify KKT conditions at solution
        // grad = H*U + g
        double grad[2];
        mpc_linalg::gemv(n, n, H, ws.U, grad);
        grad[0] += g[0];
        grad[1] += g[1];

        // For each variable:
        //   - If u_min < U[i] < u_max (free), grad[i] should be ~0
        //   - If U[i] == u_min (clamped low), grad[i] >= 0
        //   - If U[i] == u_max (clamped high), grad[i] <= 0
        bool kkt_ok = true;
        for (int i = 0; i < n; i++) {
            bool at_lower = std::fabs(ws.U[i] - u_min) < 1e-10;
            bool at_upper = std::fabs(ws.U[i] - u_max) < 1e-10;
            if (!at_lower && !at_upper) {
                // Free variable: gradient should be ~0
                if (std::fabs(grad[i]) > 1e-8) kkt_ok = false;
            } else if (at_lower) {
                if (grad[i] < -1e-8) kkt_ok = false;
            } else if (at_upper) {
                if (grad[i] > 1e-8) kkt_ok = false;
            }
        }

        std::snprintf(label, sizeof(label),
                      "KKT conditions satisfied (grad=[%.6e, %.6e], U=[%.6f, %.6f])",
                      grad[0], grad[1], ws.U[0], ws.U[1]);
        check(kkt_ok, label);

        // Verify all within bounds
        bool in_bounds = true;
        for (int i = 0; i < n; i++) {
            if (ws.U[i] < u_min - 1e-12 || ws.U[i] > u_max + 1e-12)
                in_bounds = false;
        }
        check(in_bounds, "Solution within bounds");
    }
}

// ---------------------------------------------------------------------------
// Test 4: Larger problem (N=5 horizon, NU=4 -> 20 variables)
// ---------------------------------------------------------------------------
static void test_large_problem() {
    std::printf("\n--- Test 4: Larger problem (20x20) ---\n");

    const int n = 20;  // N=5, NU=4
    double H[n * n], B[n * n], L[n * n];
    double g[n];

    // Build SPD Hessian with deterministic random values
    lcg_seed(7777);
    make_spd(n, H, B, 2.0);  // eps=2.0 for strong convexity

    // Set gradient
    for (int i = 0; i < n; i++) g[i] = lcg_rand() * 10.0;

    int ret = cholesky_factor(n, H, L);
    check(ret == 0, "Cholesky factor succeeds (20x20)");

    // Tight bounds
    double u_min = -2.0;
    double u_max =  2.0;

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = box_qp_solve(H, L, g, u_min, u_max, n, 50, ws);

    char label[256];

    // (a) All elements within bounds
    bool in_bounds = true;
    for (int i = 0; i < n; i++) {
        if (ws.U[i] < u_min - 1e-12 || ws.U[i] > u_max + 1e-12) {
            in_bounds = false;
            std::printf("    U[%d] = %.10f out of bounds [%.1f, %.1f]\n",
                        i, ws.U[i], u_min, u_max);
        }
    }
    check(in_bounds, "All elements within bounds");

    // (b) KKT conditions
    double grad[n];
    mpc_linalg::gemv(n, n, H, ws.U, grad);
    for (int i = 0; i < n; i++) grad[i] += g[i];

    bool kkt_ok = true;
    int n_free = 0, n_clamped = 0;
    double max_free_grad = 0.0;
    for (int i = 0; i < n; i++) {
        bool at_lower = std::fabs(ws.U[i] - u_min) < 1e-10;
        bool at_upper = std::fabs(ws.U[i] - u_max) < 1e-10;
        if (!at_lower && !at_upper) {
            // Free variable: gradient should be ~0
            n_free++;
            double ag = std::fabs(grad[i]);
            if (ag > max_free_grad) max_free_grad = ag;
            if (ag > 1e-8) {
                kkt_ok = false;
                std::printf("    KKT violation: free var %d, grad=%.3e\n",
                            i, grad[i]);
            }
        } else {
            n_clamped++;
            if (at_lower && grad[i] < -1e-8) {
                kkt_ok = false;
                std::printf("    KKT violation: clamped-low var %d, grad=%.3e (should be >= 0)\n",
                            i, grad[i]);
            }
            if (at_upper && grad[i] > 1e-8) {
                kkt_ok = false;
                std::printf("    KKT violation: clamped-high var %d, grad=%.3e (should be <= 0)\n",
                            i, grad[i]);
            }
        }
    }

    std::snprintf(label, sizeof(label),
                  "KKT conditions (n_free=%d, n_clamped=%d, max_free_grad=%.3e)",
                  n_free, n_clamped, max_free_grad);
    check(kkt_ok, label);

    // (c) Convergence in reasonable iterations
    std::snprintf(label, sizeof(label),
                  "Converged in %d iterations (<= 10)", iters);
    check(iters <= 10, label);
}

// ---------------------------------------------------------------------------
// Helper: compute largest eigenvalue via power iteration (for FISTA step size)
// ---------------------------------------------------------------------------
static double test_power_iteration(const double* H, int n) {
    double v[N_MAX * NU], Hv[N_MAX * NU];
    double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
    for (int i = 0; i < n; ++i) v[i] = inv_sqrt_n;

    double lambda = 0.0;
    for (int iter = 0; iter < 30; ++iter) {
        mpc_linalg::gemv(n, n, H, v, Hv);
        lambda = mpc_linalg::dot(n, v, Hv);
        double norm = std::sqrt(mpc_linalg::dot(n, Hv, Hv));
        if (norm < 1e-15) break;
        double inv_norm = 1.0 / norm;
        mpc_linalg::scal(n, inv_norm, Hv);
        std::memcpy(v, Hv, static_cast<std::size_t>(n) * sizeof(double));
    }
    return lambda;
}

// ---------------------------------------------------------------------------
// Helper: verify KKT conditions at a point (for FISTA tests)
// ---------------------------------------------------------------------------
static bool verify_kkt(const double* H, const double* g, const double* U,
                       double u_min, double u_max, int n, double tol) {
    double grad[N_MAX * NU];
    mpc_linalg::gemv(n, n, H, U, grad);
    for (int i = 0; i < n; ++i) grad[i] += g[i];

    for (int i = 0; i < n; ++i) {
        bool at_lower = std::fabs(U[i] - u_min) < 1e-10;
        bool at_upper = std::fabs(U[i] - u_max) < 1e-10;
        if (!at_lower && !at_upper) {
            if (std::fabs(grad[i]) > tol) return false;
        } else if (at_lower) {
            if (grad[i] < -tol) return false;
        } else if (at_upper) {
            if (grad[i] > tol) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// FISTA Test 1: Unconstrained case (8x8)
// ---------------------------------------------------------------------------
static void test_fista_unconstrained() {
    std::printf("\n--- FISTA Test 1: Unconstrained case (8x8) ---\n");

    const int n = 8;
    double H[n * n], B[n * n];
    double g[n];

    lcg_seed(100);
    make_spd(n, H, B, 1.0);
    for (int i = 0; i < n; i++) g[i] = lcg_rand() * 5.0;

    double u_min = -1000.0, u_max = 1000.0;
    double step_size = 1.0 / test_power_iteration(H, n);

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = fista_box_qp_solve(H, g, u_min, u_max, n, 200, step_size, ws);

    char label[128];
    std::snprintf(label, sizeof(label),
                  "FISTA unconstrained converged in %d iterations", iters);
    check(iters <= 200, label);

    double residual[n];
    mpc_linalg::gemv(n, n, H, ws.U, residual);
    for (int i = 0; i < n; i++) residual[i] += g[i];
    double err = max_abs(n, residual);
    std::snprintf(label, sizeof(label),
                  "H*U + g ~ 0, max_err=%.3e", err);
    check(err < 1e-6, label);
}

// ---------------------------------------------------------------------------
// FISTA Test 2: Simple 2D bound-constrained
// ---------------------------------------------------------------------------
static void test_fista_2d_clamped() {
    std::printf("\n--- FISTA Test 2: Simple 2D bound-constrained ---\n");

    const int n = 2;
    double H[4] = {2.0, 0.0, 0.0, 2.0};
    double g[2] = {-6.0, -6.0};
    double u_min = -1.0, u_max = 2.0;
    double step_size = 1.0 / test_power_iteration(H, n);

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = fista_box_qp_solve(H, g, u_min, u_max, n, 200, step_size, ws);

    char label[128];
    std::snprintf(label, sizeof(label),
                  "FISTA 2D converged in %d iterations", iters);
    check(iters <= 200, label);

    double err0 = std::fabs(ws.U[0] - 2.0);
    double err1 = std::fabs(ws.U[1] - 2.0);
    std::snprintf(label, sizeof(label),
                  "U = [%.6f, %.6f], expected [2, 2], err=[%.3e, %.3e]",
                  ws.U[0], ws.U[1], err0, err1);
    check(err0 < 1e-6 && err1 < 1e-6, label);

    check(ws.U[0] >= u_min && ws.U[0] <= u_max &&
          ws.U[1] >= u_min && ws.U[1] <= u_max, "Solution within bounds");
}

// ---------------------------------------------------------------------------
// FISTA Test 3: Mixed free and clamped
// ---------------------------------------------------------------------------
static void test_fista_mixed() {
    std::printf("\n--- FISTA Test 3: Mixed free and clamped ---\n");

    const int n = 2;
    double H[4] = {4.0, 1.0, 1.0, 4.0};
    double g[2] = {-10.0, -2.0};
    double u_min = -5.0, u_max = 2.0;
    double step_size = 1.0 / test_power_iteration(H, n);

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = fista_box_qp_solve(H, g, u_min, u_max, n, 200, step_size, ws);

    char label[128];
    std::snprintf(label, sizeof(label),
                  "FISTA mixed converged in %d iterations", iters);
    check(iters <= 200, label);

    bool kkt_ok = verify_kkt(H, g, ws.U, u_min, u_max, n, 1e-6);
    std::snprintf(label, sizeof(label),
                  "KKT conditions satisfied (U=[%.6f, %.6f])", ws.U[0], ws.U[1]);
    check(kkt_ok, label);

    bool in_bounds = ws.U[0] >= u_min - 1e-12 && ws.U[0] <= u_max + 1e-12 &&
                     ws.U[1] >= u_min - 1e-12 && ws.U[1] <= u_max + 1e-12;
    check(in_bounds, "Solution within bounds");
}

// ---------------------------------------------------------------------------
// FISTA Test 4: Larger problem (20x20)
// ---------------------------------------------------------------------------
static void test_fista_large() {
    std::printf("\n--- FISTA Test 4: Larger problem (20x20) ---\n");

    const int n = 20;
    double H[n * n], B[n * n];
    double g[n];

    lcg_seed(7777);
    make_spd(n, H, B, 2.0);
    for (int i = 0; i < n; i++) g[i] = lcg_rand() * 10.0;

    double u_min = -2.0, u_max = 2.0;
    double step_size = 1.0 / test_power_iteration(H, n);

    BoxQPWorkspace ws;
    std::memset(&ws, 0, sizeof(ws));

    int iters = fista_box_qp_solve(H, g, u_min, u_max, n, 500, step_size, ws);

    char label[256];
    std::snprintf(label, sizeof(label),
                  "FISTA large converged in %d iterations", iters);
    check(iters <= 500, label);

    bool in_bounds = true;
    for (int i = 0; i < n; i++) {
        if (ws.U[i] < u_min - 1e-12 || ws.U[i] > u_max + 1e-12) {
            in_bounds = false;
            std::printf("    U[%d] = %.10f out of bounds\n", i, ws.U[i]);
        }
    }
    check(in_bounds, "All elements within bounds");

    bool kkt_ok = verify_kkt(H, g, ws.U, u_min, u_max, n, 1e-4);
    std::snprintf(label, sizeof(label), "KKT conditions satisfied (tol=1e-4)");
    check(kkt_ok, label);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== Box-constrained QP solver tests ===\n\n");

    test_unconstrained();
    test_2d_clamped();
    test_mixed_free_clamped();
    test_large_problem();

    std::printf("\n=== FISTA solver tests ===\n");

    test_fista_unconstrained();
    test_fista_2d_clamped();
    test_fista_mixed();
    test_fista_large();

    std::printf("\n=== Summary: %d / %d tests passed ===\n",
                g_tests_passed, g_tests_run);

    if (g_tests_passed == g_tests_run) {
        std::printf("All box QP tests passed\n");
        return 0;
    } else {
        std::printf("SOME TESTS FAILED\n");
        return 1;
    }
}
