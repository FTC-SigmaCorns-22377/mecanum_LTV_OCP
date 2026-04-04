// test_waypoint_tracking.cpp — Unit tests for zero-final-velocity waypoint
// tracking via MecanumLTV::solve_waypoint()
//
// Tests:
//  1. Returns -1 when model params not set
//  2. Returns -1 when MPC config not set
//  3. Returns 0 on a valid solve call
//  4. Control outputs lie within [u_min, u_max]
//  5. Near-zero controls when starting exactly at target (LQR mode)
//  6. Near-zero controls when starting exactly at target (Hermite mode)
//  7. Closed-loop XY convergence — LQR mode, zero-velocity target
//  8. Closed-loop XY convergence — Hermite mode, zero-velocity target
//  9. Heading convergence — LQR mode with heading change required
// 10. Short t_remaining clamps N_eff = 1 and still returns bounded controls
// 11. Controls change direction appropriately as target switches sides

#include "mecanum_ltv.h"
#include "mecanum_model.h"
#include "mpc_types.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ModelParams make_params()
{
    ModelParams p{};
    p.mass            = 10.0;
    p.inertia         = 0.5;
    p.damping_linear  = 2.0;
    p.damping_angular = 0.3;
    p.wheel_radius    = 0.05;
    p.lx              = 0.15;
    p.ly              = 0.15;
    p.stall_torque    = 6.0;
    p.free_speed      = 435.0;
    compute_mecanum_jacobian(p);
    return p;
}

// N=15 gives room for horizon shortening tests while staying under N_MAX
static MPCConfig make_config(double dt = 0.02, int N = 15)
{
    MPCConfig cfg{};
    cfg.N     = N;
    cfg.dt    = dt;
    cfg.u_min = -1.0;
    cfg.u_max =  1.0;

    std::memset(cfg.Q,  0, sizeof(cfg.Q));
    std::memset(cfg.R,  0, sizeof(cfg.R));
    std::memset(cfg.Qf, 0, sizeof(cfg.Qf));

    cfg.Q[0 + NX * 0] = 20.0;   // px
    cfg.Q[1 + NX * 1] = 20.0;   // py
    cfg.Q[2 + NX * 2] =  5.0;   // theta
    cfg.Q[3 + NX * 3] =  2.0;   // vx
    cfg.Q[4 + NX * 4] =  2.0;   // vy
    cfg.Q[5 + NX * 5] =  0.5;   // omega

    for (int i = 0; i < NU; ++i)
        cfg.R[i + NU * i] = 0.1;

    // Terminal cost: stronger position and velocity penalty to stop cleanly
    for (int i = 0; i < NX * NX; ++i)
        cfg.Qf[i] = 5.0 * cfg.Q[i];

    return cfg;
}

// Forward-simulate one Euler step using the linearised continuous dynamics.
// Accurate enough for short dt used in tests.
static void sim_step(double x[NX], const double u[NU],
                     const ModelParams& params, double dt)
{
    double Ac[NX * NX], Bc[NX * NU];
    continuous_dynamics(x[2], params, Ac, Bc);

    double xdot[NX] = {};
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NX; ++j) xdot[i] += Ac[i + j * NX] * x[j];
        for (int j = 0; j < NU; ++j) xdot[i] += Bc[i + j * NX] * u[j];
    }
    for (int i = 0; i < NX; ++i)
        x[i] += dt * xdot[i];
}

static double pos_error(const double x[NX], const double target[NX])
{
    double dx = x[0] - target[0];
    double dy = x[1] - target[1];
    return std::sqrt(dx * dx + dy * dy);
}

static double vec_inf(const double* v, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::fabs(v[i]));
    return m;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

// 1. Returns -1 when model params not set
static bool test_unconfigured_no_params()
{
    std::printf("1. Returns -1 without setModelParams ... ");

    MecanumLTV ltv;
    // Only config, no params
    MPCConfig cfg = make_config();
    ltv.setConfig(cfg);

    double x0[NX]     = {};
    double x_tgt[NX]  = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double u_out[NU]  = {};

    int ret = ltv.solve_waypoint(x0, x_tgt, 0.5, 0.02, true, u_out);
    bool ok = (ret == -1);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// 2. Returns -1 when MPC config not set
static bool test_unconfigured_no_config()
{
    std::printf("2. Returns -1 without setConfig ... ");

    MecanumLTV ltv;
    ModelParams p = make_params();
    ltv.setModelParams(p);
    // no setConfig

    double x0[NX]    = {};
    double x_tgt[NX] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double u_out[NU] = {};

    int ret = ltv.solve_waypoint(x0, x_tgt, 0.5, 0.02, true, u_out);
    bool ok = (ret == -1);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// 3. Returns 0 on a valid solve call
static bool test_returns_zero_on_success()
{
    std::printf("3. Returns 0 when fully configured ... ");

    MecanumLTV ltv;
    ltv.setModelParams(make_params());
    ltv.setConfig(make_config());

    double x0[NX]    = {};
    double x_tgt[NX] = {1.0, 0.5, 0.0, 0.0, 0.0, 0.0};
    double u_out[NU] = {};

    int ret = ltv.solve_waypoint(x0, x_tgt, 0.4, 0.02, true, u_out);
    bool ok = (ret == 0);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// 4. Control outputs lie within [u_min, u_max]
static bool test_control_bounds()
{
    std::printf("4. Control outputs lie within [u_min, u_max] ... ");

    ModelParams params = make_params();
    MPCConfig   cfg    = make_config();
    MecanumLTV  ltv;
    ltv.setModelParams(params);
    ltv.setConfig(cfg);

    // Test across multiple starting states and horizons
    struct Case { double x0[NX]; double xt[NX]; double t_rem; };
    Case cases[] = {
        {{0,0,0,0,0,0},    {1.0, 0.0, 0,0,0,0},   0.4},
        {{0,0,0,0.5,0,0},  {2.0, 1.0, 0,0,0,0},   0.6},
        {{1,1,0.5,0,0,0},  {0.0, 0.0, 0,0,0,0},   0.3},
        {{0,0,0,0,0,0},    {0.0, 0.0, 0,0,0,0},   0.5},  // already at target
        {{0,0,0,0,0,0},    {0.5, 0.5, 0,0,0,0},   0.02}, // one-step horizon
    };

    bool ok = true;
    for (auto& c : cases) {
        double u_out[NU] = {};
        ltv.solve_waypoint(c.x0, c.xt, c.t_rem, 0.02, true, u_out);
        for (int i = 0; i < NU; ++i) {
            if (u_out[i] < cfg.u_min - 1e-6 || u_out[i] > cfg.u_max + 1e-6)
                ok = false;
        }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// 5. Near-zero controls when starting exactly at zero-velocity target (LQR mode)
static bool test_at_target_lqr_near_zero()
{
    std::printf("5. Near-zero controls at target (LQR mode) ... ");

    MecanumLTV ltv;
    ltv.setModelParams(make_params());
    ltv.setConfig(make_config());

    // x0 == x_target, both zero velocity
    double x[NX]     = {0.5, 0.3, 0.1, 0.0, 0.0, 0.0};
    double x_tgt[NX] = {0.5, 0.3, 0.1, 0.0, 0.0, 0.0};
    double u_out[NU] = {};

    ltv.solve_waypoint(x, x_tgt, 0.4, 0.02, /*lqr_ref=*/true, u_out);

    double ctrl_mag = vec_inf(u_out, NU);
    bool ok = (ctrl_mag < 1e-2);
    std::printf("max|u|=%.2e %s\n", ctrl_mag, ok ? "PASS" : "FAIL");
    return ok;
}

// 6. Near-zero controls when starting exactly at zero-velocity target (Hermite mode)
static bool test_at_target_hermite_near_zero()
{
    std::printf("6. Near-zero controls at target (Hermite mode) ... ");

    MecanumLTV ltv;
    ltv.setModelParams(make_params());
    ltv.setConfig(make_config());

    double x[NX]     = {-0.2, 0.7, 0.0, 0.0, 0.0, 0.0};
    double x_tgt[NX] = {-0.2, 0.7, 0.0, 0.0, 0.0, 0.0};
    double u_out[NU] = {};

    ltv.solve_waypoint(x, x_tgt, 0.4, 0.02, /*lqr_ref=*/false, u_out);

    double ctrl_mag = vec_inf(u_out, NU);
    bool ok = (ctrl_mag < 1e-2);
    std::printf("max|u|=%.2e %s\n", ctrl_mag, ok ? "PASS" : "FAIL");
    return ok;
}

// 7. Closed-loop XY convergence — LQR mode, zero-velocity target
static bool test_closed_loop_lqr_converges()
{
    std::printf("7. Closed-loop XY convergence (LQR, zero-vel target) ... ");

    ModelParams params = make_params();
    MPCConfig   cfg    = make_config(0.02, 15);
    MecanumLTV  ltv;
    ltv.setModelParams(params);
    ltv.setConfig(cfg);

    double x[NX]     = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x_tgt[NX] = {1.0, 0.5, 0.0, 0.0, 0.0, 0.0};

    const double dt         = cfg.dt;
    const int    n_steps    = 60;     // 1.2 s at 50 Hz
    const double t_total    = 0.8;    // aim to arrive in 0.8 s

    double err_initial = pos_error(x, x_tgt);
    double err_prev    = err_initial;
    double err_final   = err_initial;
    int    n_improving = 0;

    for (int k = 0; k < n_steps; ++k) {
        double t_rem = t_total - k * dt;
        if (t_rem < dt) t_rem = dt;   // keep at least one step

        double u_out[NU] = {};
        ltv.solve_waypoint(x, x_tgt, t_rem, dt, /*lqr_ref=*/true, u_out);
        sim_step(x, u_out, params, dt);

        double err = pos_error(x, x_tgt);
        if (err < err_prev) ++n_improving;
        err_prev  = err;
        err_final = err;
    }

    // Error must reduce significantly; allow for small oscillation near target
    bool ok = (err_final < 0.25 && err_final < 0.5 * err_initial);
    std::printf("err %.3f->%.3f, improving=%d/%d %s\n",
                err_initial, err_final, n_improving, n_steps,
                ok ? "PASS" : "FAIL");
    return ok;
}

// 8. Closed-loop XY convergence — Hermite mode, zero-velocity target
static bool test_closed_loop_hermite_converges()
{
    std::printf("8. Closed-loop XY convergence (Hermite, zero-vel target) ... ");

    ModelParams params = make_params();
    MPCConfig   cfg    = make_config(0.02, 15);
    MecanumLTV  ltv;
    ltv.setModelParams(params);
    ltv.setConfig(cfg);

    double x[NX]     = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x_tgt[NX] = {0.8, -0.6, 0.0, 0.0, 0.0, 0.0};

    const double dt      = cfg.dt;
    const int    n_steps = 60;
    const double t_total = 0.8;

    double err_initial = pos_error(x, x_tgt);
    double err_final   = err_initial;

    for (int k = 0; k < n_steps; ++k) {
        double t_rem = t_total - k * dt;
        if (t_rem < dt) t_rem = dt;

        double u_out[NU] = {};
        ltv.solve_waypoint(x, x_tgt, t_rem, dt, /*lqr_ref=*/false, u_out);
        sim_step(x, u_out, params, dt);
        err_final = pos_error(x, x_tgt);
    }

    bool ok = (err_final < 0.15 && err_final < 0.5 * err_initial);
    std::printf("err %.3f->%.3f %s\n", err_initial, err_final,
                ok ? "PASS" : "FAIL");
    return ok;
}

// 9. Heading convergence — LQR mode with heading change
static bool test_heading_convergence_lqr()
{
    std::printf("9. Heading convergence (LQR mode) ... ");

    ModelParams params = make_params();
    MPCConfig   cfg    = make_config(0.02, 15);
    MecanumLTV  ltv;
    ltv.setModelParams(params);
    ltv.setConfig(cfg);

    // Target: same position, but rotated 90°, zero velocity
    double x[NX]     = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x_tgt[NX] = {0.0, 0.0, M_PI / 2.0, 0.0, 0.0, 0.0};

    const double dt      = cfg.dt;
    const int    n_steps = 80;
    const double t_total = 1.0;

    auto angle_wrap = [](double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    };

    double h_err_initial = std::fabs(angle_wrap(x[2] - x_tgt[2]));
    double h_err_final   = h_err_initial;

    for (int k = 0; k < n_steps; ++k) {
        double t_rem = t_total - k * dt;
        if (t_rem < dt) t_rem = dt;

        double u_out[NU] = {};
        ltv.solve_waypoint(x, x_tgt, t_rem, dt, /*lqr_ref=*/true, u_out);
        sim_step(x, u_out, params, dt);
        h_err_final = std::fabs(angle_wrap(x[2] - x_tgt[2]));
    }

    bool ok = (h_err_final < 0.2 && h_err_final < 0.5 * h_err_initial);
    std::printf("heading_err %.3f->%.3f rad %s\n",
                h_err_initial, h_err_final, ok ? "PASS" : "FAIL");
    return ok;
}

// 10. Short t_remaining (< dt) clamps N_eff = 1 and still returns bounded controls
static bool test_short_t_remaining_bounded()
{
    std::printf("10. Short t_remaining clamps to N_eff=1, still bounded ... ");

    MecanumLTV ltv;
    ltv.setModelParams(make_params());
    MPCConfig cfg = make_config(0.02, 15);
    ltv.setConfig(cfg);

    double x[NX]     = {0.5, 0.5, 0.0, 0.3, 0.1, 0.0};
    double x_tgt[NX] = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0};

    // t_remaining < dt → N_eff = ceil(t_rem/dt) = 1
    double u_out[NU] = {};
    int ret = ltv.solve_waypoint(x, x_tgt, 0.005, 0.02, /*lqr_ref=*/true, u_out);

    bool ok = (ret == 0);
    for (int i = 0; i < NU; ++i)
        if (u_out[i] < cfg.u_min - 1e-6 || u_out[i] > cfg.u_max + 1e-6)
            ok = false;

    std::printf("ret=%d max|u|=%.3f %s\n", ret, vec_inf(u_out, NU),
                ok ? "PASS" : "FAIL");
    return ok;
}

// 11. Controls point in opposite directions for targets on opposite sides
static bool test_controls_flip_with_target_direction()
{
    std::printf("11. Controls reverse when target switches sides ... ");

    ModelParams params = make_params();
    MecanumLTV  ltv;
    ltv.setModelParams(params);
    ltv.setConfig(make_config(0.02, 15));

    // Robot at origin heading 0 (facing +x).
    // Target A: +x (forward), Target B: -x (backward)
    double x0[NX]      = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x_tgt_A[NX] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x_tgt_B[NX] = {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    double u_A[NU] = {}, u_B[NU] = {};
    ltv.solve_waypoint(x0, x_tgt_A, 0.5, 0.02, /*lqr_ref=*/true, u_A);
    ltv.solve_waypoint(x0, x_tgt_B, 0.5, 0.02, /*lqr_ref=*/true, u_B);

    // Net longitudinal force is sum(u) for mecanum (symmetric wheel layout).
    // Forward target → positive sum; backward target → negative sum (or opposite sign).
    double sum_A = u_A[0] + u_A[1] + u_A[2] + u_A[3];
    double sum_B = u_B[0] + u_B[1] + u_B[2] + u_B[3];

    // Both should be non-negligible and have opposite signs
    bool ok = (sum_A * sum_B < 0.0 &&
               std::fabs(sum_A) > 1e-3 &&
               std::fabs(sum_B) > 1e-3);
    std::printf("sum_u: A=%.3f B=%.3f %s\n", sum_A, sum_B,
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    std::printf("=== test_waypoint_tracking ===\n\n");

    bool all_pass = true;
    all_pass &= test_unconfigured_no_params();
    all_pass &= test_unconfigured_no_config();
    all_pass &= test_returns_zero_on_success();
    all_pass &= test_control_bounds();
    all_pass &= test_at_target_lqr_near_zero();
    all_pass &= test_at_target_hermite_near_zero();
    all_pass &= test_closed_loop_lqr_converges();
    all_pass &= test_closed_loop_hermite_converges();
    all_pass &= test_heading_convergence_lqr();
    all_pass &= test_short_t_remaining_bounded();
    all_pass &= test_controls_flip_with_target_direction();

    std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
