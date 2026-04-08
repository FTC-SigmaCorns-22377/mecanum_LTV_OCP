#pragma once

#include <cstdint>
#include <cstring>

// Fixed problem dimensions
constexpr int NX = 6;      // state: [px, py, theta, vx, vy, omega]
constexpr int NU = 4;      // input: [V1, V2, V3, V4]
constexpr int N_MAX = 30;  // maximum horizon length
constexpr int N_AUG = NX + NU;  // augmented system dimension (10)

// Robot parameters
struct ModelParams {
    double mass;              // kg
    double inertia;           // kg*m^2
    double damping_linear;    // N*s/m
    double damping_angular;   // N*m*s/rad
    double wheel_radius;      // m
    double lx, ly;            // half-distances center to wheel (m)
    double stall_torque;      // motor stall torque (N*m)
    double free_speed;        // motor no-load speed (rad/s)
    double J_mec[3 * NU];    // 3x4 mecanum Jacobian, column-major
};

// MPC tuning
struct MPCConfig {
    int N;                    // horizon length (<= N_MAX)
    double Q[NX * NX];       // state cost, column-major
    double R[NU * NU];       // input cost, column-major
    double Qf[NX * NX];      // terminal cost, column-major
    double u_min;             // input lower bound (duty cycle, typically -1)
    double u_max;             // input upper bound (duty cycle, typically +1)
    double dt;                // control timestep (s)
    // Anti-tip acceleration limits (m/s²) in robot body frame.
    // Prevents wheelies by adding a log-barrier on net body-frame acceleration
    // at each step, parameterised by the predicted velocity at that step.
    // Naturally allows full input authority at low speeds (the constraint only
    // tightens when damping + braking torque would exceed the tip threshold).
    //   a_tip_x = g * (half fore-aft wheelbase) / h_com
    //   a_tip_y = g * (half lateral track width) / h_com
    // Set to 0.0 (default) to disable.
    double a_tip_x = 0.0;
    double a_tip_y = 0.0;
};

// Reference trajectory node
struct RefNode {
    double x_ref[NX];        // reference state
    double u_ref[NU];        // reference input (feedforward)
    double theta;             // heading (explicit copy of x_ref[2])
    double omega;             // angular velocity (explicit copy of x_ref[5])
    double t;                 // timestamp
};

// Precomputed QP data for one MPC window (output of offline phase)
struct PrecomputedWindow {
    double H[N_MAX * NU * N_MAX * NU];     // condensed Hessian, column-major
    double L[N_MAX * NU * N_MAX * NU];     // lower Cholesky factor
    double F[N_MAX * NU * NX];             // gradient linear term: g = F*e0 + f_const
    double f_const[N_MAX * NU];            // gradient constant term
    double x_ref_0[NX];                    // reference state at window start
    int N;                                  // horizon length
    int n_vars;                             // N * NU
    double lambda_max;                      // largest eigenvalue of H (FISTA step size = 1/lambda_max)
};

// Online QP solution
struct QPSolution {
    double U[N_MAX * NU];       // full optimized input sequence
    double u0[NU];              // first control: [V1, V2, V3, V4]
    int n_active;               // number of active constraints at solution
    int n_iterations;           // solver iterations taken
    double solve_time_ns;       // timing
};

// Preallocated workspace for box QP solver (no heap allocation online)
struct BoxQPWorkspace {
    double grad[N_MAX * NU];
    double U[N_MAX * NU];
    double rhs[N_MAX * NU];
    double temp[N_MAX * NU];

    // FISTA workspace
    double V[N_MAX * NU];       // momentum/extrapolation point
    double U_old[N_MAX * NU];   // previous iterate (for restart check)

    // Warm-start state (zero-init sets warm_valid = false)
    double U_prev[N_MAX * NU];  // previous solution (shifted for warm-start)
    int    prev_n_vars;          // n_vars of previous solve
    bool   warm_valid;           // true if U_prev is usable
};

// Serialization header
struct MPCFileHeader {
    uint32_t magic;       // 0x4D504351 ("MPCQ")
    uint32_t version;     // 2
    uint32_t n_windows;
    uint32_t N;           // horizon
    uint32_t nx;
    uint32_t nu;
    double u_min;
    double u_max;
    double dt;            // control timestep (v2+)
};

constexpr uint32_t MPC_FILE_MAGIC = 0x4D504351;
constexpr uint32_t MPC_FILE_VERSION = 2;

// ---------------------------------------------------------------------------
// Cost-based window selection config
// ---------------------------------------------------------------------------

struct WindowSelConfig {
    double pos_weight     = 1.0;   // weight on position+heading distance squared
    double time_weight    = 0.5;   // weight on (i - time_idx_float)^2
    double heading_weight = 0.3;   // scale applied to dtheta in distance metric
    int    search_radius  = 10;    // max windows to search forward from prev_idx
    int    max_jump       = 5;     // hard cap on advance per solve call
    double hold_radius    = 0.3;   // metres: if robot is farther than this from the current
                                   // reference window, freeze the window (don't advance) and
                                   // don't accumulate elapsed time until it re-approaches.
};

// ---------------------------------------------------------------------------
// Heading-lookup LTV mode data structures
// ---------------------------------------------------------------------------

// Trig decomposition: B_d(θ) = B_d0 + cos(θ)·B_dc + sin(θ)·B_ds
struct HeadingLookupData {
    double A_d[NX * NX];                         // constant discrete state matrix
    double B_d0[NX * NU];                         // heading-independent B_d component
    double B_dc[NX * NU];                         // cos(θ) coefficient
    double B_ds[NX * NU];                         // sin(θ) coefficient
    double A_d_pow[(N_MAX + 1) * NX * NX];        // A_d^k for k=0..N_MAX
    double dt;
};

// Heading schedule configuration
struct HeadingScheduleConfig {
    double alpha_0;        // max angular accel at zero ω (rad/s²)
    double omega_max;      // max angular velocity (rad/s)
    double v_max;          // max field velocity for derating (m/s)
    double heading_gain;   // proportional gain for heading tracking (1/s)
};
