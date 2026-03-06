#pragma once

#include "mpc_types.h"
#include "qp_solvers.h"

// ---------------------------------------------------------------------------
// Offline precomputation
// ---------------------------------------------------------------------------

// Trig decomposition: precompute A_d, B_d0, B_dc, B_ds, A_d_pow
// Uses exact_discretize at θ=0, π/2, π to extract components.
// Returns max decomposition error at a verification heading.
double heading_lookup_precompute(const ModelParams& params, double dt,
                                 HeadingLookupData& data);

// Heading table: precompute B_d at M evenly-spaced headings
void heading_table_precompute(const ModelParams& params, double dt, int M,
                              HeadingTableData& table);

// ---------------------------------------------------------------------------
// Online B_d reconstruction
// ---------------------------------------------------------------------------

// Trig decomposition: B_k = B_d0 + cos(θ_k)·B_dc + sin(θ_k)·B_ds
void heading_lookup_build_B_list(const HeadingLookupData& data,
                                 const double* theta_list, int N,
                                 double* B_list);

// Table interpolation: linearly interpolate B_d from table
void heading_table_build_B_list(const HeadingTableData& table,
                                const double* theta_list, int N,
                                double* B_list);

// ---------------------------------------------------------------------------
// Heading schedule generation
// ---------------------------------------------------------------------------

// Derive default HeadingScheduleConfig from model parameters
HeadingScheduleConfig heading_schedule_config_from_params(const ModelParams& params);

// Generate a torque-feasible heading schedule from current state and reference
// theta_out has N+1 entries (heading at start of each interval + terminal)
void generate_heading_schedule(const double x0[NX], const RefNode* ref_window,
                               int N, double dt,
                               const HeadingScheduleConfig& sched_config,
                               double* theta_out);

// ---------------------------------------------------------------------------
// Online condensed solve (heading-lookup mode)
// ---------------------------------------------------------------------------

// Solve using trig decomposition
QPSolution heading_lookup_solve_condensed(const HeadingLookupData& data,
                                          const RefNode* ref_window,
                                          const double x0[NX],
                                          const MPCConfig& config,
                                          const HeadingScheduleConfig& sched_config,
                                          QpSolverType solver_type,
                                          SolverContext& ctx);

// Solve using table interpolation
QPSolution heading_table_solve_condensed(const HeadingTableData& table,
                                         const RefNode* ref_window,
                                         const double x0[NX],
                                         const MPCConfig& config,
                                         const HeadingScheduleConfig& sched_config,
                                         QpSolverType solver_type,
                                         SolverContext& ctx);

// ---------------------------------------------------------------------------
// Kernel-based fast solve (precomputed cost-to-go kernels)
// ---------------------------------------------------------------------------

// Precompute cost kernels P[j], G[j] (offline, ~3μs)
void heading_kernel_precompute(const HeadingLookupData& data,
                                const MPCConfig& config,
                                HeadingKernelData& kern);

// Fast solve using precomputed kernels (for FISTA/HPIPM dense/qpOASES)
QPSolution heading_lookup_solve_fast(const HeadingLookupData& data,
                                     const HeadingKernelData& kern,
                                     const RefNode* ref_window,
                                     const double x0[NX],
                                     const MPCConfig& config,
                                     const HeadingScheduleConfig& sched_config,
                                     QpSolverType solver_type,
                                     SolverContext& ctx);

// ---------------------------------------------------------------------------
// HPIPM OCP direct solve (no condensing)
// ---------------------------------------------------------------------------

#ifdef MPC_USE_HPIPM
// Direct OCP solve via HPIPM Riccati (no condensing at all)
QPSolution heading_lookup_solve_ocp(const HeadingLookupData& data,
                                    const RefNode* ref_window,
                                    const double x0[NX],
                                    const MPCConfig& config,
                                    const HeadingScheduleConfig& sched_config,
                                    SolverContext& ctx);
#endif
