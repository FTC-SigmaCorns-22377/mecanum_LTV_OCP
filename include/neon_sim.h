// neon_sim.h -- Scalar simulation of ARM NEON intrinsics for x86 testing.
// Drop-in replacement for <arm_neon.h>: same types and function signatures,
// implemented with plain C++ so the NEON kernels compile and run on any host.
#pragma once

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef float float32_t;

struct float32x2_t { float v[2]; };
struct float32x4_t { float v[4]; };
struct float32x4x3_t { float32x4_t val[3]; };
struct float32x4x4_t { float32x4_t val[4]; };
struct float64x2_t { double v[2]; };
struct uint64x2_t  { unsigned long long v[2]; };

// ---------------------------------------------------------------------------
// Load / Store
// ---------------------------------------------------------------------------

inline float32x4_t vld1q_f32(const float* p) {
    float32x4_t r; std::memcpy(r.v, p, 16); return r;
}
inline void vst1q_f32(float* p, float32x4_t a) {
    std::memcpy(p, a.v, 16);
}
inline float64x2_t vld1q_f64(const double* p) {
    float64x2_t r; r.v[0] = p[0]; r.v[1] = p[1]; return r;
}
inline void vst1q_f64(double* p, float64x2_t a) {
    p[0] = a.v[0]; p[1] = a.v[1];
}

// ---------------------------------------------------------------------------
// Broadcast / Duplicate
// ---------------------------------------------------------------------------

inline float32x4_t vdupq_n_f32(float a) {
    return {{ a, a, a, a }};
}
inline float64x2_t vdupq_n_f64(double a) {
    float64x2_t r; r.v[0] = a; r.v[1] = a; return r;
}

// ---------------------------------------------------------------------------
// Lane access
// ---------------------------------------------------------------------------

template<int lane>
inline float vgetq_lane_f32_tpl(float32x4_t a) { return a.v[lane]; }

// Macro to match the arm_neon API: vgetq_lane_f32(vec, imm)
#define vgetq_lane_f32(a, lane) ((a).v[lane])
#define vgetq_lane_u64(a, lane) ((a).v[lane])

// vsetq_lane is not used but included for completeness
#define vsetq_lane_f32(val, vec, lane) ([&]{ auto _v = (vec); _v.v[lane] = (val); return _v; }())

// ---------------------------------------------------------------------------
// Arithmetic: 4-wide float
// ---------------------------------------------------------------------------

inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
    return {{ a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2], a.v[3]+b.v[3] }};
}
inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) {
    return {{ a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2], a.v[3]-b.v[3] }};
}
inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
    return {{ a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2], a.v[3]*b.v[3] }};
}
inline float32x4_t vnegq_f32(float32x4_t a) {
    return {{ -a.v[0], -a.v[1], -a.v[2], -a.v[3] }};
}
inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) {
    return {{ std::fmax(a.v[0],b.v[0]), std::fmax(a.v[1],b.v[1]),
              std::fmax(a.v[2],b.v[2]), std::fmax(a.v[3],b.v[3]) }};
}
inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) {
    return {{ std::fmin(a.v[0],b.v[0]), std::fmin(a.v[1],b.v[1]),
              std::fmin(a.v[2],b.v[2]), std::fmin(a.v[3],b.v[3]) }};
}

// ---------------------------------------------------------------------------
// Multiply by scalar
// ---------------------------------------------------------------------------

inline float32x4_t vmulq_n_f32(float32x4_t a, float b) {
    return {{ a.v[0]*b, a.v[1]*b, a.v[2]*b, a.v[3]*b }};
}

// ---------------------------------------------------------------------------
// Fused multiply-add/sub: 4-wide float
// ---------------------------------------------------------------------------

// vfmaq_f32(a, b, c) = a + b*c
inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
    return {{ a.v[0]+b.v[0]*c.v[0], a.v[1]+b.v[1]*c.v[1],
              a.v[2]+b.v[2]*c.v[2], a.v[3]+b.v[3]*c.v[3] }};
}
// vfmsq_f32(a, b, c) = a - b*c
inline float32x4_t vfmsq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
    return {{ a.v[0]-b.v[0]*c.v[0], a.v[1]-b.v[1]*c.v[1],
              a.v[2]-b.v[2]*c.v[2], a.v[3]-b.v[3]*c.v[3] }};
}
// vfmaq_n_f32(a, b, c) = a + b*c  (c is scalar)
inline float32x4_t vfmaq_n_f32(float32x4_t a, float32x4_t b, float c) {
    return {{ a.v[0]+b.v[0]*c, a.v[1]+b.v[1]*c, a.v[2]+b.v[2]*c, a.v[3]+b.v[3]*c }};
}
// vfmsq_n_f32(a, b, c) = a - b*c  (c is scalar)
inline float32x4_t vfmsq_n_f32(float32x4_t a, float32x4_t b, float c) {
    return {{ a.v[0]-b.v[0]*c, a.v[1]-b.v[1]*c, a.v[2]-b.v[2]*c, a.v[3]-b.v[3]*c }};
}

// ---------------------------------------------------------------------------
// Lane-indexed multiply / fma: vmulq_laneq_f32, vfmaq_laneq_f32
// ---------------------------------------------------------------------------

// vmulq_laneq_f32(a, b, lane) = a * b[lane]
#define vmulq_laneq_f32(a, b, lane) vmulq_n_f32((a), (b).v[lane])

// vfmaq_laneq_f32(a, b, c, lane) = a + b * c[lane]
#define vfmaq_laneq_f32(a, b, c, lane) vfmaq_n_f32((a), (b), (c).v[lane])

// vfmsq_laneq_f32 not used by riccati_tracking but included for completeness
#define vfmsq_laneq_f32(a, b, c, lane) \
    vfmsq_f32((a), (b), vdupq_n_f32((c).v[lane]))

// ---------------------------------------------------------------------------
// Horizontal operations
// ---------------------------------------------------------------------------

// vaddvq_f32: horizontal add all 4 lanes
inline float vaddvq_f32(float32x4_t a) {
    return a.v[0] + a.v[1] + a.v[2] + a.v[3];
}

// vpaddq_f32: pairwise add
inline float32x4_t vpaddq_f32(float32x4_t a, float32x4_t b) {
    return {{ a.v[0]+a.v[1], a.v[2]+a.v[3], b.v[0]+b.v[1], b.v[2]+b.v[3] }};
}

// vaddvq_f64: horizontal add 2 double lanes
inline double vaddvq_f64(float64x2_t a) {
    return a.v[0] + a.v[1];
}

// ---------------------------------------------------------------------------
// Reciprocal square root estimate + Newton step helper
// ---------------------------------------------------------------------------

// vrsqrtes_f32: scalar rsqrt estimate (we just compute exact for simulation)
inline float32_t vrsqrtes_f32(float32_t a) {
    return 1.0f / std::sqrt(a);
}

// vrsqrtss_f32(a, b): Newton step helper = (3 - a*b) / 2
inline float32_t vrsqrtss_f32(float32_t a, float32_t b) {
    return (3.0f - a * b) * 0.5f;
}

// ---------------------------------------------------------------------------
// Double-precision FMA (for the utility functions)
// ---------------------------------------------------------------------------

inline float64x2_t vfmaq_f64(float64x2_t a, float64x2_t b, float64x2_t c) {
    float64x2_t r; r.v[0] = a.v[0]+b.v[0]*c.v[0]; r.v[1] = a.v[1]+b.v[1]*c.v[1]; return r;
}
inline float64x2_t vfmsq_f64(float64x2_t a, float64x2_t b, float64x2_t c) {
    float64x2_t r; r.v[0] = a.v[0]-b.v[0]*c.v[0]; r.v[1] = a.v[1]-b.v[1]*c.v[1]; return r;
}
