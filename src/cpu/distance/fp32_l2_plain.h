#ifndef PARASELECT_CPU_DISTANCE_FP32_L2_PLAIN_H
#define PARASELECT_CPU_DISTANCE_FP32_L2_PLAIN_H

namespace paraselect {
namespace cpu_distance {

#if defined(__GNUC__) || defined(__clang__)
#define PARASELECT_FP32_L2_PLAIN_INLINE __attribute__((always_inline)) inline
#else
#define PARASELECT_FP32_L2_PLAIN_INLINE inline
#endif

PARASELECT_FP32_L2_PLAIN_INLINE void fp32_prefetch_plain(const float* p, int dim) {
#if defined(__GNUC__) || defined(__clang__)
    for (int i = 0; i < dim; i += 16) {
        __builtin_prefetch(p + i, 0, 3);
    }
#else
    (void)p;
    (void)dim;
#endif
}

PARASELECT_FP32_L2_PLAIN_INLINE float fp32_l2_plain(const float* q, const float* v, int dim) {
    float out = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = q[i] - v[i];
        out += d * d;
    }
    return out;
}

PARASELECT_FP32_L2_PLAIN_INLINE void fp32_l2_q4_plain(
    const float* q0,
    const float* q1,
    const float* q2,
    const float* q3,
    const float* v,
    int dim,
    float& out0,
    float& out1,
    float& out2,
    float& out3) {
    out0 = out1 = out2 = out3 = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float a0 = q0[i] - v[i]; out0 += a0 * a0;
        float a1 = q1[i] - v[i]; out1 += a1 * a1;
        float a2 = q2[i] - v[i]; out2 += a2 * a2;
        float a3 = q3[i] - v[i]; out3 += a3 * a3;
    }
}

#undef PARASELECT_FP32_L2_PLAIN_INLINE

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_FP32_L2_PLAIN_H
