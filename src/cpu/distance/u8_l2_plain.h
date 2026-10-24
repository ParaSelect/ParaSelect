#ifndef PARASELECT_CPU_DISTANCE_U8_L2_PLAIN_H
#define PARASELECT_CPU_DISTANCE_U8_L2_PLAIN_H

#include <cstdint>

namespace paraselect {
namespace cpu_distance {

#if defined(__GNUC__) || defined(__clang__)
#define PARASELECT_U8_L2_PLAIN_INLINE __attribute__((always_inline)) inline
#else
#define PARASELECT_U8_L2_PLAIN_INLINE inline
#endif

PARASELECT_U8_L2_PLAIN_INLINE void u8_prefetch_plain(const uint8_t* p, int dim) {
#if defined(__GNUC__) || defined(__clang__)
    for (int i = 0; i < dim; i += 32) __builtin_prefetch(p + i, 0, 3);
#else
    (void)p;
    (void)dim;
#endif
}

PARASELECT_U8_L2_PLAIN_INLINE int32_t u8_l2_plain(const uint8_t* q, const uint8_t* v, int dim) {
    int32_t out = 0;
    for (int i = 0; i < dim; ++i) {
        int d = static_cast<int>(q[i]) - static_cast<int>(v[i]);
        out += d * d;
    }
    return out;
}

PARASELECT_U8_L2_PLAIN_INLINE void u8_l2_q4_plain(
    const uint8_t* q0,
    const uint8_t* q1,
    const uint8_t* q2,
    const uint8_t* q3,
    const uint8_t* v,
    int dim,
    int32_t& out0,
    int32_t& out1,
    int32_t& out2,
    int32_t& out3) {
    out0 = out1 = out2 = out3 = 0;
    for (int i = 0; i < dim; ++i) {
        int s0 = static_cast<int>(q0[i]) - static_cast<int>(v[i]); out0 += s0 * s0;
        int s1 = static_cast<int>(q1[i]) - static_cast<int>(v[i]); out1 += s1 * s1;
        int s2 = static_cast<int>(q2[i]) - static_cast<int>(v[i]); out2 += s2 * s2;
        int s3 = static_cast<int>(q3[i]) - static_cast<int>(v[i]); out3 += s3 * s3;
    }
}

#undef PARASELECT_U8_L2_PLAIN_INLINE

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_U8_L2_PLAIN_H
