#ifndef PARASELECT_KERNEL_FP32_AVX512_H
#define PARASELECT_KERNEL_FP32_AVX512_H

#include "cpu/distance/fp32_l2_avx512.h"

namespace paraselect {
namespace cpu_fine {

struct Fp32Avx512Kernel {
    static inline void prefetch_candidate(const float* p, int dim) {
        cpu_distance::fp32_prefetch_avx512(p, dim);
    }

    static inline void distance_q4(
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
        cpu_distance::fp32_l2_q4_avx512(q0, q1, q2, q3, v, dim, out0, out1, out2, out3);
    }

    static inline float distance_q1(const float* q, const float* v, int dim) {
        return cpu_distance::fp32_l2_avx512(q, v, dim);
    }
};

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_KERNEL_FP32_AVX512_H
