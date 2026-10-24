#ifndef PARASELECT_KERNEL_U8_AVX2_H
#define PARASELECT_KERNEL_U8_AVX2_H

#include "cpu/distance/u8_l2_avx2.h"

namespace paraselect {
namespace cpu_fine {

struct U8Avx2Kernel {
    static inline void prefetch_candidate(const uint8_t* p, int dim) {
        cpu_distance::u8_prefetch_avx2(p, dim);
    }

    static inline void distance_q4(
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
        cpu_distance::u8_l2_q4_avx2(q0, q1, q2, q3, v, dim, out0, out1, out2, out3);
    }

    static inline int32_t distance_q1(const uint8_t* q, const uint8_t* v, int dim) {
        return cpu_distance::u8_l2_avx2(q, v, dim);
    }
};

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_KERNEL_U8_AVX2_H
