#include "cpu/distance/distance_dispatch.h"

#include "cpu/distance/fp32_l2_avx2.h"
#include "cpu/distance/u8_l2_avx2.h"

namespace paraselect {
namespace cpu_distance {

float fp32_l2_avx2_fn(const float* q, const float* v, int dim) {
    return fp32_l2_avx2(q, v, dim);
}

void fp32_prefetch_avx2_fn(const float* p, int dim) {
    fp32_prefetch_avx2(p, dim);
}

int32_t u8_l2_avx2_fn(const uint8_t* q, const uint8_t* v, int dim) {
    return u8_l2_avx2(q, v, dim);
}

void u8_prefetch_avx2_fn(const uint8_t* p, int dim) {
    u8_prefetch_avx2(p, dim);
}

}  // namespace cpu_distance
}  // namespace paraselect
