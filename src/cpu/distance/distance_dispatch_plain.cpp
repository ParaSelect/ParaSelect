#include "cpu/distance/distance_dispatch.h"

#include "cpu/distance/fp32_l2_plain.h"
#include "cpu/distance/u8_l2_plain.h"

namespace paraselect {
namespace cpu_distance {

float fp32_l2_plain_fn(const float* q, const float* v, int dim) {
    return fp32_l2_plain(q, v, dim);
}

void fp32_prefetch_plain_fn(const float* p, int dim) {
    fp32_prefetch_plain(p, dim);
}

int32_t u8_l2_plain_fn(const uint8_t* q, const uint8_t* v, int dim) {
    return u8_l2_plain(q, v, dim);
}

void u8_prefetch_plain_fn(const uint8_t* p, int dim) {
    u8_prefetch_plain(p, dim);
}

}  // namespace cpu_distance
}  // namespace paraselect
