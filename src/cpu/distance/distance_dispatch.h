#ifndef PARASELECT_CPU_DISTANCE_DISPATCH_H
#define PARASELECT_CPU_DISTANCE_DISPATCH_H

#include "cpu/distance/distance_select.h"

#include <cstdint>

namespace paraselect {
namespace cpu_distance {

using Fp32L2Fn = float (*)(const float* q, const float* v, int dim);
using Fp32PrefetchFn = void (*)(const float* p, int dim);
using U8L2Fn = int32_t (*)(const uint8_t* q, const uint8_t* v, int dim);
using U8PrefetchFn = void (*)(const uint8_t* p, int dim);

float fp32_l2_plain_fn(const float* q, const float* v, int dim);
float fp32_l2_avx2_fn(const float* q, const float* v, int dim);
float fp32_l2_avx512_fn(const float* q, const float* v, int dim);
void fp32_prefetch_plain_fn(const float* p, int dim);
void fp32_prefetch_avx2_fn(const float* p, int dim);
void fp32_prefetch_avx512_fn(const float* p, int dim);

int32_t u8_l2_plain_fn(const uint8_t* q, const uint8_t* v, int dim);
int32_t u8_l2_avx2_fn(const uint8_t* q, const uint8_t* v, int dim);
int32_t u8_l2_avx512_fn(const uint8_t* q, const uint8_t* v, int dim);
void u8_prefetch_plain_fn(const uint8_t* p, int dim);
void u8_prefetch_avx2_fn(const uint8_t* p, int dim);
void u8_prefetch_avx512_fn(const uint8_t* p, int dim);

Fp32L2Fn select_fp32_l2_fn(Fp32DistanceIsa isa);
Fp32PrefetchFn select_fp32_prefetch_fn(Fp32DistanceIsa isa);
U8L2Fn select_u8_l2_fn(U8DistanceIsa isa);
U8PrefetchFn select_u8_prefetch_fn(U8DistanceIsa isa);

inline Fp32L2Fn select_fp32_l2_fn() {
    return select_fp32_l2_fn(select_fp32_distance_isa());
}

inline Fp32PrefetchFn select_fp32_prefetch_fn() {
    return select_fp32_prefetch_fn(select_fp32_distance_isa());
}

inline U8L2Fn select_u8_l2_fn() {
    return select_u8_l2_fn(select_u8_distance_isa());
}

inline U8PrefetchFn select_u8_prefetch_fn() {
    return select_u8_prefetch_fn(select_u8_distance_isa());
}

inline const char* selected_fp32_l2_isa_name() {
    return fp32_distance_isa_name(select_fp32_distance_isa());
}

inline const char* selected_u8_l2_isa_name() {
    return u8_distance_isa_name(select_u8_distance_isa());
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_DISPATCH_H
