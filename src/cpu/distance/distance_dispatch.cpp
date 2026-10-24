#include "cpu/distance/distance_dispatch.h"

namespace paraselect {
namespace cpu_distance {

Fp32L2Fn select_fp32_l2_fn(Fp32DistanceIsa isa) {
    switch (isa) {
#if PARASELECT_ENABLE_X86_SIMD
        case Fp32DistanceIsa::Avx512: return fp32_l2_avx512_fn;
        case Fp32DistanceIsa::Avx2: return fp32_l2_avx2_fn;
#endif
        case Fp32DistanceIsa::Plain: return fp32_l2_plain_fn;
    }
    return fp32_l2_plain_fn;
}

Fp32PrefetchFn select_fp32_prefetch_fn(Fp32DistanceIsa isa) {
    switch (isa) {
#if PARASELECT_ENABLE_X86_SIMD
        case Fp32DistanceIsa::Avx512: return fp32_prefetch_avx512_fn;
        case Fp32DistanceIsa::Avx2: return fp32_prefetch_avx2_fn;
#endif
        case Fp32DistanceIsa::Plain: return fp32_prefetch_plain_fn;
    }
    return fp32_prefetch_plain_fn;
}

U8L2Fn select_u8_l2_fn(U8DistanceIsa isa) {
    switch (isa) {
#if PARASELECT_ENABLE_X86_SIMD
        case U8DistanceIsa::Avx512: return u8_l2_avx512_fn;
        case U8DistanceIsa::Avx2: return u8_l2_avx2_fn;
#endif
        case U8DistanceIsa::Plain: return u8_l2_plain_fn;
    }
    return u8_l2_plain_fn;
}

U8PrefetchFn select_u8_prefetch_fn(U8DistanceIsa isa) {
    switch (isa) {
#if PARASELECT_ENABLE_X86_SIMD
        case U8DistanceIsa::Avx512: return u8_prefetch_avx512_fn;
        case U8DistanceIsa::Avx2: return u8_prefetch_avx2_fn;
#endif
        case U8DistanceIsa::Plain: return u8_prefetch_plain_fn;
    }
    return u8_prefetch_plain_fn;
}

}  // namespace cpu_distance
}  // namespace paraselect
