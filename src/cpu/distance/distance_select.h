#ifndef PARASELECT_CPU_DISTANCE_SELECT_H
#define PARASELECT_CPU_DISTANCE_SELECT_H

#include "cpu/distance/cpu_isa.h"

namespace paraselect {
namespace cpu_distance {

enum class Fp32DistanceIsa {
    Plain,
    Avx2,
    Avx512,
};

enum class U8DistanceIsa {
    Plain,
    Avx2,
    Avx512,
};

inline Fp32DistanceIsa select_fp32_distance_isa() {
#if PARASELECT_ENABLE_X86_SIMD
    if (host_supports_fp32_avx512()) return Fp32DistanceIsa::Avx512;
    if (host_supports_fp32_avx2()) return Fp32DistanceIsa::Avx2;
#endif
    return Fp32DistanceIsa::Plain;
}

inline U8DistanceIsa select_u8_distance_isa() {
#if PARASELECT_ENABLE_X86_SIMD
    if (host_supports_u8_avx512()) return U8DistanceIsa::Avx512;
    if (host_supports_avx2()) return U8DistanceIsa::Avx2;
#endif
    return U8DistanceIsa::Plain;
}

inline const char* fp32_distance_isa_name(Fp32DistanceIsa isa) {
    switch (isa) {
        case Fp32DistanceIsa::Avx512: return "avx512";
        case Fp32DistanceIsa::Avx2: return "avx2";
        case Fp32DistanceIsa::Plain: return "plain";
    }
    return "unknown";
}

inline const char* u8_distance_isa_name(U8DistanceIsa isa) {
    switch (isa) {
        case U8DistanceIsa::Avx512: return "avx512";
        case U8DistanceIsa::Avx2: return "avx2";
        case U8DistanceIsa::Plain: return "plain";
    }
    return "unknown";
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_SELECT_H
