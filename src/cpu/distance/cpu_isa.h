#ifndef PARASELECT_CPU_DISTANCE_CPU_ISA_H
#define PARASELECT_CPU_DISTANCE_CPU_ISA_H

namespace paraselect {
namespace cpu_distance {
#ifndef PARASELECT_ENABLE_X86_SIMD
#define PARASELECT_ENABLE_X86_SIMD 0
#endif

inline constexpr bool x86_simd_compiled() {
    return PARASELECT_ENABLE_X86_SIMD != 0;
}

inline bool host_supports_avx2() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return true;
#endif
#else
    return false;
#endif
}

inline bool host_supports_avx512f() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f");
#else
    return true;
#endif
#else
    return false;
#endif
}

inline bool host_supports_fma() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("fma");
#else
    return true;
#endif
#else
    return false;
#endif
}

inline bool host_supports_avx512bw() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512bw");
#else
    return true;
#endif
#else
    return false;
#endif
}

inline bool host_supports_u8_avx512() {
    return host_supports_avx512f() && host_supports_avx512bw();
}

inline bool host_supports_fp32_avx2() {
    return host_supports_avx2() && host_supports_fma();
}

inline bool host_supports_fp32_avx512() {
    return host_supports_avx512f() && host_supports_fma();
}

}  // namespace cpu_distance
}  // namespace paraselect

#endif  // PARASELECT_CPU_DISTANCE_CPU_ISA_H
