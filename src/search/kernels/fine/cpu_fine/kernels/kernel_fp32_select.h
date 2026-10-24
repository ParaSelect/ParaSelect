#ifndef PARASELECT_KERNEL_FP32_SELECT_H
#define PARASELECT_KERNEL_FP32_SELECT_H

#include "search/kernels/fine/cpu_fine/dispatch/cpu_isa.h"

namespace paraselect {
namespace cpu_fine {

enum class Fp32KernelIsa {
    Plain,
    Avx2,
    Avx512,
};

inline Fp32KernelIsa select_fp32_kernel_isa() {
#if PARASELECT_ENABLE_X86_SIMD
    if (host_supports_fp32_avx512()) return Fp32KernelIsa::Avx512;
    if (host_supports_fp32_avx2()) return Fp32KernelIsa::Avx2;
#endif
    return Fp32KernelIsa::Plain;
}

inline const char* fp32_kernel_isa_name(Fp32KernelIsa isa) {
    switch (isa) {
        case Fp32KernelIsa::Avx512: return "avx512";
        case Fp32KernelIsa::Avx2: return "avx2";
        case Fp32KernelIsa::Plain: return "plain";
    }
    return "unknown";
}

}  // namespace cpu_fine
}  // namespace paraselect

#if PARASELECT_ENABLE_X86_SIMD
#define PARASELECT_FP32_BACKEND_ARGS(PLAIN, AVX2, AVX512) PLAIN, AVX2, AVX512
#else
#define PARASELECT_FP32_BACKEND_ARGS(PLAIN, AVX2, AVX512) PLAIN, PLAIN, PLAIN
#endif

#endif  // PARASELECT_KERNEL_FP32_SELECT_H
