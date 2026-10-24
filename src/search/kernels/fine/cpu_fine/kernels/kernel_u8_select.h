#ifndef PARASELECT_KERNEL_U8_SELECT_H
#define PARASELECT_KERNEL_U8_SELECT_H

#include "search/kernels/fine/cpu_fine/dispatch/cpu_isa.h"

namespace paraselect {
namespace cpu_fine {

enum class U8KernelIsa {
    Plain,
    Avx2,
    Avx512,
};

inline U8KernelIsa select_u8_kernel_isa() {
#if PARASELECT_ENABLE_X86_SIMD
    if (host_supports_u8_avx512()) return U8KernelIsa::Avx512;
    if (host_supports_avx2()) return U8KernelIsa::Avx2;
#endif
    return U8KernelIsa::Plain;
}

inline const char* u8_kernel_isa_name(U8KernelIsa isa) {
    switch (isa) {
        case U8KernelIsa::Avx512: return "avx512";
        case U8KernelIsa::Avx2: return "avx2";
        case U8KernelIsa::Plain: return "plain";
    }
    return "unknown";
}

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_KERNEL_U8_SELECT_H
