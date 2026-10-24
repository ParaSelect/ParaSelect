#include "search/kernels/fine/cpu_fine/runners/u8_rowmajor_numa_runners.h"

#include "search/kernels/fine/cpu_fine/kernels/kernel_u8_select.h"
#include "search/kernels/fine/cpu_fine/runners/u8_isa_runners.h"

namespace paraselect {
namespace cpu_fine {

long long u8_rowmajor_cluster_dynamic_numa_scratch(const CpuFineRunArgs& args) {
    switch (select_u8_kernel_isa()) {
#if PARASELECT_ENABLE_X86_SIMD
        case U8KernelIsa::Avx512: return u8_rowmajor_cluster_dynamic_numa_scratch_avx512(args);
        case U8KernelIsa::Avx2: return u8_rowmajor_cluster_dynamic_numa_scratch_avx2(args);
#endif
        case U8KernelIsa::Plain: return u8_rowmajor_cluster_dynamic_numa_scratch_plain(args);
    }
    return u8_rowmajor_cluster_dynamic_numa_scratch_plain(args);
}

long long u8_rowmajor_cluster_dynamic_numa_metadata(const CpuFineRunArgs& args) {
    switch (select_u8_kernel_isa()) {
#if PARASELECT_ENABLE_X86_SIMD
        case U8KernelIsa::Avx512: return u8_rowmajor_cluster_dynamic_numa_metadata_avx512(args);
        case U8KernelIsa::Avx2: return u8_rowmajor_cluster_dynamic_numa_metadata_avx2(args);
#endif
        case U8KernelIsa::Plain: return u8_rowmajor_cluster_dynamic_numa_metadata_plain(args);
    }
    return u8_rowmajor_cluster_dynamic_numa_metadata_plain(args);
}

long long u8_rowmajor_cluster_dynamic_numa_compact_metadata(const CpuFineRunArgs& args) {
    switch (select_u8_kernel_isa()) {
#if PARASELECT_ENABLE_X86_SIMD
        case U8KernelIsa::Avx512: return u8_rowmajor_cluster_dynamic_numa_compact_metadata_avx512(args);
        case U8KernelIsa::Avx2: return u8_rowmajor_cluster_dynamic_numa_compact_metadata_avx2(args);
#endif
        case U8KernelIsa::Plain: return u8_rowmajor_cluster_dynamic_numa_compact_metadata_plain(args);
    }
    return u8_rowmajor_cluster_dynamic_numa_compact_metadata_plain(args);
}

}  // namespace cpu_fine
}  // namespace paraselect
