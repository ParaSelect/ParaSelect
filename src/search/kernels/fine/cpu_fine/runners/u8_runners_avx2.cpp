#include "search/kernels/fine/cpu_fine/runners/u8_isa_runners.h"

#include "search/kernels/fine/cpu_fine/kernels/kernel_u8_avx2.h"
#include "search/kernels/fine/cpu_fine/runners/paged_runner_impl.h"
#include "search/kernels/fine/cpu_fine/runners/u8_rowmajor_numa_runner_impl.h"
#include "search/kernels/fine/cpu_fine/runners/u8_rowmajor_runner_impl.h"

namespace paraselect {
namespace cpu_fine {

long long u8_rowmajor_cluster_dynamic_scratch_avx2(const CpuFineRunArgs& args) {
    return u8_rowmajor_runner_impl::run_scratch<U8Avx2Kernel>(args);
}
long long u8_rowmajor_cluster_dynamic_metadata_avx2(const CpuFineRunArgs& args) {
    return u8_rowmajor_runner_impl::run_metadata<U8Avx2Kernel>(args);
}
long long u8_rowmajor_cluster_dynamic_compact_metadata_avx2(const CpuFineRunArgs& args) {
    return u8_rowmajor_runner_impl::run_compact_metadata<U8Avx2Kernel>(args);
}
long long u8_rowmajor_cluster_dynamic_numa_scratch_avx2(const CpuFineRunArgs& args) {
    return u8_rowmajor_numa_runner_impl::run_scratch<U8Avx2Kernel>(args);
}
long long u8_rowmajor_cluster_dynamic_numa_metadata_avx2(const CpuFineRunArgs& args) {
    return u8_rowmajor_numa_runner_impl::run_metadata<U8Avx2Kernel>(args);
}
long long u8_rowmajor_cluster_dynamic_numa_compact_metadata_avx2(const CpuFineRunArgs& args) {
    return u8_rowmajor_numa_runner_impl::run_compact_metadata<U8Avx2Kernel>(args);
}

long long u8_paged_cluster_dynamic_scratch_avx2(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_scratch<uint8_t, U8Avx2Kernel>(args);
}
long long u8_paged_cluster_dynamic_metadata_avx2(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_metadata<uint8_t, U8Avx2Kernel>(args);
}
long long u8_paged_cluster_dynamic_compact_metadata_avx2(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_compact_metadata<uint8_t, U8Avx2Kernel>(args);
}

}  // namespace cpu_fine
}  // namespace paraselect
