#include "search/kernels/fine/cpu_fine/runners/fp32_isa_runners.h"

#include "search/kernels/fine/cpu_fine/runners/paged_runner_impl.h"
#include "search/kernels/fine/cpu_fine/kernels/kernel_fp32_avx512.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_numa_runner_impl.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_runner_impl.h"

namespace paraselect {
namespace cpu_fine {

long long fp32_rowmajor_cluster_dynamic_scratch_avx512(const CpuFineRunArgs& args) {
    return fp32_rowmajor_runner_impl::run_scratch<Fp32Avx512Kernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_metadata_avx512(const CpuFineRunArgs& args) {
    return fp32_rowmajor_runner_impl::run_metadata<Fp32Avx512Kernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_compact_metadata_avx512(const CpuFineRunArgs& args) {
    return fp32_rowmajor_runner_impl::run_compact_metadata<Fp32Avx512Kernel>(args);
}

long long fp32_rowmajor_cluster_dynamic_numa_scratch_avx512(const CpuFineRunArgs& args) {
    return fp32_rowmajor_numa_runner_impl::run_scratch<Fp32Avx512Kernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_numa_metadata_avx512(const CpuFineRunArgs& args) {
    return fp32_rowmajor_numa_runner_impl::run_metadata<Fp32Avx512Kernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_numa_compact_metadata_avx512(const CpuFineRunArgs& args) {
    return fp32_rowmajor_numa_runner_impl::run_compact_metadata<Fp32Avx512Kernel>(args);
}

long long fp32_paged_cluster_dynamic_scratch_avx512(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_scratch<float, Fp32Avx512Kernel>(args);
}
long long fp32_paged_cluster_dynamic_metadata_avx512(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_metadata<float, Fp32Avx512Kernel>(args);
}
long long fp32_paged_cluster_dynamic_compact_metadata_avx512(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_compact_metadata<float, Fp32Avx512Kernel>(args);
}
}  // namespace cpu_fine
}  // namespace paraselect
