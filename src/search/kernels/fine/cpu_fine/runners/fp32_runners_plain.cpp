#include "search/kernels/fine/cpu_fine/runners/fp32_isa_runners.h"

#include "search/kernels/fine/cpu_fine/runners/paged_runner_impl.h"
#include "search/kernels/fine/cpu_fine/kernels/kernel_fp32_plain.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_numa_runner_impl.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_runner_impl.h"

namespace paraselect {
namespace cpu_fine {

long long fp32_rowmajor_cluster_dynamic_scratch_plain(const CpuFineRunArgs& args) {
    return fp32_rowmajor_runner_impl::run_scratch<Fp32PlainKernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_metadata_plain(const CpuFineRunArgs& args) {
    return fp32_rowmajor_runner_impl::run_metadata<Fp32PlainKernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_compact_metadata_plain(const CpuFineRunArgs& args) {
    return fp32_rowmajor_runner_impl::run_compact_metadata<Fp32PlainKernel>(args);
}

long long fp32_rowmajor_cluster_dynamic_numa_scratch_plain(const CpuFineRunArgs& args) {
    return fp32_rowmajor_numa_runner_impl::run_scratch<Fp32PlainKernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_numa_metadata_plain(const CpuFineRunArgs& args) {
    return fp32_rowmajor_numa_runner_impl::run_metadata<Fp32PlainKernel>(args);
}
long long fp32_rowmajor_cluster_dynamic_numa_compact_metadata_plain(const CpuFineRunArgs& args) {
    return fp32_rowmajor_numa_runner_impl::run_compact_metadata<Fp32PlainKernel>(args);
}

long long fp32_paged_cluster_dynamic_scratch_plain(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_scratch<float, Fp32PlainKernel>(args);
}
long long fp32_paged_cluster_dynamic_metadata_plain(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_metadata<float, Fp32PlainKernel>(args);
}
long long fp32_paged_cluster_dynamic_compact_metadata_plain(const CpuFineRunArgs& args) {
    return paged_runner_impl::run_compact_metadata<float, Fp32PlainKernel>(args);
}
}  // namespace cpu_fine
}  // namespace paraselect
