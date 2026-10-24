#include "search/kernels/fine/cpu_fine/runners/fp32_paged_runners.h"

#include "search/kernels/fine/cpu_fine/kernels/kernel_fp32_select.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_isa_runners.h"
#include "search/kernels/fine/cpu_fine/runners/fp32_rowmajor_runner_impl.h"

namespace paraselect {
namespace cpu_fine {
namespace {

template <typename PlainFn, typename Avx2Fn, typename Avx512Fn>
long long dispatch_fp32_paged(
    const CpuFineRunArgs& args, PlainFn plain_fn, Avx2Fn avx2_fn, Avx512Fn avx512_fn) {
    switch (select_fp32_kernel_isa()) {
        case Fp32KernelIsa::Avx512: return avx512_fn(args);
        case Fp32KernelIsa::Avx2: return avx2_fn(args);
        case Fp32KernelIsa::Plain: return plain_fn(args);
    }
    return plain_fn(args);
}

}  // namespace

long long fp32_paged_cluster_dynamic_scratch(const CpuFineRunArgs& args) {
    return dispatch_fp32_paged(
        args, PARASELECT_FP32_BACKEND_ARGS(fp32_paged_cluster_dynamic_scratch_plain, fp32_paged_cluster_dynamic_scratch_avx2, fp32_paged_cluster_dynamic_scratch_avx512));
}

long long fp32_paged_cluster_dynamic_metadata(const CpuFineRunArgs& args) {
    return dispatch_fp32_paged(
        args, PARASELECT_FP32_BACKEND_ARGS(fp32_paged_cluster_dynamic_metadata_plain, fp32_paged_cluster_dynamic_metadata_avx2, fp32_paged_cluster_dynamic_metadata_avx512));
}

long long fp32_paged_cluster_dynamic_compact_metadata(const CpuFineRunArgs& args) {
    return dispatch_fp32_paged(
        args, PARASELECT_FP32_BACKEND_ARGS(fp32_paged_cluster_dynamic_compact_metadata_plain, fp32_paged_cluster_dynamic_compact_metadata_avx2, fp32_paged_cluster_dynamic_compact_metadata_avx512));
}

}  // namespace cpu_fine
}  // namespace paraselect
