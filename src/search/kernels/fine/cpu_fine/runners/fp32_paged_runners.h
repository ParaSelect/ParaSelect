#ifndef PARASELECT_FP32_PAGED_RUNNERS_H
#define PARASELECT_FP32_PAGED_RUNNERS_H

#include "search/kernels/fine/cpu_fine/types/cpu_fine_run.h"

namespace paraselect {
namespace cpu_fine {

long long fp32_paged_cluster_dynamic_scratch(const CpuFineRunArgs& args);

long long fp32_paged_cluster_dynamic_metadata(const CpuFineRunArgs& args);

long long fp32_paged_cluster_dynamic_compact_metadata(const CpuFineRunArgs& args);



}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_FP32_PAGED_RUNNERS_H
