#ifndef PARASELECT_U8_ROWMAJOR_NUMA_RUNNERS_H
#define PARASELECT_U8_ROWMAJOR_NUMA_RUNNERS_H

#include "search/kernels/fine/cpu_fine/types/cpu_fine_run.h"

namespace paraselect {
namespace cpu_fine {

long long u8_rowmajor_cluster_dynamic_numa_scratch(const CpuFineRunArgs& args);

long long u8_rowmajor_cluster_dynamic_numa_metadata(const CpuFineRunArgs& args);

long long u8_rowmajor_cluster_dynamic_numa_compact_metadata(const CpuFineRunArgs& args);


}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_U8_ROWMAJOR_NUMA_RUNNERS_H
