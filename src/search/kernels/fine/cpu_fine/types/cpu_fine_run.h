#ifndef PARASELECT_CPU_FINE_RUN_H
#define PARASELECT_CPU_FINE_RUN_H

#include "utils/pch.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"
#include "search/kernels/fine/cpu_fine/types/cpu_ivf_storage.h"

namespace paraselect {
namespace cpu_fine {

using CpuFineQueryPrepareFn = const float* (*)(
    const CpuFineBatchView*,
    int,
    int);

struct CpuFineRunArgs {
    const CpuIvfStorageView* storage = nullptr;
    const float* query = nullptr;
    const idx_t* top_nprobe = nullptr;
    const CpuFineBatchView* batch = nullptr;
    int n_probes = 0;
    int topk = 0;
    int distance_mode = 0;
    int num_threads = 0;
    const int* numa_cluster_boundaries = nullptr;
    int numa_cluster_boundary_count = 0;
};

using CpuFineClusterDynamicFn = long long (*)(const CpuFineRunArgs&);

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_RUN_H
