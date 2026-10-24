#ifndef PARASELECT_IVFFLAT_PROBE_METADATA_CUH
#define PARASELECT_IVFFLAT_PROBE_METADATA_CUH

#include "search/ivfflat/ivfflat_search_ctx.h"
#include "search/ivfflat/stage/ivfflat_stages.cuh"

#include <cuda_runtime.h>

namespace paraselect {
namespace cpu_fine {

struct CpuProbeMetadataBuffers {
    int* h_cluster_query_offset = nullptr;
    int* h_cluster_query_data = nullptr;
    int* h_active_cluster_ids = nullptr;
};

struct CpuProbeMetadataResult {
    int active_cluster_count = 0;
    double d2h_ms = 0.0;
};

CpuProbeMetadataResult build_cpu_probe_metadata(
    const idx_t* top_nprobe_batch,
    int n_query,
    int n_probes,
    int n_total_clusters,
    bool use_compact_metadata,
    int* h_cluster_query_offset,
    int* h_cluster_query_data,
    int* h_active_cluster_ids);

CpuProbeMetadataResult transfer_cpu_probe_metadata(
    IVFSearchCtx* ctx,
    int pool_slot,
    int n_query,
    cudaStream_t stream,
    CpuProbeMetadataBuffers buffers,
    const char* error_prefix);

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_IVFFLAT_PROBE_METADATA_CUH
