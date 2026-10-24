#pragma once
#include "utils/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdint>

namespace raft { class resources; }
namespace paraselect::coarse {
class Workspace;
struct CoarseHistogramTiming {
    double estimate_ms = 0.0;
    double compact_ms = 0.0;
    double refine_ms = 0.0;
    double sort_ms = 0.0;
    double gather_ms = 0.0;
};
struct Request {
    bool fp16_coarse = false;
    int distance_mode = 0;
    int n_query = 0;
    int dim = 0;
    int nlist = 0;
    int nprobe = 0;
    int cluster_tile_size = 0;
    // Incremented by an owner that mutates the center buffer in place.  Zero
    // keeps pointer identity as the compatibility key for standalone callers.
    std::uint64_t center_version = 0;

    const float* d_query = nullptr;
    const __half* d_query_fp16 = nullptr;
    const float* d_query_norm = nullptr;
    const float* d_cluster_centers = nullptr;
    const __half* d_cluster_centers_fp16 = nullptr;
    const float* d_cluster_centers_norm = nullptr;

    float* d_top_nprobe_dist = nullptr;
    idx_t* d_top_nprobe_index = nullptr;

    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = 0;
    raft::resources* cuvs_res = nullptr;
};

struct Timing {
    int gemm_fused_calls = 0;
    int materialized_calls = 0;
    int materialized_query_tiles = 0;
    double init_topk_ms = 0.0;
    double tile_fill_ms = 0.0;
    double gemm_ms = 0.0;
    // Prepared fused/PC execution reports its indivisible total here.
    double topk_ms = 0.0;
    double merge_ms = 0.0;
    CoarseHistogramTiming histogram;
};

// Selects for the actual shape and executes with reusable scratch.
// Optional timing synchronizes this stream.
void search(const Request&, Workspace&, Timing* = nullptr);
}
