#ifndef PARASELECT_RESIDUAL_PQ_GPU_GROUPED_IMPL_CUH
#define PARASELECT_RESIDUAL_PQ_GPU_GROUPED_IMPL_CUH

#include "search/ivfrespq/stage/residual_pq_gpu.cuh"

namespace paraselect::search {

void residual_pq_grouped_chunk_search_impl_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws);

void residual_pq_grouped_chunk_search_top512_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws);

void residual_pq_grouped_chunk_search_top64_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws);

}  // namespace paraselect::search

#endif  // PARASELECT_RESIDUAL_PQ_GPU_GROUPED_IMPL_CUH
