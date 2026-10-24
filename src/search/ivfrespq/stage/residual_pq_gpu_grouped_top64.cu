#include "search/ivfrespq/stage/residual_pq_gpu_grouped_impl.cuh"

#include <stdexcept>

namespace paraselect::search {

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
    ResidualPQGroupedWorkspace* ws)
{
    if (spec.global_k != 64) {
        throw std::invalid_argument("residual_pq_grouped_chunk_search_top64_gpu: invalid global_k");
    }
    residual_pq_grouped_chunk_search_impl_gpu(
        distance_mode,
        d_queries,
        d_centroids,
        d_top_nprobe,
        d_pq_codes,
        d_cluster_offsets,
        d_cluster_counts,
        d_codebook,
        n_query,
        dim,
        n_total_clusters,
        n_probes,
        spec,
        d_global_candidate_idx,
        d_global_candidate_dist,
        stream,
        timing,
        ws);
}

}  // namespace paraselect::search
