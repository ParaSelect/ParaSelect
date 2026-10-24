#ifndef PARASELECT_RESIDUAL_PQ_GPU_CUH
#define PARASELECT_RESIDUAL_PQ_GPU_CUH

#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_workspace.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_combos.h"

void residual_pq_reduce_candidates_gpu(
    const int* d_probe_candidate_idx,
    const float* d_probe_candidate_dist,
    int n_query,
    int n_probes,
    int local_k,
    int global_k,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream);

void residual_pq_pad_queries_gpu(
    const float* d_queries,
    int n_query,
    int pq_m,
    int d_sub,
    int d_sub_padded,
    float* d_queries_padded,
    cudaStream_t stream);

// Production fused path: build ADC LUT, scan PQ codes, and reduce to candidates.
void residual_pq_search_gpu(
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
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing = nullptr,
    paraselect::search::ResidualPQGroupedWorkspace* grouped_workspace = nullptr);

#endif // PARASELECT_RESIDUAL_PQ_GPU_CUH
