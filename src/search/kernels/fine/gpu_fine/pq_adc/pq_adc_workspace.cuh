#ifndef PARASELECT_PQ_ADC_WORKSPACE_CUH
#define PARASELECT_PQ_ADC_WORKSPACE_CUH

#include "utils/pch.h"
#include "search/kernels/fine/fine_search_strategy.h"
#include "search/ivfrespq/stage/residual_pq_workspace_budget.h"

#include <cuda_runtime.h>

struct ResidualPQGpuDebugTiming {
    float query_prepare_ms = 0.0f;
    float seed_count_ms = 0.0f;
    float candidate_readback_ms = 0.0f;
    float query_lut_build_ms = 0.0f;
    float lut_build_ms = 0.0f;
    float adc_scan_ms = 0.0f;
    float global_reduce_ms = 0.0f;
    float probe_topk_ms = 0.0f;
    float desc_ms = 0.0f;
    float group_merge_ms = 0.0f;
    float final_merge_ms = 0.0f;
    float candidate_desc_ms = 0.0f;
    float pq_scan_ms = 0.0f;
    float candidate_reduce_ms = 0.0f;
    float candidate_cap_select_ms = 0.0f;
    float final_copy_ms = 0.0f;
    // Threshold-path detail. Legacy aggregate fields above remain populated.
    float seed_compact_ms = 0.0f;
    float seed_adc_ms = 0.0f;
    float threshold_policy_ms = 0.0f;
    float scan_setup_ms = 0.0f;
    float scan_kernel_ms = 0.0f;
};

namespace paraselect::search {

struct ResidualPQChunkDesc {
    int query = 0;
    int cluster = 0;
    int start = 0;
    int length = 0;
};

struct ResidualPQGroupDesc {
    int query = 0;
    int source_start = 0;
    int source_count = 0;
};

struct ResidualPQGroupedWorkspace {
    int* d_pair_counts = nullptr;
    int* d_pair_offsets = nullptr;
    int* d_query_counts = nullptr;
    int* d_query_starts = nullptr;
    int* d_next_counts = nullptr;
    int* d_next_starts = nullptr;
    ResidualPQChunkDesc* d_chunk_desc = nullptr;
    ResidualPQGroupDesc* d_group_desc = nullptr;
    float* d_merge_a_dist = nullptr;
    int* d_merge_a_idx = nullptr;
    float* d_merge_b_dist = nullptr;
    int* d_merge_b_idx = nullptr;
    void* d_scan_temp = nullptr;
    int* h_query_chunks = nullptr;
    unsigned long long* d_scanned_codes = nullptr;
    unsigned long long* h_scanned_codes = nullptr;
    unsigned long long scanned_codes = 0;
    size_t scan_temp_bytes = 0;
    int pair_capacity = 0;
    int query_capacity = 0;
    int chunk_capacity = 0;
    int group_capacity = 0;
    int merge_a_item_capacity = 0;
    int merge_b_item_capacity = 0;
    int host_query_capacity = 0;
};

struct PQAdcThresholdWorkspace {
    // Reused event pairs: seed, compact, policy, scan and cap-select boundaries.
    cudaEvent_t timing_events[10]{};
    unsigned long long* h_seed_scanned_codes = nullptr;
    unsigned long long* d_scanned_codes = nullptr;
    unsigned long long* h_scanned_codes = nullptr;
    // Optional audit counter: all threshold passes, before global-pool overflow.
    unsigned long long* d_accepted_count = nullptr;
    unsigned long long* d_physical_code_rows = nullptr;
    idx_t* d_seed_top_nprobe = nullptr;
    // Global threshold pool. The pool is grouped and cap-selected after scan.
    int* d_pool_idx = nullptr;
    float* d_pool_dist = nullptr;
    int* d_pool_query = nullptr;
    int* d_global_count = nullptr;
    int* d_candidate_idx = nullptr;
    float* d_candidate_dist = nullptr;
    int* d_candidate_count = nullptr;
    int* d_overflow = nullptr;
    float* d_threshold = nullptr;
    // Query-major raw ADC distance histogram used by threshold seeding.
    // The producer is optional and its capacity grows on demand.
    unsigned int* d_seed_histogram = nullptr;
    uint64_t* d_sort_keys_in = nullptr;
    uint64_t* d_sort_keys_out = nullptr;
    int* d_sort_values_out = nullptr;
    int* d_offsets = nullptr;
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    int max_queries = 0;
    int max_seed_probes = 0;
    int candidate_capacity = 0;
    int pool_capacity = 0;
    int histogram_bins = 0;
    unsigned long long scanned_codes = 0;
    unsigned long long physical_code_rows = 0;
};

void pq_count_scanned_codes_async(
    const idx_t* d_top_nprobe,
    const int* d_cluster_counts,
    int n_query,
    int n_probes,
    cudaStream_t stream,
    unsigned long long* device_count, unsigned long long* host_count);

void pq_adc_threshold_workspace_reserve(
    PQAdcThresholdWorkspace* ws,
    int max_queries,
    int candidate_capacity,
    int seed_probes);
std::size_t pq_adc_threshold_workspace_required_bytes(
    int max_queries,
    int candidate_capacity,
    int seed_probes);
void pq_adc_threshold_workspace_release(PQAdcThresholdWorkspace* ws);

void residual_pq_grouped_workspace_reserve(
    paraselect::search::ResidualPQGroupedWorkspace* ws,
    int fine_query_tile,
    int nprobe,
    int max_chunks_per_probe,
    int group_chunks,
    int global_k,
    std::size_t reserved_after_grouped_bytes = 0);

void residual_pq_grouped_workspace_release(
    paraselect::search::ResidualPQGroupedWorkspace* ws);

void residual_pq_grouped_chunk_search_gpu(
    paraselect::search::PQAdcDistanceMode distance_mode,
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
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    paraselect::search::ResidualPQGroupedWorkspace* ws);

}  // namespace paraselect::search

#endif  // PARASELECT_PQ_ADC_WORKSPACE_CUH
