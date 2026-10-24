#ifndef PARASELECT_IVFPQ_SEARCH_CONTEXT_H
#define PARASELECT_IVFPQ_SEARCH_CONTEXT_H

#include "search/ivf_common/ivf_search_types.h"
#include "search/ivf_common/ivf_search_stats_collector.h"
#include "search/ivf_common/ivf_storage.h"
#include "search/ivfrespq/stage/residual_pq_gpu.cuh"
#include "search/kernels/fine/fine_search_strategy.h"

#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <vector>
#include <string>

using namespace paraselect::search;

struct IVFPQStorageView {
    const float* rerank_fp32 = nullptr;
    const uint8_t* rerank_u8 = nullptr;
    IVFPQCodeStorageView pq;
};

struct IVFPQIndexView {
    IVFCoarseIndexView coarse;
    IVFListLayoutView lists;
    IVFPQStorageView storage;
};

struct IVFPQSearchOptions {
    const float* transformed_queries = nullptr;
    int n_rerank = 0;
    int cpu_rerank_threads = 0;
    bool pin_rerank_threads = false;
    int rerank_prefetch = 4;
    // Sort retained IDs for CPU locality; does not change candidate selection.
    bool order_candidate_ids = false;
    IVFExactRerankDType rerank_dtype = IVFExactRerankDType::Disabled;
    bool resident_codes = false;
    std::string resident_combo;
    paraselect::search::PQAdcSearchStrategy adc_strategy =
        paraselect::search::PQAdcSearchStrategy::Dense;
    paraselect::search::PQAdcThresholdParams threshold;
};

struct IVFPQCoarseContext {
    std::unique_ptr<paraselect::coarse::Workspace> coarse_workspace;
    int enabled = 0;
    int query_batch_size = 0;
    int coarse_cluster_tile_size = 0;
    float* d_cluster_centers = nullptr;
    __half* d_cluster_centers_fp16 = nullptr;
    float* d_cluster_centers_norm = nullptr;
    float* d_query_batch = nullptr;
    __half* d_query_fp16 = nullptr;
    float* d_query_norm = nullptr;
    float* d_top_nprobe_dist = nullptr;
    idx_t* d_top_nprobe_index = nullptr;
    idx_t* d_top_nprobe_batch = nullptr;
    int d_top_nprobe_batch_valid = 0;
    cudaStream_t stream = 0;
    cublasHandle_t cublas_handle = nullptr;
};

struct IVFPQResidentCodeContext {
    int enabled = 0;
    int storage_prepared = 0;
    int query_prepared = 0;
    int max_batch_n = 0;
    char combo_name[64] = {0};
    float* d_queries = nullptr;
    idx_t* d_top_nprobe = nullptr;
    uint8_t* d_pq_codes = nullptr;
    long long* d_cluster_offsets = nullptr;
    int* d_cluster_counts = nullptr;
    float* d_pq_codebook = nullptr;
    int* d_adc_candidate_idx = nullptr;
    float* d_adc_candidate_dist = nullptr;
    paraselect::search::ResidualPQGpuSearchSpec adc_spec;
    paraselect::search::ResidualPQGroupedWorkspace adc_workspace;
    paraselect::search::PQAdcThresholdWorkspace threshold_workspace;
    int local_k = 0;
    cudaStream_t stream = nullptr;
};

struct IVFPQSearchContext {
    IVFPQIndexView index;
    IVFPQSearchOptions options;
    IVFQueryBatchView queries;
    IVFSearchParams params;
    IVFSearchOutputView output;
    paraselect::search::IVFSearchStatsCollector* collector = nullptr;
    int query_batch_size = 0;
    std::vector<long long> cluster_offsets;
    std::vector<int> cluster_counts;
    std::vector<idx_t> top_nprobe_batch;
    std::vector<int> topk_local_idx_batch;
    std::vector<int> rerank_candidate_idx_batch;
    std::vector<int> rerank_candidate_count_batch;
    IVFPQCoarseContext coarse;
    int pq_stage1_k = 0;
    int pq_rerank_enabled = 0;
    IVFPQResidentCodeContext resident_codes;
    long long pq_scanned_codes_total = 0;
    long long pq_adc_lookups_total = 0;
    long long threshold_candidate_total = 0;
    int threshold_query_total = 0;
    int threshold_candidate_max = 0;
};

#endif  // PARASELECT_IVFPQ_SEARCH_CONTEXT_H
