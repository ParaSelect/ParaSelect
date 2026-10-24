#ifndef PARASELECT_IVFRESPQ_SEARCH_CTX_H
#define PARASELECT_IVFRESPQ_SEARCH_CTX_H

#include "search/ivf_common/quantized/residual_codec.h"
#include "transform/opq_transform.cuh"

#include "utils/pch.h"
#include "search/ivfrespq/ivfrespq_search.cuh"
#include "search/ivfrespq/ivfrespq_search_internal.h"
#include "search/ivfflat/ivfflat_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_pq_storage.h"
#include "search/ivf_common/file/ivf_file_reader.h"
#include "search/kernels/fine/fine_search_strategy.h"
#include "search/ivfrespq/stage/residual_pq_gpu.cuh"
#include "search/ivf_common/ivf_pca_refine.h"
#include "search/ivf_common/ivf_probe_schedule.cuh"
#include "search/ivf_common/ivf_search_stats_collector.h"
#include "search/kernels/coarse/runtime/coarse_runtime.h"

#include <memory>
#include <cstdint>
#include <vector>

struct IVFRESPQSearchCtx;
class IVFRESPQCpuChunkExecutor;
using IVFRESPQFineRunnerFn = void (*)(IVFRESPQSearchCtx*, int, int);

struct IVFRESPQCoarseCtx {
    const int* h_cluster_sizes = nullptr;
    const float* h_cluster_centers = nullptr;

    int query_batch_size = 0;
    int query_tile = 0;

    float* d_cluster_centers = nullptr;
    __half* d_cluster_centers_fp16 = nullptr;
    float* d_cluster_centers_norm = nullptr;

    float* d_query_batch = nullptr;
    __half* d_query_fp16 = nullptr;
    float* d_query_norm = nullptr;
    float* d_top_nprobe_dist = nullptr;
    idx_t* d_top_nprobe_index = nullptr;
    std::unique_ptr<paraselect::coarse::Workspace> coarse_workspace;

    cudaStream_t stream = 0;
    cublasHandle_t cublas_handle = nullptr;
};

// Immutable host-side assets shared by all fine-search backends.
// These pointers describe the index; query-batch buffers do not belong here.
struct IVFRESPQFineAssets {
    paraselect::transform::OPQTransform* residual_rotation = nullptr;
    const uint8_t* h_pq_codes = nullptr;
    paraselect::cpu_fine::CpuPqCodeStorageView pq_storage;
    const float* h_pq_codebook = nullptr;
    const float* h_base_fp32 = nullptr;
    const uint8_t* h_base_u8 = nullptr;
    IVFFileStorageView file_storage;
    std::shared_ptr<paraselect::search::IVFFileReader> file_reader;
    const float* h_queries_fp32 = nullptr;
    const uint8_t* h_queries_u8 = nullptr;
    const idx_t* h_reordered_indices = nullptr;
    const int* h_cluster_sizes = nullptr;
    const float* h_cluster_centers = nullptr;

    int pq_M = 0;
    int pq_K = 0;
    const paraselect::search::ResidualPQGpuCombo* rpq_gpu_combo = nullptr;
    paraselect::search::ResidualPQGpuSearchSpec residual_pq_gpu_spec;
    std::shared_ptr<paraselect::search::IVFPCAGpuRefiner> pca_gpu_refiner;
};

// Device allocations owned by the static fine-search context.
// Keeping ownership here makes release and re-prepare symmetrical.
struct IVFRESPQFineDeviceBuffers {
    struct RowCorrectionDeviceAllocation {
        float* fp32 = nullptr;
        uint16_t* u16 = nullptr;
    } query_lut_bias_device;

    float* d_pq_codebook = nullptr;
    __half* d_pq_codebook_fp16 = nullptr;
    float* d_pq_lut_batch = nullptr;
    float* d_query_codeword_dot_lut = nullptr;
    uint8_t* d_query_codeword_dot_lut_u8 = nullptr;
    float* d_query_lut_scale = nullptr;
    float* d_query_lut_lo_sum = nullptr;
    float* d_query_lut_inflation = nullptr;
    float* d_top_nprobe_dist_batch = nullptr;
    float* d_query_padded = nullptr;
    float* d_cluster_centers_padded = nullptr;
    idx_t* d_top_nprobe_batch = nullptr;
    paraselect::search::ResidualCodec* residual_codec = nullptr;
    uint8_t* d_pq_codes = nullptr;
    long long* d_cluster_offsets = nullptr;
    int* d_cluster_counts = nullptr;
    int rpq_total_physical_sources = 0;
    int rpq_max_sources_per_cluster = 1;
    int* d_probe_candidate_idx = nullptr;
    float* d_probe_candidate_dist = nullptr;
    int* d_rerank_candidate_idx = nullptr;
    float* d_rerank_candidate_dist = nullptr;
    int8_t* d_pca_vectors = nullptr;
    int* d_pca_candidate_idx = nullptr;
    const float* d_resident_interleaved_vectors = nullptr;
    const uint8_t* d_resident_interleaved_vectors_u8 = nullptr;
    const float* d_resident_interleaved_norms = nullptr;
    const idx_t* d_resident_interleaved_reordered_indices = nullptr;
    int* d_resident_compact_to_padded = nullptr;
    int* d_resident_topk_local_idx = nullptr;
    float* d_resident_topk_dist = nullptr;
    int use_resident_rerank = 0;
    char resident_combo_name[RESIDENT_FINE_COMBO_NAME_MAX_LEN] = {0};
    paraselect::search::ResidualPQGroupedWorkspace rpq_grouped_workspace;
    paraselect::search::PQAdcThresholdWorkspace rpq_threshold_workspace;
    std::shared_ptr<paraselect::search::IVFProbeScheduleBuilder> rpq_probe_schedule;
};

// Host-side staging buffers. They are separate from index assets because they
// are resized for each query batch.
struct IVFRESPQFineHostBuffers {
    long long* h_cluster_offsets = nullptr;
    idx_t* h_top_nprobe_batch = nullptr;
    int* h_topk_local_idx_batch = nullptr;
    int* h_rerank_candidate_idx = nullptr;
    int* h_shortlist_candidate_idx = nullptr;
    float* h_pq_lut_batch = nullptr;
    size_t pq_lut_batch_capacity = 0;
};

// Per-search policy and timing state. Values are resolved during prepare and
// then read by the stage runners; no data ownership is implied here.
struct IVFRESPQFineRuntime {
    int pq_rerank_n = 0;
    int pq_local_k = 0;
    int candidate_width = 0;
    int fixed_tile_rows = 512;
    bool order_threshold_ids = false;
    int pq_use_float_rerank = 0;
    int adaptive_probe_padding = 0;
    cudaEvent_t ev_rpq_lut_start = nullptr;
    cudaEvent_t ev_rpq_lut_stop = nullptr;
    cudaEvent_t ev_rpq_schedule_start = nullptr;
    cudaEvent_t ev_rpq_schedule_stop = nullptr;
    cudaEvent_t ev_rpq_tile_stop = nullptr;
    cudaEvent_t ev_rpq_gpu_start = nullptr;
    cudaEvent_t ev_rpq_gpu_stop = nullptr;
    cudaEvent_t ev_rpq_d2h_start = nullptr;
    cudaEvent_t ev_rpq_d2h_stop = nullptr;
    int d_query_batch_valid = 0;
    int d_top_nprobe_batch_valid = 0;
    int d_top_nprobe_dist_batch_valid = 0;
    int h_top_nprobe_batch_valid = 0;
    int use_gpu_lut = 0;
    int use_residual_pq_gpu = 0;
    int consume_probe_on_gpu = 0;
    int reuse_query_on_gpu = 0;
    int cpu_fine_num_threads = 0;
    paraselect::search::IVFResPQCpuConfig cpu_config;
    int phase_log_enabled = 0;
    int rpq_kernel_breakdown = 0;
    int rpq_rerank_breakdown = 0;
    IVFRESPQFineRunnerFn fine_runner = nullptr;
};

// Static ResPQ fine-search state. The base classes intentionally preserve the
// existing fine.<field> access in kernels and stage code while making ownership
// boundaries explicit for future fine-search implementations.
struct IVFRESPQFineCtx {
    IVFRESPQFineAssets assets;
    IVFRESPQFineDeviceBuffers device;
    IVFRESPQFineHostBuffers host;
    IVFRESPQFineRuntime runtime;
};

struct IVFRESPQSearchCtx {
    std::shared_ptr<IVFRESPQCpuChunkExecutor> cpu_executor;
    std::unique_ptr<paraselect::transform::OPQTransform> residual_rotation_owner;
    std::unique_ptr<paraselect::search::ResidualCodec> residual_codec_owner;
    IVFResSearchState arg = {};
    IVFRESPQCoarseCtx coarse = {};
    IVFRESPQFineCtx fine = {};
    paraselect::search::IVFSearchStatsCollector* collector = nullptr;
    long long pq_scanned_codes_total = 0;
    long long pq_adc_lookups_total = 0;
    long long pq_physical_code_rows_total = 0;
    long long threshold_candidate_total = 0;
    long long threshold_query_total = 0;
    int threshold_candidate_max = 0;
    std::vector<long long> file_block_candidate_ref_histogram;
    long long file_blocks_considered_total = 0;
    long long file_blocks_skipped_total = 0;
    std::vector<int> cluster_counts;
    std::vector<uint8_t> owned_query_u8;
    std::vector<std::uint16_t> owned_query_fp16;
    std::vector<uint8_t> owned_base_u8;
};

#endif  // PARASELECT_IVFRESPQ_SEARCH_CTX_H
