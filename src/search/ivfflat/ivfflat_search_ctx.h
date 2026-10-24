/**
 * IVF : cuda / host /  ctx  Arg.
 *  CPU_FP32_FINE_SEARCH .
 */
#ifndef PARASELECT_IVF_SEARCH_CTX_H
#define PARASELECT_IVF_SEARCH_CTX_H

#include "utils/pch.h"
#include "search/ivf_common/ivf_search_types.h"
#include "search/kernels/coarse/coarse_policy.h"
#ifdef __cplusplus
#include "search/kernels/fine/cpu_fine/types/cpu_ivf_storage.h"
#include "search/kernels/fine/cpu_fine/types/cpu_fine_run.h"
#endif
#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include <cublas_v2.h>
#include <cstdlib>
#include <cstring>

using paraselect::search::IVFFlatFineKind;
using paraselect::search::IVFFlatThresholdParams;

// Forward declaration for cuVS resource
namespace raft {
class resources;
}
#ifdef __cplusplus
namespace paraselect {
namespace cpu_fine {
struct CpuFineKernelCombo;
}
} // namespace paraselect
#endif

// Helper to delete cuvs_res without full type (defined in ivfflat_search.cu)
extern void ivfflat_search_delete_cuvs_res(raft::resources* res);

#ifdef __cplusplus
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CPU_FINE_COMBO_NAME_MAX_LEN 64
#define RESIDENT_FINE_COMBO_NAME_MAX_LEN 64

typedef enum CpuFineClusterMetadataMode {
    CPU_FINE_CLUSTER_METADATA_SCRATCH = 0,
    CPU_FINE_CLUSTER_METADATA_FULL = 1,
    CPU_FINE_CLUSTER_METADATA_COMPACT = 2
} CpuFineClusterMetadataMode;

/** ivf_search / */
typedef struct IVFSearchArg {
    int n_query;
    int query_batch_size;
    int n_dim;
    int n_total_clusters;
    int n_total_vectors;
    int n_probes;
    int k;
    int distance_mode; /* DistanceType:L2_DISTANCE / COSINE_DISTANCE */
    char cpu_fine_combo_name[CPU_FINE_COMBO_NAME_MAX_LEN];
    char resident_combo_name[RESIDENT_FINE_COMBO_NAME_MAX_LEN];
    int resident_chunk_blocks;
    int max_probe_slots;
    bool fp16_coarse;
    IVFFlatFineKind fine_kind;
    IVFFlatThresholdParams threshold;
    bool cpu_batch_pipeline_overlap;
} IVFSearchArg;

/** IVFFlat : coarse  */
typedef struct IVFFlatCoarseCtx {
    /* host inputs/outputs */
    const int* h_cluster_sizes;
    const float* h_cluster_centers;
    const float* h_queries;
    const uint8_t* h_queries_u8;
    idx_t** h_coarse_index;
    float** h_coarse_dist;
    float* h_topk_dist_out;
    idx_t* h_topk_index_out;
    int current_n_querys;
    int coarse_sub_batch_size;

    /* device + streams */
    int* d_cluster_size;
    float* d_cluster_centers;
    __half* d_cluster_centers_fp16;
    int* d_probe_vector_offset;
    int* d_probe_vector_offset_compact;
    float* d_cluster_centers_norm;
    int* d_cluster_query_count[2];
    int* d_cluster_write_pos[2];
    int* d_pool_cluster_offset[2];
    // Persistent scan scratch so metadata can run without cudaMalloc/cudaFree
    // (which would device-synchronize and kill pipeline overlap).
    int* d_scan_block_sums;
    // Separate scratch for the fine stage's per-batch prefix scans (entry/seed
    // builds); metadata may run concurrently with fine in the overlapped path.
    int* d_fine_scan_block_sums;
    int* d_cluster_query_data[2];
    int* d_cluster_query_probe_indices[2];
    int* d_active_cluster_ids[2];
    int* d_active_cluster_count[2];
    int* d_cluster_to_active_pos[2];
    int* d_compact_cluster_query_count[2];
    int* d_compact_cluster_query_offset[2];
    int* d_compact_cluster_write_pos[2];
    int* d_compact_cluster_query_data[2];
    idx_t* d_pool_topk_tmp[2];
    float* d_pool_topk_dist[2];
    idx_t* d_pool_topk_index[2];
    float* d_pool_query_batch[2];
    uint8_t* d_pool_query_batch_u8[2];
    __half* d_pool_query_fp16[2];
    float* d_pool_query_norm[2];
    float* d_pool_top_nprobe_dist[2];
    idx_t* d_top_nprobe_index[2];
    int coarse_cluster_tile_size;
    paraselect::coarse::Workspace* coarse_workspace[2];
    int n_padded_cache;
    cublasHandle_t cublas_handle;
    cudaStream_t stream_0;
    cudaStream_t stream_1;
    cudaEvent_t ev_coarse_done[2];
    cudaEvent_t ev_fine_done;
    double debug_coarse_gemm_ms;
    double debug_coarse_topk_ms;
    raft::resources* cuvs_res;
} IVFFlatCoarseCtx;

/** IVFFlat : CPU fine  */
typedef struct IVFFlatFineCtx {
    const int* h_cluster_sizes;
    const float* h_queries;
    const uint8_t* h_queries_u8;
    float* h_topk_dist_out;
    idx_t* h_topk_index_out;
    const float* h_cluster_vectors;
    const float* h_cluster_vectors_aosoa;
    const uint8_t* h_cluster_vectors_u8;
    const idx_t* h_reordered_indices;
    const float* d_resident_interleaved_vectors;
    const uint8_t* d_resident_interleaved_vectors_u8;
    const float* d_resident_interleaved_norms;
    const idx_t* d_resident_interleaved_reordered_indices;
    int resident_interleaved_enabled;
    int resident_interleaved_u8;
    char resident_combo_name[RESIDENT_FINE_COMBO_NAME_MAX_LEN];
    float* d_resident_probe_topk_dist[2];
    idx_t* d_resident_probe_topk_index[2];
    float* d_resident_final_topk_dist[2];
    idx_t* d_resident_final_topk_index[2];
    int* d_resident_entry_count_per_cluster[2];
    int* d_resident_entry_offset[2];
    int* d_resident_entry_cluster_id[2];
    int* d_resident_entry_query_start[2];
    int* d_resident_entry_query_count[2];
    int* d_resident_entry_queries[2];
    int* d_resident_entry_probe_indices[2];
    int* d_resident_threshold_seed_entry_count_per_cluster[2];
    int* d_resident_threshold_seed_entry_offset[2];
    int* d_resident_threshold_seed_entry_cluster_id[2];
    int* d_resident_threshold_seed_entry_query_start[2];
    int* d_resident_threshold_seed_entry_query_count[2];
    int* d_resident_threshold_seed_entry_queries[2];
    int* d_resident_threshold_seed_entry_probe_indices[2];
    int* d_resident_chunk_task_count[2];
    int* d_resident_row_chunk_count[2];
    int* d_resident_row_chunk_offset[2];
    int* d_resident_row_chunk_write_pos[2];
    int* d_resident_chunk_task_row[2];
    int* d_resident_chunk_task_cluster[2];
    int* d_resident_chunk_task_first_block[2];
    int* d_resident_chunk_task_num_blocks[2];
    float* d_resident_chunk_candidate_dist[2];
    idx_t* d_resident_chunk_candidate_index[2];
    float* d_resident_threshold[2];
    int* d_resident_threshold_candidate_count[2];
    int* d_resident_threshold_overflow[2];
    float* d_resident_threshold_candidate_dist[2];
    idx_t* d_resident_threshold_candidate_index[2];
    int resident_threshold_seed_probes;
    int resident_threshold_candidate_capacity;
    int resident_chunk_blocks;
    int resident_max_chunk_tasks;
    const float* h_cluster_centers;
    float* h_cluster_norm;
    const long long* h_cluster_offsets_aosoa_ll;
    int h_cluster_aosoa_tile;

    int* h_cluster_offsets;
    idx_t* h_top_nprobe_batch;
    int* h_topk_local_idx_batch;
    int* h_cluster_query_offset_batch;
    int* h_cluster_query_data_batch;
    int* h_active_cluster_ids_batch;
    int h_active_cluster_count_batch;
    int* h_numa_cluster_boundaries;
    int h_numa_cluster_boundary_count;
    int h_use_compact_metadata;
    long long* h_cluster_offsets_ll;
    uint8_t* h_query_u8_tmp;
    float* h_thread_dists;
    int* h_thread_idxs;
    int* h_thread_sizes;
    uint8_t* h_thread_touched;
    size_t h_thread_topk_capacity;
    size_t h_thread_query_capacity;
#ifdef __cplusplus
    const paraselect::cpu_fine::CpuFineKernelCombo* cached_combo;
    paraselect::cpu_fine::CpuIvfStorageView cached_storage_view;
    paraselect::cpu_fine::CpuFineQueryPrepareFn cached_query_prepare;
    paraselect::cpu_fine::CpuFineClusterDynamicFn cached_cluster_runner;
#else
    const void* cached_combo;
    void* cached_query_prepare;
    void* cached_cluster_runner;
#endif
    int cached_cluster_metadata_mode;
    double debug_resident_entry_build_ms;
    double debug_resident_probe_init_ms;
    double debug_resident_scan_ms;
    double debug_resident_merge_map_ms;
    double debug_resident_threshold_seed_build_ms;
    double debug_resident_threshold_sketch_ms;
    double debug_resident_threshold_scan_ms;
    double debug_resident_threshold_reduce_ms;
    double debug_resident_threshold_avg_candidates;
    double debug_resident_threshold_max_candidates;
    double debug_resident_d2h_ms;
} IVFFlatFineCtx;

/** IVFPQ (,) */
typedef struct IVFPQFineCtx {
    const uint8_t* h_pq_codes;
    const float* h_pq_codebook;
    const float* h_base_fp32;
    const float* h_queries_fp32;
    int pq_M;
    int pq_K;
    int pq_rerank_n;
    int pq_use_float_rerank;
} IVFPQFineCtx;

/**  */
typedef struct IVFSearchCtx {
    IVFSearchArg arg;
    IVFFlatCoarseCtx flat_coarse;
    IVFFlatFineCtx flat_fine;
    IVFPQFineCtx pq_fine;
} IVFSearchCtx;

inline void ivf_search_ctx_zero(IVFSearchCtx* ctx) {
    if (!ctx) return;
    std::memset(ctx, 0, sizeof(IVFSearchCtx));
}

inline void ivf_search_flat_coarse_free_all(IVFFlatCoarseCtx* ctx) {
    if (!ctx) return;
#define FREE_DEVICE(p)          \
    do {                        \
        if ((ctx)->p) {         \
            cudaFree((ctx)->p); \
            (ctx)->p = nullptr; \
        }                       \
    } while (0)
    FREE_DEVICE(d_cluster_size);
    FREE_DEVICE(d_cluster_centers);
    FREE_DEVICE(d_cluster_centers_fp16);
    FREE_DEVICE(d_probe_vector_offset);
    FREE_DEVICE(d_probe_vector_offset_compact);
    FREE_DEVICE(d_cluster_centers_norm);
    FREE_DEVICE(d_cluster_query_count[0]);
    FREE_DEVICE(d_cluster_query_count[1]);
    FREE_DEVICE(d_cluster_write_pos[0]);
    FREE_DEVICE(d_cluster_write_pos[1]);
    FREE_DEVICE(d_pool_cluster_offset[0]);
    FREE_DEVICE(d_pool_cluster_offset[1]);
    FREE_DEVICE(d_scan_block_sums);
    FREE_DEVICE(d_fine_scan_block_sums);
    FREE_DEVICE(d_cluster_query_data[0]);
    FREE_DEVICE(d_cluster_query_data[1]);
    FREE_DEVICE(d_cluster_query_probe_indices[0]);
    FREE_DEVICE(d_cluster_query_probe_indices[1]);
    FREE_DEVICE(d_active_cluster_ids[0]);
    FREE_DEVICE(d_active_cluster_ids[1]);
    FREE_DEVICE(d_active_cluster_count[0]);
    FREE_DEVICE(d_active_cluster_count[1]);
    FREE_DEVICE(d_cluster_to_active_pos[0]);
    FREE_DEVICE(d_cluster_to_active_pos[1]);
    FREE_DEVICE(d_compact_cluster_query_count[0]);
    FREE_DEVICE(d_compact_cluster_query_count[1]);
    FREE_DEVICE(d_compact_cluster_query_offset[0]);
    FREE_DEVICE(d_compact_cluster_query_offset[1]);
    FREE_DEVICE(d_compact_cluster_write_pos[0]);
    FREE_DEVICE(d_compact_cluster_write_pos[1]);
    FREE_DEVICE(d_compact_cluster_query_data[0]);
    FREE_DEVICE(d_compact_cluster_query_data[1]);
    FREE_DEVICE(d_pool_topk_tmp[0]);
    FREE_DEVICE(d_pool_topk_tmp[1]);
    FREE_DEVICE(d_pool_topk_dist[0]);
    FREE_DEVICE(d_pool_topk_dist[1]);
    FREE_DEVICE(d_pool_topk_index[0]);
    FREE_DEVICE(d_pool_topk_index[1]);
    FREE_DEVICE(d_pool_query_batch[0]);
    FREE_DEVICE(d_pool_query_batch[1]);
    FREE_DEVICE(d_pool_query_batch_u8[0]);
    FREE_DEVICE(d_pool_query_batch_u8[1]);
    FREE_DEVICE(d_pool_query_fp16[0]);
    FREE_DEVICE(d_pool_query_fp16[1]);
    FREE_DEVICE(d_pool_query_norm[0]);
    FREE_DEVICE(d_pool_query_norm[1]);
    FREE_DEVICE(d_pool_top_nprobe_dist[0]);
    FREE_DEVICE(d_pool_top_nprobe_dist[1]);
    for(auto& runtime:ctx->coarse_workspace) {delete runtime;runtime=nullptr;}
    FREE_DEVICE(d_top_nprobe_index[0]);
    FREE_DEVICE(d_top_nprobe_index[1]);
#undef FREE_DEVICE
    if (ctx->stream_0) {
        cudaStreamDestroy(ctx->stream_0);
        ctx->stream_0 = 0;
    }
    if (ctx->stream_1) {
        cudaStreamDestroy(ctx->stream_1);
        ctx->stream_1 = 0;
    }
    if (ctx->cublas_handle) {
        cublasDestroy(ctx->cublas_handle);
        ctx->cublas_handle = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (ctx->ev_coarse_done[i]) {
            cudaEventDestroy(ctx->ev_coarse_done[i]);
            ctx->ev_coarse_done[i] = 0;
        }
    }
    if (ctx->ev_fine_done) {
        cudaEventDestroy(ctx->ev_fine_done);
        ctx->ev_fine_done = 0;
    }
    if (ctx->cuvs_res) {
        ivfflat_search_delete_cuvs_res(ctx->cuvs_res);
        ctx->cuvs_res = nullptr;
    }
}

inline void ivf_search_flat_fine_free_all(IVFFlatFineCtx* ctx) {
    if (!ctx) return;
    if (ctx->h_cluster_norm) {
        ctx->h_cluster_norm = nullptr;
    }
#define FREE_FINE_DEVICE(p)     \
    do {                        \
        if ((ctx)->p) {         \
            cudaFree((ctx)->p); \
            (ctx)->p = nullptr; \
        }                       \
    } while (0)
    FREE_FINE_DEVICE(d_resident_probe_topk_dist[0]);
    FREE_FINE_DEVICE(d_resident_probe_topk_dist[1]);
    FREE_FINE_DEVICE(d_resident_probe_topk_index[0]);
    FREE_FINE_DEVICE(d_resident_probe_topk_index[1]);
    FREE_FINE_DEVICE(d_resident_final_topk_dist[0]);
    FREE_FINE_DEVICE(d_resident_final_topk_dist[1]);
    FREE_FINE_DEVICE(d_resident_final_topk_index[0]);
    FREE_FINE_DEVICE(d_resident_final_topk_index[1]);
    FREE_FINE_DEVICE(d_resident_entry_count_per_cluster[0]);
    FREE_FINE_DEVICE(d_resident_entry_count_per_cluster[1]);
    FREE_FINE_DEVICE(d_resident_entry_offset[0]);
    FREE_FINE_DEVICE(d_resident_entry_offset[1]);
    FREE_FINE_DEVICE(d_resident_entry_cluster_id[0]);
    FREE_FINE_DEVICE(d_resident_entry_cluster_id[1]);
    FREE_FINE_DEVICE(d_resident_entry_query_start[0]);
    FREE_FINE_DEVICE(d_resident_entry_query_start[1]);
    FREE_FINE_DEVICE(d_resident_entry_query_count[0]);
    FREE_FINE_DEVICE(d_resident_entry_query_count[1]);
    FREE_FINE_DEVICE(d_resident_entry_queries[0]);
    FREE_FINE_DEVICE(d_resident_entry_queries[1]);
    FREE_FINE_DEVICE(d_resident_entry_probe_indices[0]);
    FREE_FINE_DEVICE(d_resident_entry_probe_indices[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_count_per_cluster[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_count_per_cluster[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_offset[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_offset[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_cluster_id[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_cluster_id[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_query_start[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_query_start[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_query_count[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_query_count[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_queries[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_queries[1]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_probe_indices[0]);
    FREE_FINE_DEVICE(d_resident_threshold_seed_entry_probe_indices[1]);
    FREE_FINE_DEVICE(d_resident_chunk_task_count[0]);
    FREE_FINE_DEVICE(d_resident_chunk_task_count[1]);
    FREE_FINE_DEVICE(d_resident_row_chunk_count[0]);
    FREE_FINE_DEVICE(d_resident_row_chunk_count[1]);
    FREE_FINE_DEVICE(d_resident_row_chunk_offset[0]);
    FREE_FINE_DEVICE(d_resident_row_chunk_offset[1]);
    FREE_FINE_DEVICE(d_resident_row_chunk_write_pos[0]);
    FREE_FINE_DEVICE(d_resident_row_chunk_write_pos[1]);
    FREE_FINE_DEVICE(d_resident_chunk_task_row[0]);
    FREE_FINE_DEVICE(d_resident_chunk_task_row[1]);
    FREE_FINE_DEVICE(d_resident_chunk_task_cluster[0]);
    FREE_FINE_DEVICE(d_resident_chunk_task_cluster[1]);
    FREE_FINE_DEVICE(d_resident_chunk_task_first_block[0]);
    FREE_FINE_DEVICE(d_resident_chunk_task_first_block[1]);
    FREE_FINE_DEVICE(d_resident_chunk_task_num_blocks[0]);
    FREE_FINE_DEVICE(d_resident_chunk_task_num_blocks[1]);
    FREE_FINE_DEVICE(d_resident_chunk_candidate_dist[0]);
    FREE_FINE_DEVICE(d_resident_chunk_candidate_dist[1]);
    FREE_FINE_DEVICE(d_resident_chunk_candidate_index[0]);
    FREE_FINE_DEVICE(d_resident_chunk_candidate_index[1]);
    FREE_FINE_DEVICE(d_resident_threshold[0]);
    FREE_FINE_DEVICE(d_resident_threshold[1]);
    FREE_FINE_DEVICE(d_resident_threshold_candidate_count[0]);
    FREE_FINE_DEVICE(d_resident_threshold_candidate_count[1]);
    FREE_FINE_DEVICE(d_resident_threshold_overflow[0]);
    FREE_FINE_DEVICE(d_resident_threshold_overflow[1]);
    FREE_FINE_DEVICE(d_resident_threshold_candidate_dist[0]);
    FREE_FINE_DEVICE(d_resident_threshold_candidate_dist[1]);
    FREE_FINE_DEVICE(d_resident_threshold_candidate_index[0]);
    FREE_FINE_DEVICE(d_resident_threshold_candidate_index[1]);
#undef FREE_FINE_DEVICE
    ctx->d_resident_interleaved_vectors = nullptr;
    ctx->d_resident_interleaved_vectors_u8 = nullptr;
    ctx->d_resident_interleaved_norms = nullptr;
    ctx->d_resident_interleaved_reordered_indices = nullptr;
    ctx->resident_interleaved_enabled = 0;
    ctx->resident_interleaved_u8 = 0;
    ctx->resident_combo_name[0] = '\0';
    ctx->resident_chunk_blocks = 0;
    ctx->resident_max_chunk_tasks = 0;
    ctx->resident_threshold_seed_probes = 0;
    ctx->resident_threshold_candidate_capacity = 0;
    if (ctx->h_cluster_offsets) {
        cudaFreeHost(ctx->h_cluster_offsets);
        ctx->h_cluster_offsets = nullptr;
    }
    if (ctx->h_top_nprobe_batch) {
        cudaFreeHost(ctx->h_top_nprobe_batch);
        ctx->h_top_nprobe_batch = nullptr;
    }
    if (ctx->h_topk_local_idx_batch) {
        std::free(ctx->h_topk_local_idx_batch);
        ctx->h_topk_local_idx_batch = nullptr;
    }
    if (ctx->h_cluster_query_offset_batch) {
        cudaFreeHost(ctx->h_cluster_query_offset_batch);
        ctx->h_cluster_query_offset_batch = nullptr;
    }
    if (ctx->h_cluster_query_data_batch) {
        cudaFreeHost(ctx->h_cluster_query_data_batch);
        ctx->h_cluster_query_data_batch = nullptr;
    }
    if (ctx->h_active_cluster_ids_batch) {
        cudaFreeHost(ctx->h_active_cluster_ids_batch);
        ctx->h_active_cluster_ids_batch = nullptr;
    }
    ctx->h_active_cluster_count_batch = 0;
    if (ctx->h_numa_cluster_boundaries) {
        std::free(ctx->h_numa_cluster_boundaries);
        ctx->h_numa_cluster_boundaries = nullptr;
    }
    ctx->h_numa_cluster_boundary_count = 0;
    ctx->h_use_compact_metadata = 0;
    if (ctx->h_cluster_offsets_ll) {
        std::free(ctx->h_cluster_offsets_ll);
        ctx->h_cluster_offsets_ll = nullptr;
    }
    ctx->h_cluster_offsets_aosoa_ll = nullptr;
    ctx->h_cluster_aosoa_tile = 0;
    if (ctx->h_query_u8_tmp) {
        std::free(ctx->h_query_u8_tmp);
        ctx->h_query_u8_tmp = nullptr;
    }
    if (ctx->h_thread_dists) {
        std::free(ctx->h_thread_dists);
        ctx->h_thread_dists = nullptr;
    }
    if (ctx->h_thread_idxs) {
        std::free(ctx->h_thread_idxs);
        ctx->h_thread_idxs = nullptr;
    }
    if (ctx->h_thread_sizes) {
        std::free(ctx->h_thread_sizes);
        ctx->h_thread_sizes = nullptr;
    }
    if (ctx->h_thread_touched) {
        std::free(ctx->h_thread_touched);
        ctx->h_thread_touched = nullptr;
    }
    ctx->h_thread_topk_capacity = 0;
    ctx->h_thread_query_capacity = 0;
    ctx->cached_combo = nullptr;
    ctx->cached_storage_view = {};
    ctx->cached_query_prepare = nullptr;
    ctx->cached_cluster_runner = nullptr;
    ctx->cached_cluster_metadata_mode = CPU_FINE_CLUSTER_METADATA_SCRATCH;
    ctx->debug_resident_entry_build_ms = 0.0;
    ctx->debug_resident_probe_init_ms = 0.0;
    ctx->debug_resident_scan_ms = 0.0;
    ctx->debug_resident_merge_map_ms = 0.0;
    ctx->debug_resident_threshold_seed_build_ms = 0.0;
    ctx->debug_resident_threshold_sketch_ms = 0.0;
    ctx->debug_resident_threshold_scan_ms = 0.0;
    ctx->debug_resident_threshold_reduce_ms = 0.0;
    ctx->debug_resident_threshold_avg_candidates = 0.0;
    ctx->debug_resident_threshold_max_candidates = 0.0;
    ctx->debug_resident_d2h_ms = 0.0;
}

inline void ivf_search_ctx_free_all(IVFSearchCtx* ctx) {
    if (!ctx) return;
    ivf_search_flat_fine_free_all(&ctx->flat_fine);
    ivf_search_flat_coarse_free_all(&ctx->flat_coarse);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <stdexcept>
#define CUDA_CTX_ERROR(ctx, msg)                 \
    do {                                         \
        ivf_search_flat_coarse_free_all(&(ctx)); \
        throw std::runtime_error(msg);           \
    } while (0)
#endif

#endif /* PARASELECT_IVF_SEARCH_CTX_H */
