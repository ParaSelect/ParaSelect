#ifndef PARASELECT_IVFFLAT_STAGES_H
#define PARASELECT_IVFFLAT_STAGES_H

#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/ivfflat_search_ctx.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_batch_view.h"

namespace paraselect::search::ivfflat_stage {

struct CpuFinePrepareConfig {
    int query_batch_size = 0;
    int coarse_sub_batch_size = 0;
    int n_dim = 0;
    int n_total_clusters = 0;
    int n_total_vectors = 0;
    int n_probes = 0;
    int topk = 0;
    const int* h_cluster_sizes = nullptr;
    const idx_t* h_cluster_offsets = nullptr;
    const float* h_cluster_vectors = nullptr;
    const float* h_cluster_vectors_aosoa = nullptr;
    const uint8_t* h_cluster_vectors_u8 = nullptr;
    const idx_t* h_reordered_indices = nullptr;
    const long long* h_cluster_offsets_aosoa_ll = nullptr;
    int h_cluster_aosoa_tile = 0;
    const char* cpu_fine_combo_name = nullptr;
    bool allow_precomputed_metadata = false;
};

struct CpuFineBatchBindConfig {
    int n_query = 0;
    int topk = 0;
    const float* h_queries = nullptr;
    const uint8_t* h_queries_u8 = nullptr;
    float* h_topk_dist_out = nullptr;
    idx_t* h_topk_index_out = nullptr;
    idx_t* h_top_nprobe_batch = nullptr;
    int* h_cluster_query_offset_batch = nullptr;
    int* h_cluster_query_data_batch = nullptr;
    int* h_active_cluster_ids_batch = nullptr;
    int active_cluster_count = 0;
    int* h_topk_local_idx_batch = nullptr;
    uint8_t* h_query_u8_tmp = nullptr;
    float* h_thread_dists = nullptr;
    int* h_thread_idxs = nullptr;
    int* h_thread_sizes = nullptr;
    uint8_t* h_thread_touched = nullptr;
    size_t h_thread_topk_capacity = 0;
    size_t h_thread_query_capacity = 0;
};

struct CpuFineRunConfig {
    int n_query = 0;
    int n_dim = 0;
    int n_probes = 0;
    int topk = 0;
    int distance_mode = 0;
};

struct CpuFineMetadataBuildConfig {
    int n_query = 0;
    int n_probes = 0;
    int n_total_clusters = 0;
    bool use_compact_metadata = false;
};

void prepare_cpu_fine_stage(IVFFlatFineCtx* fine, const CpuFinePrepareConfig& cfg);
void bind_cpu_fine_batch(
    paraselect::cpu_fine::CpuFineBatchView* view,
    const CpuFineBatchBindConfig& cfg);
void build_cpu_fine_metadata(
    paraselect::cpu_fine::CpuFineBatchView* view,
    const CpuFineMetadataBuildConfig& cfg);
void run_cpu_fine_batch(
    const IVFFlatFineCtx* fine,
    const CpuFineRunConfig& cfg,
    const paraselect::cpu_fine::CpuFineBatchView& view);

}  // namespace paraselect::search::ivfflat_stage

/**
 * ( coarse/fine/arg )
 *  coarse  device/stream; fine ; arg .
 */
void ivfflat_prepare(IVFSearchCtx* ctx);

/**
 * Stage1: query norm + (fp32  fp16 GEMM + topk)
 * @param pid pool slot id (0  1, 0)
 */
void ivfflat_coarse(IVFSearchCtx* ctx, int pid, cudaStream_t stream = 0,
                    bool measure = true);

/**
 * Stage2:  cluster id  entry data
 * @param pid pool slot id (0  1, 0)
 */
void ivfflat_metadata(IVFSearchCtx* ctx, int pid, cudaStream_t stream = 0,
                      bool sync = true);
void ivfflat_metadata_compact(IVFSearchCtx* ctx, int pid, cudaStream_t stream = 0);

/** Stage3 CPU: host (,) */
void ivfflat_fine_cpu(IVFSearchCtx* ctx, const paraselect::cpu_fine::CpuFineBatchView& view);

/** Stage3 GPU resident: interleaved  + probe topk merge +  */
void ivfflat_fine_gpu_resident(IVFSearchCtx* ctx, int pid);

#endif /* PARASELECT_IVFFLAT_STAGES_H */
