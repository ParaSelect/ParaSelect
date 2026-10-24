#ifndef PARASELECT_IVFRESPQ_STAGES_H
#define PARASELECT_IVFRESPQ_STAGES_H

#include "search/ivfrespq/ivfrespq_search_ctx.h"

struct IVFRESPQBatchView {
    int batch_id = 0;
    int batch_start = 0;
    int batch_n = 0;
    const float* h_queries_fp32 = nullptr;
    const uint8_t* h_queries_u8 = nullptr;
    float* h_topk_dist_out = nullptr;
    idx_t* h_topk_index_out = nullptr;
    idx_t* h_top_nprobe_batch = nullptr;
    int* h_rerank_candidate_idx = nullptr;
    int* h_topk_local_idx_batch = nullptr;
};

void ivfrespq_prepare(IVFRESPQSearchCtx* ctx, const IVFResSearchState* user_arg);
void ivfrespq_prepare_cpu_pipeline(IVFRESPQSearchCtx* ctx);
void ivfrespq_search_ctx_release(IVFRESPQSearchCtx* ctx);

void ivfrespq_coarse(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query,
    idx_t* h_top_nprobe_out);

void ivfrespq_coarse_for_gpu(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query);

void ivfrespq_coarse_to_device(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query,
    idx_t* d_top_nprobe_out,
    float* d_top_nprobe_dist_out = nullptr);

void ivfrespq_coarse_preload_query_batch(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query);

void ivfrespq_coarse_preloaded_to_device(
    IVFRESPQSearchCtx* ctx,
    int n_query,
    idx_t* d_top_nprobe_out,
    float* d_top_nprobe_dist_out = nullptr);

void ivfrespq_fine(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n);

void ivfrespq_fine_residual_pq_gpu(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n);

void ivfrespq_fine_residual_pq_gpu_candidates(
    IVFRESPQSearchCtx* ctx,
    const IVFRESPQBatchView& view);

void ivfrespq_rerank_residual_pq_candidates_cpu(
    IVFRESPQSearchCtx* ctx,
    const IVFRESPQBatchView& view);

void ivfrespq_prewarm_residual_pq_rerank_cpu(
    IVFRESPQSearchCtx* ctx,
    int batch_n);

void ivfrespq_fine_gpu_lut_cpu_scan(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n);

void ivfrespq_fine_cpu_lut(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n);

void ivfrespq_map_ids(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n);

void ivfrespq_map_ids_view(
    IVFRESPQSearchCtx* ctx,
    const IVFRESPQBatchView& view);

#endif  // PARASELECT_IVFRESPQ_STAGES_H
