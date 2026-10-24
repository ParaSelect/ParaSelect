#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "utils/check_cuda_error.cuh"
#include <cstdlib>

void ivfrespq_search_ctx_release(IVFRESPQSearchCtx* ctx) {
    if (!ctx) return;
    // Join the CPU worker before releasing anything borrowed by its jobs.
    ctx->cpu_executor.reset();
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    IVFRESPQFineCtx& fine = ctx->fine;
    if(coarse.coarse_workspace) { CUDA_CHECK(cudaStreamSynchronize(coarse.stream));coarse.coarse_workspace.reset(); }
    fine.assets.pca_gpu_refiner.reset();
    fine.device.rpq_probe_schedule.reset();

#define FREE_DEVICE(p) do { if ((p)) { cudaFree((p)); (p) = nullptr; } } while (0)
    FREE_DEVICE(coarse.d_cluster_centers);
    FREE_DEVICE(coarse.d_cluster_centers_fp16);
    FREE_DEVICE(coarse.d_cluster_centers_norm);
    FREE_DEVICE(coarse.d_query_batch);
    FREE_DEVICE(coarse.d_query_fp16);
    FREE_DEVICE(coarse.d_query_norm);
    FREE_DEVICE(coarse.d_top_nprobe_dist);
    FREE_DEVICE(coarse.d_top_nprobe_index);
    FREE_DEVICE(fine.device.d_pq_codebook);
    FREE_DEVICE(fine.device.d_pq_codebook_fp16);
    FREE_DEVICE(fine.device.d_pq_lut_batch);
    FREE_DEVICE(fine.device.query_lut_bias_device.fp32);
    FREE_DEVICE(fine.device.query_lut_bias_device.u16);
    FREE_DEVICE(fine.device.d_query_codeword_dot_lut);
    FREE_DEVICE(fine.device.d_query_codeword_dot_lut_u8);
    FREE_DEVICE(fine.device.d_query_lut_scale);
    FREE_DEVICE(fine.device.d_query_lut_lo_sum);
    FREE_DEVICE(fine.device.d_query_lut_inflation);
    FREE_DEVICE(fine.device.d_top_nprobe_dist_batch);
    FREE_DEVICE(fine.device.d_query_padded);
    FREE_DEVICE(fine.device.d_cluster_centers_padded);
    FREE_DEVICE(fine.device.d_top_nprobe_batch);
    fine.device.residual_codec = nullptr;
    ctx->residual_codec_owner.reset();
    ctx->residual_rotation_owner.reset();
    fine.assets.residual_rotation = nullptr;
    FREE_DEVICE(fine.device.d_pq_codes);
    FREE_DEVICE(fine.device.d_cluster_offsets);
    FREE_DEVICE(fine.device.d_cluster_counts);
    FREE_DEVICE(fine.device.d_probe_candidate_idx);
    FREE_DEVICE(fine.device.d_probe_candidate_dist);
    FREE_DEVICE(fine.device.d_rerank_candidate_idx);
    FREE_DEVICE(fine.device.d_rerank_candidate_dist);
    FREE_DEVICE(fine.device.d_pca_vectors);
    FREE_DEVICE(fine.device.d_pca_candidate_idx);
    FREE_DEVICE(fine.device.d_resident_compact_to_padded);
    FREE_DEVICE(fine.device.d_resident_topk_local_idx);
    FREE_DEVICE(fine.device.d_resident_topk_dist);
#undef FREE_DEVICE
    paraselect::search::residual_pq_grouped_workspace_release(&fine.device.rpq_grouped_workspace);
    paraselect::search::pq_adc_threshold_workspace_release(&fine.device.rpq_threshold_workspace);

#define DESTROY_EVENT(e) do { if ((e)) { cudaEventDestroy((e)); (e) = nullptr; } } while (0)
    DESTROY_EVENT(fine.runtime.ev_rpq_lut_start);
    DESTROY_EVENT(fine.runtime.ev_rpq_lut_stop);
    DESTROY_EVENT(fine.runtime.ev_rpq_schedule_start);
    DESTROY_EVENT(fine.runtime.ev_rpq_schedule_stop);
    DESTROY_EVENT(fine.runtime.ev_rpq_tile_stop);
    DESTROY_EVENT(fine.runtime.ev_rpq_gpu_start);
    DESTROY_EVENT(fine.runtime.ev_rpq_gpu_stop);
    DESTROY_EVENT(fine.runtime.ev_rpq_d2h_start);
    DESTROY_EVENT(fine.runtime.ev_rpq_d2h_stop);
#undef DESTROY_EVENT

    if (coarse.cublas_handle) {
        cublasDestroy(coarse.cublas_handle);
        coarse.cublas_handle = nullptr;
    }
    if (coarse.stream) {
        cudaStreamDestroy(coarse.stream);
        coarse.stream = 0;
    }
    if (fine.host.h_cluster_offsets) {
        std::free(fine.host.h_cluster_offsets);
        fine.host.h_cluster_offsets = nullptr;
    }
    if (fine.host.h_top_nprobe_batch) {
        cudaFreeHost(fine.host.h_top_nprobe_batch);
        fine.host.h_top_nprobe_batch = nullptr;
    }
    if (fine.host.h_topk_local_idx_batch) {
        std::free(fine.host.h_topk_local_idx_batch);
        fine.host.h_topk_local_idx_batch = nullptr;
    }
    if (fine.host.h_pq_lut_batch) {
        std::free(fine.host.h_pq_lut_batch);
        fine.host.h_pq_lut_batch = nullptr;
    }
    if (fine.host.h_rerank_candidate_idx) {
        cudaFreeHost(fine.host.h_rerank_candidate_idx);
        fine.host.h_rerank_candidate_idx = nullptr;
    }
    if (fine.host.h_shortlist_candidate_idx) {
        std::free(fine.host.h_shortlist_candidate_idx);
        fine.host.h_shortlist_candidate_idx = nullptr;
    }
    ctx->cluster_counts.clear();
}
