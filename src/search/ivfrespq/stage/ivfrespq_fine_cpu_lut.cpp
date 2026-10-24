#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "search/ivfrespq/stage/ivfrespq_cpu_rerank.h"
#include "search/ivfrespq/stage/ivfrespq_gpu_lut.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>

static const uint8_t* batch_query_u8_or_null(
    const IVFRESPQFineCtx& fine,
    int batch_start,
    int dim)
{
    return fine.assets.h_queries_u8 ? fine.assets.h_queries_u8 + (size_t)batch_start * dim : nullptr;
}

static void fill_stage2_outputs(IVFRESPQFineCtx& fine, int batch_n, int topk) {
    std::fill(
        fine.host.h_topk_local_idx_batch,
        fine.host.h_topk_local_idx_batch + (size_t)batch_n * topk,
        -1);
}

static void ivfrespq_fine_cpu_lut_common(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n,
    const float* precomputed_lut,
    double gpu_lut_build_ms,
    double gpu_lut_d2h_ms)
{
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    const int topk = arg.k;
    const int nprobe = arg.n_probes;
    const float* batch_query_fp32 = arg.h_queries_fp32 + (size_t)batch_start * dim;
    const uint8_t* batch_query_u8 = batch_query_u8_or_null(fine, batch_start, dim);
    float* batch_dist_out = arg.h_topk_dist_out + (size_t)batch_start * topk;
    const int num_threads = fine.runtime.cpu_fine_num_threads;

    paraselect::cpu_fine::PQResidCpuLUTTiming pq_timing;
    (void)paraselect::cpu_fine::cpu_fine_kernel_pq_resid_cpu_lut_timed(
        fine.assets.h_base_u8,
        fine.host.h_cluster_offsets,
        fine.assets.h_cluster_sizes,
        fine.assets.pq_storage,
        fine.assets.h_pq_codebook,
        fine.assets.h_cluster_centers,
        batch_query_u8,
        batch_query_fp32,
        precomputed_lut,
        fine.assets.h_base_fp32,
        fine.runtime.pq_use_float_rerank,
        fine.host.h_top_nprobe_batch,
        batch_n,
        dim,
        arg.lists.nlist,
        nprobe,
        topk,
        fine.runtime.pq_rerank_n,
        fine.assets.pq_M,
        fine.assets.pq_K,
        num_threads,
        fine.host.h_topk_local_idx_batch,
        batch_dist_out,
        fine.runtime.cpu_config.scheduler,
        fine.runtime.cpu_config.lut_tile,
        &pq_timing);
    if (precomputed_lut) {
        pq_timing.lut_build_ms = gpu_lut_build_ms;
        pq_timing.lut_d2h_ms = gpu_lut_d2h_ms;
    }
    if (ctx->collector) {
        ctx->collector->addStageTime("pq_workspace_alloc", pq_timing.workspace_ms);
        ctx->collector->addStageTime("pq_adc_lut_scan", pq_timing.adc_lut_scan_ms);
        ctx->collector->addStageTime("pq_residual", pq_timing.residual_ms);
        ctx->collector->addStageTime("pq_lut_build", pq_timing.lut_build_ms);
        ctx->collector->addStageTime("pq_lut_d2h", pq_timing.lut_d2h_ms);
        ctx->collector->addStageTime("pq_adc_lookup", pq_timing.adc_lookup_ms);
        ctx->collector->addStageTime("pq_heap_push", pq_timing.heap_push_ms);
        ctx->collector->addStageTime("pq_rerank", pq_timing.rerank_ms);
        ctx->collector->addField("pq_heap_pushes", (double)pq_timing.heap_pushes);
    }
    ctx->pq_scanned_codes_total += pq_timing.scanned_codes;
    ctx->pq_adc_lookups_total += pq_timing.adc_lookups;
}


void ivfrespq_fine_gpu_lut_cpu_scan(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n)
{
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    const int nprobe = arg.n_probes;
    const int topk = arg.k;
    const float* batch_query_fp32 = arg.h_queries_fp32 + (size_t)batch_start * dim;
    const size_t lut_elems =
        (size_t)batch_n * (size_t)nprobe * (size_t)fine.assets.pq_M * (size_t)fine.assets.pq_K;
    double gpu_lut_build_ms = 0.0;
    double gpu_lut_d2h_ms = 0.0;

    fill_stage2_outputs(fine, batch_n, topk);
    const auto lut_t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
        ctx->coarse.d_query_batch,
        batch_query_fp32,
        (size_t)batch_n * dim * sizeof(float),
        cudaMemcpyHostToDevice,
        ctx->coarse.stream));
    CUDA_CHECK(cudaMemcpyAsync(
        fine.device.d_top_nprobe_batch,
        fine.host.h_top_nprobe_batch,
        (size_t)batch_n * nprobe * sizeof(idx_t),
        cudaMemcpyHostToDevice,
        ctx->coarse.stream));
    ivfrespq_build_lut_gpu(
        ctx->coarse.d_query_batch,
        ctx->coarse.d_cluster_centers,
        fine.device.d_top_nprobe_batch,
        fine.device.d_pq_codebook,
        batch_n,
        dim,
        arg.lists.nlist,
        nprobe,
        fine.assets.pq_M,
        fine.assets.pq_K,
        fine.device.d_pq_lut_batch,
        ctx->coarse.stream);
    const auto copy_t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
        fine.host.h_pq_lut_batch,
        fine.device.d_pq_lut_batch,
        lut_elems * sizeof(float),
        cudaMemcpyDeviceToHost,
        ctx->coarse.stream));
    CUDA_SYNC_CHECK(ctx->coarse.stream, "ivfrespq gpu lut d2h");
    const auto copy_t1 = std::chrono::high_resolution_clock::now();
    gpu_lut_build_ms = std::chrono::duration<double, std::milli>(copy_t1 - lut_t0).count();
    gpu_lut_d2h_ms = std::chrono::duration<double, std::milli>(copy_t1 - copy_t0).count();
    ivfrespq_fine_cpu_lut_common(
        ctx,
        batch_start,
        batch_n,
        fine.host.h_pq_lut_batch,
        gpu_lut_build_ms,
        gpu_lut_d2h_ms);
}

void ivfrespq_fine_cpu_lut(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n)
{
    const int topk = ctx->arg.k;
    fill_stage2_outputs(ctx->fine, batch_n, topk);
    ivfrespq_fine_cpu_lut_common(ctx, batch_start, batch_n, nullptr, 0.0, 0.0);
}
