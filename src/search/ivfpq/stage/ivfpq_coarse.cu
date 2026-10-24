#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/stage/ivfpq_stages.cuh"
#include "search/kernels/coarse/coarse.h"
#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include "l2norm/l2norm.cuh"
#include "quantization/linear/linear_fp16.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>

namespace {

bool ivfpq_detailed_timing_enabled() {
    const char* env = std::getenv("PARASELECT_IVFPQ_DETAILED_TIMING");
    return env && env[0] == '1';
}

paraselect::coarse::Request make_coarse_request(
    IVFPQSearchContext* ctx,
    int n_query)
{
    const auto& index = ctx->index;
    const auto& params = ctx->params;
    IVFPQCoarseContext& coarse = ctx->coarse;

    paraselect::coarse::Request run_args;
    run_args.fp16_coarse = params.fp16_coarse;
    run_args.distance_mode = params.distance;
    run_args.n_query = n_query;
    run_args.dim = index.coarse.dim;
    run_args.nlist = index.lists.nlist;
    run_args.nprobe = params.nprobe;
    run_args.cluster_tile_size = coarse.coarse_cluster_tile_size;
    run_args.d_query = coarse.d_query_batch;
    run_args.d_query_fp16 = coarse.d_query_fp16;
    run_args.d_query_norm = coarse.d_query_norm;
    run_args.d_cluster_centers = coarse.d_cluster_centers;
    run_args.d_cluster_centers_fp16 = coarse.d_cluster_centers_fp16;
    run_args.d_cluster_centers_norm = coarse.d_cluster_centers_norm;
    run_args.d_top_nprobe_dist = coarse.d_top_nprobe_dist;
    run_args.d_top_nprobe_index = coarse.d_top_nprobe_index;
    run_args.cublas_handle = coarse.cublas_handle;
    run_args.stream = coarse.stream;
    return run_args;
}

void preprocess_query(IVFPQSearchContext* ctx, const float* h_query, int n_query) {
    const auto& params = ctx->params;
    IVFPQCoarseContext& coarse = ctx->coarse;
    const int dim = ctx->index.coarse.dim;

    CUDA_CHECK(cudaMemcpyAsync(
        coarse.d_query_batch,
        h_query,
        (size_t)n_query * dim * sizeof(float),
        cudaMemcpyHostToDevice,
        coarse.stream));
    compute_l2_norm_gpu(
        coarse.d_query_batch,
        coarse.d_query_norm,
        n_query,
        dim,
        L2NORM_AUTO,
        coarse.stream);

    if (params.fp16_coarse) {
        // Convert the already uploaded FP32 query on the same stream.  This
        // removes the host-side temporary vector and preserves the exact
        // round-to-nearest FP16 cast used by the old CPU path.
        ivf::LinearQuantizerFP16GPU quantizer(1.f, 0.f);
        quantizer.encode_async(
            coarse.d_query_batch, n_query, dim, coarse.d_query_fp16, coarse.stream);
    }
}

void ivfpq_coarse_impl(
    IVFPQSearchContext* ctx,
    const float* h_query,
    int n_query,
    idx_t* h_top_nprobe_out,
    idx_t* d_top_nprobe_out,
    bool copy_top_nprobe_to_host)
{
    IVFPQCoarseContext& coarse = ctx->coarse;
    const auto& params = ctx->params;
    const bool detailed = ivfpq_detailed_timing_enabled() && ctx->collector;

    const auto prep_t0 = std::chrono::high_resolution_clock::now();
    preprocess_query(ctx, h_query, n_query);
    if (detailed) {
        CUDA_SYNC_CHECK(coarse.stream, "ivfpq coarse preprocess");
        const auto prep_t1 = std::chrono::high_resolution_clock::now();
        ctx->collector->addStageTime(
            "ivfpq_coarse_query_preprocess",
            std::chrono::duration<double, std::milli>(prep_t1 - prep_t0).count());
    }

    paraselect::coarse::Timing timing;
    auto run_args=make_coarse_request(ctx,n_query);

    paraselect::coarse::search(run_args, *coarse.coarse_workspace,detailed ? &timing : nullptr);
    if (detailed) {
        ctx->collector->addStageTime("ivfpq_coarse_init_topk", timing.init_topk_ms);
        ctx->collector->addStageTime("ivfpq_coarse_tile_fill", timing.tile_fill_ms);
        ctx->collector->addStageTime("ivfpq_coarse_gemm", timing.gemm_ms);
        ctx->collector->addStageTime("ivfpq_coarse_topk", timing.topk_ms);
        ctx->collector->addStageTime("ivfpq_coarse_merge", timing.merge_ms);
    }

    if (d_top_nprobe_out) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_top_nprobe_out,
            coarse.d_top_nprobe_index,
            (size_t)n_query * params.nprobe * sizeof(idx_t),
            cudaMemcpyDeviceToDevice,
            coarse.stream));
        CUDA_SYNC_CHECK(coarse.stream, "ivfpq coarse to device");
        coarse.d_top_nprobe_batch_valid = 1;
        if (detailed) ctx->collector->addStageTime("ivfpq_coarse_d2h_topnprobe", 0.0);
        return;
    }

    if (copy_top_nprobe_to_host) {
        const auto copy_t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(
            h_top_nprobe_out,
            coarse.d_top_nprobe_index,
            (size_t)n_query * params.nprobe * sizeof(idx_t),
            cudaMemcpyDeviceToHost,
            coarse.stream));
        CUDA_SYNC_CHECK(coarse.stream, "ivfpq coarse to host");
        const auto copy_t1 = std::chrono::high_resolution_clock::now();
        if (detailed) {
            ctx->collector->addStageTime(
                "ivfpq_coarse_d2h_topnprobe",
                std::chrono::duration<double, std::milli>(copy_t1 - copy_t0).count());
        }
    } else {
        CUDA_SYNC_CHECK(coarse.stream, "ivfpq coarse");
        if (detailed) ctx->collector->addStageTime("ivfpq_coarse_d2h_topnprobe", 0.0);
    }
}

}  // namespace

void ivfpq_coarse(IVFPQSearchContext* ctx, const float* h_query, int n_query, idx_t* h_top_nprobe_out) {
    ivfpq_coarse_impl(ctx, h_query, n_query, h_top_nprobe_out, nullptr, true);
}

void ivfpq_coarse_to_device(IVFPQSearchContext* ctx, const float* h_query, int n_query) {
    ivfpq_coarse_impl(ctx, h_query, n_query, nullptr, ctx->coarse.d_top_nprobe_batch, false);
}
