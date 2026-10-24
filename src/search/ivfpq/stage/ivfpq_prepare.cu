#include "search/kernels/coarse/coarse_prepare.h"
#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/stage/ivfpq_stages.cuh"
#include "search/ivfpq/stage/ivfpq_resident_codes.cuh"
#include "l2norm/l2norm.cuh"
#include "quantization/linear/prepare_fp16_host.h"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
namespace {

void clear_query_workspace(IVFPQSearchContext* ctx) {
    ctx->top_nprobe_batch.clear();
    ctx->topk_local_idx_batch.clear();
    ctx->rerank_candidate_idx_batch.clear();
    ctx->rerank_candidate_count_batch.clear();
    ctx->query_batch_size = 0;
    ctx->pq_stage1_k = 0;
    ctx->pq_rerank_enabled = 0;
}

void ivfpq_prepare_gpu_coarse(IVFPQSearchContext* ctx) {
    IVFPQCoarseContext& coarse = ctx->coarse;
    const auto& index = ctx->index;
    const auto& params = ctx->params;
    const int qb = ctx->query_batch_size;
    const int dim = index.coarse.dim;
    const int nlist = index.lists.nlist;
    const int nprobe = params.nprobe;

    coarse.enabled = 1;
    coarse.query_batch_size = qb;
    coarse.coarse_cluster_tile_size = nlist;
    coarse.coarse_workspace=paraselect::coarse::prepare_coarse(
        params.fp16_coarse,params.distance,qb,dim,nlist,nprobe);

    CUDA_CHECK(cudaStreamCreate(&coarse.stream));
    if (cublasCreate(&coarse.cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("ivfpq_prepare: cublasCreate failed");
    }
    cublasSetStream(coarse.cublas_handle, coarse.stream);

    CUDA_CHECK(cudaMalloc(&coarse.d_query_batch, (size_t)qb * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coarse.d_query_norm, (size_t)qb * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coarse.d_top_nprobe_dist, (size_t)qb * nprobe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coarse.d_top_nprobe_index, (size_t)qb * nprobe * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&coarse.d_top_nprobe_batch, (size_t)qb * nprobe * sizeof(idx_t)));


    CUDA_CHECK(cudaMalloc(&coarse.d_cluster_centers,size_t(nlist)*dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(coarse.d_cluster_centers,index.coarse.centroids,
        size_t(nlist)*dim*sizeof(float),cudaMemcpyHostToDevice,coarse.stream));
    CUDA_CHECK(cudaMalloc(&coarse.d_cluster_centers_norm,size_t(nlist)*sizeof(float)));
    compute_l2_norm_gpu(coarse.d_cluster_centers,coarse.d_cluster_centers_norm,
        nlist,dim,L2NORM_AUTO,coarse.stream);
    if(params.fp16_coarse) {
        CUDA_CHECK(cudaMalloc(&coarse.d_query_fp16,size_t(qb)*dim*sizeof(__half)));
        ivf::convert_fp32_to_fp16_upload(index.coarse.centroids,nlist,dim,&coarse.d_cluster_centers_fp16);
    }
    CUDA_SYNC_CHECK(coarse.stream, "ivfpq prepare gpu coarse");
}

void ivfpq_release_gpu_coarse(IVFPQSearchContext* ctx) {
    IVFPQCoarseContext& coarse = ctx->coarse;
    if(coarse.stream)CUDA_CHECK(cudaStreamSynchronize(coarse.stream));
    coarse.coarse_workspace.reset();
#define FREE_DEVICE(p) do { if ((p)) { cudaFree((p)); (p) = nullptr; } } while (0)
    FREE_DEVICE(coarse.d_cluster_centers);
    FREE_DEVICE(coarse.d_cluster_centers_fp16);
    FREE_DEVICE(coarse.d_cluster_centers_norm);
    FREE_DEVICE(coarse.d_query_batch);
    FREE_DEVICE(coarse.d_query_fp16);
    FREE_DEVICE(coarse.d_query_norm);
    FREE_DEVICE(coarse.d_top_nprobe_dist);
    FREE_DEVICE(coarse.d_top_nprobe_index);
    FREE_DEVICE(coarse.d_top_nprobe_batch);
#undef FREE_DEVICE
    if (coarse.cublas_handle) {
        cublasDestroy(coarse.cublas_handle);
        coarse.cublas_handle = nullptr;
    }
    if (coarse.stream) {
        cudaStreamDestroy(coarse.stream);
        coarse.stream = 0;
    }
    coarse = IVFPQCoarseContext{};
}

}  // namespace

void ivfpq_prepare_index(IVFPQSearchContext* ctx) {
    if (!ctx) return;
    const auto& index = ctx->index;
    ctx->cluster_counts = derive_ivf_list_counts(index.lists);
    ctx->cluster_offsets.assign(
        index.lists.offsets, index.lists.offsets + index.lists.nlist + 1);
    ivfpq_resident_codes_prepare_storage(ctx);
}

void ivfpq_prepare(IVFPQSearchContext* ctx) {
    if (!ctx) return;
    const auto& index = ctx->index;
    const auto& queries = ctx->queries;
    const auto& params = ctx->params;
    const auto& options = ctx->options;
    ivfpq_release_gpu_coarse(ctx);
    clear_query_workspace(ctx);
    ctx->pq_rerank_enabled =
        (options.rerank_dtype != IVFExactRerankDType::Disabled &&
         options.n_rerank > params.topk) ? 1 : 0;
    ctx->pq_stage1_k = ctx->pq_rerank_enabled
        ? std::min(options.n_rerank, static_cast<int>(index.lists.n_vectors))
        : params.topk;
    if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
        ctx->pq_stage1_k = std::min(
            options.n_rerank,
            static_cast<int>(index.lists.n_vectors));
    }
    ctx->query_batch_size = std::min(params.batch_size, queries.count);
    ctx->top_nprobe_batch.assign(
        (size_t)ctx->query_batch_size * params.nprobe, (idx_t)-1);
    ctx->topk_local_idx_batch.assign(
        (size_t)ctx->query_batch_size * params.topk, -1);
    ctx->rerank_candidate_idx_batch.assign((size_t)ctx->query_batch_size * ctx->pq_stage1_k, -1);
    ctx->rerank_candidate_count_batch.assign((size_t)ctx->query_batch_size, 0);
    ctx->pq_scanned_codes_total = 0;
    ctx->pq_adc_lookups_total = 0;
    try {
        ivfpq_prepare_gpu_coarse(ctx);
        ivfpq_resident_codes_prepare_workspace(ctx);
    } catch (...) {
        ivfpq_release_gpu_coarse(ctx);
        clear_query_workspace(ctx);
        throw;
    }
}

void ivfpq_release(IVFPQSearchContext* ctx) {
    if (!ctx) return;
    ivfpq_resident_codes_release(ctx);
    ivfpq_release_gpu_coarse(ctx);
    clear_query_workspace(ctx);
    ctx->cluster_offsets.clear();
    ctx->cluster_counts.clear();
}
