#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_prepare_internal.cuh"
#include "l2norm/l2norm.cuh"
#include "quantization/linear/prepare_fp16_host.h"
#include "utils/check_cuda_error.cuh"

void ivfrespq_upload_coarse_centers(IVFRESPQSearchCtx* ctx) {
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    const int dim = arg.n_dim;
    const int nlist = arg.lists.nlist;

    CUDA_CHECK(cudaMalloc(&coarse.d_cluster_centers, (size_t)nlist * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(
        coarse.d_cluster_centers,
        arg.h_cluster_centers,
        (size_t)nlist * dim * sizeof(float),
        cudaMemcpyHostToDevice,
        coarse.stream));
    CUDA_CHECK(cudaMalloc(&coarse.d_cluster_centers_norm, (size_t)nlist * sizeof(float)));
    compute_l2_norm_gpu(
        coarse.d_cluster_centers,
        coarse.d_cluster_centers_norm,
        nlist,
        dim,
        L2NORM_AUTO,
        coarse.stream);
    if (arg.fp16_coarse) {
        ivf::convert_fp32_to_fp16_upload(
            arg.h_cluster_centers, nlist, dim, &coarse.d_cluster_centers_fp16);
    }
    CUDA_SYNC_CHECK(coarse.stream, "ivfrespq upload coarse centers");
}

