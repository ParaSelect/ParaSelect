#include "utils/pch.h"
#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/kernels/coarse/coarse.h"
#include "l2norm/l2norm.cuh"
#include "utils/cudatimer.h"
#include "utils/utils.cuh"

// ---------- Stage1: query norm + (fp32  fp16 GEMM + topk) ----------
void ivfflat_coarse(IVFSearchCtx* ctx, int pid, cudaStream_t stream_override,
                    bool measure) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFSearchArg* arg = &ctx->arg;
    cudaStream_t stream = stream_override
                              ? stream_override
                              : (coarse->stream_0 ? coarse->stream_0 : 0);

    const int current_n_querys = coarse->current_n_querys;
    const int n_dim = arg->n_dim;
    const int n_probes = arg->n_probes;
    const int n_total_clusters = arg->n_total_clusters;

    paraselect::coarse::Request run_args;
    run_args.fp16_coarse = arg->fp16_coarse;
    run_args.distance_mode = arg->distance_mode;
    run_args.n_query = current_n_querys;
    run_args.dim = n_dim;
    run_args.nlist = n_total_clusters;
    run_args.nprobe = n_probes;
    run_args.cluster_tile_size = coarse->coarse_cluster_tile_size;
    run_args.d_query = coarse->d_pool_query_batch[pid];
    run_args.d_query_fp16 = coarse->d_pool_query_fp16[pid];
    run_args.d_query_norm = coarse->d_pool_query_norm[pid];
    run_args.d_cluster_centers = coarse->d_cluster_centers;
    run_args.d_cluster_centers_fp16 = coarse->d_cluster_centers_fp16;
    run_args.d_cluster_centers_norm = coarse->d_cluster_centers_norm;
    run_args.d_top_nprobe_dist = coarse->d_pool_top_nprobe_dist[pid];
    run_args.d_top_nprobe_index = coarse->d_top_nprobe_index[pid];
    run_args.cublas_handle = coarse->cublas_handle;
    run_args.stream = stream;
    run_args.cuvs_res = coarse->cuvs_res;

    NVTX_PUSH("IVFFlat_Coarse");
    CUDATimer timer("Step 1: Coarse Search", ENABLE_CUDA_TIMING);
    paraselect::coarse::Timing timing;
    // In an overlapped pipeline the coarse stage must stay fully asynchronous;
    // its internal event timing would synchronize the coarse stream and defeat
    // the overlap. The pipeline measures the coarse span with its own events.

    paraselect::coarse::search(run_args, *coarse->coarse_workspace[pid], measure ? &timing : nullptr);
    if (measure) {
        coarse->debug_coarse_gemm_ms += timing.gemm_ms;
        coarse->debug_coarse_topk_ms += timing.topk_ms;
    }
    NVTX_POP();

}
