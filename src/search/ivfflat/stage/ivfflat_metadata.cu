#include "utils/pch.h"
#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "utils/cudatimer.h"
#include "utils/utils.cuh"
#include "utils/check_cuda_error.cuh"
#include <algorithm>

// ---------- Stage2:  CPU fine metadata(cluster → query list) ----------
void ivfflat_metadata(IVFSearchCtx* ctx, int pid, cudaStream_t stream_override,
                      bool sync) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    cudaStream_t stream = stream_override
                              ? stream_override
                              : (coarse->stream_0 ? coarse->stream_0 : 0);
    int current_n_querys = coarse->current_n_querys;
    int n_total_clusters = ctx->arg.n_total_clusters;
    int n_probes = ctx->arg.n_probes;

    int* d_pool_cluster_offset = coarse->d_pool_cluster_offset[pid];
    int* d_cluster_query_data  = coarse->d_cluster_query_data[pid];
    int* d_cluster_query_probe_indices = coarse->d_cluster_query_probe_indices[pid];
    int* d_cluster_query_count = coarse->d_cluster_query_count[pid];
    int* d_cluster_write_pos   = coarse->d_cluster_write_pos[pid];
    idx_t* d_top_nprobe_index = coarse->d_top_nprobe_index[pid];

    NVTX_PUSH("IVFFlat_Metadata");
    {
        CUDATimer timer("Step 2: Build CPU metadata (GPU)", ENABLE_CUDA_TIMING);

        CUDA_CHECK(cudaMemsetAsync(d_cluster_query_count, 0, (n_total_clusters + 1) * sizeof(int), stream));
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 after cudaMemsetAsync");

        int block_size = 256;
        int count_probe_block = (n_probes + block_size - 1) / block_size;
        dim3 grid(current_n_querys, count_probe_block);
        count_cluster_queries_kernel<<<grid, block_size, 0, stream>>>(
            d_top_nprobe_index, d_cluster_query_count,
            current_n_querys, n_probes, n_total_clusters);
        CUDA_LAUNCH_CHECK("count_cluster_queries_kernel");
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 after count_cluster_queries_kernel");

        compute_prefix_sum_scratch(d_cluster_query_count, d_pool_cluster_offset,
                                   n_total_clusters, coarse->d_scan_block_sums, stream);
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 after compute_prefix_sum 1");

        CUDA_CHECK(cudaMemcpyAsync(d_cluster_write_pos, d_pool_cluster_offset,
                        n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 after cudaMemcpyAsync 1");

        const int mapping_block_size = 256;
        const int probe_blocks = (n_probes + mapping_block_size - 1) / mapping_block_size;
        dim3 mapping_grid(current_n_querys, probe_blocks);
        build_cluster_query_mapping_kernel<<<mapping_grid, mapping_block_size, 0, stream>>>(
            d_top_nprobe_index,
            d_pool_cluster_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos, current_n_querys, n_probes, n_total_clusters);
        CUDA_LAUNCH_CHECK("build_cluster_query_mapping_kernel");
        if (sync) {
            CUDA_SYNC_CHECK(stream, "Stage2 after build_cluster_query_mapping_kernel");
        }
    }
    NVTX_POP();
}


void ivfflat_metadata_compact(IVFSearchCtx* ctx, int pid, cudaStream_t stream_override) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    cudaStream_t stream = stream_override
                              ? stream_override
                              : (coarse->stream_0 ? coarse->stream_0 : 0);
    const int current_n_querys = coarse->current_n_querys;
    const int n_total_clusters = ctx->arg.n_total_clusters;
    const int n_probes = ctx->arg.n_probes;

    int* d_cluster_query_count = coarse->d_cluster_query_count[pid];
    int* d_active_cluster_ids = coarse->d_active_cluster_ids[pid];
    int* d_active_cluster_count = coarse->d_active_cluster_count[pid];
    int* d_cluster_to_active_pos = coarse->d_cluster_to_active_pos[pid];
    int* d_compact_count = coarse->d_compact_cluster_query_count[pid];
    int* d_compact_offset = coarse->d_compact_cluster_query_offset[pid];
    int* d_compact_write_pos = coarse->d_compact_cluster_write_pos[pid];
    int* d_compact_data = coarse->d_compact_cluster_query_data[pid];
    idx_t* d_top_nprobe_index = coarse->d_top_nprobe_index[pid];

    NVTX_PUSH("IVFFlat_Metadata_Compact");
    {
        CUDATimer timer("Step 2: Build compact CPU metadata (GPU)", ENABLE_CUDA_TIMING);
        const int block_size = 256;
        const int count_probe_block = (n_probes + block_size - 1) / block_size;
        dim3 grid(current_n_querys, count_probe_block);

        CUDA_CHECK(cudaMemsetAsync(d_cluster_query_count, 0, (size_t)(n_total_clusters + 1) * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_active_cluster_count, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_cluster_to_active_pos, 0xff, (size_t)n_total_clusters * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_compact_count, 0, (size_t)n_total_clusters * sizeof(int), stream));
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 compact after memset");

        count_cluster_queries_kernel<<<grid, block_size, 0, stream>>>(
            d_top_nprobe_index, d_cluster_query_count,
            current_n_querys, n_probes, n_total_clusters);
        CUDA_LAUNCH_CHECK("count_cluster_queries_kernel compact");
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 compact after count_cluster_queries_kernel");

        const int active_grid = (n_total_clusters + block_size - 1) / block_size;
        build_active_cluster_compact_kernel<<<active_grid, block_size, 0, stream>>>(
            d_cluster_query_count,
            d_active_cluster_ids,
            d_active_cluster_count,
            d_cluster_to_active_pos,
            d_compact_count,
            n_total_clusters);
        CUDA_LAUNCH_CHECK("build_active_cluster_compact_kernel");
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 compact after build_active_cluster_compact_kernel");

        compute_prefix_sum(d_compact_count, d_compact_offset, n_total_clusters, stream);
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 compact after compute_prefix_sum");

        CUDA_CHECK(cudaMemcpyAsync(d_compact_write_pos, d_compact_offset,
                    (size_t)n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_DEBUG_SYNC_CHECK(stream, "Stage2 compact after write_pos copy");

        build_compact_cluster_query_mapping_kernel<<<grid, block_size, 0, stream>>>(
            d_top_nprobe_index,
            d_cluster_to_active_pos,
            d_compact_offset,
            d_compact_data,
            d_compact_write_pos,
            current_n_querys,
            n_probes,
            n_total_clusters);
        CUDA_LAUNCH_CHECK("build_compact_cluster_query_mapping_kernel");
        CUDA_SYNC_CHECK(stream, "Stage2 compact after build_compact_cluster_query_mapping_kernel");
    }
    NVTX_POP();
}
