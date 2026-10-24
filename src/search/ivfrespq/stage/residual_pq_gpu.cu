#include "search/ivfrespq/stage/residual_pq_gpu.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_detail.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_kernels.cuh"
#include "utils/check_cuda_error.cuh"

#include <cfloat>
#include <stdexcept>

void residual_pq_pad_queries_gpu(
    const float* d_queries,
    int n_query,
    int pq_m,
    int d_sub,
    int d_sub_padded,
    float* d_queries_padded,
    cudaStream_t stream) {
    paraselect::search::rpq_kernels::launch_residual_pq_pad_queries(
        d_queries, n_query, pq_m, d_sub, d_sub_padded, d_queries_padded, stream);
    CUDA_LAUNCH_CHECK("residual_pq_pad_query_kernel");
}

void pq_adc_search_gpu(
    paraselect::search::PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    paraselect::search::ResidualPQGroupedWorkspace* grouped_workspace) {
    constexpr int kFixedTopK = 10;
    const int local_k = spec.local_k;
    const int thread_k = spec.thread_k;
    const int global_k = spec.global_k;
    if (n_query <= 0 || n_probes <= 0 || spec.topk <= 0 || global_k <= 0) return;
    const bool use_direct_codebook =
        spec.scan_mode == paraselect::search::ResidualPQGpuScanMode::DirectCodebook;
    if (spec.pq_k != 256 || spec.topk != kFixedTopK || global_k > 512 ||
        spec.pq_m <= 0 || dim != spec.pq_dim_padded ||
        (spec.pq_d_sub_padded != 2 && spec.pq_d_sub_padded != 4 &&
         spec.pq_d_sub_padded != 8) ||
        (local_k != 10 && local_k != 16 && local_k != 32 && local_k != 256)) {
        throw std::invalid_argument("residual_pq_search_gpu: unsupported prepared spec");
    }

    const bool use_probe_chunk = spec.impl == paraselect::search::ResidualPQGpuImpl::ProbeChunk;
    const bool use_vector_chunk =
        spec.impl == paraselect::search::ResidualPQGpuImpl::VectorChunk;
    const bool use_grouped_vector_chunk =
        spec.impl == paraselect::search::ResidualPQGpuImpl::GroupedVectorChunk;
    if (distance_mode == paraselect::search::PQAdcDistanceMode::Plain &&
        !use_grouped_vector_chunk) {
        throw std::invalid_argument("pq_adc_search_gpu: plain ADC currently requires grouped vector chunk");
    }
    if (use_grouped_vector_chunk) {
        if (local_k != 256 || (global_k != 64 && global_k != 512) ||
            spec.vector_chunk_size <= 0 || spec.group_chunks <= 0) {
            throw std::invalid_argument(
                "residual_pq_search_gpu: unsupported grouped vector chunk spec");
        }
        paraselect::search::residual_pq_grouped_chunk_search_gpu(
            distance_mode,
            d_queries,
            d_centroids,
            d_top_nprobe,
            d_pq_codes,
            d_cluster_offsets,
            d_cluster_counts,
            d_codebook,
            n_query,
            dim,
            n_total_clusters,
            n_probes,
            spec,
            d_global_candidate_idx,
            d_global_candidate_dist,
            stream,
            timing,
            grouped_workspace);
        return;
    }
    if (!use_vector_chunk && local_k == 256) {
        throw std::invalid_argument("residual_pq_search_gpu: local_k=256 requires vector chunk impl");
    }
    if (use_probe_chunk && (spec.probes_per_chunk != 4 || local_k != 32 || thread_k != 8)) {
        throw std::invalid_argument("residual_pq_search_gpu: unsupported probe chunk spec");
    }
    if (use_vector_chunk &&
        (spec.vector_chunk_size <= 0 || spec.max_vector_chunks_per_probe <= 0 ||
         local_k != 256 || global_k != 512)) {
        throw std::invalid_argument("residual_pq_search_gpu: unsupported vector chunk spec");
    }
    const int reduce_sources = use_probe_chunk
                                   ? (n_probes + spec.probes_per_chunk - 1) / spec.probes_per_chunk
                                   : (use_vector_chunk ? n_probes * spec.max_vector_chunks_per_probe : n_probes);
    const size_t lut_shmem = use_direct_codebook
                                 ? 0
                                 : (size_t)spec.pq_m * paraselect::search::rpq_detail::kResidualPqLutStride *
                                       sizeof(float);
    const size_t chunk_shmem = use_probe_chunk
                                   ? (size_t)spec.probes_per_chunk * (size_t)local_k * (sizeof(float) + sizeof(int))
                                   : 0;
    const size_t shmem = lut_shmem + chunk_shmem;
    cudaEvent_t probe_start = nullptr;
    cudaEvent_t probe_stop = nullptr;
    cudaEvent_t reduce_stop = nullptr;
    if (timing) {
        CUDA_CHECK(cudaEventCreate(&probe_start));
        CUDA_CHECK(cudaEventCreate(&probe_stop));
        CUDA_CHECK(cudaEventCreate(&reduce_stop));
        CUDA_CHECK(cudaEventRecord(probe_start, stream));
    }

    if (use_direct_codebook && use_vector_chunk) {
        const dim3 chunk_grid(n_query, reduce_sources);
        paraselect::search::rpq_kernels::launch_residual_pq_vector_chunk_direct_codebook_runtime_m<2>(
            chunk_grid, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim,
            n_total_clusters, n_probes, spec.pq_m, spec.vector_chunk_size,
            spec.max_vector_chunks_per_probe, reduce_sources, d_probe_candidate_idx,
            d_probe_candidate_dist);
        CUDA_LAUNCH_CHECK("residual_pq_vector_chunk_direct_codebook_kernel");
    } else if (!use_direct_codebook && use_vector_chunk) {
        const dim3 chunk_grid(n_query, reduce_sources);
        if (spec.pq_d_sub_padded == 2) {
            paraselect::search::rpq_kernels::launch_residual_pq_vector_chunk_lut_runtime_m<2>(
                chunk_grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim,
                n_total_clusters, n_probes, spec.pq_m, spec.vector_chunk_size,
                spec.max_vector_chunks_per_probe, reduce_sources, d_probe_candidate_idx,
                d_probe_candidate_dist);
        } else if (spec.pq_d_sub_padded == 4) {
            paraselect::search::rpq_kernels::launch_residual_pq_vector_chunk_lut_runtime_m<4>(
                chunk_grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim,
                n_total_clusters, n_probes, spec.pq_m, spec.vector_chunk_size,
                spec.max_vector_chunks_per_probe, reduce_sources, d_probe_candidate_idx,
                d_probe_candidate_dist);
        } else {
            paraselect::search::rpq_kernels::launch_residual_pq_vector_chunk_lut_runtime_m<8>(
                chunk_grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim,
                n_total_clusters, n_probes, spec.pq_m, spec.vector_chunk_size,
                spec.max_vector_chunks_per_probe, reduce_sources, d_probe_candidate_idx,
                d_probe_candidate_dist);
        }
        CUDA_LAUNCH_CHECK("residual_pq_vector_chunk_lut_kernel");
    } else if (use_direct_codebook && use_probe_chunk) {
        const dim3 chunk_grid(n_query, reduce_sources);
        paraselect::search::rpq_kernels::launch_residual_pq_probe_chunk_direct_codebook_runtime_m<2>(
            spec.probes_per_chunk, chunk_grid, shmem, stream, d_queries, d_centroids,
            d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
            n_query, dim, n_total_clusters, n_probes, spec.pq_m, local_k,
            reduce_sources, d_probe_candidate_idx, d_probe_candidate_dist);
        CUDA_LAUNCH_CHECK("residual_pq_probe_chunk_direct_codebook_kernel");
    } else if (use_direct_codebook) {
        const dim3 dense_grid(n_query, n_probes);
        paraselect::search::rpq_kernels::launch_residual_pq_probe_direct_codebook_runtime_m<2>(
            local_k, thread_k, dense_grid, stream, d_queries, d_centroids,
            d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
            n_query, dim, n_total_clusters, n_probes, spec.pq_m,
            d_probe_candidate_idx, d_probe_candidate_dist);
        CUDA_LAUNCH_CHECK("residual_pq_probe_direct_codebook_kernel");
    } else if (use_probe_chunk) {
        const dim3 chunk_grid(n_query, reduce_sources);
        if (spec.pq_d_sub_padded == 2) {
            paraselect::search::rpq_kernels::launch_residual_pq_probe_chunk_runtime_m<2>(
                spec.probes_per_chunk, chunk_grid, shmem, stream, d_queries, d_centroids,
                d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                n_query, dim, n_total_clusters, n_probes, spec.pq_m, local_k,
                reduce_sources, d_probe_candidate_idx, d_probe_candidate_dist);
        } else if (spec.pq_d_sub_padded == 4) {
            paraselect::search::rpq_kernels::launch_residual_pq_probe_chunk_runtime_m<4>(
                spec.probes_per_chunk, chunk_grid, shmem, stream, d_queries, d_centroids,
                d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                n_query, dim, n_total_clusters, n_probes, spec.pq_m, local_k,
                reduce_sources, d_probe_candidate_idx, d_probe_candidate_dist);
        } else {
            paraselect::search::rpq_kernels::launch_residual_pq_probe_chunk_runtime_m<8>(
                spec.probes_per_chunk, chunk_grid, shmem, stream, d_queries, d_centroids,
                d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                n_query, dim, n_total_clusters, n_probes, spec.pq_m, local_k,
                reduce_sources, d_probe_candidate_idx, d_probe_candidate_dist);
        }
        CUDA_LAUNCH_CHECK("residual_pq_probe_chunk_topk_kernel");
    } else {
        const dim3 dense_grid(n_query, n_probes);
        if (spec.pq_d_sub_padded == 2) {
            paraselect::search::rpq_kernels::launch_residual_pq_probe_runtime_m<2>(
                local_k, thread_k, dense_grid, shmem, stream, d_queries, d_centroids,
                d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                n_query, dim, n_total_clusters, n_probes, spec.pq_m,
                d_probe_candidate_idx, d_probe_candidate_dist);
        } else if (spec.pq_d_sub_padded == 4) {
            paraselect::search::rpq_kernels::launch_residual_pq_probe_runtime_m<4>(
                local_k, thread_k, dense_grid, shmem, stream, d_queries, d_centroids,
                d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                n_query, dim, n_total_clusters, n_probes, spec.pq_m,
                d_probe_candidate_idx, d_probe_candidate_dist);
        } else {
            paraselect::search::rpq_kernels::launch_residual_pq_probe_runtime_m<8>(
                local_k, thread_k, dense_grid, shmem, stream, d_queries, d_centroids,
                d_top_nprobe, d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook,
                n_query, dim, n_total_clusters, n_probes, spec.pq_m,
                d_probe_candidate_idx, d_probe_candidate_dist);
        }
        CUDA_LAUNCH_CHECK("residual_pq_probe_topk_kernel");
    }

    if (timing) CUDA_CHECK(cudaEventRecord(probe_stop, stream));

    residual_pq_reduce_candidates_gpu(
        d_probe_candidate_idx,
        d_probe_candidate_dist,
        n_query,
        reduce_sources,
        local_k,
        global_k,
        d_global_candidate_idx,
        d_global_candidate_dist,
        stream);

    if (timing) {
        CUDA_CHECK(cudaEventRecord(reduce_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(reduce_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->probe_topk_ms, probe_start, probe_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->global_reduce_ms, probe_stop, reduce_stop));
        timing->candidate_desc_ms = 0.0f;
        timing->pq_scan_ms = timing->probe_topk_ms;
        timing->candidate_reduce_ms = timing->global_reduce_ms;
        timing->final_copy_ms = 0.0f;
        CUDA_CHECK(cudaEventDestroy(probe_start));
        CUDA_CHECK(cudaEventDestroy(probe_stop));
        CUDA_CHECK(cudaEventDestroy(reduce_stop));
    }
}
void residual_pq_search_gpu(
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    paraselect::search::ResidualPQGroupedWorkspace* grouped_workspace) {
    pq_adc_search_gpu(
        paraselect::search::PQAdcDistanceMode::Residual,
        d_queries,
        d_centroids,
        d_top_nprobe,
        d_pq_codes,
        d_cluster_offsets,
        d_cluster_counts,
        d_codebook,
        n_query,
        dim,
        n_total_clusters,
        n_probes,
        spec,
        d_probe_candidate_idx,
        d_probe_candidate_dist,
        d_global_candidate_idx,
        d_global_candidate_dist,
        stream,
        timing,
        grouped_workspace);
}
