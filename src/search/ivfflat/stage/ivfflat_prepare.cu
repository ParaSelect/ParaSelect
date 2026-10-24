#include "search/kernels/coarse/coarse_prepare.h"
#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
#include "utils/pch.h"
#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_query_prepare.h"
#include "search/kernels/fine/cpu_fine/types/cpu_ivf_storage_aosoa.h"
#include "search/ivfflat/pipeline/ivf_cpu_compact_metadata.h"
#include "search/kernels/fine/gpu_fine/resident/resident_l2_topk_combos.cuh"
#include "l2norm/l2norm.cuh"
#include "quantization/linear/prepare_fp16_host.h"
#include "utils/cudatimer.h"
#include "utils/utils.cuh"
#include "utils/check_cuda_error.cuh"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/numa_cluster_owner.h"

#if PARASELECT_ENABLE_CUVS
#include <raft/core/resources.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <cstdlib>

/**  cluster size  32 ( cuvs kIndexGroupSize , interleaved  probe_vector_offset) */
__global__ void round_up_to_interleaved_group_kernel(
    const int* __restrict__ d_in,
    int* __restrict__ d_out,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = static_cast<int>(d_in[i]);
        d_out[i] = ((v + 31) / 32) * 32;
    }
}

static int choose_coarse_sub_batch_size(
    int requested_batch_size,
    int n_total_clusters,
    int n_probes,
    int dim) {
    if (requested_batch_size <= 0) return requested_batch_size;

    constexpr long long kGiB = 1024LL * 1024 * 1024;
    constexpr long long kMaxGemmElems = 1900000000LL;
    constexpr int kAlign = 32;

    size_t gpu_free = 0, gpu_total = 0;
    if (cudaMemGetInfo(&gpu_free, &gpu_total) != cudaSuccess) {
        return requested_batch_size;
    }

    const long long headroom = std::max(4LL * kGiB, (long long)gpu_total / 8);
    long long usable = (long long)gpu_free - headroom;
    if (usable < (long long)gpu_free / 5) usable = (long long)gpu_free / 5;
    usable = usable * 9 / 10;

    const long long bytes_per_query =
        (long long)n_total_clusters * sizeof(float) +
        (long long)n_probes * (sizeof(float) + sizeof(idx_t)) +
        (long long)dim * sizeof(float) +
        4096;
    int max_sub_mem = (int)std::min(
        (long long)requested_batch_size,
        usable / std::max(bytes_per_query, 1LL));
    int max_sub_int = (int)(kMaxGemmElems / std::max(n_total_clusters, 1));
    int max_sub = std::min(max_sub_mem, max_sub_int);
    if (max_sub < 1) max_sub = 1;
    if (max_sub >= kAlign) max_sub = (max_sub / kAlign) * kAlign;
    if (max_sub < 1) max_sub = 1;

    if (max_sub < requested_batch_size) {
        std::fprintf(stderr,
                     "[ivfflat_prepare] coarse auto-tiling enabled: "
                     "coarse_sub_batch_size=%d (query_batch_size=%d, nlist=%d, nprobe=%d)\n",
                     max_sub, requested_batch_size, n_total_clusters, n_probes);
    }
    return max_sub;
}

static int choose_coarse_cluster_tile_size(
    int query_tile_size,
    int n_total_clusters,
    int n_probes) {
    if (query_tile_size <= 0 || n_total_clusters <= 0) return n_total_clusters;

    constexpr long long kGiB = 1024LL * 1024 * 1024;
    constexpr long long kMaxGemmElems = 1900000000LL;
    constexpr int kAlign = 256;

    size_t gpu_free = 0, gpu_total = 0;
    if (cudaMemGetInfo(&gpu_free, &gpu_total) != cudaSuccess) {
        return n_total_clusters;
    }

    const long long headroom = std::max(4LL * kGiB, (long long)gpu_total / 8);
    long long usable = (long long)gpu_free - headroom;
    if (usable < (long long)gpu_free / 5) usable = (long long)gpu_free / 5;
    usable = usable * 9 / 10;

    // GEMM output dominates. cuVS large-k path may also allocate a rowwise index
    // matrix, so budget for both float distances and idx_t indices.
    const long long bytes_per_cluster =
        (long long)query_tile_size * (sizeof(float) + sizeof(idx_t));
    int max_tile_mem = (int)std::min(
        (long long)n_total_clusters,
        usable / std::max(bytes_per_cluster, 1LL));
    int max_tile_int = (int)(kMaxGemmElems / std::max(query_tile_size, 1));
    int max_tile = std::min(max_tile_mem, max_tile_int);
    max_tile = std::max(max_tile, std::min(n_total_clusters, n_probes));
    if (max_tile >= kAlign) max_tile = (max_tile / kAlign) * kAlign;
    if (max_tile < n_probes) max_tile = n_probes;
    if (max_tile > n_total_clusters) max_tile = n_total_clusters;

    if (max_tile < n_total_clusters) {
        std::fprintf(stderr,
                     "[ivfflat_prepare] coarse cluster tiling enabled: "
                     "coarse_cluster_tile_size=%d (nlist=%d, nprobe=%d)\n",
                     max_tile, n_total_clusters, n_probes);
    }
    return max_tile;
}

/**
 * Stage0 :prefix sum,cluster center norm.
 */
void ivfflat_prepare(
    IVFSearchCtx* ctx) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFFlatFineCtx* fine = &ctx->flat_fine;
    IVFSearchArg* arg = &ctx->arg;

    const int n_query = arg->n_query;
    const int n_total_clusters = arg->n_total_clusters;
    const int n_probes = arg->n_probes;
    const int k = arg->k;
    const int max_batch = arg->query_batch_size;
    const int dim = arg->n_dim;
    const int max_cpu_threads = paraselect::resolve_cpu_fine_thread_count();
    if (coarse->coarse_sub_batch_size <= 0) {
        coarse->coarse_sub_batch_size = choose_coarse_sub_batch_size(
            max_batch, n_total_clusters, n_probes, dim);
    } else {
        coarse->coarse_sub_batch_size = std::min(coarse->coarse_sub_batch_size, max_batch);
    }
    const int coarse_max_batch = coarse->coarse_sub_batch_size;
    const int max_pairs = max_batch * arg->n_probes;
    const int max_entries = max_pairs;
    const bool resident = arg->fine_kind == IVFFlatFineKind::Resident;
    int resident_chunk_blocks = 8;
    int resident_max_chunk_tasks = 0;
    const paraselect::resident_fine::ResidentL2TopKCombo* resident_combo = nullptr;
    if (resident) {
        resident_combo = paraselect::resident_fine::resolve_resident_l2_topk_combo_ptr_or_throw(
            arg->resident_combo_name);
        std::strncpy(fine->resident_combo_name, resident_combo->name, RESIDENT_FINE_COMBO_NAME_MAX_LEN - 1);
        fine->resident_combo_name[RESIDENT_FINE_COMBO_NAME_MAX_LEN - 1] = '\0';
        resident_chunk_blocks = resident_combo->default_chunk_blocks;
        if (resident_combo->kind ==
                paraselect::resident_fine::ResidentL2TopKKernelKind::Chunked &&
            arg->resident_chunk_blocks > 0) {
            resident_chunk_blocks = arg->resident_chunk_blocks;
        }

        int max_cluster_size = 0;
        for (int i = 0; i < n_total_clusters; ++i) {
            max_cluster_size = std::max(max_cluster_size, coarse->h_cluster_sizes[i]);
        }
        const int max_vector_blocks = (max_cluster_size + 31) / 32;
        const int chunk_blocks_for_capacity = resident_chunk_blocks > 0 ? resident_chunk_blocks : std::max(1, max_vector_blocks);
        const int max_chunks_per_entry = std::max(1, (max_vector_blocks + chunk_blocks_for_capacity - 1) / chunk_blocks_for_capacity);
        resident_max_chunk_tasks = max_entries * max_chunks_per_entry;
        fine->resident_chunk_blocks = resident_chunk_blocks;
        fine->resident_max_chunk_tasks = resident_max_chunk_tasks;
        if (paraselect::resident_fine::is_threshold_kind(resident_combo->kind)) {
            if (arg->threshold.seed_probes <= 0) {
                throw std::invalid_argument(
                    "resident threshold seed_probes must be positive");
            }
            if (arg->threshold.candidate_capacity < k) {
                throw std::invalid_argument(
                    "resident threshold candidate_capacity must be at least top-k");
            }
        }
        fine->resident_threshold_seed_probes = arg->threshold.seed_probes;
        fine->resident_threshold_candidate_capacity = arg->threshold.candidate_capacity;
    }
    if (resident) {
        const bool has_resident_fp32 = fine->d_resident_interleaved_vectors != nullptr;
        const bool has_resident_u8 = fine->d_resident_interleaved_vectors_u8 != nullptr;
        if (has_resident_fp32 == has_resident_u8 || !fine->d_resident_interleaved_norms || !fine->d_resident_interleaved_reordered_indices) {
            throw std::runtime_error("ivfflat_prepare: missing or ambiguous resident GPU storage");
        }
        if (has_resident_u8 && !fine->h_queries_u8) {
            throw std::runtime_error("ivfflat_prepare: u8 resident storage requires u8 queries");
        }
        fine->cached_combo = nullptr;
        fine->cached_cluster_runner = nullptr;
        fine->cached_query_prepare = nullptr;
        fine->cached_cluster_metadata_mode = CPU_FINE_CLUSTER_METADATA_FULL;
        fine->h_use_compact_metadata = 0;
    }
    const int max_topk = max_batch * arg->k;
    constexpr long long kCoarseClusterTilingAutoElems = 512LL * 1024 * 1024;
    const bool use_coarse_cluster_tiling =
        (long long)max_batch * (long long)n_total_clusters >= kCoarseClusterTilingAutoElems;
    coarse->coarse_cluster_tile_size = (arg->fp16_coarse || !use_coarse_cluster_tiling)
                                           ? n_total_clusters
                                           : choose_coarse_cluster_tile_size(coarse_max_batch, n_total_clusters, n_probes);

    if (n_query <= 0 || arg->n_dim <= 0 || n_total_clusters <= 0 || k <= 0) {
        fprintf(stderr, "[ERROR] ivfflat_prepare:  - n_query=%d, n_dim=%d, n_total_clusters=%d, k=%d\n",
                n_query, arg->n_dim, n_total_clusters, k);
        throw std::invalid_argument("invalid ivf_search configuration");
    }
    if (arg->distance_mode != (int)L2_DISTANCE) {
        throw std::invalid_argument("ivfflat_prepare: cached CPU fine path only supports L2 distance");
    }
    if (!coarse->h_cluster_sizes) {
        fprintf(stderr, "[ERROR] ivfflat_prepare: h_cluster_sizes is null\n");
        throw std::invalid_argument("h_cluster_sizes must be provided");
    }

    // d_cluster_size  prepare ()
    if (coarse->d_cluster_size == nullptr) {
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_size, (size_t)n_total_clusters * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(coarse->d_cluster_size, coarse->h_cluster_sizes,
                              (size_t)n_total_clusters * sizeof(int), cudaMemcpyHostToDevice));
    }

    if (n_probes <= 0 || n_probes > n_total_clusters) {
        fprintf(stderr, "[ERROR] ivfflat_prepare: n_probes - n_probes=%d, n_total_clusters=%d\n",
                n_probes, n_total_clusters);
        throw std::invalid_argument("invalid n_probes");
    }

    cudaStream_t stream = coarse->stream_0 ? coarse->stream_0 : 0;

    CUDA_CHECK(cudaMalloc(&coarse->d_probe_vector_offset, (size_t)(n_total_clusters + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&coarse->d_probe_vector_offset_compact, (size_t)(n_total_clusters + 1) * sizeof(int)));
    CHECK_CUDA_ERRORS;

    //  prefix sum  n_padded,
    //  probe_vector_offset(cluster size  prefix sum,padded  32 )
    int* d_cluster_size_padded = nullptr;
    cudaMalloc(&d_cluster_size_padded, (size_t)(n_total_clusters + 1) * sizeof(int));
    dim3 block_round(256);
    dim3 grid_round((n_total_clusters + block_round.x - 1) / block_round.x);
    round_up_to_interleaved_group_kernel<<<grid_round, block_round, 0, stream>>>(
        coarse->d_cluster_size, d_cluster_size_padded, n_total_clusters);
    CUDA_LAUNCH_CHECK("round_up_to_interleaved_group_kernel");
    compute_prefix_sum(d_cluster_size_padded, coarse->d_probe_vector_offset, n_total_clusters, stream);
    CUDA_SYNC_CHECK(stream, "after compute_prefix_sum 1");
    // compact offset( padded)
    compute_prefix_sum(coarse->d_cluster_size, coarse->d_probe_vector_offset_compact, n_total_clusters, stream);
    CUDA_SYNC_CHECK(stream, "after compute_prefix_sum 2");

    //  padded (interleaved )
    int n_padded = 0;
    CUDA_CHECK(cudaMemcpy(&n_padded, coarse->d_probe_vector_offset + n_total_clusters, sizeof(int), cudaMemcpyDeviceToHost));
    coarse->n_padded_cache = n_padded;

    //  host,
    if (!resident) {
        std::vector<idx_t> h_cluster_offsets((size_t)n_total_clusters + 1u, 0);
        for (int i = 0; i < n_total_clusters; ++i) {
            h_cluster_offsets[(size_t)i + 1u] =
                h_cluster_offsets[(size_t)i] + static_cast<idx_t>(std::max(0, coarse->h_cluster_sizes[i]));
        }
        paraselect::search::ivfflat_stage::prepare_cpu_fine_stage(
            fine,
            paraselect::search::ivfflat_stage::CpuFinePrepareConfig{
                max_batch,
                coarse->coarse_sub_batch_size,
                dim,
                n_total_clusters,
                arg->n_total_vectors,
                n_probes,
                k,
                coarse->h_cluster_sizes,
                h_cluster_offsets.data(),
                fine->h_cluster_vectors,
                fine->h_cluster_vectors_aosoa,
                fine->h_cluster_vectors_u8,
                fine->h_reordered_indices,
                fine->h_cluster_offsets_aosoa_ll,
                fine->h_cluster_aosoa_tile,
                arg->cpu_fine_combo_name,
                coarse->coarse_sub_batch_size >= max_batch,
            });
    }

    cudaFree(d_cluster_size_padded);

    // device :, stage
    //  [2]: slot 0/1 , overlap

    // Metadata prefix-scan scratch (ceil(nlist/256) ints); preallocated so the
    // overlapped pipeline never hits cudaMalloc/cudaFree (device sync).
    CUDA_CHECK(cudaMalloc(&coarse->d_scan_block_sums,
                          (size_t)(n_total_clusters / 256 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&coarse->d_fine_scan_block_sums,
                          (size_t)(n_total_clusters / 256 + 1) * sizeof(int)));

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_query_count[s], (size_t)(n_total_clusters + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_write_pos[s], (size_t)(n_total_clusters + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_cluster_offset[s], (size_t)(n_total_clusters + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_query_data[s], (size_t)max_entries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_query_probe_indices[s], (size_t)max_entries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_active_cluster_ids[s], (size_t)n_total_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_active_cluster_count[s], sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_to_active_pos[s], (size_t)n_total_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_compact_cluster_query_count[s], (size_t)n_total_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_compact_cluster_query_offset[s], (size_t)(n_total_clusters + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_compact_cluster_write_pos[s], (size_t)n_total_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_compact_cluster_query_data[s], (size_t)max_entries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_topk_tmp[s], (size_t)max_topk * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_topk_dist[s], (size_t)max_batch * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_topk_index[s], (size_t)max_batch * k * sizeof(idx_t)));

        CUDA_CHECK(cudaMalloc(&coarse->d_pool_top_nprobe_dist[s], (size_t)coarse_max_batch * n_probes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&coarse->d_top_nprobe_index[s], (size_t)coarse_max_batch * n_probes * sizeof(idx_t)));
        coarse->coarse_workspace[s]=paraselect::coarse::prepare_coarse(
            arg->fp16_coarse,arg->distance_mode,coarse_max_batch,dim,n_total_clusters,n_probes).release();
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_query_batch[s], (size_t)coarse_max_batch * dim * sizeof(float)));
        if (resident && fine->resident_interleaved_u8) {
            CUDA_CHECK(cudaMalloc(&coarse->d_pool_query_batch_u8[s], (size_t)coarse_max_batch * dim * sizeof(uint8_t)));
        }
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_query_norm[s], (size_t)coarse_max_batch * sizeof(float)));
        if (arg->fp16_coarse) {
            // The pipeline converts each query batch for Tensor Core GEMM.
            CUDA_CHECK(cudaMalloc(&coarse->d_pool_query_fp16[s],
                                  (size_t)coarse_max_batch * dim * sizeof(__half)));
        }
        if (resident) {
            CUDA_CHECK(cudaMalloc(&fine->d_resident_probe_topk_dist[s], (size_t)max_batch * n_probes * k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_probe_topk_index[s], (size_t)max_batch * n_probes * k * sizeof(idx_t)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_final_topk_dist[s], (size_t)max_batch * k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_final_topk_index[s], (size_t)max_batch * k * sizeof(idx_t)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_count_per_cluster[s], (size_t)n_total_clusters * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_offset[s], (size_t)(n_total_clusters + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_cluster_id[s], (size_t)max_entries * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_query_start[s], (size_t)max_entries * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_query_count[s], (size_t)max_entries * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_queries[s], (size_t)max_entries * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_entry_probe_indices[s], (size_t)max_entries * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_task_count[s], sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_row_chunk_count[s], ((size_t)max_batch * n_probes + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_row_chunk_offset[s], ((size_t)max_batch * n_probes + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_row_chunk_write_pos[s], (size_t)max_batch * n_probes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_task_row[s], (size_t)resident_max_chunk_tasks * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_task_cluster[s], (size_t)resident_max_chunk_tasks * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_task_first_block[s], (size_t)resident_max_chunk_tasks * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_task_num_blocks[s], (size_t)resident_max_chunk_tasks * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_candidate_dist[s], (size_t)resident_max_chunk_tasks * k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&fine->d_resident_chunk_candidate_index[s], (size_t)resident_max_chunk_tasks * k * sizeof(idx_t)));
            if (paraselect::resident_fine::is_threshold_kind(resident_combo->kind)) {
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_count_per_cluster[s], (size_t)n_total_clusters * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_offset[s], (size_t)(n_total_clusters + 1) * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_cluster_id[s], (size_t)max_entries * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_query_start[s], (size_t)max_entries * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_query_count[s], (size_t)max_entries * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_queries[s], (size_t)max_entries * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_seed_entry_probe_indices[s], (size_t)max_entries * sizeof(int)));
                const size_t threshold_candidates =
                    (size_t)max_batch * fine->resident_threshold_candidate_capacity;
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold[s],
                                      (size_t)max_batch * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_candidate_count[s],
                                      (size_t)max_batch * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_overflow[s],
                                      sizeof(int)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_candidate_dist[s],
                                      threshold_candidates * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&fine->d_resident_threshold_candidate_index[s],
                                      threshold_candidates * sizeof(idx_t)));
            }
        }
    }

    CHECK_CUDA_ERRORS;

    // cublas handle  batch
    cublasStatus_t cublas_err = cublasCreate(&coarse->cublas_handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ivfflat_prepare ERROR] cublasCreate failed: %d\n", cublas_err);
    }
    if (coarse->stream_0)
        cublasSetStream(coarse->cublas_handle, coarse->stream_0);

    //  cluster centers, norm()
    //  prepare ,
    //
    // FP32 centers / norms are required unconditionally: the coarse L2 top-k
    // stage (tile/cub/adaptive) consumes the FP32 centroid norm even when the
    // GEMM itself runs in FP16. Allocating them only for the FP32 path made
    // every FP16 coarse run fail with "tile topk policy is unsupported".
    CUDA_CHECK(cudaMalloc(&coarse->d_cluster_centers, (size_t)n_total_clusters * dim * sizeof(float)));
    cudaMemcpy(coarse->d_cluster_centers, coarse->h_cluster_centers, (size_t)n_total_clusters * dim * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMalloc(&coarse->d_cluster_centers_norm, (size_t)n_total_clusters * sizeof(float)));
    compute_l2_norm_gpu(coarse->d_cluster_centers, coarse->d_cluster_centers_norm, n_total_clusters, dim, L2NORM_AUTO, stream);
    CUDA_SYNC_CHECK(stream, "after compute_l2_norm_gpu for cluster_centers");
    if (arg->fp16_coarse) {
        // FP16 : host , Tensor Core GEMM
        if (coarse->h_cluster_centers == nullptr) {
            fprintf(stderr, "[ivfflat_prepare] ERROR: h_cluster_centers is nullptr!\n");
            fflush(stderr);
        }
        ivf::convert_fp32_to_fp16_upload(coarse->h_cluster_centers, n_total_clusters, dim, &coarse->d_cluster_centers_fp16);

    }

#if PARASELECT_ENABLE_CUVS
    if (!coarse->cuvs_res) {
        coarse->cuvs_res = new raft::resources();
    }
#endif
}
