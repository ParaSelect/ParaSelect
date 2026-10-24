#include "utils/pch.h"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/kernels/fine/gpu_fine/resident/resident_chunked_l2_topk.cuh"
#include "search/kernels/fine/gpu_fine/resident/resident_full_l2_topk.cuh"
#include "search/kernels/fine/gpu_fine/resident/resident_l2_topk_combos.cuh"
#include "search/kernels/fine/gpu_fine/resident/resident_threshold_l2_topk.cuh"
#include "utils/utils.cuh"
#include "utils/check_cuda_error.cuh"
#include <cfloat>
#include <stdexcept>

namespace {

__global__ void merge_probe_topk_and_map_ids_kernel(
    const float* __restrict__ d_probe_dist,
    const idx_t* __restrict__ d_probe_index,
    const idx_t* __restrict__ d_reordered_indices,
    float* __restrict__ d_final_dist,
    idx_t* __restrict__ d_final_index,
    int n_query,
    int n_probes,
    int k,
    int n_total_vectors) {
    const int q = blockIdx.x;
    if (q >= n_query || threadIdx.x != 0) return;
    float best_dist[256];
    idx_t best_idx[256];
    if (k > 256) return;
    for (int i = 0; i < k; ++i) {
        best_dist[i] = FLT_MAX;
        best_idx[i] = (idx_t)-1;
    }
    const int candidates = n_probes * k;
    const size_t row_base = (size_t)q * (size_t)candidates;
    for (int c = 0; c < candidates; ++c) {
        const float dist = d_probe_dist[row_base + (size_t)c];
        const idx_t compact_id = d_probe_index[row_base + (size_t)c];
        if (compact_id < 0 || compact_id >= n_total_vectors) continue;
        int pos = -1;
        for (int r = 0; r < k; ++r) {
            if (dist < best_dist[r]) {
                pos = r;
                break;
            }
        }
        if (pos < 0) continue;
        for (int r = k - 1; r > pos; --r) {
            best_dist[r] = best_dist[r - 1];
            best_idx[r] = best_idx[r - 1];
        }
        best_dist[pos] = dist;
        best_idx[pos] = d_reordered_indices[(int)compact_id];
    }
    for (int r = 0; r < k; ++r) {
        d_final_dist[(size_t)q * (size_t)k + (size_t)r] = best_dist[r];
        d_final_index[(size_t)q * (size_t)k + (size_t)r] = best_idx[r];
    }
}

__device__ __forceinline__ bool resident_merge_pair_less(float da, int ia, float db, int ib) {
    return (da < db) || (da == db && ia < ib);
}

template <int KCap>
__device__ void resident_merge_insert_candidate(
    float dist,
    int idx,
    int k,
    float* best_dist,
    int* best_idx) {
    if (idx < 0) return;
    if (!resident_merge_pair_less(dist, idx, best_dist[k - 1], best_idx[k - 1])) return;
    int pos = k - 1;
    while (pos > 0 && resident_merge_pair_less(dist, idx, best_dist[pos - 1], best_idx[pos - 1])) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
    }
    best_dist[pos] = dist;
    best_idx[pos] = idx;
}

template <int KCap>
__global__ void merge_probe_topk_single_warp_and_map_kernel(
    const float* __restrict__ d_probe_dist,
    const idx_t* __restrict__ d_probe_index,
    const idx_t* __restrict__ d_reordered_indices,
    float* __restrict__ d_final_dist,
    idx_t* __restrict__ d_final_index,
    int n_query,
    int n_probes,
    int k,
    int n_total_vectors) {
    const int q = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (q >= n_query) return;

    float local_dist[KCap];
    int local_idx[KCap];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = FLT_MAX;
        local_idx[i] = INT_MAX;
    }

    const int width = n_probes * k;
    const size_t base = (size_t)q * (size_t)width;
    for (int col = lane; col < width; col += 32) {
        const idx_t raw_id = d_probe_index[base + (size_t)col];
        if (raw_id < 0 || raw_id >= n_total_vectors || raw_id > (idx_t)INT_MAX) continue;
        resident_merge_insert_candidate<KCap>(
            d_probe_dist[base + (size_t)col],
            (int)raw_id,
            k,
            local_dist,
            local_idx);
    }

    __shared__ float warp_dist[32 * KCap];
    __shared__ int warp_idx[32 * KCap];
    for (int i = 0; i < k; ++i) {
        warp_dist[(size_t)lane * KCap + (size_t)i] = local_dist[i];
        warp_idx[(size_t)lane * KCap + (size_t)i] = local_idx[i];
    }
    __syncwarp();

    if (lane == 0) {
        float final_dist[KCap];
        int final_idx[KCap];
        for (int i = 0; i < k; ++i) {
            final_dist[i] = FLT_MAX;
            final_idx[i] = INT_MAX;
        }
        for (int l = 0; l < 32; ++l) {
            for (int i = 0; i < k; ++i) {
                resident_merge_insert_candidate<KCap>(
                    warp_dist[(size_t)l * KCap + (size_t)i],
                    warp_idx[(size_t)l * KCap + (size_t)i],
                    k,
                    final_dist,
                    final_idx);
            }
        }
        for (int i = 0; i < k; ++i) {
            const int compact_id = final_idx[i];
            d_final_dist[(size_t)q * (size_t)k + (size_t)i] = final_dist[i];
            d_final_index[(size_t)q * (size_t)k + (size_t)i] =
                compact_id == INT_MAX ? (idx_t)-1 : d_reordered_indices[compact_id];
        }
    }
}

void launch_merge_probe_topk_warpsort_and_map(
    const float* d_probe_dist,
    const idx_t* d_probe_index,
    const idx_t* d_reordered_indices,
    float* d_final_dist,
    idx_t* d_final_index,
    int n_query,
    int n_probes,
    int k,
    int n_total_vectors,
    cudaStream_t stream) {
    if (n_query <= 0 || n_probes <= 0 || k <= 0) return;

    if (k <= 64) {
        merge_probe_topk_single_warp_and_map_kernel<64><<<n_query, 32, 0, stream>>>(
            d_probe_dist, d_probe_index, d_reordered_indices, d_final_dist, d_final_index,
            n_query, n_probes, k, n_total_vectors);
    } else if (k <= 128) {
        merge_probe_topk_single_warp_and_map_kernel<128><<<n_query, 32, 0, stream>>>(
            d_probe_dist, d_probe_index, d_reordered_indices, d_final_dist, d_final_index,
            n_query, n_probes, k, n_total_vectors);
    } else {
        merge_probe_topk_and_map_ids_kernel<<<n_query, 1, 0, stream>>>(
            d_probe_dist, d_probe_index, d_reordered_indices, d_final_dist, d_final_index,
            n_query, n_probes, k, n_total_vectors);
    }
}

} // namespace

void ivfflat_fine_gpu_resident(IVFSearchCtx* ctx, int pid) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFFlatFineCtx* fine = &ctx->flat_fine;
    IVFSearchArg* arg = &ctx->arg;
    cudaStream_t stream = coarse->stream_0 ? coarse->stream_0 : 0;
    const int n_query = coarse->current_n_querys;
    const int nlist = arg->n_total_clusters;
    const int n_probes = arg->n_probes;
    const int k = arg->k;
    const int max_pairs = n_query * n_probes;
    if (arg->distance_mode != (int)L2_DISTANCE) {
        throw std::runtime_error("resident IVFFlat currently supports L2 only");
    }
    const bool resident_u8 = fine->resident_interleaved_u8 != 0;
    const bool has_resident_fp32 = fine->d_resident_interleaved_vectors != nullptr;
    const bool has_resident_u8 = fine->d_resident_interleaved_vectors_u8 != nullptr;
    if (ctx->arg.fine_kind != IVFFlatFineKind::Resident || has_resident_fp32 == has_resident_u8 || !fine->d_resident_interleaved_norms ||
        !fine->d_resident_interleaved_reordered_indices) {
        throw std::runtime_error("resident IVFFlat fine called without valid resident storage");
    }
    if (resident_u8 && !coarse->d_pool_query_batch_u8[pid]) {
        throw std::runtime_error("u8 resident IVFFlat fine called without u8 device queries");
    }
    fine->debug_resident_entry_build_ms = 0.0;
    fine->debug_resident_probe_init_ms = 0.0;
    fine->debug_resident_scan_ms = 0.0;
    fine->debug_resident_merge_map_ms = 0.0;
    fine->debug_resident_threshold_seed_build_ms = 0.0;
    fine->debug_resident_threshold_sketch_ms = 0.0;
    fine->debug_resident_threshold_scan_ms = 0.0;
    fine->debug_resident_threshold_reduce_ms = 0.0;
    fine->debug_resident_threshold_avg_candidates = 0.0;
    fine->debug_resident_threshold_max_candidates = 0.0;
    fine->debug_resident_d2h_ms = 0.0;

    auto time_gpu_part = [&](auto&& fn) -> double {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return (double)ms;
    };

    dim3 grid_nlist(nlist);
    int n_entry = 0;
    fine->debug_resident_entry_build_ms = time_gpu_part([&]() {
        count_entries_per_cluster_kernel<<<grid_nlist, 1, 0, stream>>>(
            coarse->d_pool_cluster_offset[pid],
            fine->d_resident_entry_count_per_cluster[pid],
            nlist,
            8);
        CUDA_LAUNCH_CHECK("resident count_entries_per_cluster_kernel");
        compute_prefix_sum_scratch(
            fine->d_resident_entry_count_per_cluster[pid],
            fine->d_resident_entry_offset[pid],
            nlist,
            coarse->d_fine_scan_block_sums,
            stream);
        CUDA_CHECK(cudaMemcpyAsync(&n_entry,
                                   fine->d_resident_entry_offset[pid] + nlist,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        build_entry_data_kernel<<<grid_nlist, 1, 0, stream>>>(
            coarse->d_pool_cluster_offset[pid],
            coarse->d_cluster_query_data[pid],
            coarse->d_cluster_query_probe_indices[pid],
            fine->d_resident_entry_offset[pid],
            coarse->d_pool_cluster_offset[pid],
            fine->d_resident_entry_cluster_id[pid],
            fine->d_resident_entry_query_start[pid],
            fine->d_resident_entry_query_count[pid],
            fine->d_resident_entry_queries[pid],
            fine->d_resident_entry_probe_indices[pid],
            nlist,
            8);
        CUDA_LAUNCH_CHECK("resident build_entry_data_kernel");
    });
    if (n_entry < 0 || n_entry > max_pairs) {
        throw std::runtime_error("resident IVFFlat invalid entry count");
    }

    fine->debug_resident_probe_init_ms = time_gpu_part([&]() {
        const int total_slots = n_query * n_probes * k;
        dim3 block(256, 1, 1);
        dim3 grid((total_slots + block.x - 1) / block.x, 1, 1);
        init_invalid_values_kernel<<<grid, block, 0, stream>>>(
            fine->d_resident_probe_topk_dist[pid],
            fine->d_resident_probe_topk_index[pid],
            total_slots);
        CUDA_LAUNCH_CHECK("resident init probe topk");
    });

    const auto* resident_combo =
        paraselect::resident_fine::resolve_resident_l2_topk_combo_ptr_or_throw(
            fine->resident_combo_name[0] ? fine->resident_combo_name
                                         : "chunked");
    // Experimental vec4 layout: the same buffer permutation is consumed by both
    // the threshold sketch and the scan kernel of the dedicated vec4 combos.
    const bool resident_vec4_layout =
        resident_combo->kind ==
            paraselect::resident_fine::ResidentL2TopKKernelKind::ThresholdGroupedVec4 ||
        resident_combo->kind ==
            paraselect::resident_fine::ResidentL2TopKKernelKind::ThresholdGroupedVec4Shq;
    const auto build_chunk_tasks = [&]() {
        const int n_rows = n_query * n_probes;
        CUDA_CHECK(cudaMemsetAsync(
            fine->d_resident_row_chunk_count[pid], 0,
            ((size_t)n_rows + 1) * sizeof(int), stream));
        launch_resident_count_row_chunks(
            fine->d_resident_entry_cluster_id[pid],
            fine->d_resident_entry_query_start[pid],
            fine->d_resident_entry_query_count[pid],
            fine->d_resident_entry_queries[pid],
            fine->d_resident_entry_probe_indices[pid],
            coarse->d_cluster_size, n_entry, n_probes,
            fine->resident_chunk_blocks,
            fine->d_resident_row_chunk_count[pid], stream);
        CUDA_LAUNCH_CHECK("resident count row chunks");
        compute_prefix_sum(
            fine->d_resident_row_chunk_count[pid],
            fine->d_resident_row_chunk_offset[pid], n_rows, stream);
        CUDA_CHECK(cudaMemcpyAsync(
            fine->d_resident_row_chunk_write_pos[pid],
            fine->d_resident_row_chunk_offset[pid],
            (size_t)n_rows * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        int n_tasks = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &n_tasks, fine->d_resident_row_chunk_offset[pid] + n_rows,
            sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (n_tasks > fine->resident_max_chunk_tasks)
            throw std::runtime_error(
                "resident IVFFlat chunk task overflow");
        launch_resident_fill_row_chunk_tasks(
            fine->d_resident_entry_cluster_id[pid],
            fine->d_resident_entry_query_start[pid],
            fine->d_resident_entry_query_count[pid],
            fine->d_resident_entry_queries[pid],
            fine->d_resident_entry_probe_indices[pid],
            coarse->d_cluster_size, n_entry, n_probes,
            fine->resident_chunk_blocks, fine->resident_max_chunk_tasks,
            fine->d_resident_row_chunk_write_pos[pid],
            fine->d_resident_chunk_task_row[pid],
            fine->d_resident_chunk_task_cluster[pid],
            fine->d_resident_chunk_task_first_block[pid],
            fine->d_resident_chunk_task_num_blocks[pid], stream);
        CUDA_LAUNCH_CHECK("resident fill row chunk tasks");
        return n_tasks;
    };

    bool threshold_direct_output = false;
    fine->debug_resident_scan_ms = time_gpu_part([&]() {
        if (paraselect::resident_fine::is_threshold_kind(
                resident_combo->kind)) {
            if (!fine->d_resident_threshold[pid] ||
                !fine->d_resident_threshold_seed_entry_count_per_cluster[pid] ||
                !fine->d_resident_threshold_seed_entry_offset[pid] ||
                !fine->d_resident_threshold_seed_entry_cluster_id[pid] ||
                !fine->d_resident_threshold_seed_entry_query_start[pid] ||
                !fine->d_resident_threshold_seed_entry_query_count[pid] ||
                !fine->d_resident_threshold_seed_entry_queries[pid] ||
                !fine->d_resident_threshold_seed_entry_probe_indices[pid] ||
                !fine->d_resident_threshold_candidate_count[pid] ||
                !fine->d_resident_threshold_overflow[pid] ||
                !fine->d_resident_threshold_candidate_dist[pid] ||
                !fine->d_resident_threshold_candidate_index[pid])
                throw std::runtime_error(
                    "resident threshold IVFFlat workspace is missing");

            const int seed_probes =
                std::min(fine->resident_threshold_seed_probes, n_probes);
            int n_seed_entry = 0;
            fine->debug_resident_threshold_seed_build_ms =
                time_gpu_part([&]() {
                    launch_resident_threshold_count_seed_entries(
                        coarse->d_pool_cluster_offset[pid],
                        coarse->d_cluster_query_probe_indices[pid], nlist,
                        seed_probes,
                        fine->d_resident_threshold_seed_entry_count_per_cluster[pid],
                        stream);
                    CUDA_LAUNCH_CHECK(
                        "resident threshold count seed entries");
                    compute_prefix_sum_scratch(
                        fine->d_resident_threshold_seed_entry_count_per_cluster[pid],
                        fine->d_resident_threshold_seed_entry_offset[pid],
                        nlist,
                        coarse->d_fine_scan_block_sums,
                        stream);
                    CUDA_CHECK(cudaMemcpyAsync(
                        &n_seed_entry,
                        fine->d_resident_threshold_seed_entry_offset[pid] +
                            nlist,
                        sizeof(int), cudaMemcpyDeviceToHost, stream));
                    launch_resident_threshold_fill_seed_entries(
                        coarse->d_pool_cluster_offset[pid],
                        coarse->d_cluster_query_data[pid],
                        coarse->d_cluster_query_probe_indices[pid],
                        fine->d_resident_threshold_seed_entry_offset[pid],
                        nlist, seed_probes,
                        fine->d_resident_threshold_seed_entry_cluster_id[pid],
                        fine->d_resident_threshold_seed_entry_query_start[pid],
                        fine->d_resident_threshold_seed_entry_query_count[pid],
                        fine->d_resident_threshold_seed_entry_queries[pid],
                        fine->d_resident_threshold_seed_entry_probe_indices[pid],
                        stream);
                    CUDA_LAUNCH_CHECK(
                        "resident threshold fill seed entries");
                });
            if (n_seed_entry < 0 || n_seed_entry > max_pairs)
                throw std::runtime_error(
                    "resident threshold IVFFlat invalid seed entry count");

            CUDA_CHECK(cudaMemsetAsync(
                fine->d_resident_threshold_candidate_count[pid], 0,
                (size_t)n_query * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(
                fine->d_resident_threshold_overflow[pid], 0,
                sizeof(int), stream));

            fine->debug_resident_threshold_sketch_ms =
                time_gpu_part([&]() {
                    if (resident_u8) {
                        launch_resident_u8_l2_topk_dp4a_distances(
                            k, dim3(256, 1, 1), arg->n_dim,
                            coarse->d_pool_query_batch_u8[pid],
                            fine->d_resident_interleaved_vectors_u8,
                            coarse->d_probe_vector_offset,
                            coarse->d_probe_vector_offset_compact,
                            coarse->d_cluster_size,
                            fine->d_resident_threshold_seed_entry_cluster_id[pid],
                            fine->d_resident_threshold_seed_entry_query_start[pid],
                            fine->d_resident_threshold_seed_entry_query_count[pid],
                            fine->d_resident_threshold_seed_entry_queries[pid],
                            fine->d_resident_threshold_seed_entry_probe_indices[pid],
                            coarse->d_pool_query_norm[pid],
                            fine->d_resident_interleaved_norms,
                            n_seed_entry, n_probes,
                            fine->d_resident_probe_topk_dist[pid],
                            resident_vec4_layout, stream);
                    } else {
                        launch_resident_l2_topk_full(
                            k, dim3(256, 1, 1), arg->n_dim,
                            coarse->d_pool_query_batch[pid],
                            fine->d_resident_interleaved_vectors,
                            coarse->d_probe_vector_offset,
                            coarse->d_probe_vector_offset_compact,
                            coarse->d_cluster_size,
                            fine->d_resident_threshold_seed_entry_cluster_id[pid],
                            fine->d_resident_threshold_seed_entry_query_start[pid],
                            fine->d_resident_threshold_seed_entry_query_count[pid],
                            fine->d_resident_threshold_seed_entry_queries[pid],
                            fine->d_resident_threshold_seed_entry_probe_indices[pid],
                            coarse->d_pool_query_norm[pid],
                            fine->d_resident_interleaved_norms,
                            n_seed_entry, n_probes,
                            fine->d_resident_probe_topk_dist[pid],
                            fine->d_resident_probe_topk_index[pid], stream);
                    }
                    launch_resident_threshold_from_probe_topk(
                        k, n_query, n_probes, seed_probes,
                        fine->d_resident_probe_topk_dist[pid],
                        fine->d_resident_threshold[pid], stream);
                    CUDA_LAUNCH_CHECK("resident threshold prepass");
                });

            std::vector<int> candidate_counts((size_t)n_query);
            int overflow = 0;
            fine->debug_resident_threshold_scan_ms =
                time_gpu_part([&]() {
                    if (resident_combo->kind ==
                        paraselect::resident_fine::ResidentL2TopKKernelKind::
                            ThresholdQuery) {
                        const int n_tasks = build_chunk_tasks();
                        launch_resident_threshold_query_scan(
                            arg->n_dim, n_tasks, n_probes,
                            fine->resident_threshold_candidate_capacity,
                            coarse->d_pool_query_batch[pid],
                            coarse->d_pool_query_batch_u8[pid],
                            fine->d_resident_interleaved_vectors,
                            fine->d_resident_interleaved_vectors_u8,
                            coarse->d_probe_vector_offset,
                            coarse->d_probe_vector_offset_compact,
                            coarse->d_cluster_size,
                            fine->d_resident_chunk_task_row[pid],
                            fine->d_resident_chunk_task_cluster[pid],
                            fine->d_resident_chunk_task_first_block[pid],
                            fine->d_resident_chunk_task_num_blocks[pid],
                            coarse->d_pool_query_norm[pid],
                            fine->d_resident_interleaved_norms,
                            fine->d_resident_threshold[pid], resident_u8,
                            fine->d_resident_threshold_candidate_count[pid],
                            fine->d_resident_threshold_overflow[pid],
                            fine->d_resident_threshold_candidate_dist[pid],
                            fine->d_resident_threshold_candidate_index[pid],
                            stream);
                    } else if (resident_u8 &&
                               resident_combo->kind ==
                                   paraselect::resident_fine::
                                       ResidentL2TopKKernelKind::
                                           ThresholdGroupedPair) {
                        launch_resident_threshold_grouped_u8_pair_scan(
                            arg->n_dim, n_entry,
                            fine->resident_threshold_candidate_capacity,
                            coarse->d_pool_query_batch_u8[pid],
                            fine->d_resident_interleaved_vectors_u8,
                            coarse->d_probe_vector_offset,
                            coarse->d_probe_vector_offset_compact,
                            coarse->d_cluster_size,
                            fine->d_resident_entry_cluster_id[pid],
                            fine->d_resident_entry_query_start[pid],
                            fine->d_resident_entry_query_count[pid],
                            fine->d_resident_entry_queries[pid],
                            coarse->d_pool_query_norm[pid],
                            fine->d_resident_interleaved_norms,
                            fine->d_resident_threshold[pid],
                            fine->d_resident_threshold_candidate_count[pid],
                            fine->d_resident_threshold_overflow[pid],
                            fine->d_resident_threshold_candidate_dist[pid],
                            fine->d_resident_threshold_candidate_index[pid],
                            stream);
                    } else if (resident_u8 &&
                               resident_combo->kind ==
                                   paraselect::resident_fine::
                                       ResidentL2TopKKernelKind::
                                           ThresholdGroupedShq &&
                               launch_resident_threshold_grouped_shq_scan(
                                   arg->n_dim, n_entry,
                                   fine->resident_threshold_candidate_capacity,
                                   coarse->d_pool_query_batch_u8[pid],
                                   fine->d_resident_interleaved_vectors_u8,
                                   coarse->d_probe_vector_offset,
                                   coarse->d_probe_vector_offset_compact,
                                   coarse->d_cluster_size,
                                   fine->d_resident_entry_cluster_id[pid],
                                   fine->d_resident_entry_query_start[pid],
                                   fine->d_resident_entry_query_count[pid],
                                   fine->d_resident_entry_queries[pid],
                                   coarse->d_pool_query_norm[pid],
                                   fine->d_resident_interleaved_norms,
                                   fine->d_resident_threshold[pid],
                                   fine->d_resident_threshold_candidate_count[pid],
                                   fine->d_resident_threshold_overflow[pid],
                                   fine->d_resident_threshold_candidate_dist[pid],
                                   fine->d_resident_threshold_candidate_index[pid],
                                   stream)) {
                        // shared-query kernel handled the request
                    } else if (resident_u8 &&
                               resident_combo->kind ==
                                   paraselect::resident_fine::
                                       ResidentL2TopKKernelKind::
                                           ThresholdGroupedVec4 &&
                               launch_resident_threshold_grouped_vec4_scan(
                                   arg->n_dim, n_entry,
                                   fine->resident_threshold_candidate_capacity,
                                   coarse->d_pool_query_batch_u8[pid],
                                   fine->d_resident_interleaved_vectors_u8,
                                   coarse->d_probe_vector_offset,
                                   coarse->d_probe_vector_offset_compact,
                                   coarse->d_cluster_size,
                                   fine->d_resident_entry_cluster_id[pid],
                                   fine->d_resident_entry_query_start[pid],
                                   fine->d_resident_entry_query_count[pid],
                                   fine->d_resident_entry_queries[pid],
                                   coarse->d_pool_query_norm[pid],
                                   fine->d_resident_interleaved_norms,
                                   fine->d_resident_threshold[pid],
                                   fine->d_resident_threshold_candidate_count[pid],
                                   fine->d_resident_threshold_overflow[pid],
                                   fine->d_resident_threshold_candidate_dist[pid],
                                   fine->d_resident_threshold_candidate_index[pid],
                                   stream)) {
                        // vec4 kernel handled the request
                    } else if (resident_u8 &&
                               resident_combo->kind ==
                                   paraselect::resident_fine::
                                       ResidentL2TopKKernelKind::
                                           ThresholdGroupedVec4Shq &&
                               launch_resident_threshold_grouped_vec4_shq_scan(
                                   arg->n_dim, n_entry,
                                   fine->resident_threshold_candidate_capacity,
                                   coarse->d_pool_query_batch_u8[pid],
                                   fine->d_resident_interleaved_vectors_u8,
                                   coarse->d_probe_vector_offset,
                                   coarse->d_probe_vector_offset_compact,
                                   coarse->d_cluster_size,
                                   fine->d_resident_entry_cluster_id[pid],
                                   fine->d_resident_entry_query_start[pid],
                                   fine->d_resident_entry_query_count[pid],
                                   fine->d_resident_entry_queries[pid],
                                   coarse->d_pool_query_norm[pid],
                                   fine->d_resident_interleaved_norms,
                                   fine->d_resident_threshold[pid],
                                   fine->d_resident_threshold_candidate_count[pid],
                                   fine->d_resident_threshold_overflow[pid],
                                   fine->d_resident_threshold_candidate_dist[pid],
                                   fine->d_resident_threshold_candidate_index[pid],
                                   stream)) {
                        // vec4 + shared-query kernel handled the request
                    } else if (resident_u8 &&
                               resident_combo->kind ==
                                   paraselect::resident_fine::
                                       ResidentL2TopKKernelKind::
                                           ThresholdGroupedMlp2 &&
                               launch_resident_threshold_grouped_mlp2_scan(
                                   arg->n_dim, n_entry,
                                   fine->resident_threshold_candidate_capacity,
                                   coarse->d_pool_query_batch_u8[pid],
                                   fine->d_resident_interleaved_vectors_u8,
                                   coarse->d_probe_vector_offset,
                                   coarse->d_probe_vector_offset_compact,
                                   coarse->d_cluster_size,
                                   fine->d_resident_entry_cluster_id[pid],
                                   fine->d_resident_entry_query_start[pid],
                                   fine->d_resident_entry_query_count[pid],
                                   fine->d_resident_entry_queries[pid],
                                   coarse->d_pool_query_norm[pid],
                                   fine->d_resident_interleaved_norms,
                                   fine->d_resident_threshold[pid],
                                   fine->d_resident_threshold_candidate_count[pid],
                                   fine->d_resident_threshold_overflow[pid],
                                   fine->d_resident_threshold_candidate_dist[pid],
                                   fine->d_resident_threshold_candidate_index[pid],
                                   stream)) {
                        // mlp2 kernel handled the request
                    } else if (resident_u8 &&
                               resident_combo->kind ==
                                   paraselect::resident_fine::
                                       ResidentL2TopKKernelKind::
                                           ThresholdGroupedCpasync &&
                               launch_resident_threshold_grouped_cpasync_scan(
                                   arg->n_dim, n_entry,
                                   fine->resident_threshold_candidate_capacity,
                                   coarse->d_pool_query_batch_u8[pid],
                                   fine->d_resident_interleaved_vectors_u8,
                                   coarse->d_probe_vector_offset,
                                   coarse->d_probe_vector_offset_compact,
                                   coarse->d_cluster_size,
                                   fine->d_resident_entry_cluster_id[pid],
                                   fine->d_resident_entry_query_start[pid],
                                   fine->d_resident_entry_query_count[pid],
                                   fine->d_resident_entry_queries[pid],
                                   coarse->d_pool_query_norm[pid],
                                   fine->d_resident_interleaved_norms,
                                   fine->d_resident_threshold[pid],
                                   fine->d_resident_threshold_candidate_count[pid],
                                   fine->d_resident_threshold_overflow[pid],
                                   fine->d_resident_threshold_candidate_dist[pid],
                                   fine->d_resident_threshold_candidate_index[pid],
                                   stream)) {
                        // cp.async kernel handled the request
                    } else {
                        launch_resident_threshold_grouped_scan(
                            arg->n_dim, n_entry,
                            fine->resident_threshold_candidate_capacity,
                            coarse->d_pool_query_batch[pid],
                            coarse->d_pool_query_batch_u8[pid],
                            fine->d_resident_interleaved_vectors,
                            fine->d_resident_interleaved_vectors_u8,
                            coarse->d_probe_vector_offset,
                            coarse->d_probe_vector_offset_compact,
                            coarse->d_cluster_size,
                            fine->d_resident_entry_cluster_id[pid],
                            fine->d_resident_entry_query_start[pid],
                            fine->d_resident_entry_query_count[pid],
                            fine->d_resident_entry_queries[pid],
                            coarse->d_pool_query_norm[pid],
                            fine->d_resident_interleaved_norms,
                            fine->d_resident_threshold[pid], resident_u8,
                            fine->d_resident_threshold_candidate_count[pid],
                            fine->d_resident_threshold_overflow[pid],
                            fine->d_resident_threshold_candidate_dist[pid],
                            fine->d_resident_threshold_candidate_index[pid],
                            stream);
                    }
                    CUDA_LAUNCH_CHECK("resident threshold scan");
                    CUDA_CHECK(cudaMemcpyAsync(
                        candidate_counts.data(),
                        fine->d_resident_threshold_candidate_count[pid],
                        (size_t)n_query * sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        &overflow,
                        fine->d_resident_threshold_overflow[pid],
                        sizeof(int), cudaMemcpyDeviceToHost, stream));
                });
            long long candidate_sum = 0;
            int candidate_max = 0;
            for (int count : candidate_counts) {
                candidate_sum += count;
                candidate_max = std::max(candidate_max, count);
            }
            fine->debug_resident_threshold_avg_candidates =
                n_query > 0
                    ? (double)candidate_sum / (double)n_query
                    : 0.0;
            fine->debug_resident_threshold_max_candidates =
                (double)candidate_max;

            if (overflow) {
                throw std::runtime_error(
                    "resident threshold IVFFlat candidate capacity exceeded: max=" +
                    std::to_string(candidate_max) + " capacity=" +
                    std::to_string(fine->resident_threshold_candidate_capacity));
            }

            fine->debug_resident_threshold_reduce_ms =
                time_gpu_part([&]() {
                    launch_resident_threshold_reduce(
                        k, n_query,
                        fine->resident_threshold_candidate_capacity,
                        arg->n_total_vectors,
                        fine->d_resident_threshold_candidate_count[pid],
                        fine->d_resident_threshold_candidate_dist[pid],
                        fine->d_resident_threshold_candidate_index[pid],
                        fine->d_resident_interleaved_reordered_indices,
                        fine->d_resident_final_topk_dist[pid],
                        fine->d_resident_final_topk_index[pid], stream);
                    CUDA_LAUNCH_CHECK("resident threshold reduce");
                });
            threshold_direct_output = true;
            return;
        }

        if (resident_combo->kind ==
            paraselect::resident_fine::ResidentL2TopKKernelKind::Dp4a) {
            if (!resident_u8)
                throw std::runtime_error(
                    "resident IVFFlat dp4a combo requires u8 storage");
            launch_resident_u8_l2_topk_dp4a(
                k, dim3(256, 1, 1), arg->n_dim,
                coarse->d_pool_query_batch_u8[pid],
                fine->d_resident_interleaved_vectors_u8,
                coarse->d_probe_vector_offset,
                coarse->d_probe_vector_offset_compact,
                coarse->d_cluster_size,
                fine->d_resident_entry_cluster_id[pid],
                fine->d_resident_entry_query_start[pid],
                fine->d_resident_entry_query_count[pid],
                fine->d_resident_entry_queries[pid],
                fine->d_resident_entry_probe_indices[pid],
                coarse->d_pool_query_norm[pid],
                fine->d_resident_interleaved_norms, n_entry, n_probes,
                fine->d_resident_probe_topk_dist[pid],
                fine->d_resident_probe_topk_index[pid], stream);
            CUDA_LAUNCH_CHECK("resident u8 dp4a l2_topk");
            return;
        }

        if (resident_combo->kind ==
            paraselect::resident_fine::ResidentL2TopKKernelKind::Full) {
            if (resident_u8) {
                launch_resident_u8_l2_topk_full(
                    k, dim3(256, 1, 1), arg->n_dim,
                    coarse->d_pool_query_batch_u8[pid],
                    fine->d_resident_interleaved_vectors_u8,
                    coarse->d_probe_vector_offset,
                    coarse->d_probe_vector_offset_compact,
                    coarse->d_cluster_size,
                    fine->d_resident_entry_cluster_id[pid],
                    fine->d_resident_entry_query_start[pid],
                    fine->d_resident_entry_query_count[pid],
                    fine->d_resident_entry_queries[pid],
                    fine->d_resident_entry_probe_indices[pid],
                    coarse->d_pool_query_norm[pid],
                    fine->d_resident_interleaved_norms, n_entry, n_probes,
                    fine->d_resident_probe_topk_dist[pid],
                    fine->d_resident_probe_topk_index[pid], stream);
                CUDA_LAUNCH_CHECK(
                    "resident u8 gemv_interleaved_l2_topk_full");
            } else {
                launch_resident_l2_topk_full(
                    k, dim3(256, 1, 1), arg->n_dim,
                    coarse->d_pool_query_batch[pid],
                    fine->d_resident_interleaved_vectors,
                    coarse->d_probe_vector_offset,
                    coarse->d_probe_vector_offset_compact,
                    coarse->d_cluster_size,
                    fine->d_resident_entry_cluster_id[pid],
                    fine->d_resident_entry_query_start[pid],
                    fine->d_resident_entry_query_count[pid],
                    fine->d_resident_entry_queries[pid],
                    fine->d_resident_entry_probe_indices[pid],
                    coarse->d_pool_query_norm[pid],
                    fine->d_resident_interleaved_norms, n_entry, n_probes,
                    fine->d_resident_probe_topk_dist[pid],
                    fine->d_resident_probe_topk_index[pid], stream);
                CUDA_LAUNCH_CHECK(
                    "resident gemv_interleaved_l2_topk_full");
            }
            return;
        }

        const int n_rows = n_query * n_probes;
        const int n_tasks = build_chunk_tasks();
        if (resident_u8) {
            launch_resident_u8_l2_topk_chunk_scan(
                k, dim3(256, 1, 1), arg->n_dim,
                coarse->d_pool_query_batch_u8[pid],
                fine->d_resident_interleaved_vectors_u8,
                coarse->d_probe_vector_offset,
                coarse->d_probe_vector_offset_compact,
                coarse->d_cluster_size,
                fine->d_resident_chunk_task_row[pid],
                fine->d_resident_chunk_task_cluster[pid],
                fine->d_resident_chunk_task_first_block[pid],
                fine->d_resident_chunk_task_num_blocks[pid],
                coarse->d_pool_query_norm[pid],
                fine->d_resident_interleaved_norms, n_tasks, n_probes,
                fine->d_resident_chunk_candidate_dist[pid],
                fine->d_resident_chunk_candidate_index[pid], stream);
        } else {
            launch_resident_l2_topk_chunk_scan(
                k, dim3(256, 1, 1), arg->n_dim,
                coarse->d_pool_query_batch[pid],
                fine->d_resident_interleaved_vectors,
                coarse->d_probe_vector_offset,
                coarse->d_probe_vector_offset_compact,
                coarse->d_cluster_size,
                fine->d_resident_chunk_task_row[pid],
                fine->d_resident_chunk_task_cluster[pid],
                fine->d_resident_chunk_task_first_block[pid],
                fine->d_resident_chunk_task_num_blocks[pid],
                coarse->d_pool_query_norm[pid],
                fine->d_resident_interleaved_norms, n_tasks, n_probes,
                fine->d_resident_chunk_candidate_dist[pid],
                fine->d_resident_chunk_candidate_index[pid], stream);
        }
        CUDA_LAUNCH_CHECK("resident chunk scan candidates");
        launch_resident_reduce_row_chunks(
            k, dim3(256, 1, 1),
            fine->d_resident_row_chunk_offset[pid],
            fine->d_resident_chunk_candidate_dist[pid],
            fine->d_resident_chunk_candidate_index[pid], n_rows,
            fine->d_resident_probe_topk_dist[pid],
            fine->d_resident_probe_topk_index[pid], stream);
        CUDA_LAUNCH_CHECK("resident reduce row chunks");
    });

    if (!threshold_direct_output) {
        fine->debug_resident_merge_map_ms = time_gpu_part([&]() {
            launch_merge_probe_topk_warpsort_and_map(
                fine->d_resident_probe_topk_dist[pid],
                fine->d_resident_probe_topk_index[pid],
                fine->d_resident_interleaved_reordered_indices,
                fine->d_resident_final_topk_dist[pid],
                fine->d_resident_final_topk_index[pid], n_query, n_probes,
                k, arg->n_total_vectors, stream);
            CUDA_LAUNCH_CHECK(
                "resident merge_probe_topk_warpsort_and_map");
        });
    }

    fine->debug_resident_d2h_ms = time_gpu_part([&]() {
        CUDA_CHECK(cudaMemcpyAsync(
            fine->h_topk_dist_out,
            fine->d_resident_final_topk_dist[pid],
            (size_t)n_query * k * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            fine->h_topk_index_out,
            fine->d_resident_final_topk_index[pid],
            (size_t)n_query * k * sizeof(idx_t),
            cudaMemcpyDeviceToHost, stream));
    });
}
