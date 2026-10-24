#include "search/ivfrespq/stage/residual_pq_gpu.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include "utils/check_cuda_error.cuh"

#if PARASELECT_ENABLE_CUVS
#include <cuvs/selection/select_k.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <optional>
#endif

#include <cfloat>
#include <stdexcept>

namespace {

#if PARASELECT_ENABLE_CUVS
cudaError_t select_existing_candidates_cuvs(
    const float* input_vals,
    const int* input_idx,
    int batch_size,
    int len,
    int k,
    float* output_vals,
    int* output_idx,
    cudaStream_t stream)
{
    if (!input_vals || !input_idx || !output_vals || !output_idx ||
        batch_size <= 0 || len <= 0 || k <= 0 || k > len) {
        return cudaErrorInvalidValue;
    }

    raft::resources res;
    raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));

    auto in_vals = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        input_vals, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));
    auto in_idx_view = raft::make_device_matrix_view<const int, int64_t, raft::row_major>(
        input_idx, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));
    auto out_vals = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        output_vals, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));
    auto out_idx = raft::make_device_matrix_view<int, int64_t, raft::row_major>(
        output_idx, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));
    std::optional<decltype(in_idx_view)> in_idx_opt = in_idx_view;

    cuvs::selection::select_k(res, in_vals, in_idx_opt, out_vals, out_idx, true, true);
    raft::resource::sync_stream(res);
    return cudaGetLastError();
}
#endif

template <int GLOBAL_CAP, int WARPS_PER_BLOCK>
__global__ void residual_pq_query_reduce_warpsort_kernel(
    const int* __restrict__ probe_candidate_idx,
    const float* __restrict__ probe_candidate_dist,
    int n_query,
    int n_probes,
    int local_k,
    int global_k,
    int* __restrict__ global_idx)
{
    const int q = blockIdx.x;
    if (q >= n_query) return;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int total = n_probes * local_k;
    const float* qd = probe_candidate_dist + (size_t)q * total;
    const int* qi = probe_candidate_idx + (size_t)q * total;

    constexpr int kStoreCap = GLOBAL_CAP > 512 ? 512 : GLOBAL_CAP;
    __shared__ float warp_dist[WARPS_PER_BLOCK * kStoreCap];
    __shared__ int warp_idx[WARPS_PER_BLOCK * kStoreCap];
    __shared__ float final_dist[kStoreCap];
    __shared__ int final_idx[kStoreCap];

    paraselect::warpsort::WarpSortFiltered<GLOBAL_CAP, true, float, int> queue(global_k);
    for (int probe = warp_id; probe < n_probes; probe += WARPS_PER_BLOCK) {
        const int probe_base = probe * local_k;
        const int rounded_local_k = ((local_k + 31) / 32) * 32;
        for (int r = lane; r < rounded_local_k; r += 32) {
            const int pos = probe_base + r;
            const bool valid_pos = r < local_k;
            const float d = valid_pos ? qd[pos] : FLT_MAX;
            const int id = valid_pos ? qi[pos] : -1;
            const bool keep = (id >= 0 && d < queue.threshold());
            queue.add(keep ? d : FLT_MAX, keep ? id : -1);
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();
    queue.store(
        warp_dist + (size_t)warp_id * kStoreCap,
        warp_idx + (size_t)warp_id * kStoreCap);
    __syncthreads();

    if (warp_id == 0) {
        paraselect::warpsort::WarpSortFiltered<GLOBAL_CAP, true, float, int> merge_queue(global_k);
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            for (int r = lane; r < global_k; r += 32) {
                const float d = warp_dist[(size_t)w * kStoreCap + r];
                const int id = warp_idx[(size_t)w * kStoreCap + r];
                const bool keep = (id >= 0 && d < merge_queue.threshold());
                merge_queue.add(keep ? d : FLT_MAX, keep ? id : -1);
            }
        }
        __syncwarp();
        merge_queue.done();
        __syncwarp();
        merge_queue.store(final_dist, final_idx);
        __syncwarp();

        for (int i = lane; i < global_k; i += 32) {
            global_idx[(size_t)q * global_k + i] = final_idx[i];
        }
    }
}

template <int GLOBAL_CAP, int WARPS_PER_BLOCK>
__global__ void residual_pq_query_reduce_compact_warpsort_kernel(
    const int* __restrict__ source_candidate_idx,
    const float* __restrict__ source_candidate_dist,
    const int* __restrict__ source_offsets,
    int n_query,
    int n_probes,
    int local_k,
    int global_k,
    int* __restrict__ global_idx)
{
    const int q = blockIdx.x;
    if (q >= n_query) return;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int normal_begin = q * n_probes;
    const int normal_count = n_probes;
    const int normal_total = n_query * n_probes;
    const int extra_begin = source_offsets[q * n_probes];
    const int extra_end = source_offsets[(q + 1) * n_probes];
    const int extra_count = max(0, extra_end - extra_begin);
    const int source_count = normal_count + extra_count;

    constexpr int kStoreCap = GLOBAL_CAP > 512 ? 512 : GLOBAL_CAP;
    __shared__ float warp_dist[WARPS_PER_BLOCK * kStoreCap];
    __shared__ int warp_idx[WARPS_PER_BLOCK * kStoreCap];
    __shared__ float final_dist[kStoreCap];
    __shared__ int final_idx[kStoreCap];

    paraselect::warpsort::WarpSortFiltered<GLOBAL_CAP, true, float, int> queue(global_k);
    for (int source = warp_id; source < source_count; source += WARPS_PER_BLOCK) {
        const int compact_source = (source < normal_count)
            ? (normal_begin + source)
            : (normal_total + extra_begin + source - normal_count);
        const int source_base = compact_source * local_k;
        const int rounded_local_k = ((local_k + 31) / 32) * 32;
        for (int r = lane; r < rounded_local_k; r += 32) {
            const int pos = source_base + r;
            const bool valid_pos = r < local_k;
            const float d = valid_pos ? source_candidate_dist[pos] : FLT_MAX;
            const int id = valid_pos ? source_candidate_idx[pos] : -1;
            const bool keep = (id >= 0 && d < queue.threshold());
            queue.add(keep ? d : FLT_MAX, keep ? id : -1);
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();
    queue.store(
        warp_dist + (size_t)warp_id * kStoreCap,
        warp_idx + (size_t)warp_id * kStoreCap);
    __syncthreads();

    if (warp_id == 0) {
        paraselect::warpsort::WarpSortFiltered<GLOBAL_CAP, true, float, int> merge_queue(global_k);
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            for (int r = lane; r < global_k; r += 32) {
                const float d = warp_dist[(size_t)w * kStoreCap + r];
                const int id = warp_idx[(size_t)w * kStoreCap + r];
                const bool keep = (id >= 0 && d < merge_queue.threshold());
                merge_queue.add(keep ? d : FLT_MAX, keep ? id : -1);
            }
        }
        __syncwarp();
        merge_queue.done();
        __syncwarp();
        merge_queue.store(final_dist, final_idx);
        __syncwarp();

        for (int i = lane; i < global_k; i += 32) {
            global_idx[(size_t)q * global_k + i] = final_idx[i];
        }
    }
}

}  // namespace

void residual_pq_reduce_candidates_gpu(
    const int* d_probe_candidate_idx,
    const float* d_probe_candidate_dist,
    int n_query,
    int n_probes,
    int local_k,
    int global_k,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream)
{
    if (n_query <= 0 || n_probes <= 0 || local_k <= 0 || global_k <= 0) return;

    constexpr int kReduceWarps = 8;
    if (global_k <= 64) {
        residual_pq_query_reduce_warpsort_kernel<128, kReduceWarps>
            <<<n_query, kReduceWarps * 32, 0, stream>>>(
                d_probe_candidate_idx, d_probe_candidate_dist, n_query, n_probes,
                local_k, global_k, d_global_candidate_idx);
    } else if (global_k <= 128) {
        residual_pq_query_reduce_warpsort_kernel<256, kReduceWarps>
            <<<n_query, kReduceWarps * 32, 0, stream>>>(
                d_probe_candidate_idx, d_probe_candidate_dist, n_query, n_probes,
                local_k, global_k, d_global_candidate_idx);
    } else if (global_k <= 512) {
        residual_pq_query_reduce_warpsort_kernel<1024, kReduceWarps>
            <<<n_query, kReduceWarps * 32, 0, stream>>>(
                d_probe_candidate_idx, d_probe_candidate_dist, n_query, n_probes,
                local_k, global_k, d_global_candidate_idx);
    } else {
#if PARASELECT_ENABLE_CUVS
        CUDA_CHECK(select_existing_candidates_cuvs(
            d_probe_candidate_dist,
            d_probe_candidate_idx,
            n_query,
            n_probes * local_k,
            global_k,
            d_global_candidate_dist,
            d_global_candidate_idx,
            stream));
        return;
#else
        throw std::runtime_error("global_k > 512 requires PARASELECT_ENABLE_CUVS");
#endif
    }
    CUDA_LAUNCH_CHECK("residual_pq_query_reduce_warpsort_kernel");
}

void residual_pq_reduce_compact_candidates_gpu(
    const int* d_probe_candidate_idx,
    const float* d_probe_candidate_dist,
    const int* d_source_offsets,
    int n_query,
    int n_probes,
    int local_k,
    int global_k,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream)
{
    (void)d_global_candidate_dist;
    if (n_query <= 0 || n_probes <= 0 || local_k <= 0 || global_k <= 0) return;
    if (!d_source_offsets) {
        residual_pq_reduce_candidates_gpu(
            d_probe_candidate_idx,
            d_probe_candidate_dist,
            n_query,
            n_probes,
            local_k,
            global_k,
            d_global_candidate_idx,
            d_global_candidate_dist,
            stream);
        return;
    }

    constexpr int kReduceWarps = 8;
    if (global_k <= 64) {
        residual_pq_query_reduce_compact_warpsort_kernel<128, kReduceWarps>
            <<<n_query, kReduceWarps * 32, 0, stream>>>(
                d_probe_candidate_idx, d_probe_candidate_dist, d_source_offsets, n_query,
                n_probes, local_k, global_k, d_global_candidate_idx);
    } else if (global_k <= 128) {
        residual_pq_query_reduce_compact_warpsort_kernel<256, kReduceWarps>
            <<<n_query, kReduceWarps * 32, 0, stream>>>(
                d_probe_candidate_idx, d_probe_candidate_dist, d_source_offsets, n_query,
                n_probes, local_k, global_k, d_global_candidate_idx);
    } else {
        residual_pq_query_reduce_compact_warpsort_kernel<1024, kReduceWarps>
            <<<n_query, kReduceWarps * 32, 0, stream>>>(
                d_probe_candidate_idx, d_probe_candidate_dist, d_source_offsets, n_query,
                n_probes, local_k, min(global_k, 512), d_global_candidate_idx);
    }
    CUDA_LAUNCH_CHECK("residual_pq_query_reduce_compact_warpsort_kernel");
}
