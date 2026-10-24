#include "search/kernels/fine/gpu_fine/resident/resident_full_l2_topk.cuh"
#include "layout/ivf_interleaved_layout.cuh"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"

#include <cfloat>
#include <stdexcept>

namespace {

#ifndef kWarpSize
#define kWarpSize 32
#endif

__device__ __forceinline__ int resident_u8_full_lane_id() {
    return threadIdx.x & 31;
}

template<int Capacity, bool Ascending, int QueriesPerBlock, bool PCA>
__global__ __launch_bounds__(256, 1)
void resident_byte_l2_topk_full_kernel(
    const float* __restrict__ d_weighted_query_group,
    const uint8_t* __restrict__ d_query_group,
    const uint8_t* __restrict__ d_cluster_vector,
    const int* __restrict__ d_probe_vector_offset,
    const int* __restrict__ d_probe_vector_offset_compact,
    const int* __restrict__ d_probe_vector_count,
    const int* __restrict__ d_entry_cluster_id,
    const int* __restrict__ d_entry_query_start,
    const int* __restrict__ d_entry_query_count,
    const int* __restrict__ d_entry_queries,
    const int* __restrict__ d_entry_probe_indices,
    float* __restrict__ d_query_norm,
    const float* __restrict__ d_cluster_vector_norm,
    int n_entry,
    int n_probes,
    int n_dim,
    int k,
    float* __restrict__ d_topk_dist,
    idx_t* __restrict__ d_topk_index)
{
    __shared__ float s_query_norm[QueriesPerBlock];

    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;

    const int cluster_id = d_entry_cluster_id[entry_id];
    const int vector_offset = d_probe_vector_offset[cluster_id];
    const int compact_offset = d_probe_vector_offset_compact[cluster_id];
    const int vector_count = d_probe_vector_count[cluster_id];

    const int entry_query_start = d_entry_query_start[entry_id];
    const int entry_query_count = d_entry_query_count[entry_id];

    const int local_query_idx = threadIdx.x / kWarpSize;
    const int lane = resident_u8_full_lane_id();
    const bool active_warp = local_query_idx < entry_query_count;

    int query_global_id = -1;
    int probe_index_in_query = -1;
    if (active_warp) {
        const int entry_query_idx = entry_query_start + local_query_idx;
        query_global_id = d_entry_queries[entry_query_idx];
        probe_index_in_query = d_entry_probe_indices[entry_query_idx];
        if (lane == 0) s_query_norm[local_query_idx] = d_query_norm[query_global_id];
    }
    __syncthreads();
    if (!active_warp) return;

    const uint8_t* query_global_ptr = PCA
        ? nullptr
        : d_query_group + (size_t)query_global_id * n_dim;
    const float* weighted_query_global_ptr = PCA
        ? d_weighted_query_group + (size_t)query_global_id * n_dim
        : nullptr;
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kGroup * n_dim;
    const uint8_t* cluster_interleaved_base =
        d_cluster_vector + (size_t)(vector_offset / kGroup) * words_per_block;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);

    const int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;
    for (int iter = 0; iter < max_iterations; ++iter) {
        const int vec_idx_padded = vector_offset + iter * kWarpSize + lane;
        const bool has_valid_vec = vec_idx_padded < vector_offset + vector_count;
        const uint8_t* data = cluster_interleaved_base + (size_t)iter * words_per_block;
        float dot = 0.0f;
        const int dimBlocks = (n_dim / kWarpSize) * kWarpSize;
        for (int dBase = 0; dBase < dimBlocks; dBase += kWarpSize) {
            if constexpr (PCA) {
                const float qReg = weighted_query_global_ptr[dBase + lane];
                float partial0 = 0.0f;
                float partial1 = 0.0f;
                float partial2 = 0.0f;
                float partial3 = 0.0f;
                for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kGroup) {
                    partial0 += __shfl_sync(0xffffffff, qReg, i * 4 + 0) *
                        static_cast<float>(reinterpret_cast<const int8_t*>(data)[0 * kGroup + lane]);
                    partial1 += __shfl_sync(0xffffffff, qReg, i * 4 + 1) *
                        static_cast<float>(reinterpret_cast<const int8_t*>(data)[1 * kGroup + lane]);
                    partial2 += __shfl_sync(0xffffffff, qReg, i * 4 + 2) *
                        static_cast<float>(reinterpret_cast<const int8_t*>(data)[2 * kGroup + lane]);
                    partial3 += __shfl_sync(0xffffffff, qReg, i * 4 + 3) *
                        static_cast<float>(reinterpret_cast<const int8_t*>(data)[3 * kGroup + lane]);
                }
                dot += partial0 + partial1 + partial2 + partial3;
            } else {
                const int qReg = (int)query_global_ptr[dBase + lane];
                int partial = 0;
                for (int i = 0; i < kWarpSize / 4; ++i, data += 4 * kGroup) {
                    const int v0 = (int)data[0 * kGroup + lane];
                    const int v1 = (int)data[1 * kGroup + lane];
                    const int v2 = (int)data[2 * kGroup + lane];
                    const int v3 = (int)data[3 * kGroup + lane];
                    partial += __shfl_sync(0xffffffff, qReg, i * 4 + 0) * v0;
                    partial += __shfl_sync(0xffffffff, qReg, i * 4 + 1) * v1;
                    partial += __shfl_sync(0xffffffff, qReg, i * 4 + 2) * v2;
                    partial += __shfl_sync(0xffffffff, qReg, i * 4 + 3) * v3;
                }
                dot += static_cast<float>(partial);
            }
        }
        if (dimBlocks < n_dim) {
            const int dLoad = dimBlocks + lane;
            if constexpr (PCA) {
                const float qReg = dLoad < n_dim ? weighted_query_global_ptr[dLoad] : 0.0f;
                for (int dSub = 0; dSub < n_dim - dimBlocks; ++dSub, data += kGroup) {
                    dot += __shfl_sync(0xffffffff, qReg, dSub) *
                        static_cast<float>(reinterpret_cast<const int8_t*>(data)[lane]);
                }
            } else {
                const int qReg = (dLoad < n_dim) ? (int)query_global_ptr[dLoad] : 0;
                for (int dSub = 0; dSub < n_dim - dimBlocks; ++dSub, data += kGroup) {
                    dot += __shfl_sync(0xffffffff, qReg, dSub) * (int)data[lane];
                }
            }
        }
        if (!has_valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            const float data_norm = d_cluster_vector_norm[vec_idx_padded];
            const float query_norm = s_query_norm[local_query_idx];
            const float l2_distance = PCA
                ? query_norm + data_norm - 2.0f * dot
                : query_norm * query_norm + data_norm * data_norm - 2.0f * dot;
            const int local = vec_idx_padded - vector_offset;
            const idx_t vec_idx_compact = compact_offset + local;
            queue.add(l2_distance, vec_idx_compact);
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();

    float* row_dist = d_topk_dist + ((size_t)query_global_id * n_probes + probe_index_in_query) * k;
    idx_t* row_idx = d_topk_index + ((size_t)query_global_id * n_probes + probe_index_in_query) * k;
    queue.store(row_dist, row_idx);
}

}  // namespace

void launch_resident_u8_l2_topk_full(
    int k,
    dim3 block,
    int dim,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    float* d_query_norm,
    const float* d_norms,
    int n_entry,
    int n_probes,
    float* d_probe_dist,
    idx_t* d_probe_index,
    cudaStream_t stream)
{
    if (n_entry <= 0) return;
    if (k <= 32) {
        resident_byte_l2_topk_full_kernel<64, true, 8, false><<<n_entry, block, 0, stream>>>(
            nullptr, d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else if (k <= 96) {
        resident_byte_l2_topk_full_kernel<128, true, 8, false><<<n_entry, block, 0, stream>>>(
            nullptr, d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else if (k <= 224) {
        resident_byte_l2_topk_full_kernel<256, true, 8, false><<<n_entry, block, 0, stream>>>(
            nullptr, d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else if (k <= 256) {
        resident_byte_l2_topk_full_kernel<512, true, 8, false><<<n_entry, block, 0, stream>>>(
            nullptr, d_queries_u8, d_vectors_u8, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else {
        throw std::runtime_error("resident u8 IVFFlat supports topk <= 256");
    }
}

void launch_resident_pca_i8_l2_topk_full(
    int k,
    dim3 block,
    int dim,
    const float* d_weighted_queries,
    const int8_t* d_vectors_i8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    float* d_query_norm,
    const float* d_norms,
    int n_entry,
    int n_probes,
    float* d_probe_dist,
    idx_t* d_probe_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
#define LAUNCH_PCA(CAP) \
    resident_byte_l2_topk_full_kernel<CAP, true, 8, true><<<n_entry, block, 0, stream>>>( \
        d_weighted_queries, nullptr, reinterpret_cast<const uint8_t*>(d_vectors_i8), \
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id, \
        d_entry_query_start, d_entry_query_count, d_entry_queries, d_entry_probe_indices, \
        d_query_norm, d_norms, n_entry, n_probes, dim, k, d_probe_dist, d_probe_index)
    if (k <= 32) {
        LAUNCH_PCA(64);
    } else if (k <= 96) {
        LAUNCH_PCA(128);
    } else if (k <= 224) {
        LAUNCH_PCA(256);
    } else if (k <= 256) {
        LAUNCH_PCA(512);
    } else {
        throw std::runtime_error("resident PCA IVFFlat supports topk <= 256");
    }
#undef LAUNCH_PCA
}
