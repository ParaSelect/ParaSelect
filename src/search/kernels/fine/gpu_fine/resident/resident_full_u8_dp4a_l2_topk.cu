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

__device__ __forceinline__ int resident_dp4a_lane_id() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ unsigned int load_u8x4_zero_tail(const uint8_t* ptr, int remaining) {
    unsigned int v = 0;
    uint8_t* out = reinterpret_cast<uint8_t*>(&v);
    if (remaining > 0) out[0] = ptr[0];
    if (remaining > 1) out[1] = ptr[1];
    if (remaining > 2) out[2] = ptr[2];
    if (remaining > 3) out[3] = ptr[3];
    return v;
}

template<int Capacity, bool Ascending, int QueriesPerBlock, typename SortIdxT = idx_t, bool StoreIndices = true, bool Vec4Layout = false>
__global__ __launch_bounds__(256, 1)
void resident_u8_l2_topk_dp4a_kernel(
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
    const int lane = resident_dp4a_lane_id();
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

    const uint8_t* query = d_query_group + (size_t)query_global_id * n_dim;
    constexpr int kGroup = paraselect::kInterleavedGroupSize;
    const int dim4 = (n_dim + 3) / 4;
    const size_t bytes_per_block = (size_t)dim4 * (size_t)kGroup * 4;
    const uint8_t* cluster_base = d_cluster_vector + (size_t)(vector_offset / kGroup) * bytes_per_block;

    using WarpSortBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, SortIdxT>;
    const float dummy_val = WarpSortBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, SortIdxT> queue(k);

    const int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;
    for (int iter = 0; iter < max_iterations; ++iter) {
        const int vec_idx_padded = vector_offset + iter * kWarpSize + lane;
        const bool valid_vec = vec_idx_padded < vector_offset + vector_count;
        const uint8_t* data = cluster_base + (size_t)iter * bytes_per_block;
        unsigned int dot = 0;
        for (int d4_base = 0; d4_base < dim4; d4_base += kWarpSize) {
            const int lane_d4 = d4_base + lane;
            unsigned int q4_lane = 0;
            if (lane_d4 < dim4) {
                const int d = lane_d4 * 4;
                const int remaining = n_dim - d;
                q4_lane = (remaining >= 4)
                    ? *reinterpret_cast<const unsigned int*>(query + d)
                    : load_u8x4_zero_tail(query + d, remaining);
            }
            const int d4_limit = min(kWarpSize, dim4 - d4_base);
            if constexpr (Vec4Layout) {
                int j = 0;
                for (; j + 3 < d4_limit; j += 4) {
                    const int g = (d4_base + j) / 4;
                    const uint4 u = *reinterpret_cast<const uint4*>(
                        data + ((size_t)g * kGroup + (size_t)lane) * 16);
                    dot = __dp4a(__shfl_sync(0xffffffff, q4_lane, j + 0), u.x, dot);
                    dot = __dp4a(__shfl_sync(0xffffffff, q4_lane, j + 1), u.y, dot);
                    dot = __dp4a(__shfl_sync(0xffffffff, q4_lane, j + 2), u.z, dot);
                    dot = __dp4a(__shfl_sync(0xffffffff, q4_lane, j + 3), u.w, dot);
                }
                for (; j < d4_limit; ++j) {
                    const int d4 = d4_base + j;
                    const unsigned int q4 = __shfl_sync(0xffffffff, q4_lane, j);
                    const unsigned int v4 = *reinterpret_cast<const unsigned int*>(
                        data + ((size_t)d4 * kGroup + (size_t)lane) * 4);
                    dot = __dp4a(q4, v4, dot);
                }
            } else {
                for (int j = 0; j < d4_limit; ++j) {
                    const int d4 = d4_base + j;
                    const unsigned int q4 = __shfl_sync(0xffffffff, q4_lane, j);
                    const unsigned int v4 = *reinterpret_cast<const unsigned int*>(
                        data + ((size_t)d4 * kGroup + (size_t)lane) * 4);
                    dot = __dp4a(q4, v4, dot);
                }
            }
        }
        if (!valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            const float data_norm = d_cluster_vector_norm[vec_idx_padded];
            const float query_norm = s_query_norm[local_query_idx];
            const float dist = query_norm * query_norm + data_norm * data_norm - 2.0f * (float)dot;
            const SortIdxT compact_id = static_cast<SortIdxT>(compact_offset + (vec_idx_padded - vector_offset));
            queue.add(dist, compact_id);
        }
    }
    __syncwarp();
    queue.done();
    __syncwarp();

    float* row_dist = d_topk_dist + ((size_t)query_global_id * n_probes + probe_index_in_query) * k;
    if constexpr (StoreIndices) {
        idx_t* row_idx = d_topk_index +
                         ((size_t)query_global_id * n_probes + probe_index_in_query) * k;
        queue.store(row_dist, row_idx);
    } else {
        queue.store_values(row_dist);
    }
}

__device__ __forceinline__ int pack_i8x4_interleaved_v1(
    const int8_t* data, int dim, int lane) {
    const unsigned packed =
        static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 0) * kWarpSize + lane])) |
        (static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 1) * kWarpSize + lane])) << 8) |
        (static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 2) * kWarpSize + lane])) << 16) |
        (static_cast<unsigned>(static_cast<uint8_t>(data[(dim + 3) * kWarpSize + lane])) << 24);
    return static_cast<int>(packed);
}

template<int Version, int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1)
void resident_pca_i8_l2_topk_dp4a_kernel(
    const int8_t* __restrict__ queries,
    const float* __restrict__ query_dot_scales,
    const int8_t* __restrict__ vectors,
    const int* __restrict__ padded_offsets,
    const int* __restrict__ compact_offsets,
    const int* __restrict__ counts,
    const int* __restrict__ entry_cluster_id,
    const int* __restrict__ entry_query_start,
    const int* __restrict__ entry_query_count,
    const int* __restrict__ entry_queries,
    const int* __restrict__ entry_probe_indices,
    const float* __restrict__ query_squared_norms,
    const float* __restrict__ vector_squared_norms,
    int n_entry,
    int n_probes,
    int dim,
    int k,
    float* __restrict__ topk_dist,
    idx_t* __restrict__ topk_index) {
    __shared__ float shared_query_norm[QueriesPerBlock];
    __shared__ float shared_query_scale[QueriesPerBlock];
    const int entry = blockIdx.x;
    if (entry >= n_entry) return;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = resident_dp4a_lane_id();
    const bool active = warp < entry_query_count[entry];
    int query_id = -1;
    int probe = -1;
    if (active) {
        const int ref = entry_query_start[entry] + warp;
        query_id = entry_queries[ref];
        probe = entry_probe_indices[ref];
        if (lane == 0) {
            shared_query_norm[warp] = query_squared_norms[query_id];
            shared_query_scale[warp] = query_dot_scales[query_id];
        }
    }
    __syncthreads();
    if (!active) return;

    const int cluster = entry_cluster_id[entry];
    const int padded_start = padded_offsets[cluster];
    const int compact_start = compact_offsets[cluster];
    const int count = counts[cluster];
    const int8_t* query = queries + (size_t)query_id * dim;
    const int8_t* vector_base =
        vectors + (size_t)(padded_start / kWarpSize) * kWarpSize * dim;
    const int dim4 = dim / 4;
    using QueueBase = paraselect::warpsort::WarpSort<Capacity, Ascending, float, idx_t>;
    const float dummy = QueueBase::kDummy();
    paraselect::warpsort::WarpSortFiltered<Capacity, Ascending, float, idx_t> queue(k);
    const int iterations = (count + kWarpSize - 1) / kWarpSize;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int local = iteration * kWarpSize + lane;
        const int padded = padded_start + local;
        const int8_t* data = vector_base + (size_t)iteration * kWarpSize * dim;
        const int query4_lane = lane < dim4
            ? reinterpret_cast<const int*>(query)[lane]
            : 0;
        int dot = 0;
        for (int d4 = 0; d4 < dim4; ++d4) {
            const int query4 = __shfl_sync(0xffffffffu, query4_lane, d4);
            int vector4 = 0;
            if constexpr (Version == 1) {
                vector4 = pack_i8x4_interleaved_v1(data, d4 * 4, lane);
            } else {
                vector4 = *reinterpret_cast<const int*>(
                    data + ((size_t)d4 * kWarpSize + lane) * 4);
            }
            dot = __dp4a(query4, vector4, dot);
        }
        if (local >= count) {
            queue.add(dummy, -1);
        } else {
            const float distance = shared_query_norm[warp] +
                vector_squared_norms[padded] -
                2.0f * shared_query_scale[warp] * static_cast<float>(dot);
            queue.add(distance, compact_start + local);
        }
    }
    queue.done();
    float* row_dist = topk_dist + ((size_t)query_id * n_probes + probe) * k;
    idx_t* row_index = topk_index + ((size_t)query_id * n_probes + probe) * k;
    queue.store(row_dist, row_index);
}

}  // namespace

void launch_resident_u8_l2_topk_dp4a(
    int k,
    dim3 block,
    int dim,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8interleaved,
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
    if (k <= 64) {
        resident_u8_l2_topk_dp4a_kernel<64, true, 8><<<n_entry, block, 0, stream>>>(
            d_queries_u8, d_vectors_u8interleaved, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else if (k <= 128) {
        resident_u8_l2_topk_dp4a_kernel<128, true, 8><<<n_entry, block, 0, stream>>>(
            d_queries_u8, d_vectors_u8interleaved, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else if (k <= 256) {
        resident_u8_l2_topk_dp4a_kernel<256, true, 8><<<n_entry, block, 0, stream>>>(
            d_queries_u8, d_vectors_u8interleaved, d_padded_offsets, d_compact_offsets, d_counts,
            d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
            d_entry_queries, d_entry_probe_indices, d_query_norm, d_norms,
            n_entry, n_probes, dim, k, d_probe_dist, d_probe_index);
    } else {
        throw std::runtime_error("resident u8 dp4a IVFFlat supports topk <= 256");
    }
}

void launch_resident_u8_l2_topk_dp4a_distances(
    int k,
    dim3 block,
    int dim,
    const uint8_t* d_queries_u8,
    const uint8_t* d_vectors_u8interleaved,
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
    bool vec4_layout,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    if (k <= 0 || k > 64)
        throw std::runtime_error(
            "resident u8 dp4a distance sketch supports topk 1..64");
    if (vec4_layout) {
        resident_u8_l2_topk_dp4a_kernel<64, true, 8, int, false, true>
            <<<n_entry, block, 0, stream>>>(
                d_queries_u8, d_vectors_u8interleaved, d_padded_offsets,
                d_compact_offsets, d_counts, d_entry_cluster_id,
                d_entry_query_start, d_entry_query_count, d_entry_queries,
                d_entry_probe_indices, d_query_norm, d_norms, n_entry,
                n_probes, dim, k, d_probe_dist, nullptr);
    } else {
        resident_u8_l2_topk_dp4a_kernel<64, true, 8, int, false, false>
            <<<n_entry, block, 0, stream>>>(
                d_queries_u8, d_vectors_u8interleaved, d_padded_offsets,
                d_compact_offsets, d_counts, d_entry_cluster_id,
                d_entry_query_start, d_entry_query_count, d_entry_queries,
                d_entry_probe_indices, d_query_norm, d_norms, n_entry,
                n_probes, dim, k, d_probe_dist, nullptr);
    }
}

template <int Version>
void launch_resident_pca_i8_l2_topk_dp4a_impl(
    int k,
    dim3 block,
    int dim,
    const int8_t* d_queries_i8,
    const float* d_query_dot_scales,
    const int8_t* d_vectors_i8,
    const int* d_padded_offsets,
    const int* d_compact_offsets,
    const int* d_counts,
    const int* d_entry_cluster_id,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const float* d_query_squared_norms,
    const float* d_vector_squared_norms,
    int n_entry,
    int n_probes,
    float* d_probe_dist,
    idx_t* d_probe_index,
    cudaStream_t stream) {
    if (n_entry <= 0) return;
    if (dim <= 0 || dim % 4 != 0 || dim / 4 > kWarpSize) {
        throw std::invalid_argument(
            "resident PCA dp4a requires a dimension divisible by 4 and at most 128");
    }
#define LAUNCH_PCA_DP4A(CAP) \
    resident_pca_i8_l2_topk_dp4a_kernel<Version, CAP, true, 8> \
        <<<n_entry, block, 0, stream>>>( \
        d_queries_i8, d_query_dot_scales, d_vectors_i8, d_padded_offsets, \
        d_compact_offsets, d_counts, d_entry_cluster_id, d_entry_query_start, \
        d_entry_query_count, d_entry_queries, d_entry_probe_indices, \
        d_query_squared_norms, d_vector_squared_norms, n_entry, n_probes, \
        dim, k, d_probe_dist, d_probe_index)
    if (k <= 32) {
        LAUNCH_PCA_DP4A(64);
    } else if (k <= 96) {
        LAUNCH_PCA_DP4A(128);
    } else if (k <= 224) {
        LAUNCH_PCA_DP4A(256);
    } else if (k <= 256) {
        LAUNCH_PCA_DP4A(512);
    } else {
        throw std::invalid_argument("resident PCA dp4a supports topk <= 256");
    }
#undef LAUNCH_PCA_DP4A
}

#define DEFINE_PCA_DP4A_LAUNCH(VERSION) \
void launch_resident_pca_i8_l2_topk_dp4a_v##VERSION( \
    int k, dim3 block, int dim, const int8_t* d_queries_i8, \
    const float* d_query_dot_scales, const int8_t* d_vectors_i8, \
    const int* d_padded_offsets, const int* d_compact_offsets, \
    const int* d_counts, const int* d_entry_cluster_id, \
    const int* d_entry_query_start, const int* d_entry_query_count, \
    const int* d_entry_queries, const int* d_entry_probe_indices, \
    const float* d_query_squared_norms, const float* d_vector_squared_norms, \
    int n_entry, int n_probes, float* d_probe_dist, idx_t* d_probe_index, \
    cudaStream_t stream) { \
    launch_resident_pca_i8_l2_topk_dp4a_impl<VERSION>( \
        k, block, dim, d_queries_i8, d_query_dot_scales, d_vectors_i8, \
        d_padded_offsets, d_compact_offsets, d_counts, d_entry_cluster_id, \
        d_entry_query_start, d_entry_query_count, d_entry_queries, \
        d_entry_probe_indices, d_query_squared_norms, d_vector_squared_norms, \
        n_entry, n_probes, d_probe_dist, d_probe_index, stream); \
}

DEFINE_PCA_DP4A_LAUNCH(1)
DEFINE_PCA_DP4A_LAUNCH(2)

#undef DEFINE_PCA_DP4A_LAUNCH
