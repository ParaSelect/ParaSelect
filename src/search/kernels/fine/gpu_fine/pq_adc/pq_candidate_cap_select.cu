#include "search/kernels/fine/gpu_fine/pq_adc/pq_candidate_cap_select.cuh"

#include <cub/cub.cuh>

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

// Query IDs are dense in [0,n_query). All remaining high key bits are zero.
int query_distance_end_bit(int n_query) {
    int bits = 32;
    for (unsigned q = unsigned(n_query - 1); q; q >>= 1) ++bits;
    return bits;
}

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float unordered_float_bits(uint32_t ordered) {
    const uint32_t bits = ordered & 0x80000000u
        ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__global__ void build_sort_keys_kernel(
    const float* pool_dist, const int* pool_query, int total, uint64_t* keys) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += blockDim.x * gridDim.x) {
        const uint32_t q = uint32_t(pool_query[i]);
        const uint32_t distance = ordered_float_bits(pool_dist[i]);
        keys[i] = (uint64_t(q) << 32) | distance;
    }
}

__global__ void build_offsets_kernel(const int* counts, int n_query, int* offsets) {
    int total = 0;
    offsets[0] = 0;
    for (int q = 0; q < n_query; ++q) {
        total += max(counts[q], 0);
        offsets[q + 1] = total;
    }
}

__global__ void compact_sorted_candidates_kernel(
    const uint64_t* keys,
    const int* values,
    const int* offsets,
    int n_query,
    int output_stride,
    int n_rerank,
    int* output_idx,
    float* output_dist,
    int* output_counts) {
    const int q = blockIdx.x;
    if (q >= n_query) return;
    const int begin = offsets[q];
    const int end = offsets[q + 1];
    __shared__ int valid_kept;
    if (threadIdx.x == 0) {
        int lo=begin, hi=min(end,begin+n_rerank);
        while(lo<hi) {
            const int mid=lo+(hi-lo)/2;
            if (values[mid]>=0 && isfinite(unordered_float_bits(uint32_t(keys[mid])))) lo=mid+1;
            else hi=mid;
        }
        valid_kept=lo-begin;
    }
    __syncthreads();
    const int kept = valid_kept;
    for (int rank = threadIdx.x; rank < output_stride; rank += blockDim.x) {
        const size_t out = static_cast<size_t>(q) * output_stride + rank;
        if (rank < kept) {
            const uint64_t key = keys[begin + rank];
            output_idx[out] = values[begin + rank];
            output_dist[out] = unordered_float_bits(
                static_cast<uint32_t>(key & 0xffffffffULL));
        } else {
            output_idx[out] = -1;
            output_dist[out] = INFINITY;
        }
    }
    if (threadIdx.x == 0) output_counts[q] = kept;

}


}  // namespace

namespace paraselect::search {

std::size_t pq_candidate_cap_select_temp_bytes(int max_candidates, int max_queries) {
    if (max_candidates <= 0) return 0;
    if (max_queries <= 0) throw std::invalid_argument("invalid cap-select query capacity");
    std::size_t bytes = 0;
    cudaError_t err;
    // A partial final batch can use fewer query bits. Query all supported
    // ranges at workspace creation; no allocation or scratch query in search.
    for (int end_bit = 32; end_bit <= query_distance_end_bit(max_queries); ++end_bit) {
        std::size_t required = 0;
        err = cub::DeviceRadixSort::SortPairs(
            nullptr, required,
            static_cast<const uint64_t*>(nullptr),
            static_cast<uint64_t*>(nullptr),
            static_cast<const int*>(nullptr),
            static_cast<int*>(nullptr), max_candidates, 0, end_bit);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        bytes = required > bytes ? required : bytes;
    }
    return bytes;
}

void pq_candidate_cap_select(
    int* d_pool_idx,
    const float* d_pool_dist,
    const int* d_pool_query,
    const int* d_pool_counts,
    int total_candidates,
    int n_query,
    int output_stride,
    int n_rerank,
    uint64_t* d_sort_keys_in,
    uint64_t* d_sort_keys_out,
    int* d_sort_values_out,
    int* d_offsets,
    int* d_output_idx,
    float* d_output_dist,
    int* d_output_counts,
    void* d_sort_temp,
    std::size_t sort_temp_bytes,
    cudaStream_t stream) {
    if (!d_pool_idx || !d_pool_dist || !d_pool_query || !d_pool_counts ||
        !d_sort_keys_in || !d_sort_keys_out || !d_sort_values_out ||
        !d_offsets || !d_output_idx || !d_output_dist || !d_output_counts ||
        total_candidates < 0 || n_query <= 0 || output_stride <= 0 ||
        n_rerank <= 0 || n_rerank > output_stride) {
        throw std::invalid_argument("invalid PQ candidate cap-select arguments");
    }
    build_offsets_kernel<<<1, 1, 0, stream>>>(d_pool_counts, n_query, d_offsets);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    if (total_candidates > 0) {
        build_sort_keys_kernel<<<(total_candidates + 255) / 256, 256, 0, stream>>>(
            d_pool_dist, d_pool_query, total_candidates, d_sort_keys_in);
        err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        std::size_t required = sort_temp_bytes;
        err = cub::DeviceRadixSort::SortPairs(
            d_sort_temp, required,
            d_sort_keys_in, d_sort_keys_out,
            d_pool_idx, d_sort_values_out,
            total_candidates, 0, query_distance_end_bit(n_query), stream);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    compact_sorted_candidates_kernel<<<n_query, 256, 0, stream>>>(
        d_sort_keys_out, d_sort_values_out, d_offsets, n_query,
        output_stride, n_rerank, d_output_idx, d_output_dist, d_output_counts);
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

}  // namespace paraselect::search
