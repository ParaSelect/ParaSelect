#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <stdexcept>

#include <cub/cub.cuh>

namespace paraselect { namespace search {

// Sort fixed-size tiles and merge only the requested prefix. For capacity
// 8192 this uses two 4096-item CUB sorts, avoiding the oversized shared-memory
// allocation of one 8192-item sort and avoiding an O(capacity log^2 capacity)
// bitonic network.
template <int Capacity>
static __global__ void pq_candidate_topcap_kernel(
    int* ids, float* distances, int* counts, int output_count) {
    constexpr int kThreads = 256;
    constexpr int kTile = Capacity > 4096 ? 4096 : Capacity;
    constexpr int kItems = kTile / kThreads;
    constexpr int kTiles = Capacity / kTile;
    static_assert(kTile % kThreads == 0, "top-cap tile must divide block size");
    using BlockSort = cub::BlockRadixSort<float, kThreads, kItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kItems];
    int values[kItems];
    const int q = blockIdx.x;
    const size_t base = static_cast<size_t>(q) * Capacity;
    const int count = min(max(counts[q], 0), Capacity);

    for (int tile = 0; tile < kTiles; ++tile) {
        #pragma unroll
        for (int item = 0; item < kItems; ++item) {
            const int i = tile * kTile + threadIdx.x * kItems + item;
            const bool valid = i < count && ids[base + i] >= 0;
            keys[item] = valid ? distances[base + i] : CUDART_INF_F;
            values[item] = valid ? ids[base + i] : INT_MAX;
        }
        BlockSort(sort_storage).Sort(keys, values);
        #pragma unroll
        for (int item = 0; item < kItems; ++item) {
            const int i = tile * kTile + threadIdx.x * kItems + item;
            ids[base + i] = values[item] == INT_MAX ? -1 : values[item];
            distances[base + i] = keys[item];
        }
        __syncthreads();
    }

    const int kept = min(count, output_count);
    if constexpr (kTiles == 1) {
        for (int i = threadIdx.x; i < Capacity; i += kThreads) {
            if (i >= kept) {
                ids[base + i] = -1;
                distances[base + i] = CUDART_INF_F;
            }
        }
    } else {
        // Each thread writes a disjoint output position. The rank of an item
        // in either sorted tile is found by binary search, so merge work is
        // O(output_count log tile) and fully parallel across output slots.
        for (int out = threadIdx.x; out < kept; out += kThreads) {
            int lo = max(0, out - kTile);
            int hi = min(kTile, out);
            while (lo < hi) {
                const int a = (lo + hi) / 2;
                const int b = out - 1 - a;
                const float ad = distances[base + a];
                const float bd = distances[base + kTile + b];
                if (a > 0 && b + 1 < kTile &&
                    distances[base + a - 1] > distances[base + kTile + b + 1]) {
                    hi = a;
                } else if (b >= 0 && a + 1 < kTile &&
                           distances[base + kTile + b] > distances[base + a + 1]) {
                    lo = a + 1;
                } else {
                    (void)ad;
                    (void)bd;
                    break;
                }
            }
            int a = lo;
            int b = out - a;
            if (a >= kTile) { a = kTile - 1; b = out - a; }
            if (b >= kTile) { b = kTile - 1; a = out - b; }
            const bool take_a = b >= kTile ||
                (a < kTile && distances[base + a] <= distances[base + kTile + b]);
            const int src = take_a ? a : (kTile + b);
            ids[base + out] = ids[base + src];
            distances[base + out] = distances[base + src];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < Capacity; i += kThreads) {
            if (i >= kept) {
                ids[base + i] = -1;
                distances[base + i] = CUDART_INF_F;
            }
        }
    }
    if (threadIdx.x == 0) counts[q] = kept;
}

template <int Capacity>
inline void launch_pq_candidate_topcap(
    int* ids, float* distances, int* counts, int queries,
    int output_count, cudaStream_t stream) {
    const int bytes = sizeof(typename cub::BlockRadixSort<
        float, 256, (Capacity > 4096 ? 4096 : Capacity) / 256, int>::TempStorage);
    auto err = cudaFuncSetAttribute(
        pq_candidate_topcap_kernel<Capacity>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    pq_candidate_topcap_kernel<Capacity><<<queries, 256, bytes, stream>>>(
        ids, distances, counts, output_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

inline void pq_candidate_topcap(
    int* ids, float* distances, int* counts, int queries,
    int capacity, int output_count, cudaStream_t stream) {
    if (capacity <= 0 || capacity > 8192 || (capacity & (capacity - 1)) ||
        output_count <= 0 || output_count > capacity) {
        throw std::invalid_argument(
            "PQ top-cap requires power-of-two capacity <=8192 and 0<N<=capacity");
    }
    switch (capacity) {
        case 512: launch_pq_candidate_topcap<512>(ids, distances, counts, queries, output_count, stream); break;
        case 1024: launch_pq_candidate_topcap<1024>(ids, distances, counts, queries, output_count, stream); break;
        case 2048: launch_pq_candidate_topcap<2048>(ids, distances, counts, queries, output_count, stream); break;
        case 4096: launch_pq_candidate_topcap<4096>(ids, distances, counts, queries, output_count, stream); break;
        case 8192: launch_pq_candidate_topcap<8192>(ids, distances, counts, queries, output_count, stream); break;
        default: throw std::invalid_argument("PQ top-cap capacity must be 512..8192 power of two");
    }
}

}}  // namespace paraselect::search
