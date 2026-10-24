#pragma once

#include <cstdint>
#include <climits>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>

namespace paraselect::coarse::small_topk {

template<class T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template<>
__device__ __forceinline__ float as_float<__half>(__half value) {
    return __half2float(value);
}

template<class T>
__device__ __forceinline__ float2 as_float2(const T* value) {
    return make_float2(static_cast<float>(value[0]), static_cast<float>(value[1]));
}

template<>
__device__ __forceinline__ float2 as_float2<float>(const float* value) {
    return *reinterpret_cast<const float2*>(value);
}

template<>
__device__ __forceinline__ float2 as_float2<__half>(const __half* value) {
    return __half22float2(*reinterpret_cast<const __half2*>(value));
}

template<int K>
struct WarpState {
    // Every lane may own the complete local top-K.  Keeping only K/32 items
    // is not exact when one lane's centroid stripe contains more than K/32
    // global winners.
    static constexpr int Items = K;
    float scores[Items];
    int64_t ids[Items];
};

__device__ __forceinline__ bool precedes(float a, int64_t aid, float b, int64_t bid) {
    return a > b || (a == b && aid < bid);
}

template<int K>
__device__ __forceinline__ void insert(WarpState<K>& state, float score, int64_t id) {
    constexpr int Items = WarpState<K>::Items;
    if (isnan(score)) score = -INFINITY;
    if (!precedes(score, id, state.scores[Items - 1], state.ids[Items - 1])) return;
    int slot = Items - 1;
    while (slot > 0 && precedes(score, id, state.scores[slot - 1], state.ids[slot - 1])) {
        state.scores[slot] = state.scores[slot - 1];
        state.ids[slot] = state.ids[slot - 1];
        --slot;
    }
    state.scores[slot] = score;
    state.ids[slot] = id;
}

template<int K>
__global__ void select_scores(const float* scores, int batch, int length, int stride,
                              float* values, int64_t* ids) {
    static_assert(K >= 8 && K <= 64 && K % 8 == 0);
    constexpr int Items = WarpState<K>::Items;
    constexpr int Capacity = Items * 32;
    const int lane = threadIdx.x & 31;
    const int query = blockIdx.x;
    if (query >= batch) return;

    WarpState<K> local{};
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        local.scores[i] = -INFINITY;
        local.ids[i] = INT64_MAX;
    }
    for (int pos = lane; pos < length; pos += 32)
        insert(local, scores[size_t(query) * stride + pos], pos);

    __shared__ float shared_scores[Capacity];
    __shared__ int64_t shared_ids[Capacity];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        shared_scores[i * 32 + lane] = local.scores[i];
        shared_ids[i * 32 + lane] = local.ids[i];
    }
    __syncwarp();

    for (int size = 2; size <= Capacity; size <<= 1) {
        for (int stride_sort = size >> 1; stride_sort > 0; stride_sort >>= 1) {
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const int i = item * 32 + lane;
                const int j = i ^ stride_sort;
                if (i < j) {
                    const bool up = (i & size) == 0;
                    const float a = shared_scores[i];
                    const int64_t aid = shared_ids[i];
                    const float b = shared_scores[j];
                    const int64_t bid = shared_ids[j];
                    const bool swap = up ? precedes(b, bid, a, aid)
                                         : precedes(a, aid, b, bid);
                    if (swap) {
                        shared_scores[i] = b;
                        shared_ids[i] = bid;
                        shared_scores[j] = a;
                        shared_ids[j] = aid;
                    }
                }
            }
            __syncwarp();
        }
    }

    for (int rank = lane; rank < K; rank += 32) {
        values[size_t(query) * K + rank] = shared_scores[rank];
        ids[size_t(query) * K + rank] = shared_ids[rank];
    }
}

template<int K>
__global__ void select_l2(const float* inner, const float* query_norm,
                          const float* center_norm, int batch, int length,
                          float* values, int64_t* ids) {
    static_assert(K >= 8 && K <= 64 && K % 8 == 0);
    constexpr int Items = WarpState<K>::Items;
    constexpr int Capacity = Items * 32;
    const int lane = threadIdx.x & 31;
    const int query = blockIdx.x;
    if (query >= batch) return;

    WarpState<K> local{};
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        local.scores[i] = -INFINITY;
        local.ids[i] = INT64_MAX;
    }
    __syncwarp();
    const float q = query_norm[query];
    for (int pos = lane; pos < length; pos += 32) {
        const float c = center_norm[pos];
        const float score = -__fsub_rn(__fmaf_rn(q, q, __fmul_rn(c, c)),
                                       __fadd_rn(inner[size_t(query) * length + pos],
                                                 inner[size_t(query) * length + pos]));
        insert(local, score, pos);
    }

    __shared__ float shared_scores[Capacity];
    __shared__ int64_t shared_ids[Capacity];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        shared_scores[i * 32 + lane] = local.scores[i];
        shared_ids[i * 32 + lane] = local.ids[i];
    }
    __syncwarp();
    for (int size = 2; size <= Capacity; size <<= 1) {
        for (int stride_sort = size >> 1; stride_sort > 0; stride_sort >>= 1) {
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const int i = item * 32 + lane;
                const int j = i ^ stride_sort;
                if (i < j) {
                    const bool up = (i & size) == 0;
                    const float a = shared_scores[i];
                    const int64_t aid = shared_ids[i];
                    const float b = shared_scores[j];
                    const int64_t bid = shared_ids[j];
                    const bool swap = up ? precedes(b, bid, a, aid)
                                         : precedes(a, aid, b, bid);
                    if (swap) {
                        shared_scores[i] = b;
                        shared_ids[i] = bid;
                        shared_scores[j] = a;
                        shared_ids[j] = aid;
                    }
                }
            }
            __syncwarp();
        }
    }
    for (int rank = lane; rank < K; rank += 32) {
        // Score selectors expose negative L2 in descending order.  The public
        // L2 wrapper restores positive distances after this kernel returns.
        values[size_t(query) * K + rank] = shared_scores[rank];
        ids[size_t(query) * K + rank] = shared_ids[rank];
    }
}

// Producer-consumer path for small FP16 batches.  A block owns one query and
// one 256-center tile.  Groups of eight lanes compute a center row, retain a
// sorted local K list, and merge those lists into one sorted tile list.  The
// tile lists are the only materialized scores; the full distance matrix is
// never written.
template<class T, int K, int Tile = 256, int Group = 8, int Threads = 128>
__global__ void select_l2_vectors_tile(const T* queries, const T* centers,
                                       const float* query_norm, const float* center_norm,
                                       int batch, int length, int tile_count,
                                       int stride,
                                       float* scratch) {
    static_assert(K >= 8 && K <= 64 && K % 8 == 0);
    static_assert(Group == 1 || Group == 8);
    static_assert(Threads == 32 || Threads == 128);
    static_assert(Threads % Group == 0);
    constexpr int Groups = Threads / Group;
    const int tid = threadIdx.x;
    const int group_lane = tid % Group;
    const int group = tid / Group;
    const int tile = blockIdx.x;
    const int query = blockIdx.y;
    if (query >= batch) return;
    const int begin = tile * Tile;
    const int end = min(begin + Tile, length);

    __shared__ float query_values[128];
    for (int d = tid; d < 128; d += Threads)
        query_values[d] = as_float(queries[size_t(query) * 128 + d]);
    __syncthreads();

    __shared__ float group_scores[Groups][K];
    __shared__ int64_t group_ids[Groups][K];
    if (group_lane == 0) {
#pragma unroll
        for (int i = 0; i < K; ++i) {
            group_scores[group][i] = -INFINITY;
            group_ids[group][i] = INT64_MAX;
        }
    }
    __syncthreads();
    const float q = query_norm[query];
    for (int center = begin + group; center < end; center += Groups) {
        float dot = 0.0f;
#pragma unroll
        for (int d = group_lane * 2; d < 128; d += Group * 2) {
            const float2 c = as_float2(centers + size_t(center) * 128 + d);
            dot = fmaf(query_values[d], c.x, dot);
            dot = fmaf(query_values[d + 1], c.y, dot);
        }
#pragma unroll
        for (int delta = Group / 2; delta > 0; delta >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, delta, Group);
        if (group_lane == 0) {
            const float c = center_norm[center];
            const float distance = fmaf(q, q, c * c) - 2.0f * dot;
            const float score = -distance;
            if (precedes(score, center, group_scores[group][K - 1], group_ids[group][K - 1])) {
                int slot = K - 1;
                while (slot > 0 && precedes(score, center,
                                             group_scores[group][slot - 1],
                                             group_ids[group][slot - 1])) {
                    group_scores[group][slot] = group_scores[group][slot - 1];
                    group_ids[group][slot] = group_ids[group][slot - 1];
                    --slot;
                }
                group_scores[group][slot] = score;
                group_ids[group][slot] = center;
            }
        }
    }

    // Each group list is already sorted. Merge only the Groups heads and write
    // one sorted list for this tile. The query-level merge below consumes the
    // tile lists with the same head-advance rule.
    __shared__ float merge_scores[Groups];
    __shared__ int64_t merge_ids[Groups];
    __shared__ int merge_pos[Groups];
    if (group_lane == 0) merge_pos[group] = 0;
    __syncthreads();
    for (int rank = 0; rank < K; ++rank) {
        if (group_lane == 0) {
            const int pos = merge_pos[group];
            merge_scores[group] = pos < K ? group_scores[group][pos] : -INFINITY;
            merge_ids[group] = pos < K ? group_ids[group][pos] : INT64_MAX;
        }
        __syncthreads();
        if (tid == 0) {
            int winner = 0;
#pragma unroll
            for (int candidate = 1; candidate < Groups; ++candidate) {
                if (precedes(merge_scores[candidate], merge_ids[candidate],
                             merge_scores[winner], merge_ids[winner])) {
                    winner = candidate;
                }
            }
            const size_t tile_offset = size_t(query) * stride + size_t(tile) * K;
            scratch[tile_offset + rank] = merge_scores[winner];
            // Adaptive supports nlist < 2^23, so every centroid ID is exactly
            // representable in FP32. Keeping IDs beside scores reuses the
            // existing inner-product workspace and needs no second allocation.
            scratch[tile_offset + size_t(tile_count) * K + rank] =
                static_cast<float>(merge_ids[winner]);
            ++merge_pos[winner];
        }
        __syncthreads();
    }
}

template<int K>
__global__ void merge_l2_vector_tiles(float* scratch,
                                      int batch, int tile_count, int stride,
                                      float* values, int64_t* ids) {
    const int query = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (query >= batch) return;
    const size_t row = size_t(query) * stride;
    const size_t id_base = row + size_t(tile_count) * K;
    const size_t position_base = id_base + size_t(tile_count) * K;
    for (int tile = lane; tile < tile_count; tile += 32) scratch[position_base + tile] = 0.0f;
    __syncwarp();

    for (int rank = 0; rank < K; ++rank) {
        float best = -INFINITY;
        int64_t best_id = INT64_MAX;
        int best_tile = -1;
        for (int tile = lane; tile < tile_count; tile += 32) {
            const int pos = static_cast<int>(scratch[position_base + tile]);
            if (pos >= K) continue;
            const float score = scratch[row + size_t(tile) * K + pos];
            const int64_t id = static_cast<int64_t>(scratch[id_base + size_t(tile) * K + pos]);
            if (precedes(score, id, best, best_id)) {
                best = score;
                best_id = id;
                best_tile = tile;
            }
        }
        for (int delta = 16; delta > 0; delta >>= 1) {
            const float other_score = __shfl_down_sync(0xffffffffu, best, delta);
            const int64_t other_id = __shfl_down_sync(0xffffffffu, best_id, delta);
            const int other_tile = __shfl_down_sync(0xffffffffu, best_tile, delta);
            if (precedes(other_score, other_id, best, best_id)) {
                best = other_score;
                best_id = other_id;
                best_tile = other_tile;
            }
        }
        if (lane == 0) {
            values[size_t(query) * K + rank] = -best;
            ids[size_t(query) * K + rank] = best_id;
            ++scratch[position_base + best_tile];
        }
        __syncwarp();
    }
}

template<int K>
inline void launch_scores(const float* scores, int batch, int length, int stride,
                          float* values, int64_t* ids, cudaStream_t stream) {
    select_scores<K><<<batch, 32, 0, stream>>>(scores, batch, length, stride, values, ids);
}

template<int K>
inline void launch_l2(const float* inner, const float* query_norm,
                      const float* center_norm, int batch, int length,
                      float* values, int64_t* ids, cudaStream_t stream) {
    select_l2<K><<<batch, 32, 0, stream>>>(inner, query_norm, center_norm,
                                           batch, length, values, ids);
}

inline bool launch_scores_if_supported(const float* scores, int batch, int length,
                                       int stride, int k, float* values,
                                       int64_t* ids, cudaStream_t stream) {
    switch (k) {
        case 8: launch_scores<8>(scores, batch, length, stride, values, ids, stream); return true;
        case 16: launch_scores<16>(scores, batch, length, stride, values, ids, stream); return true;
        case 32: launch_scores<32>(scores, batch, length, stride, values, ids, stream); return true;
        case 64: launch_scores<64>(scores, batch, length, stride, values, ids, stream); return true;
        default: return false;
    }
}

inline bool launch_l2_if_supported(float* inner, const float* query_norm,
                                   const float* center_norm, int batch, int length,
                                   int k, float* values, int64_t* ids,
                                   cudaStream_t stream) {
    switch (k) {
        case 8: launch_l2<8>(inner, query_norm, center_norm, batch, length, values, ids, stream); return true;
        case 16: launch_l2<16>(inner, query_norm, center_norm, batch, length, values, ids, stream); return true;
        case 32: launch_l2<32>(inner, query_norm, center_norm, batch, length, values, ids, stream); return true;
        case 64: launch_l2<64>(inner, query_norm, center_norm, batch, length, values, ids, stream); return true;
        default: return false;
    }
}

template<class T, int K, int Threads>
inline void launch_l2_vectors(const T* queries, const T* centers,
                              const float* query_norm, const float* center_norm,
                              int batch, int length, float* scratch, int scratch_stride,
                              float* values, int64_t* ids, cudaStream_t stream) {
    constexpr int Tile = 256;
    const int tile_count = (length + Tile - 1) / Tile;
    dim3 grid(tile_count, batch);
    select_l2_vectors_tile<T, K, Tile, 8, Threads><<<grid, Threads, 0, stream>>>(
        queries, centers, query_norm, center_norm, batch, length, tile_count,
        scratch_stride, scratch);
    merge_l2_vector_tiles<K><<<batch, 32, 0, stream>>>(
        scratch, batch, tile_count, scratch_stride, values, ids);
}

template<class T>
inline bool launch_l2_vectors_if_supported(
    const T* queries, const T* centers, const float* query_norm,
    const float* center_norm, int batch, int dim, int length, int k,
    float* scratch, int scratch_stride,
    float* values, int64_t* ids, cudaStream_t stream) {
    if (dim != 128 || length >= (1 << 23) || !queries || !centers || !query_norm || !center_norm ||
        !scratch || !values || !ids || batch <= 0 || length <= 0 ||
        scratch_stride <= 0) return false;
    const int tile_count = (length + 255) / 256;
    if (size_t(tile_count) * size_t(2 * k + 1) > size_t(scratch_stride)) return false;
    const char* threads_env = std::getenv("PARASELECT_ADAPTIVE_DIRECT_THREADS");
    const bool wide = threads_env && std::atoi(threads_env) == 128;
    switch (k) {
        case 8:
            if (wide) launch_l2_vectors<T, 8, 128>(queries, centers, query_norm, center_norm,
                                                     batch, length, scratch, scratch_stride,
                                                     values, ids, stream);
            else launch_l2_vectors<T, 8, 32>(queries, centers, query_norm, center_norm,
                                              batch, length, scratch, scratch_stride,
                                              values, ids, stream);
            return true;
        case 16:
            if (wide) launch_l2_vectors<T, 16, 128>(queries, centers, query_norm, center_norm,
                                                      batch, length, scratch, scratch_stride,
                                                      values, ids, stream);
            else launch_l2_vectors<T, 16, 32>(queries, centers, query_norm, center_norm,
                                               batch, length, scratch, scratch_stride,
                                               values, ids, stream);
            return true;
        case 32:
            if (wide) launch_l2_vectors<T, 32, 128>(queries, centers, query_norm, center_norm,
                                                      batch, length, scratch, scratch_stride,
                                                      values, ids, stream);
            else launch_l2_vectors<T, 32, 32>(queries, centers, query_norm, center_norm,
                                               batch, length, scratch, scratch_stride,
                                               values, ids, stream);
            return true;
        case 64:
            if (wide) launch_l2_vectors<T, 64, 128>(queries, centers, query_norm, center_norm,
                                                      batch, length, scratch, scratch_stride,
                                                      values, ids, stream);
            else launch_l2_vectors<T, 64, 32>(queries, centers, query_norm, center_norm,
                                               batch, length, scratch, scratch_stride,
                                               values, ids, stream);
            return true;
        default: return false;
    }
}

}  // namespace paraselect::coarse::small_topk
