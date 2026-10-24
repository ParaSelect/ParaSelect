#include "search/kernels/topk/reducer/topk_reducer.cuh"

#include "search/kernels/topk/fusion/fusion_dist_topk.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>


namespace {

constexpr int kCubTopkTile = 4096;
constexpr int kCubTopkThreads = 256;
constexpr int kCubTopkItems = 16;


constexpr int kCandidateWarpSortCap = 128;
constexpr int kCandidateTieredCap = 512;
constexpr int kCandidateTieredWarps = 4;


template <typename T>
T* workspace_slice(char*& cursor, size_t& remaining, size_t count) {
    const size_t align = 128;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(cursor);
    const size_t pad = (align - (addr % align)) % align;
    const size_t bytes = count * sizeof(T);
    if (remaining < pad + bytes) return nullptr;
    cursor += pad;
    remaining -= pad;
    T* ptr = reinterpret_cast<T*>(cursor);
    cursor += bytes;
    remaining -= bytes;
    return ptr;
}

size_t align_up(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}

size_t two_buffer_workspace_bytes(size_t count, size_t idx_bytes) {
    constexpr size_t align = 128;
    return 2 * (align_up(count * sizeof(float), align) + align_up(count * idx_bytes, align)) + align;
}



__device__ __forceinline__ bool topk_pair_less(float da, int ia, float db, int ib) {
    return (da < db) || (da == db && ia < ib);
}

template <int K_CAP>
__device__ void insert_candidate_topk(float dist, int idx, int k, float* best_dist, int* best_idx) {
    if (idx < 0) return;
    if (!topk_pair_less(dist, idx, best_dist[k - 1], best_idx[k - 1])) return;
    int pos = k - 1;
    while (pos > 0 && topk_pair_less(dist, idx, best_dist[pos - 1], best_idx[pos - 1])) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
    }
    best_dist[pos] = dist;
    best_idx[pos] = idx;
}

template <int K_CAP>
__global__ void candidate_topk_single_warp_kernel(
    const int* __restrict__ in_idx,
    const float* __restrict__ in_dist,
    int rows,
    int width,
    int k,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    float local_dist[K_CAP];
    int local_idx[K_CAP];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = FLT_MAX;
        local_idx[i] = INT_MAX;
    }

    const size_t base = (size_t)row * (size_t)width;
    for (int col = lane; col < width; col += 32) {
        const int idx = in_idx[base + col];
        const float dist = in_dist[base + col];
        insert_candidate_topk<K_CAP>(dist, idx, k, local_dist, local_idx);
    }

    __shared__ float warp_dist[32 * K_CAP];
    __shared__ int warp_idx[32 * K_CAP];
    for (int i = 0; i < k; ++i) {
        warp_dist[(size_t)lane * K_CAP + i] = local_dist[i];
        warp_idx[(size_t)lane * K_CAP + i] = local_idx[i];
    }
    __syncwarp();

    if (lane == 0) {
        float final_dist[K_CAP];
        int final_idx[K_CAP];
        for (int i = 0; i < k; ++i) {
            final_dist[i] = FLT_MAX;
            final_idx[i] = INT_MAX;
        }
        for (int l = 0; l < 32; ++l) {
            for (int i = 0; i < k; ++i) {
                insert_candidate_topk<K_CAP>(
                    warp_dist[(size_t)l * K_CAP + i],
                    warp_idx[(size_t)l * K_CAP + i],
                    k,
                    final_dist,
                    final_idx);
            }
        }
        for (int i = 0; i < k; ++i) {
            out_dist[(size_t)row * (size_t)k + i] = final_dist[i];
            out_idx[(size_t)row * (size_t)k + i] = final_idx[i] == INT_MAX ? -1 : final_idx[i];
        }
    }
}

template <int K_CAP, int WARPS_PER_BLOCK>
__global__ void candidate_topk_tiered_kernel(
    const int* __restrict__ in_idx,
    const float* __restrict__ in_dist,
    int rows,
    int width,
    int k,
    int thread_topk,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    const int row = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    const size_t base = (size_t)row * (size_t)width;
    __shared__ float warp_dist[WARPS_PER_BLOCK * K_CAP];
    __shared__ int warp_idx[WARPS_PER_BLOCK * K_CAP];

    const int rounded_width = ((width + 31) / 32) * 32;
    paraselect::warpsort::WarpSortFiltered<K_CAP, true, float, int> queue(thread_topk);
    for (int col = warp_id * 32 + lane; col < rounded_width; col += WARPS_PER_BLOCK * 32) {
        const bool valid = col < width;
        const int idx = valid ? in_idx[base + col] : -1;
        const float dist = valid ? in_dist[base + col] : FLT_MAX;
        const bool keep = (idx >= 0 && dist < queue.threshold());
        queue.add(keep ? dist : FLT_MAX, keep ? idx : -1);
    }
    __syncwarp();
    queue.done();
    __syncwarp();
    queue.store(
        warp_dist + (size_t)warp_id * K_CAP,
        warp_idx + (size_t)warp_id * K_CAP);
    __syncthreads();

    if (warp_id == 0) {
        paraselect::warpsort::WarpSortFiltered<K_CAP, true, float, int> merge_queue(k);
        const int rounded_thread_topk = ((thread_topk + 31) / 32) * 32;
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            for (int r = lane; r < rounded_thread_topk; r += 32) {
                const bool valid = r < thread_topk;
                const float dist = valid ? warp_dist[(size_t)w * K_CAP + r] : FLT_MAX;
                const int idx = valid ? warp_idx[(size_t)w * K_CAP + r] : -1;
                const bool keep = (idx >= 0 && dist < merge_queue.threshold());
                merge_queue.add(keep ? dist : FLT_MAX, keep ? idx : -1);
            }
        }
        __syncwarp();
        merge_queue.done();
        __syncwarp();
        float final_dist[K_CAP];
        int final_idx[K_CAP];
        merge_queue.store(final_dist, final_idx);
        for (int i = lane; i < k; i += 32) {
            out_dist[(size_t)row * (size_t)k + i] = final_dist[i];
            out_idx[(size_t)row * (size_t)k + i] = final_idx[i];
        }
    }
}

template <int K_CAP, int WARPS_PER_BLOCK>
__global__ void candidate_topk_tiered_fusion_kernel(
    const int* __restrict__ in_idx,
    const float* __restrict__ in_dist,
    int rows,
    int n_sources,
    int local_k,
    int k,
    int thread_topk,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    const int row = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    __shared__ float warp_dist[WARPS_PER_BLOCK * K_CAP];
    __shared__ int warp_idx[WARPS_PER_BLOCK * K_CAP];
    const size_t row_base = (size_t)row * (size_t)n_sources * (size_t)local_k;
    const int source_take_limit = k > thread_topk ? k : thread_topk;
    const int source_take = local_k < source_take_limit ? local_k : source_take_limit;
    const int compact_width = n_sources * source_take;
    const int rounded_compact_width = ((compact_width + 31) / 32) * 32;

    paraselect::warpsort::WarpSortFiltered<K_CAP, true, float, int> queue(thread_topk);
    for (int pos = warp_id * 32 + lane; pos < rounded_compact_width; pos += WARPS_PER_BLOCK * 32) {
        const bool valid = pos < compact_width;
        const int source = valid ? (pos / source_take) : 0;
        const int r = valid ? (pos - source * source_take) : 0;
        const size_t off = row_base + (size_t)source * (size_t)local_k + (size_t)r;
        const int idx = valid ? in_idx[off] : -1;
        const float dist = valid ? in_dist[off] : FLT_MAX;
        const bool keep = (idx >= 0 && dist < queue.threshold());
        queue.add(keep ? dist : FLT_MAX, keep ? idx : -1);
    }
    __syncwarp();
    queue.done();
    __syncwarp();
    queue.store(
        warp_dist + (size_t)warp_id * K_CAP,
        warp_idx + (size_t)warp_id * K_CAP);
    __syncthreads();

    if (warp_id == 0) {
        paraselect::warpsort::WarpSortFiltered<K_CAP, true, float, int> merge_queue(k);
        const int rounded_thread_topk = ((thread_topk + 31) / 32) * 32;
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            for (int r = lane; r < rounded_thread_topk; r += 32) {
                const bool valid = r < thread_topk;
                const float dist = valid ? warp_dist[(size_t)w * K_CAP + r] : FLT_MAX;
                const int idx = valid ? warp_idx[(size_t)w * K_CAP + r] : -1;
                const bool keep = (idx >= 0 && dist < merge_queue.threshold());
                merge_queue.add(keep ? dist : FLT_MAX, keep ? idx : -1);
            }
        }
        __syncwarp();
        merge_queue.done();
        __syncwarp();
        float final_dist[K_CAP];
        int final_idx[K_CAP];
        merge_queue.store(final_dist, final_idx);
        for (int i = lane; i < k; i += 32) {
            out_dist[(size_t)row * (size_t)k + i] = final_dist[i];
            out_idx[(size_t)row * (size_t)k + i] = final_idx[i];
        }
    }
}


__global__ void candidate_cub_block_initial_kernel(
    const int* __restrict__ in_idx,
    const float* __restrict__ in_dist,
    int rows,
    int width,
    int k,
    int n_tiles,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    using BlockSort = cub::BlockRadixSort<float, kCubTopkThreads, kCubTopkItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kCubTopkItems];
    int vals[kCubTopkItems];

    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    const int base = tile * kCubTopkTile;
    const size_t row_base = (size_t)row * (size_t)width;

    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int local = threadIdx.x * kCubTopkItems + item;
        const int col = base + local;
        if (row < rows && col < width) {
            const int id = in_idx[row_base + col];
            keys[item] = id >= 0 ? in_dist[row_base + col] : FLT_MAX;
            vals[item] = id >= 0 ? id : INT_MAX;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    BlockSort(sort_storage).Sort(keys, vals);

    const size_t out_base = ((size_t)row * (size_t)n_tiles + (size_t)tile) * (size_t)k;
    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int rank = threadIdx.x * kCubTopkItems + item;
        if (rank < k) {
            out_dist[out_base + rank] = keys[item];
            out_idx[out_base + rank] = vals[item] == INT_MAX ? -1 : vals[item];
        }
    }
}

__global__ void candidate_cub_block_merge_kernel(
    const int* __restrict__ in_idx,
    const float* __restrict__ in_dist,
    int rows,
    int in_count,
    int k,
    int n_tiles,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    using BlockSort = cub::BlockRadixSort<float, kCubTopkThreads, kCubTopkItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kCubTopkItems];
    int vals[kCubTopkItems];

    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    const int base = tile * kCubTopkTile;

    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int local = threadIdx.x * kCubTopkItems + item;
        const int pos = base + local;
        if (row < rows && pos < in_count) {
            const size_t in_off = (size_t)row * (size_t)in_count + (size_t)pos;
            const int id = in_idx[in_off];
            keys[item] = id >= 0 ? in_dist[in_off] : FLT_MAX;
            vals[item] = id >= 0 ? id : INT_MAX;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    BlockSort(sort_storage).Sort(keys, vals);

    const size_t out_base = ((size_t)row * (size_t)n_tiles + (size_t)tile) * (size_t)k;
    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int rank = threadIdx.x * kCubTopkItems + item;
        if (rank < k) {
            out_dist[out_base + rank] = keys[item];
            out_idx[out_base + rank] = vals[item] == INT_MAX ? -1 : vals[item];
        }
    }
}


template <bool Cosine>
__global__ void cub_block_topk_initial_kernel(
    const float* __restrict__ inner_product,
    const float* __restrict__ query_norm,
    const float* __restrict__ data_norm,
    int n_rows,
    int n_cols,
    int k,
    int n_tiles,
    float* __restrict__ out_dist,
    idx_t* __restrict__ out_idx)
{
    using BlockSort = cub::BlockRadixSort<float, kCubTopkThreads, kCubTopkItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kCubTopkItems];
    int vals[kCubTopkItems];

    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    const int base = tile * kCubTopkTile;
    const float qn = query_norm[row];

    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int local = threadIdx.x * kCubTopkItems + item;
        const int col = base + local;
        if (col < n_cols) {
            const float cn = data_norm[col];
            const float ip = inner_product[(size_t)row * (size_t)n_cols + (size_t)col];
            keys[item] = Cosine ? (1.0f - ip / (qn * cn)) : (qn * qn + cn * cn - 2.0f * ip);
            vals[item] = col;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    BlockSort(sort_storage).Sort(keys, vals);

    const size_t out_base = ((size_t)row * (size_t)n_tiles + (size_t)tile) * (size_t)k;
    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int rank = threadIdx.x * kCubTopkItems + item;
        if (rank < k) {
            out_dist[out_base + rank] = keys[item];
            out_idx[out_base + rank] = vals[item] == INT_MAX ? (idx_t)-1 : (idx_t)vals[item];
        }
    }
}

__global__ void cub_block_topk_merge_kernel(
    const float* __restrict__ in_dist,
    const idx_t* __restrict__ in_idx,
    int n_rows,
    int in_count,
    int k,
    int n_tiles,
    float* __restrict__ out_dist,
    idx_t* __restrict__ out_idx)
{
    using BlockSort = cub::BlockRadixSort<float, kCubTopkThreads, kCubTopkItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kCubTopkItems];
    int vals[kCubTopkItems];

    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    const int base = tile * kCubTopkTile;

    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int local = threadIdx.x * kCubTopkItems + item;
        const int pos = base + local;
        if (pos < in_count) {
            const size_t in_off = (size_t)row * (size_t)in_count + (size_t)pos;
            keys[item] = in_dist[in_off];
            const idx_t id = in_idx[in_off];
            vals[item] = id < 0 ? INT_MAX : (int)id;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    BlockSort(sort_storage).Sort(keys, vals);

    const size_t out_base = ((size_t)row * (size_t)n_tiles + (size_t)tile) * (size_t)k;
    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        const int rank = threadIdx.x * kCubTopkItems + item;
        if (rank < k) {
            out_dist[out_base + rank] = keys[item];
            out_idx[out_base + rank] = vals[item] == INT_MAX ? (idx_t)-1 : (idx_t)vals[item];
        }
    }
}

cudaError_t rowwise_topk_cub_block(const paraselect::topk::RowWiseTopKArgs& args) {
    if (args.k > kCubTopkTile || !args.select_min) return cudaErrorInvalidValue;

    const int n_tiles = (args.n_cols + kCubTopkTile - 1) / kCubTopkTile;
    if (n_tiles == 1) {
        dim3 grid(1, args.n_rows);
        if (args.distance_op == paraselect::topk::TopKDistanceOp::kCosine) {
            cub_block_topk_initial_kernel<true><<<grid, kCubTopkThreads, 0, args.stream>>>(
                args.inner_product, args.query_norm, args.data_norm,
                args.n_rows, args.n_cols, args.k, 1, args.out_dist, args.out_idx);
        } else {
            cub_block_topk_initial_kernel<false><<<grid, kCubTopkThreads, 0, args.stream>>>(
                args.inner_product, args.query_norm, args.data_norm,
                args.n_rows, args.n_cols, args.k, 1, args.out_dist, args.out_idx);
        }
        return cudaGetLastError();
    }

    const size_t cand_count = (size_t)args.n_rows * (size_t)n_tiles * (size_t)args.k;
    const size_t required = two_buffer_workspace_bytes(cand_count, sizeof(idx_t));
    if (!args.workspace || args.workspace_bytes < required) return cudaErrorInvalidValue;

    char* cursor = static_cast<char*>(args.workspace);
    size_t remaining = args.workspace_bytes;
    float* buf1_dist = workspace_slice<float>(cursor, remaining, cand_count);
    idx_t* buf1_idx = workspace_slice<idx_t>(cursor, remaining, cand_count);
    float* buf2_dist = workspace_slice<float>(cursor, remaining, cand_count);
    idx_t* buf2_idx = workspace_slice<idx_t>(cursor, remaining, cand_count);
    if (!buf1_dist || !buf1_idx || !buf2_dist || !buf2_idx) return cudaErrorInvalidValue;

    dim3 grid(n_tiles, args.n_rows);
    if (args.distance_op == paraselect::topk::TopKDistanceOp::kCosine) {
        cub_block_topk_initial_kernel<true><<<grid, kCubTopkThreads, 0, args.stream>>>(
            args.inner_product, args.query_norm, args.data_norm,
            args.n_rows, args.n_cols, args.k, n_tiles, buf1_dist, buf1_idx);
    } else {
        cub_block_topk_initial_kernel<false><<<grid, kCubTopkThreads, 0, args.stream>>>(
            args.inner_product, args.query_norm, args.data_norm,
            args.n_rows, args.n_cols, args.k, n_tiles, buf1_dist, buf1_idx);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    int in_count = n_tiles * args.k;
    float* cur_dist = buf1_dist;
    idx_t* cur_idx = buf1_idx;
    while (in_count > args.k) {
        const int merge_tiles = (in_count + kCubTopkTile - 1) / kCubTopkTile;
        const int next_count = merge_tiles * args.k;
        const bool final_pass = next_count == args.k;
        float* next_dist = final_pass ? args.out_dist : ((cur_dist == buf1_dist) ? buf2_dist : buf1_dist);
        idx_t* next_idx = final_pass ? args.out_idx : ((cur_idx == buf1_idx) ? buf2_idx : buf1_idx);
        cub_block_topk_merge_kernel<<<dim3(merge_tiles, args.n_rows), kCubTopkThreads, 0, args.stream>>>(
            cur_dist, cur_idx, args.n_rows, in_count, args.k, merge_tiles, next_dist, next_idx);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        cur_dist = next_dist;
        cur_idx = next_idx;
        in_count = next_count;
    }

    return (cur_dist == args.out_dist && cur_idx == args.out_idx) ? cudaSuccess : cudaErrorUnknown;
}


}  // namespace

namespace paraselect {
namespace topk {

const char* topk_reducer_backend_name(TopKReducerBackend backend) {
    switch (backend) {
        case TopKReducerBackend::kWarpSort:
            return "warpsort";
        case TopKReducerBackend::kCubBlock:
            return "cub_block";
        case TopKReducerBackend::kTiered:
            return "tiered";
        case TopKReducerBackend::kTieredFusion:
            return "tiered_fusion";
    }
    return "unknown";
}

TopKReducerBackend parse_topk_reducer_backend(const char* name, TopKReducerBackend fallback) {
    if (!name || name[0] == '\0') return fallback;
    if (std::strcmp(name, "warpsort") == 0) return TopKReducerBackend::kWarpSort;
    if (std::strcmp(name, "cub_block") == 0 || std::strcmp(name, "cub") == 0) {
        return TopKReducerBackend::kCubBlock;
    }
    if (std::strcmp(name, "tiered") == 0) return TopKReducerBackend::kTiered;
    if (std::strcmp(name, "tiered_fusion") == 0) return TopKReducerBackend::kTieredFusion;
    return fallback;
}

size_t rowwise_topk_workspace_bytes(const RowWiseTopKArgs& args, TopKReducerBackend backend) {
    if (backend != TopKReducerBackend::kCubBlock ||
        args.n_rows <= 0 || args.n_cols <= 0 || args.k <= 0) {
        return 0;
    }
    const int n_tiles = (args.n_cols + kCubTopkTile - 1) / kCubTopkTile;
    if (n_tiles <= 1) return 0;
    const size_t cand_count = (size_t)args.n_rows * (size_t)n_tiles * (size_t)args.k;
    return two_buffer_workspace_bytes(cand_count, sizeof(idx_t));
}

size_t candidate_topk_workspace_bytes(const CandidateTopKArgs& args, TopKReducerBackend backend) {
    if (backend != TopKReducerBackend::kCubBlock ||
        args.n_rows <= 0 || args.n_sources <= 0 || args.local_k <= 0 || args.global_k <= 0) {
        return 0;
    }
    const int width = args.n_sources * args.local_k;
    const int n_tiles = (width + kCubTopkTile - 1) / kCubTopkTile;
    if (n_tiles <= 1) return 0;
    const size_t cand_count = (size_t)args.n_rows * (size_t)n_tiles * (size_t)args.global_k;
    return two_buffer_workspace_bytes(cand_count, sizeof(int));
}


static cudaError_t rowwise_topk_warpsort(const RowWiseTopKArgs& args) {
    if (args.distance_op == TopKDistanceOp::kCosine) {
        return paraselect::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, idx_t>(
            args.query_norm,
            args.data_norm,
            args.inner_product,
            args.n_rows,
            args.n_cols,
            args.k,
            args.out_dist,
            args.out_idx,
            args.select_min,
            args.stream,
            args.cuvs_res);
    }
    return paraselect::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, idx_t>(
        args.query_norm,
        args.data_norm,
        args.inner_product,
        args.n_rows,
        args.n_cols,
        args.k,
        args.out_dist,
        args.out_idx,
        args.select_min,
        args.stream,
        args.cuvs_res);
}

cudaError_t rowwise_topk_reduce(
    const RowWiseTopKArgs& args,
    TopKReducerBackend backend)
{
    if (!args.query_norm || !args.data_norm || !args.inner_product ||
        !args.out_dist || !args.out_idx ||
        args.n_rows <= 0 || args.n_cols <= 0 || args.k <= 0 || args.k > args.n_cols) {
        return cudaErrorInvalidValue;
    }

    switch (backend) {
        case TopKReducerBackend::kWarpSort:
            return rowwise_topk_warpsort(args);
        case TopKReducerBackend::kCubBlock:
            return rowwise_topk_cub_block(args);
        case TopKReducerBackend::kTiered:
        case TopKReducerBackend::kTieredFusion:
            return cudaErrorNotSupported;
    }
    return cudaErrorInvalidValue;
}


static cudaError_t candidate_topk_warpsort(const CandidateTopKArgs& args) {
    if (args.global_k > kCandidateWarpSortCap) return cudaErrorNotSupported;
    const int width = args.n_sources * args.local_k;
    candidate_topk_single_warp_kernel<kCandidateWarpSortCap><<<args.n_rows, 32, 0, args.stream>>>(
        args.candidate_idx,
        args.candidate_dist,
        args.n_rows,
        width,
        args.global_k,
        args.out_idx,
        args.out_dist);
    return cudaGetLastError();
}

static cudaError_t candidate_topk_tiered(const CandidateTopKArgs& args) {
    if (args.global_k > kCandidateTieredCap) return cudaErrorNotSupported;
    const int thread_topk = std::max(1, std::min(args.thread_topk, kCandidateTieredCap));
    const int width = args.n_sources * args.local_k;
    candidate_topk_tiered_kernel<kCandidateTieredCap, kCandidateTieredWarps>
        <<<args.n_rows, kCandidateTieredWarps * 32, 0, args.stream>>>(
            args.candidate_idx,
            args.candidate_dist,
            args.n_rows,
            width,
            args.global_k,
            thread_topk,
            args.out_idx,
            args.out_dist);
    return cudaGetLastError();
}

static cudaError_t candidate_topk_tiered_fusion(const CandidateTopKArgs& args) {
    if (args.global_k > kCandidateTieredCap) return cudaErrorNotSupported;
    const int thread_topk = std::max(1, std::min(args.thread_topk, kCandidateTieredCap));
    candidate_topk_tiered_fusion_kernel<kCandidateTieredCap, kCandidateTieredWarps>
        <<<args.n_rows, kCandidateTieredWarps * 32, 0, args.stream>>>(
            args.candidate_idx,
            args.candidate_dist,
            args.n_rows,
            args.n_sources,
            args.local_k,
            args.global_k,
            thread_topk,
            args.out_idx,
            args.out_dist);
    return cudaGetLastError();
}

static cudaError_t candidate_topk_cub_block(const CandidateTopKArgs& args) {
    if (args.global_k > kCubTopkTile) return cudaErrorInvalidValue;

    const int width = args.n_sources * args.local_k;
    const int n_tiles = (width + kCubTopkTile - 1) / kCubTopkTile;
    if (n_tiles == 1) {
        candidate_cub_block_initial_kernel<<<dim3(1, args.n_rows), kCubTopkThreads, 0, args.stream>>>(
            args.candidate_idx,
            args.candidate_dist,
            args.n_rows,
            width,
            args.global_k,
            1,
            args.out_idx,
            args.out_dist);
        return cudaGetLastError();
    }

    const size_t cand_count = (size_t)args.n_rows * (size_t)n_tiles * (size_t)args.global_k;
    const size_t required = two_buffer_workspace_bytes(cand_count, sizeof(int));
    if (!args.workspace || args.workspace_bytes < required) return cudaErrorInvalidValue;

    char* cursor = static_cast<char*>(args.workspace);
    size_t remaining = args.workspace_bytes;
    float* buf1_dist = workspace_slice<float>(cursor, remaining, cand_count);
    int* buf1_idx = workspace_slice<int>(cursor, remaining, cand_count);
    float* buf2_dist = workspace_slice<float>(cursor, remaining, cand_count);
    int* buf2_idx = workspace_slice<int>(cursor, remaining, cand_count);
    if (!buf1_dist || !buf1_idx || !buf2_dist || !buf2_idx) return cudaErrorInvalidValue;

    candidate_cub_block_initial_kernel<<<dim3(n_tiles, args.n_rows), kCubTopkThreads, 0, args.stream>>>(
        args.candidate_idx,
        args.candidate_dist,
        args.n_rows,
        width,
        args.global_k,
        n_tiles,
        buf1_idx,
        buf1_dist);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    int in_count = n_tiles * args.global_k;
    float* cur_dist = buf1_dist;
    int* cur_idx = buf1_idx;
    while (in_count > args.global_k) {
        const int merge_tiles = (in_count + kCubTopkTile - 1) / kCubTopkTile;
        const int next_count = merge_tiles * args.global_k;
        const bool final_pass = next_count == args.global_k;
        float* next_dist = final_pass ? args.out_dist : ((cur_dist == buf1_dist) ? buf2_dist : buf1_dist);
        int* next_idx = final_pass ? args.out_idx : ((cur_idx == buf1_idx) ? buf2_idx : buf1_idx);
        candidate_cub_block_merge_kernel<<<dim3(merge_tiles, args.n_rows), kCubTopkThreads, 0, args.stream>>>(
            cur_idx,
            cur_dist,
            args.n_rows,
            in_count,
            args.global_k,
            merge_tiles,
            next_idx,
            next_dist);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        cur_dist = next_dist;
        cur_idx = next_idx;
        in_count = next_count;
    }

    return (cur_dist == args.out_dist && cur_idx == args.out_idx) ? cudaSuccess : cudaErrorUnknown;
}


cudaError_t candidate_topk_reduce(
    const CandidateTopKArgs& args,
    TopKReducerBackend backend)
{
    if (!args.candidate_idx || !args.candidate_dist || !args.out_idx || !args.out_dist ||
        args.n_rows <= 0 || args.n_sources <= 0 ||
        args.local_k <= 0 || args.global_k <= 0 ||
        args.thread_topk <= 0 ||
        args.global_k > args.n_sources * args.local_k) {
        return cudaErrorInvalidValue;
    }

    switch (backend) {
        case TopKReducerBackend::kWarpSort:
            return candidate_topk_warpsort(args);
        case TopKReducerBackend::kCubBlock:
            return candidate_topk_cub_block(args);
        case TopKReducerBackend::kTiered:
            return candidate_topk_tiered(args);
        case TopKReducerBackend::kTieredFusion:
            return candidate_topk_tiered_fusion(args);
    }
    return cudaErrorInvalidValue;
}


}  // namespace topk
}  // namespace paraselect
