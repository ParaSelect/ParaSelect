#include "fused_gemm_topk.cuh"

#include "search/kernels/coarse/topk/adaptive_threshold_topk/adaptive_threshold_topk_fp32.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace paraselect::coarse {
namespace {

constexpr int kRows = 32;
constexpr int kCols = 128;

#include "fused_gemm_top1.cuh"

__device__ __forceinline__ unsigned ordered_float(float value) {
    const unsigned bits = __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

template<int Bits, int Items>
__device__ float kth_score(
    const float (&keys)[Items], unsigned* histogram, int rank) {
    constexpr int bins = 1 << Bits;
    constexpr int bins_per_lane = bins / 32;
    constexpr unsigned digit_mask = bins - 1;
    const int lane = threadIdx.x & 31;
    unsigned prefix = 0;
    unsigned mask = 0;
    for (int shift = (31 / Bits) * Bits; shift >= 0; shift -= Bits) {
        unsigned low = ~0u;
        unsigned high = 0;
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const unsigned value = ordered_float(keys[i]);
            if ((value & mask) == prefix) {
                low = min(low, value);
                high = max(high, value);
            }
        }
        low = __reduce_min_sync(0xffffffff, low);
        high = __reduce_max_sync(0xffffffff, high);
        if (low == high) {
            const unsigned bits = high & 0x80000000u ? high ^ 0x80000000u : ~high;
            return __uint_as_float(bits);
        }
        if ((low >> shift) == (high >> shift)) {
            prefix |= ((high >> shift) & digit_mask) << shift;
            mask |= digit_mask << shift;
            continue;
        }
        for (int i = lane; i < bins; i += 32) histogram[i] = 0;
        __syncwarp();
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const unsigned value = ordered_float(keys[i]);
            if ((value & mask) == prefix) {
                atomicAdd(histogram + ((value >> shift) & digit_mask), 1u);
            }
        }
        __syncwarp();
        unsigned sum = 0;
#pragma unroll
        for (int i = 0; i < bins_per_lane; ++i) {
            sum += histogram[bins - 1 - lane * bins_per_lane - i];
        }
        unsigned inclusive = sum;
#pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            const unsigned previous = __shfl_up_sync(0xffffffff, inclusive, delta);
            if (lane >= delta) inclusive += previous;
        }
        int preceding = static_cast<int>(inclusive - sum);
        int selected = -1;
        int above = 0;
#pragma unroll
        for (int i = 0; i < bins_per_lane; ++i) {
            const int bin = bins - 1 - lane * bins_per_lane - i;
            const int count = static_cast<int>(histogram[bin]);
            if (rank > preceding && rank <= preceding + count) {
                selected = bin;
                above = preceding;
            }
            preceding += count;
        }
        selected = __reduce_max_sync(0xffffffff, selected);
        above = __reduce_max_sync(0xffffffff, above);
        rank -= above;
        prefix |= static_cast<unsigned>(selected) << shift;
        mask |= digit_mask << shift;
        __syncwarp();
    }
    const unsigned bits = prefix & 0x80000000u ? prefix ^ 0x80000000u : ~prefix;
    return __uint_as_float(bits);
}

template<int Items>
__device__ void retain_topk(
    const float (&keys)[Items], const int (&key_ids)[Items], int k, float cutoff,
    float* output, int* output_ids) {
    const int lane = threadIdx.x & 31;
    int base = 0;
    for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const bool hit = pass == 0 ? keys[i] > cutoff : keys[i] == cutoff;
            const unsigned ballot = __ballot_sync(0xffffffff, hit);
            const int position = base + __popc(ballot & ((1u << lane) - 1));
            if (hit && position < k) {
                output[position] = keys[i];
                output_ids[position] = key_ids[i];
            }
            base += __popc(ballot);
        }
    }
    __syncwarp();
}

template<int K, int Bits, int Delay, int Threads, int Rows = 32, int Cols = 128>
struct alignas(32) SharedPlan {
    static_assert(Threads == 256 || Threads == 512);
    static_assert(Rows == 16 || Rows == 32);
    static_assert(Cols == 128 || Cols == 256);
    static constexpr int kWarps = Threads / 32;
    static constexpr int kBuffer = K == 1 ? 1 : K + Delay + 32;
    static constexpr int kCols = Cols;
    static constexpr int kQueryStride = 136;
    static constexpr int kDotStride = Cols + 8;
    __half query[Rows * kQueryStride];
    float dot[Rows * kDotStride];
    float center_norm[Cols];
    float best[Rows][kBuffer];
    int candidate_ids[Rows][kBuffer];
    unsigned histogram[kWarps][1 << Bits];
    float cutoff[Rows];
    int count[Rows];
};

template<int K, int Bits, int Delay, int Threads, int Rows, int Cols>
__device__ float rebuild(
    SharedPlan<K, Bits, Delay, Threads, Rows, Cols>& shared, int row, int warp, int count) {
    constexpr int items = (SharedPlan<K, Bits, Delay, Threads, Rows, Cols>::kBuffer + 31) / 32;
    float keys[items];
    int key_ids[items];
    const int lane = threadIdx.x & 31;
    __syncwarp();
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int offset = lane + i * 32;
        keys[i] = offset < count ? shared.best[row][offset] : -INFINITY;
        key_ids[i] = offset < count ? shared.candidate_ids[row][offset] : -1;
    }
    __syncwarp();
    const float cutoff = kth_score<Bits>(keys, shared.histogram[warp], K);
    retain_topk(keys, key_ids, K, cutoff, shared.best[row], shared.candidate_ids[row]);
    return cutoff;
}

template<int K, int Bits, int Delay, int Threads = 256, bool Compact = false,
         int Rows = 32, int Cols = 128, bool CacheQuery = false>
__global__ void fused_gemm_local_topk(
    const __half* queries,
    const __half* centers,
    const float* query_norm,
    const float* center_norm,
    int batch,
    int chunk,
    int pool_stride,
    float* pool_scores,
    float* pool_ids,
    int* compact_locks) {
    extern __shared__ __align__(32) unsigned char storage[];
    using Shared = SharedPlan<K, Bits, Delay, Threads, Rows, Cols>;
    auto& shared = *reinterpret_cast<Shared*>(storage);
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    const int query_base = blockIdx.x * Rows;

    for (int i = tid; i < Rows * Shared::kQueryStride; i += Threads) {
        const int row = query_base + i / Shared::kQueryStride;
        const int column = i % Shared::kQueryStride;
        shared.query[i] = row < batch && column < 128
            ? queries[static_cast<size_t>(row) * 128 + column]
            : __float2half(0.0f);
    }
    for (int i = tid; i < Rows * Shared::kBuffer; i += Threads) {
        shared.best[i / Shared::kBuffer][i % Shared::kBuffer] = -INFINITY;
        shared.candidate_ids[i / Shared::kBuffer][i % Shared::kBuffer] = -1;
    }
    if (tid < Rows) {
        shared.cutoff[tid] = -INFINITY;
        shared.count[tid] = 0;
    }
    __syncthreads();

    using QueryFragment = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major>;
    QueryFragment cached_query[CacheQuery ? Rows / 16 * 8 : 1];
    if constexpr (CacheQuery) {
#pragma unroll
        for (int group = 0; group < Rows / 16; ++group) {
#pragma unroll
            for (int depth = 0; depth < 8; ++depth) {
                nvcuda::wmma::load_matrix_sync(
                    cached_query[group * 8 + depth],
                    shared.query + group * 16 * Shared::kQueryStride + depth * 16,
                    Shared::kQueryStride);
            }
        }
    }
    const int start = blockIdx.y * chunk;
    for (int begin = start; begin < start + chunk; begin += Shared::kCols) {
        for (int i = tid; i < Shared::kCols; i += Threads) {
            shared.center_norm[i] = center_norm[begin + i];
        }
        if (warp < 8) {
            QueryFragment query_fragment;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                nvcuda::wmma::col_major> center_fragment;
            nvcuda::wmma::fragment<
                nvcuda::wmma::accumulator, 16, 16, 16, float> accum[2];
            constexpr int query_groups = Rows / 16;
#pragma unroll
            for (int part = 0; part < Shared::kCols / 128; ++part) {
                const int column = warp * 16 + part * 128;
#pragma unroll
                for (int group = 0; group < query_groups; ++group) {
                    nvcuda::wmma::fill_fragment(accum[group], 0.0f);
                }
#pragma unroll
                for (int depth = 0; depth < 128; depth += 16) {
                    nvcuda::wmma::load_matrix_sync(
                        center_fragment,
                        centers + static_cast<size_t>(begin + column) * 128 + depth,
                        128);
#pragma unroll
                    for (int group = 0; group < query_groups; ++group) {
                        if constexpr (CacheQuery) {
                            nvcuda::wmma::mma_sync(
                                accum[group], cached_query[group * 8 + depth / 16],
                                center_fragment, accum[group]);
                        } else {
                            nvcuda::wmma::load_matrix_sync(
                                query_fragment,
                                shared.query + group * 16 * Shared::kQueryStride + depth,
                                Shared::kQueryStride);
                            nvcuda::wmma::mma_sync(
                                accum[group], query_fragment, center_fragment,
                                accum[group]);
                        }
                    }
                }
#pragma unroll
                for (int group = 0; group < query_groups; ++group) {
                    nvcuda::wmma::store_matrix_sync(
                        shared.dot + group * 16 * Shared::kDotStride + column,
                        accum[group], Shared::kDotStride,
                        nvcuda::wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        for (int round = 0; round < Rows / Shared::kWarps; ++round) {
            const int row = warp + round * Shared::kWarps;
            const int query = query_base + row;
            if (query < batch) {
                if constexpr (K == 1) {
                    // Assignment needs argmin only: reuse the WMMA/L2 tile, but
                    // compile out histogram construction and candidate buffering.
                    float best = shared.best[row][0];
                    int best_id = shared.candidate_ids[row][0];
                    // Assignment-only callers may omit query_norm.  The
                    // query term is constant across a row, so zero preserves
                    // the argmin ID ordering and finish_top1 restores the
                    // distance when the norm is available.
                    const float norm = query_norm ? query_norm[query] : 0.0f;
#pragma unroll
                    for (int i = 0; i < Shared::kCols / 32; ++i) {
                        const int column = lane + i * 32;
                        const float center = shared.center_norm[column];
                        const float dot = shared.dot[row * Shared::kDotStride + column];
                        const float score = -__fsub_rn(
                            __fmaf_rn(norm, norm, __fmul_rn(center, center)),
                            __fadd_rn(dot, dot));
                        const int id = begin + column;
                        if (score > best || (score == best && id < best_id)) {
                            best = score;
                            best_id = id;
                        }
                    }
                    const unsigned changed = __ballot_sync(0xffffffff,
                        best > shared.best[row][0] ||
                        (best == shared.best[row][0] && best_id < shared.candidate_ids[row][0]));
                    if (changed != 0) {
#pragma unroll
                        for (int delta = 16; delta > 0; delta >>= 1) {
                            const float other = __shfl_down_sync(0xffffffff, best, delta);
                            const int other_id = __shfl_down_sync(0xffffffff, best_id, delta);
                            if (lane + delta < 32 &&
                                (other > best || (other == best && other_id < best_id))) {
                                best = other;
                                best_id = other_id;
                            }
                        }
                    }
                    if (lane == 0 && changed != 0) {
                        shared.best[row][0] = best;
                        shared.candidate_ids[row][0] = best_id;
                    }
                } else {
                constexpr int items = (Shared::kCols + K + 31) / 32;
                float keys[items];
                int key_ids[items];
                int new_hits = 0;
                float cutoff = shared.cutoff[row];
                const float norm = query_norm[query];
#pragma unroll
                for (int i = 0; i < items; ++i) {
                    const int offset = lane + i * 32;
                    if (offset < Shared::kCols) {
                        const float center = shared.center_norm[offset];
                        const float dot = shared.dot[row * Shared::kDotStride + offset];
                        keys[i] = -__fsub_rn(
                            __fmaf_rn(norm, norm, __fmul_rn(center, center)),
                            __fadd_rn(dot, dot));
                        key_ids[i] = begin + offset;
                        new_hits += keys[i] > cutoff;
                    } else {
                        keys[i] = shared.best[row][offset - Shared::kCols];
                        key_ids[i] = shared.candidate_ids[row][offset - Shared::kCols];
                    }
                }
                int count = shared.count[row];
                if (begin == start) {
                    cutoff = kth_score<Bits>(keys, shared.histogram[warp], K);
                    retain_topk(keys, key_ids, K, cutoff, shared.best[row], shared.candidate_ids[row]);
                    count = K;
                } else if (__reduce_add_sync(0xffffffff, new_hits) > 0) {
#pragma unroll
                    for (int i = 0; i < Shared::kCols / 32; ++i) {
                        const bool hit = keys[i] > cutoff;
                        const unsigned ballot = __ballot_sync(0xffffffff, hit);
                        const int position = count + __popc(ballot & ((1u << lane) - 1));
                        if (hit) {
                            shared.best[row][position] = keys[i];
                            shared.candidate_ids[row][position] = key_ids[i];
                        }
                        count += __popc(ballot);
                        if (count >= K + Delay) {
                            cutoff = rebuild(shared, row, warp, count);
                            count = K;
                        }
                    }
                }
                if (begin + Shared::kCols == start + chunk && count > K) {
                    cutoff = rebuild(shared, row, warp, count);
                    count = K;
                }
                __syncwarp();
                if (lane == 0) {
                    shared.count[row] = count;
                    shared.cutoff[row] = cutoff;
                }
                } // K != 1
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int i = tid; i < Rows * K; i += Threads) {
        const int row = i / K;
        const int rank = i % K;
        const int query = query_base + row;
        if (query < batch) {
            if constexpr (!Compact) {
                const size_t output = static_cast<size_t>(query) * pool_stride + blockIdx.y * K + rank;
                pool_scores[output] = shared.best[row][rank];
                pool_ids[output] = __int_as_float(shared.candidate_ids[row][rank]);
            }
        }
    }
    if constexpr (Compact) {
        // One warp owns one query row. Serialize only the K-item merge into
        // the query's compact output; the expensive WMMA scan stays parallel
        // across center chunks and query groups.
        if (tid < Rows) {
            const int row = tid;
            const int query = query_base + row;
            if (query < batch) {
                int* lock = compact_locks + query;
                while (atomicCAS(lock, 0, 1) != 0) { }
                float merged[ K ];
                int merged_ids[ K ];
#pragma unroll
                for (int i = 0; i < K; ++i) {
                    merged[i] = pool_scores[size_t(query) * K + i];
                    merged_ids[i] = static_cast<int>(pool_ids[size_t(query) * K + i]);
                }
#pragma unroll
                for (int candidate = 0; candidate < K; ++candidate) {
                    const float score = shared.best[row][candidate];
                    const int id = shared.candidate_ids[row][candidate];
                    int slot = K - 1;
                    if (score > merged[slot] || (score == merged[slot] && id < merged_ids[slot])) {
                        while (slot > 0 && (score > merged[slot - 1] ||
                               (score == merged[slot - 1] && id < merged_ids[slot - 1]))) {
                            merged[slot] = merged[slot - 1];
                            merged_ids[slot] = merged_ids[slot - 1];
                            --slot;
                        }
                        merged[slot] = score;
                        merged_ids[slot] = id;
                    }
                }
#pragma unroll
                for (int i = 0; i < K; ++i) {
                    pool_scores[size_t(query) * K + i] = merged[i];
                    pool_ids[size_t(query) * K + i] = static_cast<float>(merged_ids[i]);
                }
                __threadfence();
                atomicExch(lock, 0);
            }
        }
    }
}

// One warp per vector merges chunk minima. Ties choose the smallest centroid ID.
__global__ void finish_top1(
    const float* scores, const float* pool_ids, int batch, int length, int stride,
    float* distances, idx_t* ids, const float* query_norm) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (row >= batch) return;
    float best = -INFINITY;
    int best_id = 0x7fffffff;
    for (int i = lane; i < length; i += 32) {
        const size_t offset = static_cast<size_t>(row) * stride + i;
        const float score = scores[offset];
        const int id = __float_as_int(pool_ids[offset]);
        if (score > best || (score == best && id < best_id)) {
            best = score;
            best_id = id;
        }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, best, delta);
        const int other_id = __shfl_down_sync(0xffffffff, best_id, delta);
        if (lane + delta < 32 &&
            (other > best || (other == best && other_id < best_id))) {
            best = other;
            best_id = other_id;
        }
    }
    if (lane == 0) {
        const float qn = query_norm ? query_norm[row] : 0.0f;
        distances[row] = __fmaf_rn(qn, qn, -best);
        ids[row] = best_id;
    }
}

__global__ void finish_topk(
    float* distances, idx_t* ids, const float* pool_ids, size_t total,
    int k, int pool_stride) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    ids[i] = static_cast<idx_t>(__float_as_int(pool_ids[(i / k) * pool_stride + ids[i]]));
    distances[i] = -distances[i];
}

__global__ void merge_k8_supertile_inplace(
    float* scores, float* ids, int batch, int chunks, int src_stride,
    int dst_stride) {
    const int query = blockIdx.x;
    const int supertile = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (query >= batch || supertile >= (chunks + 7) / 8) return;
    __shared__ float sort_s[64];
    __shared__ int sort_i[64];
    const size_t base = static_cast<size_t>(query) * src_stride;
    const int begin = supertile * 8 * 8;
    const int end = min(begin + 64, chunks * 8);
    for (int j = 0; j < 2; ++j) {
        const int p = begin + lane + j * 32;
        sort_s[lane + j * 32] = p < end ? scores[base + p] : -INFINITY;
        sort_i[lane + j * 32] = p < end ? __float_as_int(ids[base + p]) : 0x7fffffff;
    }
    __syncwarp();
    for (int width = 2; width <= 64; width <<= 1) {
        for (int step = width >> 1; step > 0; step >>= 1) {
            for (int j = 0; j < 2; ++j) {
                const int i = lane + j * 32;
                const int p = i ^ step;
                if (p > i) {
                    const float a = sort_s[i]; const int ai = sort_i[i];
                    const float b = sort_s[p]; const int bi = sort_i[p];
                    const bool up = (i & width) != 0; // global descending order
                    const bool a_before = a > b || (a == b && ai < bi);
                    const bool keep_i = up ? !a_before : a_before;
                    sort_s[i] = keep_i ? a : b; sort_i[i] = keep_i ? ai : bi;
                    sort_s[p] = keep_i ? b : a; sort_i[p] = keep_i ? bi : ai;
                }
            }
            __syncwarp();
        }
    }
    if (lane == 0) {
        const size_t out = static_cast<size_t>(query) * dst_stride + supertile * 8;
        for (int i = 0; i < 8; ++i) { scores[out+i] = sort_s[i]; ids[out+i] = __int_as_float(sort_i[i]); }
    }
}

__global__ void init_compact_topk(float* scores, float* ids, int* locks, int batch, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * k) {
        scores[i] = -INFINITY;
        ids[i] = static_cast<float>(0x7fffffff);
    }
    if (i < batch) locks[i] = 0;
}

__global__ void finish_compact_topk(float* scores, float* ids, int batch, int k,
                                    float* distances, idx_t* output_ids) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * k) return;
    distances[i] = -scores[i];
    output_ids[i] = static_cast<idx_t>(ids[i]);
}

template<int K, int Bits, int Delay, int Threads = 256, int Rows = 32, int Cols = 128,
         bool CacheQuery = false>
void launch_local(
    const __half* queries,
    const __half* centers,
    const float* query_norm,
    const float* center_norm,
    int batch,
    int nlist,
    int chunk,
    int pool_stride,
    float* pool_scores,
    float* pool_ids,
    cudaStream_t stream) {
    using Shared = SharedPlan<K, Bits, Delay, Threads, Rows, Cols>;
    auto kernel = fused_gemm_local_topk<K, Bits, Delay, Threads, false, Rows, Cols, CacheQuery>;
    kernel<<<dim3((batch + Rows - 1) / Rows, nlist / chunk),
             Threads, sizeof(Shared), stream>>>(
        queries, centers, query_norm, center_norm, batch, chunk, pool_stride,
        pool_scores, pool_ids, nullptr);
    CUDA_LAUNCH_CHECK("adaptive_threshold_topk fused GEMM local top-k");
}

void launch_compact_k8(const __half* queries, const __half* centers,
                       const float* query_norm, const float* center_norm,
                       int batch, int nlist, int chunk, float* scores, float* ids,
                       int* locks, float* distances, idx_t* output_ids,
                       cudaStream_t stream) {
    const int k = 8;
    init_compact_topk<<<(batch * k + 255) / 256, 256, 0, stream>>>(scores, ids, locks, batch, k);
    fused_gemm_local_topk<8, 8, 16, 256, true><<<dim3((batch + kRows - 1) / kRows, nlist / chunk),
                                                   256, sizeof(SharedPlan<8, 8, 16, 256>), stream>>>(
        queries, centers, query_norm, center_norm, batch, chunk, k, scores, ids, locks);
    finish_compact_topk<<<(batch * k + 255) / 256, 256, 0, stream>>>(
        scores, ids, batch, k, distances, output_ids);
}

int choose_chunk(int batch, int nlist, int k) {
    if (const char* forced = std::getenv("PARASELECT_FUSED_CHUNK")) {
        const int candidate = std::atoi(forced);
        if (candidate >= kCols && nlist % candidate == 0) return candidate;
    }
    // Auto fused-GEMM is only selected for large batches.  Keep the same
    // chunk for small tail batches so the prepared pool is bounded by a
    // useful launch shape instead of reserving a 2048-centroid pool per
    // query.  The explicit batch argument still selects the top1 large-grid
    // kernel below; it does not change the pool contract.
    int chunk = k == 1 && batch < 8192 ? 4096
                                      : (k <= 8 || batch < 1024 ? 8192 : 32768);
    // The grid is (ceil(batch/32), nlist/chunk). Eight column blocks are
    // enough for large query batches, but leave a 5090 badly under-filled for
    // batch <= 32. Keep more independent center chunks in flight for those
    // shapes; the pool remains bounded because it stores only K candidates per
    // chunk rather than the full distance matrix.
    const int max_column_blocks = std::max(1, nlist / kCols);
    const bool wide_small_k = k <= 8 && nlist >= 262144;
    const int target_column_blocks = batch <= 32
        ? std::min(wide_small_k ? 256 : 128, max_column_blocks)
        : std::min(64, max_column_blocks);
    while (nlist / chunk < target_column_blocks && chunk > kCols) chunk /= 2;
    while (chunk > nlist || nlist % chunk != 0) chunk /= 2;
    return chunk >= kCols ? chunk : 0;
}

}  // namespace

size_t adaptive_threshold_topk_gemm_l2_fp16_workspace_floats(
    int batch, int nlist, int k) {
    if (!adaptive_threshold_topk_gemm_l2_fp16_supported(batch, 128, nlist, k)) {
        throw std::invalid_argument("unsupported fused GEMM workspace shape");
    }
    // Size the pool for the actual execution batch. The previous code used
    // batch=1 unconditionally, reserving the smallest chunk layout even when
    // the runtime batch selected fewer, wider center chunks.
    const int chunk = choose_chunk(batch, nlist, k);
    const int length = (nlist / chunk) * k;
    const int stride = ((length + 255) / 256) * 256;
    return 2 * static_cast<size_t>(batch) * stride;
}

void prepare_adaptive_threshold_topk_gemm_l2_fp16() {
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<1, 0, 0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<1, 0, 0, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<8, 8, 16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<8, 8, 16, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<8, 8, 16, 256, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<8, 8, 16, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<8, 8, 16, 256, false, 16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<8, 8, 16, 256, 16>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<8, 8, 16, 256, false, 16, 256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<8, 8, 16, 256, 16, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<8, 8, 16, 256, false, 32, 256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<8, 8, 16, 256, 32, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<8, 8, 16, 256, false, 16, 256, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<8, 8, 16, 256, 16, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<16, 6, 20>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<16, 6, 20, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<16, 6, 20, 512>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<16, 6, 20, 512>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<32, 6, 16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<32, 6, 16, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<64, 6, 20>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<64, 6, 20, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<128, 6, 20>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedPlan<128, 6, 20, 256>)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_gemm_local_topk<512, 8, 16, 256, false, 16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedPlan<512, 8, 16, 256, 16>)));
}


bool adaptive_threshold_topk_gemm_l2_fp16_with_layout(
    const __half* queries,
    const __half* centers,
    const float* query_norm,
    const float* center_norm,
    int batch,
    int dim,
    int nlist,
    int k,
    float* workspace,
    size_t workspace_floats,
    float* distances,
    idx_t* ids,
    bool centers_tiled,
    cudaStream_t stream) {
    // Top1 ranks without the query norm; the final merge restores it when
    // supplied. Assignment-only callers may omit it and consume only IDs.
    if (!queries || !centers || (k != 1 && !query_norm) || !center_norm || !workspace ||
        !distances || !ids ||
        !adaptive_threshold_topk_gemm_l2_fp16_supported(batch, dim, nlist, k)) {
        return false;
    }
    // WMMA load_matrix_sync requires 256-bit alignment. Validate subviews too;
    // cudaMalloc alignment alone does not establish the public API contract.
    if ((reinterpret_cast<std::uintptr_t>(queries) & 31u) != 0 ||
        (reinterpret_cast<std::uintptr_t>(centers) & 31u) != 0) {
        return false;
    }
    const int chunk = choose_chunk(batch, nlist, k);
    if (chunk == 0) return false;
    if (k == 8 && std::getenv("PARASELECT_COMPACT_TOPK")) {
        if (workspace_floats < 2 * static_cast<size_t>(batch) * k) return false;
        const int compact_chunk = (nlist % 8192 == 0 && nlist >= 8192) ? 8192 : chunk;
        int* locks = nullptr;
        CUDA_CHECK(cudaMallocAsync(&locks, static_cast<size_t>(batch) * sizeof(int), stream));
        float* compact_scores = workspace;
        float* compact_ids = workspace + static_cast<size_t>(batch) * k;
        launch_compact_k8(queries, centers, query_norm, center_norm, batch, nlist, compact_chunk,
                          compact_scores, compact_ids, locks, distances, ids, stream);
        CUDA_CHECK(cudaFreeAsync(locks, stream));
        CUDA_LAUNCH_CHECK("adaptive compact Tensor Core top-k");
        return true;
    }
    const int pool_length = (nlist / chunk) * k;
    const int pool_stride = ((pool_length + 255) / 256) * 256;
    if (workspace_floats < 2 * static_cast<size_t>(batch) * pool_stride) return false;
    float* pool_scores = workspace;
    float* pool_ids = workspace + static_cast<size_t>(batch) * pool_stride;
    if (k == 1) {
        if (batch >= 8192) {
            if (centers_tiled) {
                fused_gemm_local_top1_128<true><<<dim3((batch + 127) / 128, nlist / chunk),
                                              256, 0, stream>>>(
                    queries, centers, query_norm, center_norm, batch, chunk, pool_stride,
                    pool_scores, pool_ids);
            } else {
                fused_gemm_local_top1_128<false><<<dim3((batch + 127) / 128, nlist / chunk),
                                               256, 0, stream>>>(
                    queries, centers, query_norm, center_norm, batch, chunk, pool_stride,
                    pool_scores, pool_ids);
            }
        } else {
            if (centers_tiled) {
                fused_gemm_local_top1<true><<<dim3((batch + 63) / 64, nlist / chunk),
                                              256, 0, stream>>>(
                    queries, centers, query_norm, center_norm, batch, chunk, pool_stride,
                    pool_scores, pool_ids);
            } else {
                fused_gemm_local_top1<false><<<dim3((batch + 63) / 64, nlist / chunk),
                                               256, 0, stream>>>(
                    queries, centers, query_norm, center_norm, batch, chunk, pool_stride,
                    pool_scores, pool_ids);
            }
        }
        CUDA_LAUNCH_CHECK("fused GEMM assignment MMA top1");
        finish_top1<<<(batch + 7) / 8, 256, 0, stream>>>(
            pool_scores, pool_ids, batch, pool_length, pool_stride, distances, ids, query_norm);
        CUDA_LAUNCH_CHECK("fused GEMM assignment top1 merge");
        return true;
    }
    if (k == 8) {
        const char* producer = std::getenv("PARASELECT_FUSED_PRODUCER");
        if (producer && std::strcmp(producer, "rows16_tile256") == 0) {
            launch_local<8, 8, 16, 256, 16, 256>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        } else if (producer && std::strcmp(producer, "rows32_tile256") == 0) {
            launch_local<8, 8, 16, 256, 32, 256>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        } else if (producer && std::strcmp(producer, "rows16_tile256_query_cache") == 0) {
            launch_local<8, 8, 16, 256, 16, 256, true>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        } else if ((producer && std::strcmp(producer, "rows16_tile128") == 0) ||
                   std::getenv("PARASELECT_MICROTILE_PRODUCER")) {
            launch_local<8, 8, 16, 256, 16>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        } else {
            launch_local<8, 8, 16>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        }
    } else if (k == 16) {
        if (batch == 32) {
            launch_local<16, 6, 20, 512>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        } else {
            launch_local<16, 6, 20>(queries, centers, query_norm, center_norm,
                batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
        }
    } else if (k == 32) {
        launch_local<32, 6, 16>(queries, centers, query_norm, center_norm,
            batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
    } else if (k == 64) {
        launch_local<64, 6, 20>(queries, centers, query_norm, center_norm,
            batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
    } else if (k == 512) {
        launch_local<512, 8, 16, 256, 16>(queries, centers, query_norm, center_norm,
            batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
    } else {
        launch_local<128, 6, 20>(queries, centers, query_norm, center_norm,
            batch, nlist, chunk, pool_stride, pool_scores, pool_ids, stream);
    }
    int final_pool_length = pool_length;
    int final_pool_stride = pool_stride;
    if (k == 8 && std::getenv("PARASELECT_SUPERTILE_MERGE")) {
        const int chunk_count = pool_length / k;
        const int super_count = (chunk_count + 7) / 8;
        final_pool_length = super_count * k;
        final_pool_stride = ((final_pool_length + 255) / 256) * 256;
        merge_k8_supertile_inplace<<<dim3(batch, super_count), 32, 0, stream>>>(
            pool_scores, pool_ids, batch, chunk_count, pool_stride, final_pool_stride);
        CUDA_LAUNCH_CHECK("fused GEMM supertile merge");
    }
    adaptive_threshold_topk_fp32(
        pool_scores, batch, final_pool_length, final_pool_stride, k, distances, ids, stream);
    finish_topk<<<(static_cast<size_t>(batch) * k + 255) / 256, 256, 0, stream>>>(
        distances, ids, pool_ids, static_cast<size_t>(batch) * k, k, final_pool_stride);
    CUDA_LAUNCH_CHECK("adaptive_threshold_topk fused GEMM finish");
    return true;
}

bool adaptive_threshold_topk_gemm_l2_fp16(
    const __half* queries,
    const __half* centers,
    const float* query_norm,
    const float* center_norm,
    int batch,
    int dim,
    int nlist,
    int k,
    float* workspace,
    size_t workspace_floats,
    float* distances,
    idx_t* ids,
    cudaStream_t stream) {
    return adaptive_threshold_topk_gemm_l2_fp16_with_layout(
        queries, centers, query_norm, center_norm, batch, dim, nlist, k,
        workspace, workspace_floats, distances, ids, false, stream);
}

}  // namespace paraselect::coarse
