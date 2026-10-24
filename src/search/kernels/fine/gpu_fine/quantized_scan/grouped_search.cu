#include "quantization/turboquant/turboquant_distance.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_grouped_impl.cuh"
#include "search/ivfrespq/stage/residual_pq_grouped_planner.h"
#include "search/ivfrespq/stage/residual_pq_workspace_budget.h"
#include "utils/check_cuda_error.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace paraselect::search {
namespace {

constexpr int kGroupedLocalK = 256;
constexpr int kGroupedGlobalK = 512;
constexpr int kDefaultGroupedBlockThreads = 256;
constexpr int kResidualPqK = 256;
constexpr int kResidualPqLutStride = 257;

static int grouped_block_threads_from_env() {
    static const int threads = [] {
        const char* value = std::getenv("PARASELECT_RPQ_GROUPED_BLOCK_THREADS");
        if (!value || !*value) return kDefaultGroupedBlockThreads;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (!end || *end != '\0' ||
            (parsed != 256 && parsed != 512 && parsed != 1024)) {
            throw std::invalid_argument(
                "PARASELECT_RPQ_GROUPED_BLOCK_THREADS must be 256, 512, or 1024");
        }
        return static_cast<int>(parsed);
    }();
    return threads;
}

static void reserve_bytes(void** ptr, size_t* capacity_bytes, size_t need_bytes) {
    if (need_bytes <= *capacity_bytes) return;
    if (*ptr) CUDA_CHECK(cudaFree(*ptr));
    CUDA_CHECK(cudaMalloc(ptr, need_bytes));
    *capacity_bytes = need_bytes;
}

template <typename T>
static void reserve_array(T** ptr, int* capacity, int need) {
    if (need <= *capacity) return;
    if (*ptr) CUDA_CHECK(cudaFree(*ptr));
    CUDA_CHECK(cudaMalloc(ptr, (size_t)need * sizeof(T)));
    *capacity = need;
}

static void reserve_merge_arrays(float** dist, int** idx, int* capacity, int need) {
    if (need <= *capacity) return;
    if (*dist) CUDA_CHECK(cudaFree(*dist));
    if (*idx) CUDA_CHECK(cudaFree(*idx));
    CUDA_CHECK(cudaMalloc(dist, (size_t)need * sizeof(float)));
    CUDA_CHECK(cudaMalloc(idx, (size_t)need * sizeof(int)));
    *capacity = need;
}

static void device_exclusive_scan_int(
    const int* in,
    int* out,
    int n,
    ResidualPQGroupedWorkspace* ws,
    cudaStream_t stream) {
    if (n <= 0) return;
    size_t need = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, need, in, out, n, stream);
    if (need > ws->scan_temp_bytes) {
        throw std::runtime_error("residual_pq_grouped: scan temp capacity exceeded");
    }
    cub::DeviceScan::ExclusiveSum(ws->d_scan_temp, ws->scan_temp_bytes, in, out, n, stream);
}

__device__ __forceinline__ void sort_pair(float& a, int& ia, float& b, int& ib, bool asc) {
    // Keep grouped ordering identical to the final cap-select key. Without the
    // row-id tie break, equal ADC values can exchange positions across merge
    // rounds at the threshold boundary.
    const bool sw = asc ? (a > b || (a == b && ia > ib))
                        : (a < b || (a == b && ia < ib));
    if (sw) {
        const float td = a;
        a = b;
        b = td;
        const int ti = ia;
        ia = ib;
        ib = ti;
    }
}

__global__ void pair_chunk_count_kernel(
    const idx_t* __restrict__ top_nprobe,
    const int* __restrict__ counts,
    int total_pairs,
    int vector_chunk_size,
    int* __restrict__ pair_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    const idx_t cid_idx = top_nprobe[idx];
    if (cid_idx < 0) {
        pair_counts[idx] = 0;
        return;
    }
    const int cid = static_cast<int>(cid_idx);
    const int cnt = counts[cid];
    pair_counts[idx] = (cnt + vector_chunk_size - 1) / vector_chunk_size;
}

__global__ void emit_chunk_desc_kernel(
    const idx_t* __restrict__ top_nprobe,
    const int* __restrict__ counts,
    const int* __restrict__ pair_offsets,
    int total_pairs,
    int nprobe,
    int vector_chunk_size,
    ResidualPQChunkDesc* __restrict__ desc) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    const idx_t cid_idx = top_nprobe[idx];
    if (cid_idx < 0) return;
    const int q = idx / nprobe;
    const int cid = static_cast<int>(cid_idx);
    const int cnt = counts[cid];
    const int out = pair_offsets[idx];
    for (int start = 0, k = 0; start < cnt; start += vector_chunk_size, ++k) {
        const int len = min(vector_chunk_size, cnt - start);
        desc[out + k] = ResidualPQChunkDesc{q, cid, start, len};
    }
}

__global__ void query_chunk_count_kernel(
    const int* __restrict__ pair_counts,
    int batch_n,
    int nprobe,
    int* __restrict__ query_counts) {
    const int q = blockIdx.x;
    if (q >= batch_n) return;
    int sum = 0;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        sum += pair_counts[(size_t)q * nprobe + p];
    }
    __shared__ int partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) query_counts[q] = partial[0];
}

__global__ void count_scanned_codes_kernel(
    const idx_t* __restrict__ top_nprobe,
    const int* __restrict__ counts,
    int batch_n,
    int nprobe,
    unsigned long long* __restrict__ total) {
    const int q = blockIdx.x;
    if (q >= batch_n) return;
    unsigned long long sum = 0;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        const idx_t cid = top_nprobe[(size_t)q * nprobe + p];
        if (cid >= 0) sum += static_cast<unsigned long long>(counts[cid]);
    }
    __shared__ unsigned long long partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

__global__ void group_count_kernel(
    const int* __restrict__ source_counts,
    int batch_n,
    int group_chunks,
    int* __restrict__ group_counts) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batch_n) return;
    const int c = source_counts[q];
    group_counts[q] = (c + group_chunks - 1) / group_chunks;
}

__global__ void emit_group_desc_kernel(
    const int* __restrict__ source_starts,
    const int* __restrict__ source_counts,
    const int* __restrict__ group_starts,
    int batch_n,
    int group_chunks,
    ResidualPQGroupDesc* __restrict__ gdesc) {
    const int q = blockIdx.x;
    if (q >= batch_n) return;
    const int count = source_counts[q];
    const int group_start = group_starts[q];
    const int source_start = source_starts[q];
    const int n_groups = (count + group_chunks - 1) / group_chunks;
    for (int j = threadIdx.x; j < n_groups; j += blockDim.x) {
        const int remain = count - j * group_chunks;
        const int len = min(group_chunks, remain);
        gdesc[group_start + j] = ResidualPQGroupDesc{q, source_start + j * group_chunks, len};
    }
}

template <int D_SUB, bool kResidual, bool TurboQuant = false>
__global__ void fused_first_group_merge_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const float* __restrict__ codebook,
    const uint8_t* __restrict__ codes,
    const long long* __restrict__ offsets,
    const ResidualPQChunkDesc* __restrict__ desc,
    const ResidualPQGroupDesc* __restrict__ gdesc,
    int n_groups,
    int pq_m,
    int dim_padded,
    int vector_chunk_size,
    int group_chunks,
    int global_k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx,
    SignResidualDeviceView tq = {}) {
    extern __shared__ unsigned char smem[];
    float* sd = reinterpret_cast<float*>(smem);
    int* si = reinterpret_cast<int*>(sd + group_chunks * kGroupedLocalK);
    float* lut = reinterpret_cast<float*>(si + group_chunks * kGroupedLocalK);

    const int gid = blockIdx.x;
    if (gid >= n_groups) return;
    const ResidualPQGroupDesc gd = gdesc[gid];

    for (int ch = 0; ch < gd.source_count; ++ch) {
        const ResidualPQChunkDesc cd = desc[gd.source_start + ch];
        if constexpr (TurboQuant) {
            turboquant::prepare_residual(tq, cd.query, cd.cluster, dim_padded, lut);
        } else {
        const float* qv = queries + (size_t)cd.query * dim_padded;
        const float* cv = kResidual ? centroids + (size_t)cd.cluster * dim_padded : nullptr;
        for (int e = threadIdx.x; e < pq_m * kResidualPqK; e += blockDim.x) {
            const int m = e / kResidualPqK;
            const int k = e - m * kResidualPqK;
            const float* qq = qv + (size_t)m * D_SUB;
            const float* cc = kResidual ? cv + (size_t)m * D_SUB : nullptr;
            const float* cb = codebook + ((size_t)m * kResidualPqK + k) * D_SUB;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D_SUB; ++d) {
                const float base = qq[d] - cb[d];
                const float r = kResidual ? base - cc[d] : base;
                s = fmaf(r, r, s);
            }
            lut[m * kResidualPqLutStride + k] = s;
        }
        }
        __syncthreads();

        const long long base = offsets[cd.cluster] + cd.start;
        for (int lane = threadIdx.x; lane < kGroupedLocalK; lane += blockDim.x) {
            float best = FLT_MAX;
            int best_idx = -1;
            for (int item = lane; item < vector_chunk_size; item += kGroupedLocalK) {
                if (item < cd.length) {
                    const int idx = static_cast<int>(base + item);
                    float dist = 0.0f;
                    if constexpr (TurboQuant) {
                        dist = turboquant::distance(tq, lut, idx);
                    } else {
                    const uint8_t* code = codes + (size_t)idx * pq_m;
                    for (int m = 0; m < pq_m; ++m) {
                        dist += lut[m * kResidualPqLutStride + code[m]];
                    }
                    }
                    if (dist < best) {
                        best = dist;
                        best_idx = idx;
                    }
                }
            }
            const int out = ch * kGroupedLocalK + lane;
            sd[out] = best;
            si[out] = best_idx;
        }
        __syncthreads();
    }

    for (int x = threadIdx.x + gd.source_count * kGroupedLocalK;
         x < group_chunks * kGroupedLocalK;
         x += blockDim.x) {
        sd[x] = FLT_MAX;
        si[x] = -1;
    }
    __syncthreads();

    const int merge_items = group_chunks * kGroupedLocalK;
    for (int k = 2; k <= merge_items; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = threadIdx.x; idx < merge_items; idx += blockDim.x) {
                const int ixj = idx ^ j;
                if (ixj > idx) {
                    const bool asc = ((idx & k) == 0);
                    float a = sd[idx], b = sd[ixj];
                    int ia = si[idx], ib = si[ixj];
                    sort_pair(a, ia, b, ib, asc);
                    sd[idx] = a;
                    si[idx] = ia;
                    sd[ixj] = b;
                    si[ixj] = ib;
                }
            }
            __syncthreads();
        }
    }

    const int valid = gd.source_count * kGroupedLocalK;
    for (int k = threadIdx.x; k < global_k; k += blockDim.x) {
        const size_t out = (size_t)gid * (size_t)global_k + k;
        if (k < valid) {
            out_dist[out] = sd[k];
            out_idx[out] = si[k];
        } else {
            out_dist[out] = FLT_MAX;
            out_idx[out] = -1;
        }
    }
}

template <int D_SUB, bool kResidual>
__global__ void fused_first_group_merge_direct_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const float* __restrict__ codebook,
    const uint8_t* __restrict__ codes,
    const long long* __restrict__ offsets,
    const ResidualPQChunkDesc* __restrict__ desc,
    const ResidualPQGroupDesc* __restrict__ gdesc,
    int n_groups,
    int pq_m,
    int dim_padded,
    int vector_chunk_size,
    int group_chunks,
    int global_k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx) {
    extern __shared__ unsigned char smem[];
    float* sd = reinterpret_cast<float*>(smem);
    int* si = reinterpret_cast<int*>(sd + group_chunks * kGroupedLocalK);

    const int gid = blockIdx.x;
    if (gid >= n_groups) return;
    const ResidualPQGroupDesc gd = gdesc[gid];

    for (int ch = 0; ch < gd.source_count; ++ch) {
        const ResidualPQChunkDesc cd = desc[gd.source_start + ch];
        const float* qv = queries + (size_t)cd.query * dim_padded;
        const float* cv = kResidual ? centroids + (size_t)cd.cluster * dim_padded : nullptr;
        const long long base = offsets[cd.cluster] + cd.start;
        for (int lane = threadIdx.x; lane < kGroupedLocalK; lane += blockDim.x) {
            float best = FLT_MAX;
            int best_idx = -1;
            for (int item = lane; item < vector_chunk_size; item += kGroupedLocalK) {
                if (item < cd.length) {
                    const int idx = static_cast<int>(base + item);
                    const uint8_t* code = codes + (size_t)idx * pq_m;
                    float dist = 0.0f;
                    for (int m = 0; m < pq_m; ++m) {
                        const int k = code[m];
                        const float* qq = qv + (size_t)m * D_SUB;
                        const float* cc = kResidual ? cv + (size_t)m * D_SUB : nullptr;
                        const float* cb = codebook + ((size_t)m * kResidualPqK + k) * D_SUB;
#pragma unroll
                        for (int d = 0; d < D_SUB; ++d) {
                            const float base = qq[d] - cb[d];
                            const float r = kResidual ? base - cc[d] : base;
                            dist = fmaf(r, r, dist);
                        }
                    }
                    if (dist < best) {
                        best = dist;
                        best_idx = idx;
                    }
                }
            }
            const int out = ch * kGroupedLocalK + lane;
            sd[out] = best;
            si[out] = best_idx;
        }
        __syncthreads();
    }

    for (int x = threadIdx.x + gd.source_count * kGroupedLocalK;
         x < group_chunks * kGroupedLocalK;
         x += blockDim.x) {
        sd[x] = FLT_MAX;
        si[x] = -1;
    }
    __syncthreads();

    const int merge_items = group_chunks * kGroupedLocalK;
    for (int k = 2; k <= merge_items; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = threadIdx.x; idx < merge_items; idx += blockDim.x) {
                const int ixj = idx ^ j;
                if (ixj > idx) {
                    const bool asc = ((idx & k) == 0);
                    float a = sd[idx], b = sd[ixj];
                    int ia = si[idx], ib = si[ixj];
                    sort_pair(a, ia, b, ib, asc);
                    sd[idx] = a;
                    si[idx] = ia;
                    sd[ixj] = b;
                    si[ixj] = ib;
                }
            }
            __syncthreads();
        }
    }

    const int valid = gd.source_count * kGroupedLocalK;
    for (int k = threadIdx.x; k < global_k; k += blockDim.x) {
        const size_t out = (size_t)gid * (size_t)global_k + k;
        if (k < valid) {
            out_dist[out] = sd[k];
            out_idx[out] = si[k];
        } else {
            out_dist[out] = FLT_MAX;
            out_idx[out] = -1;
        }
    }
}

__global__ void group_merge_kernel(
    const ResidualPQGroupDesc* __restrict__ gdesc,
    int n_groups,
    const float* __restrict__ in_dist,
    const int* __restrict__ in_idx,
    int group_chunks,
    int global_k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx) {
    extern __shared__ unsigned char smem[];
    float* sd = reinterpret_cast<float*>(smem);
    int* si = reinterpret_cast<int*>(sd + group_chunks * global_k);

    const int gid = blockIdx.x;
    if (gid >= n_groups) return;
    const ResidualPQGroupDesc gd = gdesc[gid];
    const int candidates = gd.source_count * global_k;
    const int merge_items = group_chunks * global_k;

    for (int x = threadIdx.x; x < merge_items; x += blockDim.x) {
        if (x < candidates) {
            const int source = x / global_k;
            const int slot = x - source * global_k;
            const size_t pos = ((size_t)gd.source_start + source) * (size_t)global_k + slot;
            sd[x] = in_dist[pos];
            si[x] = in_idx[pos];
        } else {
            sd[x] = FLT_MAX;
            si[x] = -1;
        }
    }
    __syncthreads();

    for (int k = 2; k <= merge_items; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = threadIdx.x; idx < merge_items; idx += blockDim.x) {
                const int ixj = idx ^ j;
                if (ixj > idx) {
                    const bool asc = ((idx & k) == 0);
                    float a = sd[idx], b = sd[ixj];
                    int ia = si[idx], ib = si[ixj];
                    sort_pair(a, ia, b, ib, asc);
                    sd[idx] = a;
                    si[idx] = ia;
                    sd[ixj] = b;
                    si[ixj] = ib;
                }
            }
            __syncthreads();
        }
    }

    for (int k = threadIdx.x; k < global_k; k += blockDim.x) {
        const size_t out = (size_t)gid * (size_t)global_k + k;
        out_dist[out] = sd[k];
        out_idx[out] = si[k];
    }
}

__global__ void copy_final_groups_kernel(
    const int* __restrict__ query_group_starts,
    const float* __restrict__ group_dist,
    const int* __restrict__ group_idx,
    int batch_n,
    int global_k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx) {
    const int q = blockIdx.x;
    if (q >= batch_n) return;
    const int g = query_group_starts[q];
    for (int k = threadIdx.x; k < global_k; k += blockDim.x) {
        const size_t src = (size_t)g * (size_t)global_k + k;
        const size_t dst = (size_t)q * (size_t)global_k + k;
        out_dist[dst] = group_dist[src];
        out_idx[dst] = group_idx[src];
    }
}

__global__ void fill_empty_candidates_kernel(
    int batch_n,
    int global_k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx) {
    const int q = blockIdx.x;
    if (q >= batch_n) return;
    for (int k = threadIdx.x; k < global_k; k += blockDim.x) {
        const size_t pos = (size_t)q * (size_t)global_k + k;
        out_dist[pos] = FLT_MAX;
        out_idx[pos] = -1;
    }
}

template <int D_SUB>
static void launch_fused_first(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const float* d_codebook,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const ResidualPQChunkDesc* d_chunk_desc,
    const ResidualPQGroupDesc* d_group_desc,
    int n_groups,
    int pq_m,
    int dim_padded,
    int vector_chunk_size,
    int group_chunks,
    int global_k,
    float* d_out_dist,
    int* d_out_idx,
    cudaStream_t stream) {
    const size_t shmem =
        (size_t)group_chunks * kGroupedLocalK * (sizeof(float) + sizeof(int)) +
        (size_t)pq_m * kResidualPqLutStride * sizeof(float);
    const int block_threads = grouped_block_threads_from_env();
    CUDA_CHECK(cudaFuncSetAttribute(
        distance_mode == PQAdcDistanceMode::Residual
            ? fused_first_group_merge_kernel<D_SUB, true>
            : fused_first_group_merge_kernel<D_SUB, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));
    if (distance_mode == PQAdcDistanceMode::Residual) {
        fused_first_group_merge_kernel<D_SUB, true>
            <<<n_groups, block_threads, shmem, stream>>>(
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                d_chunk_desc, d_group_desc, n_groups, pq_m, dim_padded, vector_chunk_size,
                group_chunks, global_k, d_out_dist, d_out_idx);
    } else {
        fused_first_group_merge_kernel<D_SUB, false>
            <<<n_groups, block_threads, shmem, stream>>>(
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                d_chunk_desc, d_group_desc, n_groups, pq_m, dim_padded, vector_chunk_size,
                group_chunks, global_k, d_out_dist, d_out_idx);
    }
}

template <int D_SUB>
static void launch_fused_first_direct(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const float* d_codebook,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const ResidualPQChunkDesc* d_chunk_desc,
    const ResidualPQGroupDesc* d_group_desc,
    int n_groups,
    int pq_m,
    int dim_padded,
    int vector_chunk_size,
    int group_chunks,
    int global_k,
    float* d_out_dist,
    int* d_out_idx,
    cudaStream_t stream) {
    const size_t shmem =
        (size_t)group_chunks * kGroupedLocalK * (sizeof(float) + sizeof(int));
    const int block_threads = grouped_block_threads_from_env();
    CUDA_CHECK(cudaFuncSetAttribute(
        distance_mode == PQAdcDistanceMode::Residual
            ? fused_first_group_merge_direct_kernel<D_SUB, true>
            : fused_first_group_merge_direct_kernel<D_SUB, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));
    if (distance_mode == PQAdcDistanceMode::Residual) {
        fused_first_group_merge_direct_kernel<D_SUB, true>
            <<<n_groups, block_threads, shmem, stream>>>(
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                d_chunk_desc, d_group_desc, n_groups, pq_m, dim_padded, vector_chunk_size,
                group_chunks, global_k, d_out_dist, d_out_idx);
    } else {
        fused_first_group_merge_direct_kernel<D_SUB, false>
            <<<n_groups, block_threads, shmem, stream>>>(
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                d_chunk_desc, d_group_desc, n_groups, pq_m, dim_padded, vector_chunk_size,
                group_chunks, global_k, d_out_dist, d_out_idx);
    }
}

static void launch_group_merge(
    const ResidualPQGroupDesc* d_group_desc,
    int n_groups,
    const float* d_in_dist,
    const int* d_in_idx,
    int group_chunks,
    int global_k,
    float* d_out_dist,
    int* d_out_idx,
    cudaStream_t stream) {
    const size_t shmem =
        (size_t)group_chunks * (size_t)global_k * (sizeof(float) + sizeof(int));
    const int block_threads = grouped_block_threads_from_env();
    CUDA_CHECK(cudaFuncSetAttribute(
        group_merge_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));
    group_merge_kernel<<<n_groups, block_threads, shmem, stream>>>(
        d_group_desc, n_groups, d_in_dist, d_in_idx, group_chunks, global_k, d_out_dist, d_out_idx);
}

static int scan_total_from_last(
    const int* d_counts,
    const int* d_offsets,
    int n,
    cudaStream_t stream) {
    if (n <= 0) return 0;
    int last_count = 0;
    int last_offset = 0;
    CUDA_CHECK(cudaMemcpyAsync(&last_count, d_counts + n - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_offset, d_offsets + n - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return last_count + last_offset;
}

static int choose_grouped_query_tile(
    const int* query_chunks,
    int n_query,
    int query_begin,
    const ResidualPQGpuSearchSpec& spec,
    const ResidualPQGroupedWorkspace* ws) {
    return choose_residual_pq_grouped_query_tile(
        query_chunks,
        n_query,
        query_begin,
        spec.group_chunks,
        spec.global_k,
        {ws->chunk_capacity,
         ws->group_capacity,
         ws->merge_a_item_capacity,
         ws->merge_b_item_capacity});
}

static void residual_pq_grouped_chunk_search_tile_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws);

} // namespace

void residual_pq_grouped_workspace_reserve(
    ResidualPQGroupedWorkspace* ws,
    int fine_query_tile,
    int nprobe,
    int max_chunks_per_probe,
    int group_chunks,
    int global_k,
    size_t reserved_after_grouped_bytes) {
    if (!ws || fine_query_tile <= 0 || nprobe <= 0 || max_chunks_per_probe <= 0 ||
        group_chunks <= 0 || global_k <= 0) {
        throw std::invalid_argument("residual_pq_grouped_workspace_reserve: invalid argument");
    }
    if (!ws->d_scanned_codes) CUDA_CHECK(cudaMalloc(&ws->d_scanned_codes, sizeof(unsigned long long)));
    if (!ws->h_scanned_codes) CUDA_CHECK(cudaMallocHost(&ws->h_scanned_codes, sizeof(unsigned long long)));
    const int pair_cap = fine_query_tile * nprobe;
    if (pair_cap > ws->pair_capacity) {
        if (ws->d_pair_counts) CUDA_CHECK(cudaFree(ws->d_pair_counts));
        if (ws->d_pair_offsets) CUDA_CHECK(cudaFree(ws->d_pair_offsets));
        CUDA_CHECK(cudaMalloc(&ws->d_pair_counts, (size_t)pair_cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws->d_pair_offsets, (size_t)pair_cap * sizeof(int)));
        ws->pair_capacity = pair_cap;
    }
    if (fine_query_tile > ws->query_capacity) {
        if (ws->d_query_counts) CUDA_CHECK(cudaFree(ws->d_query_counts));
        if (ws->d_query_starts) CUDA_CHECK(cudaFree(ws->d_query_starts));
        if (ws->d_next_counts) CUDA_CHECK(cudaFree(ws->d_next_counts));
        if (ws->d_next_starts) CUDA_CHECK(cudaFree(ws->d_next_starts));
        CUDA_CHECK(cudaMalloc(&ws->d_query_counts, (size_t)fine_query_tile * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws->d_query_starts, (size_t)fine_query_tile * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws->d_next_counts, (size_t)fine_query_tile * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws->d_next_starts, (size_t)fine_query_tile * sizeof(int)));
        ws->query_capacity = fine_query_tile;
    }
    if (fine_query_tile > ws->host_query_capacity) {
        if (ws->h_query_chunks) CUDA_CHECK(cudaFreeHost(ws->h_query_chunks));
        CUDA_CHECK(cudaMallocHost(&ws->h_query_chunks, (size_t)fine_query_tile * sizeof(int)));
        ws->host_query_capacity = fine_query_tile;
    }
    size_t pair_scan_temp = 0;
    size_t query_scan_temp = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, pair_scan_temp, ws->d_pair_counts, ws->d_pair_offsets, pair_cap);
    cub::DeviceScan::ExclusiveSum(
        nullptr, query_scan_temp, ws->d_query_counts, ws->d_query_starts, fine_query_tile);
    const size_t scan_temp = std::max(pair_scan_temp, query_scan_temp);
    reserve_bytes(&ws->d_scan_temp, &ws->scan_temp_bytes, scan_temp);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    const size_t safety_bytes = residual_pq_grouped_safety_bytes(
        free_bytes, reserved_after_grouped_bytes);
    const auto plan = plan_residual_pq_grouped_workspace(
        free_bytes,
        reserved_after_grouped_bytes,
        safety_bytes,
        {group_chunks,
         global_k,
         sizeof(ResidualPQChunkDesc),
         sizeof(ResidualPQGroupDesc)});
    if (plan.limits.group_capacity <= 0) {
        throw std::runtime_error(
            "residual_pq_grouped_workspace_reserve: insufficient GPU memory after reserved allocations"
            " free_bytes=" + std::to_string(free_bytes) +
            " reserved_bytes=" + std::to_string(reserved_after_grouped_bytes) +
            " safety_bytes=" + std::to_string(safety_bytes));
    }

    const int group_cap = plan.limits.group_capacity;
    const int chunk_cap = plan.limits.chunk_capacity;
    const int merge_item_cap = plan.limits.merge_a_item_capacity;
    reserve_array(&ws->d_chunk_desc, &ws->chunk_capacity, chunk_cap);
    reserve_array(&ws->d_group_desc, &ws->group_capacity, group_cap);
    reserve_merge_arrays(
        &ws->d_merge_a_dist,
        &ws->d_merge_a_idx,
        &ws->merge_a_item_capacity,
        merge_item_cap);
    reserve_merge_arrays(
        &ws->d_merge_b_dist,
        &ws->d_merge_b_idx,
        &ws->merge_b_item_capacity,
        merge_item_cap);

    (void)max_chunks_per_probe;
}

void residual_pq_grouped_workspace_release(ResidualPQGroupedWorkspace* ws) {
    if (!ws) return;
#define FREE_WS(p)         \
    do {                   \
        if ((p)) {         \
            cudaFree((p)); \
            (p) = nullptr; \
        }                  \
    } while (0)
    FREE_WS(ws->d_pair_counts);
    FREE_WS(ws->d_pair_offsets);
    FREE_WS(ws->d_query_counts);
    FREE_WS(ws->d_query_starts);
    FREE_WS(ws->d_next_counts);
    FREE_WS(ws->d_next_starts);
    FREE_WS(ws->d_chunk_desc);
    FREE_WS(ws->d_group_desc);
    FREE_WS(ws->d_merge_a_dist);
    FREE_WS(ws->d_merge_a_idx);
    FREE_WS(ws->d_merge_b_dist);
    FREE_WS(ws->d_merge_b_idx);
    FREE_WS(ws->d_scan_temp);
    FREE_WS(ws->d_scanned_codes);
#undef FREE_WS
    if (ws->h_query_chunks) {
        cudaFreeHost(ws->h_query_chunks);
        ws->h_query_chunks = nullptr;
    }
    if (ws->h_scanned_codes) {
        cudaFreeHost(ws->h_scanned_codes);
        ws->h_scanned_codes = nullptr;
    }
    ws->scanned_codes = 0;
    ws->scan_temp_bytes = 0;
    ws->pair_capacity = 0;
    ws->query_capacity = 0;
    ws->chunk_capacity = 0;
    ws->group_capacity = 0;
    ws->merge_a_item_capacity = 0;
    ws->merge_b_item_capacity = 0;
    ws->host_query_capacity = 0;
}

void pq_count_scanned_codes_async(
    const idx_t* d_top_nprobe,
    const int* d_cluster_counts,
    int n_query,
    int n_probes,
    cudaStream_t stream,
    unsigned long long* device_count, unsigned long long* host_count) {
    if (!d_top_nprobe || !d_cluster_counts || n_query <= 0 || n_probes <= 0 ||
        !device_count || !host_count) {
        throw std::invalid_argument("pq_count_scanned_codes_async: invalid argument");
    }
    CUDA_CHECK(cudaMemsetAsync(device_count, 0, sizeof(unsigned long long), stream));
    count_scanned_codes_kernel<<<n_query, 256, 0, stream>>>(
        d_top_nprobe, d_cluster_counts, n_query, n_probes, device_count);
    CUDA_LAUNCH_CHECK("pq_count_scanned_codes_kernel");
    CUDA_CHECK(cudaMemcpyAsync(
        host_count, device_count, sizeof(unsigned long long),
        cudaMemcpyDeviceToHost, stream));
}

void residual_pq_grouped_chunk_search_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws) {
    if (spec.global_k == 64) {
        residual_pq_grouped_chunk_search_top64_gpu(
            distance_mode, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim,
            n_total_clusters, n_probes, spec, d_global_candidate_idx,
            d_global_candidate_dist, stream, timing, ws);
        return;
    }
    if (spec.global_k == kGroupedGlobalK) {
        residual_pq_grouped_chunk_search_top512_gpu(
            distance_mode, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim,
            n_total_clusters, n_probes, spec, d_global_candidate_idx,
            d_global_candidate_dist, stream, timing, ws);
        return;
    }
    throw std::invalid_argument("residual_pq_grouped_chunk_search_gpu: unsupported global_k");
}

void residual_pq_grouped_chunk_search_impl_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws) {
    (void)n_total_clusters;
    if (n_query <= 0) return;
    if (!ws) {
        throw std::invalid_argument("residual_pq_grouped_chunk_search_gpu: missing workspace");
    }
    if (spec.local_k != kGroupedLocalK ||
        (spec.global_k != 64 && spec.global_k != kGroupedGlobalK) ||
        spec.vector_chunk_size <= 0 || spec.group_chunks <= 0) {
        throw std::invalid_argument("residual_pq_grouped_chunk_search_gpu: unsupported spec");
    }
    if (n_query > ws->query_capacity || n_query * n_probes > ws->pair_capacity ||
        n_query > ws->host_query_capacity || !ws->h_query_chunks) {
        throw std::invalid_argument("residual_pq_grouped_chunk_search_gpu: workspace capacity too small");
    }

    const int total_pairs = n_query * n_probes;
    const int threads = 256;
    const int pair_blocks = (total_pairs + threads - 1) / threads;
    pair_chunk_count_kernel<<<pair_blocks, threads, 0, stream>>>(
        d_top_nprobe, d_cluster_counts, total_pairs, spec.vector_chunk_size, ws->d_pair_counts);
    CUDA_LAUNCH_CHECK("residual_pq_pair_chunk_count_kernel");
    query_chunk_count_kernel<<<n_query, threads, 0, stream>>>(
        ws->d_pair_counts, n_query, n_probes, ws->d_query_counts);
    CUDA_LAUNCH_CHECK("residual_pq_query_chunk_count_kernel");
    pq_count_scanned_codes_async(
        d_top_nprobe, d_cluster_counts, n_query, n_probes, stream, ws->d_scanned_codes, ws->h_scanned_codes);

    CUDA_CHECK(cudaMemcpyAsync(
        ws->h_query_chunks,
        ws->d_query_counts,
        (size_t)n_query * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ws->scanned_codes = *ws->h_scanned_codes;

    if (timing) *timing = ResidualPQGpuDebugTiming{};
    for (int q0 = 0; q0 < n_query;) {
        const int qn = choose_grouped_query_tile(
            ws->h_query_chunks, n_query, q0, spec, ws);
        ResidualPQGpuDebugTiming tile_timing{};
        ResidualPQGpuDebugTiming* tile_timing_ptr = timing ? &tile_timing : nullptr;
        residual_pq_grouped_chunk_search_tile_gpu(
            distance_mode,
            d_queries + (size_t)q0 * dim,
            d_centroids,
            d_top_nprobe + (size_t)q0 * n_probes,
            d_pq_codes,
            d_cluster_offsets,
            d_cluster_counts,
            d_codebook,
            qn,
            dim,
            n_probes,
            spec,
            d_global_candidate_idx + (size_t)q0 * spec.global_k,
            d_global_candidate_dist + (size_t)q0 * spec.global_k,
            stream,
            tile_timing_ptr,
            ws);
        if (timing) {
            timing->lut_build_ms += tile_timing.lut_build_ms;
            timing->adc_scan_ms += tile_timing.adc_scan_ms;
            timing->global_reduce_ms += tile_timing.global_reduce_ms;
            timing->probe_topk_ms += tile_timing.probe_topk_ms;
            timing->desc_ms += tile_timing.desc_ms;
            timing->group_merge_ms += tile_timing.group_merge_ms;
            timing->final_merge_ms += tile_timing.final_merge_ms;
            timing->candidate_desc_ms += tile_timing.candidate_desc_ms;
            timing->pq_scan_ms += tile_timing.pq_scan_ms;
            timing->candidate_reduce_ms += tile_timing.candidate_reduce_ms;
            timing->final_copy_ms += tile_timing.final_copy_ms;
        }
        q0 += qn;
    }
}

namespace {

static void residual_pq_grouped_chunk_search_tile_gpu(
    PQAdcDistanceMode distance_mode,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_probes,
    const ResidualPQGpuSearchSpec& spec,
    int* d_global_candidate_idx,
    float* d_global_candidate_dist,
    cudaStream_t stream,
    ResidualPQGpuDebugTiming* timing,
    ResidualPQGroupedWorkspace* ws) {
    cudaEvent_t desc_start = nullptr, desc_stop = nullptr, chunk_stop = nullptr;
    cudaEvent_t group_stop = nullptr, final_stop = nullptr;
    if (timing) {
        CUDA_CHECK(cudaEventCreate(&desc_start));
        CUDA_CHECK(cudaEventCreate(&desc_stop));
        CUDA_CHECK(cudaEventCreate(&chunk_stop));
        CUDA_CHECK(cudaEventCreate(&group_stop));
        CUDA_CHECK(cudaEventCreate(&final_stop));
        CUDA_CHECK(cudaEventRecord(desc_start, stream));
    }

    const int total_pairs = n_query * n_probes;
    const int threads = 256;
    const int pair_blocks = (total_pairs + threads - 1) / threads;
    pair_chunk_count_kernel<<<pair_blocks, threads, 0, stream>>>(
        d_top_nprobe, d_cluster_counts, total_pairs, spec.vector_chunk_size, ws->d_pair_counts);
    CUDA_LAUNCH_CHECK("residual_pq_pair_chunk_count_kernel");
    device_exclusive_scan_int(ws->d_pair_counts, ws->d_pair_offsets, total_pairs, ws, stream);
    const int n_chunks = scan_total_from_last(ws->d_pair_counts, ws->d_pair_offsets, total_pairs, stream);
    if (n_chunks > ws->chunk_capacity) {
        throw std::runtime_error("residual_pq_grouped: chunk capacity exceeded");
    }
    emit_chunk_desc_kernel<<<pair_blocks, threads, 0, stream>>>(
        d_top_nprobe, d_cluster_counts, ws->d_pair_offsets, total_pairs, n_probes,
        spec.vector_chunk_size, ws->d_chunk_desc);
    CUDA_LAUNCH_CHECK("residual_pq_emit_chunk_desc_kernel");
    query_chunk_count_kernel<<<n_query, threads, 0, stream>>>(
        ws->d_pair_counts, n_query, n_probes, ws->d_query_counts);
    CUDA_LAUNCH_CHECK("residual_pq_query_chunk_count_kernel");
    device_exclusive_scan_int(ws->d_query_counts, ws->d_query_starts, n_query, ws, stream);
    if (timing) CUDA_CHECK(cudaEventRecord(desc_stop, stream));

    group_count_kernel<<<(n_query + threads - 1) / threads, threads, 0, stream>>>(
        ws->d_query_counts, n_query, spec.group_chunks, ws->d_next_counts);
    CUDA_LAUNCH_CHECK("residual_pq_first_group_count_kernel");
    device_exclusive_scan_int(ws->d_next_counts, ws->d_next_starts, n_query, ws, stream);
    int n_groups = scan_total_from_last(ws->d_next_counts, ws->d_next_starts, n_query, stream);
    if (n_groups > ws->group_capacity) {
        throw std::runtime_error("residual_pq_grouped: group capacity exceeded");
    }
    const long long first_merge_items = (long long)n_groups * (long long)spec.global_k;
    if (first_merge_items > ws->merge_a_item_capacity) {
        throw std::runtime_error("residual_pq_grouped: merge A capacity exceeded");
    }
    if (n_groups <= 0) {
        fill_empty_candidates_kernel<<<n_query, 256, 0, stream>>>(
            n_query, spec.global_k, d_global_candidate_dist, d_global_candidate_idx);
        CUDA_LAUNCH_CHECK("residual_pq_fill_empty_candidates_kernel");
        if (timing) {
            CUDA_CHECK(cudaEventRecord(final_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(final_stop));
            CUDA_CHECK(cudaEventElapsedTime(&timing->desc_ms, desc_start, final_stop));
            timing->candidate_desc_ms = timing->desc_ms;
            timing->pq_scan_ms = 0.0f;
            timing->candidate_reduce_ms = 0.0f;
            timing->final_copy_ms = 0.0f;
            CUDA_CHECK(cudaEventDestroy(desc_start));
            CUDA_CHECK(cudaEventDestroy(desc_stop));
            CUDA_CHECK(cudaEventDestroy(chunk_stop));
            CUDA_CHECK(cudaEventDestroy(group_stop));
            CUDA_CHECK(cudaEventDestroy(final_stop));
        }
        return;
    }
    emit_group_desc_kernel<<<n_query, 128, 0, stream>>>(
        ws->d_query_starts, ws->d_query_counts, ws->d_next_starts, n_query,
        spec.group_chunks, ws->d_group_desc);
    CUDA_LAUNCH_CHECK("residual_pq_emit_first_group_desc_kernel");

    const bool use_direct_codebook =
        spec.scan_mode == ResidualPQGpuScanMode::DirectCodebook;
    if (spec.sign_codec.codes) {
        const size_t shared = (size_t)spec.group_chunks * kGroupedLocalK * 8 +
                              (dim + 1) * sizeof(float);
        fused_first_group_merge_kernel<8, true, true>
            <<<n_groups, grouped_block_threads_from_env(), shared, stream>>>(
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k,
                ws->d_merge_a_dist, ws->d_merge_a_idx, spec.sign_codec);
    } else if (spec.pq_d_sub_padded == 2) {
        if (use_direct_codebook) {
            launch_fused_first_direct<2>(
                distance_mode,
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k, ws->d_merge_a_dist,
                ws->d_merge_a_idx, stream);
        } else {
            launch_fused_first<2>(
                distance_mode,
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k, ws->d_merge_a_dist,
                ws->d_merge_a_idx, stream);
        }
    } else if (spec.pq_d_sub_padded == 4) {
        if (use_direct_codebook) {
            launch_fused_first_direct<4>(
                distance_mode,
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k, ws->d_merge_a_dist,
                ws->d_merge_a_idx, stream);
        } else {
            launch_fused_first<4>(
                distance_mode,
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k, ws->d_merge_a_dist,
                ws->d_merge_a_idx, stream);
        }
    } else {
        if (use_direct_codebook) {
            launch_fused_first_direct<8>(
                distance_mode,
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k, ws->d_merge_a_dist,
                ws->d_merge_a_idx, stream);
        } else {
            launch_fused_first<8>(
                distance_mode,
                d_queries, d_centroids, d_codebook, d_pq_codes, d_cluster_offsets,
                ws->d_chunk_desc, ws->d_group_desc, n_groups, spec.pq_m, dim,
                spec.vector_chunk_size, spec.group_chunks, spec.global_k, ws->d_merge_a_dist,
                ws->d_merge_a_idx, stream);
        }
    }
    CUDA_LAUNCH_CHECK("residual_pq_fused_first_group_merge_kernel");
    if (timing) CUDA_CHECK(cudaEventRecord(chunk_stop, stream));

    float* cur_dist = ws->d_merge_a_dist;
    int* cur_idx = ws->d_merge_a_idx;
    int* cur_counts = ws->d_next_counts;
    int* cur_starts = ws->d_next_starts;
    int cur_groups = n_groups;
    float* next_dist = nullptr;
    int* next_idx = nullptr;
    int* next_counts = ws->d_query_counts;
    int* next_starts = ws->d_query_starts;

    while (true) {
        group_count_kernel<<<(n_query + threads - 1) / threads, threads, 0, stream>>>(
            cur_counts, n_query, spec.group_chunks, next_counts);
        CUDA_LAUNCH_CHECK("residual_pq_group_count_kernel");
        device_exclusive_scan_int(next_counts, next_starts, n_query, ws, stream);
        const int round_groups = scan_total_from_last(next_counts, next_starts, n_query, stream);
        if (round_groups == cur_groups) break;
        if (round_groups > ws->group_capacity) {
            throw std::runtime_error("residual_pq_grouped: group capacity exceeded");
        }
        emit_group_desc_kernel<<<n_query, 128, 0, stream>>>(
            cur_starts, cur_counts, next_starts, n_query, spec.group_chunks, ws->d_group_desc);
        CUDA_LAUNCH_CHECK("residual_pq_emit_group_desc_kernel");
        const long long round_merge_items =
            (long long)round_groups * (long long)spec.global_k;
        if (cur_dist == ws->d_merge_a_dist) {
            if (round_merge_items > ws->merge_b_item_capacity) {
                throw std::runtime_error("residual_pq_grouped: merge B capacity exceeded");
            }
            next_dist = ws->d_merge_b_dist;
            next_idx = ws->d_merge_b_idx;
        } else {
            if (round_merge_items > ws->merge_a_item_capacity) {
                throw std::runtime_error("residual_pq_grouped: merge A capacity exceeded");
            }
            next_dist = ws->d_merge_a_dist;
            next_idx = ws->d_merge_a_idx;
        }
        launch_group_merge(
            ws->d_group_desc, round_groups, cur_dist, cur_idx, spec.group_chunks,
            spec.global_k, next_dist, next_idx, stream);
        CUDA_LAUNCH_CHECK("residual_pq_group_merge_kernel");
        std::swap(cur_dist, next_dist);
        std::swap(cur_idx, next_idx);
        std::swap(cur_counts, next_counts);
        std::swap(cur_starts, next_starts);
        cur_groups = round_groups;
    }
    if (timing) CUDA_CHECK(cudaEventRecord(group_stop, stream));

    copy_final_groups_kernel<<<n_query, 256, 0, stream>>>(
        cur_starts, cur_dist, cur_idx, n_query, spec.global_k,
        d_global_candidate_dist, d_global_candidate_idx);
    CUDA_LAUNCH_CHECK("residual_pq_copy_final_groups_kernel");
    if (timing) {
        CUDA_CHECK(cudaEventRecord(final_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(final_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->desc_ms, desc_start, desc_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->probe_topk_ms, desc_stop, chunk_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->group_merge_ms, chunk_stop, group_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timing->final_merge_ms, group_stop, final_stop));
        timing->global_reduce_ms = timing->group_merge_ms + timing->final_merge_ms;
        timing->candidate_desc_ms = timing->desc_ms;
        timing->pq_scan_ms = timing->probe_topk_ms;
        timing->candidate_reduce_ms = timing->group_merge_ms;
        timing->final_copy_ms = timing->final_merge_ms;
        CUDA_CHECK(cudaEventDestroy(desc_start));
        CUDA_CHECK(cudaEventDestroy(desc_stop));
        CUDA_CHECK(cudaEventDestroy(chunk_stop));
        CUDA_CHECK(cudaEventDestroy(group_stop));
        CUDA_CHECK(cudaEventDestroy(final_stop));
    }
}

} // namespace

} // namespace paraselect::search
