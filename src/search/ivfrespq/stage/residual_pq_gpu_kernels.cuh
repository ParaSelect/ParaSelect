#ifndef PARASELECT_RESIDUAL_PQ_GPU_KERNELS_CUH
#define PARASELECT_RESIDUAL_PQ_GPU_KERNELS_CUH

#include "search/ivfrespq/stage/residual_pq_gpu_detail.cuh"
#include "search/kernels/topk/warpsort/warpsort.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>

namespace paraselect::search::rpq_kernels {

using namespace rpq_detail;

__global__ void residual_pq_pad_query_kernel(
    const float* __restrict__ query,
    int n_query,
    int pq_m,
    int d_sub,
    int d_sub_padded,
    float* __restrict__ query_padded)
{
    const int padded_dim = pq_m * d_sub_padded;
    const int total = n_query * padded_dim;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < total; x += blockDim.x * gridDim.x) {
        const int row = x / padded_dim;
        const int rem = x - row * padded_dim;
        const int m = rem / d_sub_padded;
        const int d = rem - m * d_sub_padded;
        query_padded[x] = (d < d_sub)
            ? query[(size_t)row * (size_t)(pq_m * d_sub) + (size_t)m * d_sub + d]
            : 0.0f;
    }
}

inline void launch_residual_pq_pad_queries(
    const float* d_queries,
    int n_query,
    int pq_m,
    int d_sub,
    int d_sub_padded,
    float* d_queries_padded,
    cudaStream_t stream)
{
    const int total = n_query * pq_m * d_sub_padded;
    const int block = 256;
    const int grid = std::min(65535, (total + block - 1) / block);
    residual_pq_pad_query_kernel<<<grid, block, 0, stream>>>(
        d_queries, n_query, pq_m, d_sub, d_sub_padded, d_queries_padded);
}

template <int D_SUB>
__device__ __forceinline__ float adc_lut_runtime_m(
    const float* __restrict__ lut,
    const uint8_t* __restrict__ pq_codes,
    size_t row,
    int pq_m)
{
    const uint8_t* c = pq_codes + row * pq_m;
    float s = 0.0f;
    for (int m = 0; m < pq_m; ++m) {
        s += lut[m * kResidualPqLutStride + __ldg(c + m)];
    }
    return s;
}

template <int D_SUB>
__device__ __forceinline__ float direct_codebook_runtime_m(
    const float* __restrict__ qv_base,
    const float* __restrict__ cv_base,
    const float* __restrict__ codebook,
    const uint8_t* __restrict__ pq_codes,
    size_t row,
    int pq_m)
{
    const uint8_t* c = pq_codes + row * pq_m;
    float s = 0.0f;
    for (int m = 0; m < pq_m; ++m) {
        const int k = __ldg(c + m);
        const float* qv = qv_base + (size_t)m * D_SUB;
        const float* cv = cv_base + (size_t)m * D_SUB;
        const float* cb = codebook + ((size_t)m * kResidualPqK + k) * D_SUB;
#pragma unroll
        for (int d = 0; d < D_SUB; ++d) {
            const float r = qv[d] - cv[d] - cb[d];
            s = fmaf(r, r, s);
        }
    }
    return s;
}


template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K, int PROBES_PER_CHUNK>
__global__ void residual_pq_probe_chunk_runtime_m_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const uint8_t* __restrict__ pq_codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int chunk_count,
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist)
{
    extern __shared__ float shmem[];
    float* lut = shmem;
    float* probe_dist = lut + pq_m * kResidualPqLutStride;
    int* probe_idx = reinterpret_cast<int*>(probe_dist + PROBES_PER_CHUNK * LOCAL_K);

    const int q = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (q >= n_query || chunk >= chunk_count) return;

    __shared__ float warp_dist[WARPS_PER_BLOCK * LOCAL_K];
    __shared__ int warp_idx[WARPS_PER_BLOCK * LOCAL_K];

    for (int cp = 0; cp < PROBES_PER_CHUNK; ++cp) {
        const int p = chunk * PROBES_PER_CHUNK + cp;
        const idx_t cid_idx = (p < n_probes) ? top_nprobe[(size_t)q * n_probes + p] : idx_t{-1};

        if (cid_idx >= 0 && cid_idx < n_total_clusters) {
            const int cid = static_cast<int>(cid_idx);
            for (int x = tid; x < pq_m * kResidualPqK; x += blockDim.x) {
                const int m = x / kResidualPqK;
                const int k = x - m * kResidualPqK;
                const float* qv = queries + (size_t)q * dim_padded + (size_t)m * D_SUB;
                const float* cv = centroids + (size_t)cid * dim_padded + (size_t)m * D_SUB;
                const float* cb = codebook + ((size_t)m * kResidualPqK + k) * D_SUB;
                float s = 0.0f;
#pragma unroll
                for (int d = 0; d < D_SUB; ++d) {
                    const float r = qv[d] - cv[d] - cb[d];
                    s = fmaf(r, r, s);
                }
                lut[m * kResidualPqLutStride + k] = s;
            }
        }
        __syncthreads();

        float local_dist[THREAD_K];
        int local_idx[THREAD_K];
        init_topk_fixed<THREAD_K>(local_dist, local_idx);

        if (cid_idx >= 0 && cid_idx < n_total_clusters) {
            const int cid = static_cast<int>(cid_idx);
            const long long base = cluster_offsets[cid];
            const int count = cluster_counts[cid];
            for (int vi = tid; vi < count; vi += blockDim.x) {
                const float s = adc_lut_runtime_m<D_SUB>(lut, pq_codes, (size_t)base + (size_t)vi, pq_m);
                insert_topk(s, (int)(base + vi), THREAD_K, local_dist, local_idx);
            }
        }

        sort_thread_candidates_across_warp<THREAD_K>(local_dist, local_idx);
        write_warp_topk_from_sorted_thread_candidates<LOCAL_K, THREAD_K>(
            local_dist, local_idx, warp_dist, warp_idx);
        __syncwarp();
        __syncthreads();

        if (tid < 32) {
            constexpr int kBlockMergeItems =
                ((WARPS_PER_BLOCK * LOCAL_K + 31) / 32 <= 2) ? 2 : 4;
            write_block_topk_from_warp_candidates<LOCAL_K, WARPS_PER_BLOCK, kBlockMergeItems>(
                warp_dist, warp_idx, local_k, 0, PROBES_PER_CHUNK, cp, probe_dist, probe_idx);
        }
        __syncthreads();
    }

    if (tid < 32) {
        constexpr int kChunkItems = PROBES_PER_CHUNK * LOCAL_K;
        constexpr int kItemsPerLane = (kChunkItems + 31) / 32;
        float merge_dist[kItemsPerLane];
        int merge_idx[kItemsPerLane];
#pragma unroll
        for (int i = 0; i < kItemsPerLane; ++i) {
            const int src = tid + i * 32;
            if (src < kChunkItems) {
                merge_dist[i] = probe_dist[src];
                merge_idx[i] = probe_idx[src];
            } else {
                merge_dist[i] = FLT_MAX;
                merge_idx[i] = -1;
            }
        }
        paraselect::bitonic::Bitonic<kItemsPerLane>(true, 32).sort(merge_dist, merge_idx);
        if (tid < LOCAL_K) {
            const size_t out = ((size_t)q * chunk_count + chunk) * local_k + tid;
            candidate_dist[out] = merge_dist[0];
            candidate_idx[out] = merge_idx[0];
        }
    }
}



template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K, int PROBES_PER_CHUNK>
__global__ void residual_pq_probe_chunk_direct_codebook_runtime_m_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const uint8_t* __restrict__ pq_codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int chunk_count,
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist)
{
    extern __shared__ float shmem[];
    float* probe_dist = shmem;
    int* probe_idx = reinterpret_cast<int*>(probe_dist + PROBES_PER_CHUNK * LOCAL_K);

    const int q = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (q >= n_query || chunk >= chunk_count) return;

    __shared__ float warp_dist[WARPS_PER_BLOCK * LOCAL_K];
    __shared__ int warp_idx[WARPS_PER_BLOCK * LOCAL_K];

    const float* qv_base = queries + (size_t)q * dim_padded;
    for (int cp = 0; cp < PROBES_PER_CHUNK; ++cp) {
        const int p = chunk * PROBES_PER_CHUNK + cp;
        const idx_t cid_idx = (p < n_probes) ? top_nprobe[(size_t)q * n_probes + p] : idx_t{-1};

        float local_dist[THREAD_K];
        int local_idx[THREAD_K];
        init_topk_fixed<THREAD_K>(local_dist, local_idx);

        if (cid_idx >= 0 && cid_idx < n_total_clusters) {
            const int cid = static_cast<int>(cid_idx);
            const long long base = cluster_offsets[cid];
            const int count = cluster_counts[cid];
            const float* cv_base = centroids + (size_t)cid * dim_padded;
            for (int vi = tid; vi < count; vi += blockDim.x) {
                const float s = direct_codebook_runtime_m<D_SUB>(
                    qv_base, cv_base, codebook, pq_codes, (size_t)base + (size_t)vi, pq_m);
                insert_topk(s, (int)(base + vi), THREAD_K, local_dist, local_idx);
            }
        }

        sort_thread_candidates_across_warp<THREAD_K>(local_dist, local_idx);
        write_warp_topk_from_sorted_thread_candidates<LOCAL_K, THREAD_K>(
            local_dist, local_idx, warp_dist, warp_idx);
        __syncwarp();
        __syncthreads();

        if (tid < 32) {
            constexpr int kBlockMergeItems =
                ((WARPS_PER_BLOCK * LOCAL_K + 31) / 32 <= 2) ? 2 : 4;
            write_block_topk_from_warp_candidates<LOCAL_K, WARPS_PER_BLOCK, kBlockMergeItems>(
                warp_dist, warp_idx, local_k, 0, PROBES_PER_CHUNK, cp, probe_dist, probe_idx);
        }
        __syncthreads();
    }

    if (tid < 32) {
        constexpr int kChunkItems = PROBES_PER_CHUNK * LOCAL_K;
        constexpr int kItemsPerLane = (kChunkItems + 31) / 32;
        float merge_dist[kItemsPerLane];
        int merge_idx[kItemsPerLane];
#pragma unroll
        for (int i = 0; i < kItemsPerLane; ++i) {
            const int src = tid + i * 32;
            if (src < kChunkItems) {
                merge_dist[i] = probe_dist[src];
                merge_idx[i] = probe_idx[src];
            } else {
                merge_dist[i] = FLT_MAX;
                merge_idx[i] = -1;
            }
        }
        paraselect::bitonic::Bitonic<kItemsPerLane>(true, 32).sort(merge_dist, merge_idx);
        if (tid < LOCAL_K) {
            const size_t out = ((size_t)q * chunk_count + chunk) * local_k + tid;
            candidate_dist[out] = merge_dist[0];
            candidate_idx[out] = merge_idx[0];
        }
    }
}

template <int D_SUB, int LOCAL_K>
__global__ void residual_pq_vector_chunk_direct_codebook_runtime_m_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const uint8_t* __restrict__ pq_codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int vector_chunk_size,
    int max_chunks_per_probe,
    int source_count,
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist)
{
    __shared__ float sd[LOCAL_K];
    __shared__ int si[LOCAL_K];

    const int q = blockIdx.x;
    const int source = blockIdx.y;
    const int tid = threadIdx.x;
    if (q >= n_query || source >= source_count) return;

    float best = FLT_MAX;
    int best_idx = -1;

    const int p = source / max_chunks_per_probe;
    const int chunk = source - p * max_chunks_per_probe;
    if (p < n_probes) {
        const idx_t cid_idx = top_nprobe[(size_t)q * n_probes + p];
        if (cid_idx >= 0 && cid_idx < n_total_clusters) {
            const int cid = static_cast<int>(cid_idx);
            const int count = cluster_counts[cid];
            const int start = chunk * vector_chunk_size;
            if (start < count) {
                const int len = min(vector_chunk_size, count - start);
                const long long base = cluster_offsets[cid] + start;
                const float* qv_base = queries + (size_t)q * dim_padded;
                const float* cv_base = centroids + (size_t)cid * dim_padded;
                for (int item = tid; item < len; item += blockDim.x) {
                    const float s = direct_codebook_runtime_m<D_SUB>(
                        qv_base, cv_base, codebook, pq_codes, (size_t)base + (size_t)item, pq_m);
                    if (s < best) {
                        best = s;
                        best_idx = static_cast<int>(base + item);
                    }
                }
            }
        }
    }

    sd[tid] = best;
    si[tid] = best_idx;
    __syncthreads();
    for (int k = 2; k <= LOCAL_K; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = tid; idx < LOCAL_K; idx += blockDim.x) {
                const int ixj = idx ^ j;
                if (ixj > idx) {
                    const bool asc = ((idx & k) == 0);
                    float a = sd[idx], b = sd[ixj];
                    int ia = si[idx], ib = si[ixj];
                    if ((asc && b < a) || (!asc && a < b)) {
                        sd[idx] = b;
                        si[idx] = ib;
                        sd[ixj] = a;
                        si[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
    candidate_dist[((size_t)q * source_count + source) * LOCAL_K + tid] = sd[tid];
    candidate_idx[((size_t)q * source_count + source) * LOCAL_K + tid] = si[tid];
}

template <int D_SUB, int LOCAL_K>
__global__ void residual_pq_vector_chunk_lut_runtime_m_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const uint8_t* __restrict__ pq_codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int vector_chunk_size,
    int max_chunks_per_probe,
    int source_count,
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist)
{
    extern __shared__ float lut[];
    __shared__ float sd[LOCAL_K];
    __shared__ int si[LOCAL_K];

    const int q = blockIdx.x;
    const int source = blockIdx.y;
    const int tid = threadIdx.x;
    if (q >= n_query || source >= source_count) return;

    float best = FLT_MAX;
    int best_idx = -1;

    const int p = source / max_chunks_per_probe;
    const int chunk = source - p * max_chunks_per_probe;
    idx_t cid_idx = idx_t{-1};
    int cid = -1;
    if (p < n_probes) {
        cid_idx = top_nprobe[(size_t)q * n_probes + p];
    }
    if (cid_idx >= 0 && cid_idx < n_total_clusters) {
        cid = static_cast<int>(cid_idx);
        for (int x = tid; x < pq_m * kResidualPqK; x += blockDim.x) {
            const int m = x / kResidualPqK;
            const int k = x - m * kResidualPqK;
            const float* qv = queries + (size_t)q * dim_padded + (size_t)m * D_SUB;
            const float* cv = centroids + (size_t)cid * dim_padded + (size_t)m * D_SUB;
            const float* cb = codebook + ((size_t)m * kResidualPqK + k) * D_SUB;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D_SUB; ++d) {
                const float r = qv[d] - cv[d] - cb[d];
                s = fmaf(r, r, s);
            }
            lut[m * kResidualPqLutStride + k] = s;
        }
    }
    __syncthreads();

    if (cid >= 0) {
        const int count = cluster_counts[cid];
        const int start = chunk * vector_chunk_size;
        if (start < count) {
            const int len = min(vector_chunk_size, count - start);
            const long long base = cluster_offsets[cid] + start;
            for (int item = tid; item < len; item += blockDim.x) {
                const float s = adc_lut_runtime_m<D_SUB>(lut, pq_codes, (size_t)base + item, pq_m);
                if (s < best) {
                    best = s;
                    best_idx = static_cast<int>(base + item);
                }
            }
        }
    }

    sd[tid] = best;
    si[tid] = best_idx;
    __syncthreads();
    for (int k = 2; k <= LOCAL_K; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = tid; idx < LOCAL_K; idx += blockDim.x) {
                const int ixj = idx ^ j;
                if (ixj > idx) {
                    const bool asc = ((idx & k) == 0);
                    float a = sd[idx], b = sd[ixj];
                    int ia = si[idx], ib = si[ixj];
                    if ((asc && b < a) || (!asc && a < b)) {
                        sd[idx] = b;
                        si[idx] = ib;
                        sd[ixj] = a;
                        si[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
    candidate_dist[((size_t)q * source_count + source) * LOCAL_K + tid] = sd[tid];
    candidate_idx[((size_t)q * source_count + source) * LOCAL_K + tid] = si[tid];
}

template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K>
__global__ void residual_pq_probe_direct_codebook_runtime_m_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const uint8_t* __restrict__ pq_codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist)
{
    const int q = blockIdx.x;
    const int p = blockIdx.y;
    const int tid = threadIdx.x;
    if (q >= n_query || p >= n_probes) return;

    const idx_t cid_idx = top_nprobe[(size_t)q * n_probes + p];
    if (cid_idx < 0 || cid_idx >= n_total_clusters) return;
    const int cid = static_cast<int>(cid_idx);

    __shared__ float warp_dist[WARPS_PER_BLOCK * LOCAL_K];
    __shared__ int warp_idx[WARPS_PER_BLOCK * LOCAL_K];
    float local_dist[THREAD_K];
    int local_idx[THREAD_K];
    init_topk_fixed<THREAD_K>(local_dist, local_idx);

    const long long base = cluster_offsets[cid];
    const int count = cluster_counts[cid];
    const float* qv_base = queries + (size_t)q * dim_padded;
    const float* cv_base = centroids + (size_t)cid * dim_padded;
    for (int vi = tid; vi < count; vi += blockDim.x) {
        const float s = direct_codebook_runtime_m<D_SUB>(
            qv_base, cv_base, codebook, pq_codes, (size_t)base + (size_t)vi, pq_m);
        insert_topk(s, (int)(base + vi), THREAD_K, local_dist, local_idx);
    }

    sort_thread_candidates_across_warp<THREAD_K>(local_dist, local_idx);
    write_warp_topk_from_sorted_thread_candidates<LOCAL_K, THREAD_K>(
        local_dist, local_idx, warp_dist, warp_idx);
    __syncwarp();
    __syncthreads();

    if (tid < 32) {
        constexpr int kBlockMergeItems =
            ((WARPS_PER_BLOCK * LOCAL_K + 31) / 32 <= 2) ? 2 : 4;
        write_block_topk_from_warp_candidates<LOCAL_K, WARPS_PER_BLOCK, kBlockMergeItems>(
            warp_dist, warp_idx, local_k, q, n_probes, p, candidate_dist, candidate_idx);
    }
}

template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K>
__global__ void residual_pq_probe_topk_runtime_m_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ top_nprobe,
    const uint8_t* __restrict__ pq_codes,
    const long long* __restrict__ cluster_offsets,
    const int* __restrict__ cluster_counts,
    const float* __restrict__ codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int* __restrict__ candidate_idx,
    float* __restrict__ candidate_dist)
{
    extern __shared__ float lut[];
    const int q = blockIdx.x;
    const int p = blockIdx.y;
    const int tid = threadIdx.x;
    if (q >= n_query || p >= n_probes) return;

    const idx_t cid_idx = top_nprobe[(size_t)q * n_probes + p];
    if (cid_idx < 0 || cid_idx >= n_total_clusters) return;
    const int cid = static_cast<int>(cid_idx);

    for (int x = tid; x < pq_m * kResidualPqK; x += blockDim.x) {
        const int m = x / kResidualPqK;
        const int k = x - m * kResidualPqK;
        const float* qv = queries + (size_t)q * dim_padded + (size_t)m * D_SUB;
        const float* cv = centroids + (size_t)cid * dim_padded + (size_t)m * D_SUB;
        const float* cb = codebook + ((size_t)m * kResidualPqK + k) * D_SUB;
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < D_SUB; ++d) {
            const float r = qv[d] - cv[d] - cb[d];
            s = fmaf(r, r, s);
        }
        lut[m * kResidualPqLutStride + k] = s;
    }
    __syncthreads();

    __shared__ float warp_dist[WARPS_PER_BLOCK * LOCAL_K];
    __shared__ int warp_idx[WARPS_PER_BLOCK * LOCAL_K];
    float local_dist[THREAD_K];
    int local_idx[THREAD_K];
    init_topk_fixed<THREAD_K>(local_dist, local_idx);

    const long long base = cluster_offsets[cid];
    const int count = cluster_counts[cid];
    for (int vi = tid; vi < count; vi += blockDim.x) {
        const float s = adc_lut_runtime_m<D_SUB>(lut, pq_codes, (size_t)base + (size_t)vi, pq_m);
        insert_topk(s, (int)(base + vi), THREAD_K, local_dist, local_idx);
    }

    sort_thread_candidates_across_warp<THREAD_K>(local_dist, local_idx);
    write_warp_topk_from_sorted_thread_candidates<LOCAL_K, THREAD_K>(
        local_dist, local_idx, warp_dist, warp_idx);
    __syncwarp();
    __syncthreads();

    if (tid < 32) {
        constexpr int kBlockMergeItems =
            ((WARPS_PER_BLOCK * LOCAL_K + 31) / 32 <= 2) ? 2 : 4;
        write_block_topk_from_warp_candidates<LOCAL_K, WARPS_PER_BLOCK, kBlockMergeItems>(
            warp_dist, warp_idx, local_k, q, n_probes, p, candidate_dist, candidate_idx);
    }
}

template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K>
inline void launch_runtime_m_kernel(
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    constexpr int kBlockThreads = WARPS_PER_BLOCK * 32;
    CUDA_CHECK(cudaFuncSetAttribute(
        residual_pq_probe_topk_runtime_m_kernel<D_SUB, LOCAL_K, WARPS_PER_BLOCK, THREAD_K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));
    residual_pq_probe_topk_runtime_m_kernel<D_SUB, LOCAL_K, WARPS_PER_BLOCK, THREAD_K>
        <<<grid, kBlockThreads, shmem, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, n_query, dim_padded, n_total_clusters, n_probes,
            pq_m, local_k, d_probe_candidate_idx, d_probe_candidate_dist);
}

template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK>
inline void launch_runtime_m_by_thread_k(
    int thread_k,
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    if (thread_k <= 4) {
        launch_runtime_m_kernel<D_SUB, LOCAL_K, WARPS_PER_BLOCK, 4>(
            grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
            n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    } else if (thread_k <= 8) {
        launch_runtime_m_kernel<D_SUB, LOCAL_K, WARPS_PER_BLOCK, 8>(
            grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
            n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    } else {
        launch_runtime_m_kernel<D_SUB, LOCAL_K, WARPS_PER_BLOCK, 16>(
            grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
            n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    }
}

template <int D_SUB>
inline void launch_residual_pq_probe_runtime_m(
    int local_k,
    int thread_k,
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    if (local_k == 10) {
        launch_runtime_m_by_thread_k<D_SUB, 10, 8>(
            thread_k, grid, shmem, stream, d_queries, d_centroids, d_top_nprobe,
            d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook, n_query,
            dim_padded, n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    } else if (local_k == 16) {
        launch_runtime_m_by_thread_k<D_SUB, 16, 4>(
            thread_k, grid, shmem, stream, d_queries, d_centroids, d_top_nprobe,
            d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook, n_query,
            dim_padded, n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    } else {
        launch_runtime_m_by_thread_k<D_SUB, 32, 2>(
            thread_k, grid, shmem, stream, d_queries, d_centroids, d_top_nprobe,
            d_pq_codes, d_cluster_offsets, d_cluster_counts, d_codebook, n_query,
            dim_padded, n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    }
}



template <int D_SUB>
inline void launch_residual_pq_vector_chunk_direct_codebook_runtime_m(
    dim3 grid,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int vector_chunk_size,
    int max_chunks_per_probe,
    int source_count,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    residual_pq_vector_chunk_direct_codebook_runtime_m_kernel<D_SUB, 256>
        <<<grid, 256, 0, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, n_query, dim_padded, n_total_clusters,
            n_probes, pq_m, vector_chunk_size, max_chunks_per_probe, source_count,
            d_probe_candidate_idx, d_probe_candidate_dist);
}

template <int D_SUB>
inline void launch_residual_pq_vector_chunk_lut_runtime_m(
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int vector_chunk_size,
    int max_chunks_per_probe,
    int source_count,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    CUDA_CHECK(cudaFuncSetAttribute(
        residual_pq_vector_chunk_lut_runtime_m_kernel<D_SUB, 256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));
    residual_pq_vector_chunk_lut_runtime_m_kernel<D_SUB, 256>
        <<<grid, 256, shmem, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, n_query, dim_padded, n_total_clusters,
            n_probes, pq_m, vector_chunk_size, max_chunks_per_probe, source_count,
            d_probe_candidate_idx, d_probe_candidate_dist);
}

template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K>
inline void launch_direct_codebook_runtime_m_kernel(
    dim3 grid,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    constexpr int kBlockThreads = WARPS_PER_BLOCK * 32;
    residual_pq_probe_direct_codebook_runtime_m_kernel<
        D_SUB, LOCAL_K, WARPS_PER_BLOCK, THREAD_K>
        <<<grid, kBlockThreads, 0, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, n_query, dim_padded, n_total_clusters, n_probes,
            pq_m, local_k, d_probe_candidate_idx, d_probe_candidate_dist);
}

template <int D_SUB>
inline void launch_residual_pq_probe_direct_codebook_runtime_m(
    int local_k,
    int thread_k,
    dim3 grid,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    if (local_k == 10) {
        launch_direct_codebook_runtime_m_kernel<D_SUB, 10, 8, 4>(
            grid, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
            n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    } else if (local_k == 16) {
        launch_direct_codebook_runtime_m_kernel<D_SUB, 16, 4, 8>(
            grid, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
            d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
            n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
            d_probe_candidate_dist);
    } else {
        const int effective_thread_k = thread_k <= 4 ? 4 : (thread_k <= 8 ? 8 : 16);
        if (effective_thread_k <= 4) {
            launch_direct_codebook_runtime_m_kernel<D_SUB, 32, 2, 4>(
                grid, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
                n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
                d_probe_candidate_dist);
        } else if (effective_thread_k <= 8) {
            launch_direct_codebook_runtime_m_kernel<D_SUB, 32, 2, 8>(
                grid, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
                n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
                d_probe_candidate_dist);
        } else {
            launch_direct_codebook_runtime_m_kernel<D_SUB, 32, 2, 16>(
                grid, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
                d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
                n_total_clusters, n_probes, pq_m, local_k, d_probe_candidate_idx,
                d_probe_candidate_dist);
        }
    }
}


template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K, int PROBES_PER_CHUNK>
inline void launch_probe_chunk_direct_codebook_runtime_m_kernel(
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int chunk_count,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    constexpr int kBlockThreads = WARPS_PER_BLOCK * 32;
    residual_pq_probe_chunk_direct_codebook_runtime_m_kernel<
        D_SUB, LOCAL_K, WARPS_PER_BLOCK, THREAD_K, PROBES_PER_CHUNK>
        <<<grid, kBlockThreads, shmem, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, n_query, dim_padded, n_total_clusters, n_probes,
            pq_m, local_k, chunk_count, d_probe_candidate_idx, d_probe_candidate_dist);
}

template <int D_SUB>
inline void launch_residual_pq_probe_chunk_direct_codebook_runtime_m(
    int probes_per_chunk,
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int chunk_count,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    if (probes_per_chunk != 4 || local_k != 32) return;
    launch_probe_chunk_direct_codebook_runtime_m_kernel<D_SUB, 32, 2, 8, 4>(
        grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
        d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
        n_total_clusters, n_probes, pq_m, local_k, chunk_count, d_probe_candidate_idx,
        d_probe_candidate_dist);
}

template <int D_SUB, int LOCAL_K, int WARPS_PER_BLOCK, int THREAD_K, int PROBES_PER_CHUNK>
inline void launch_probe_chunk_runtime_m_kernel(
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int chunk_count,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    constexpr int kBlockThreads = WARPS_PER_BLOCK * 32;
    CUDA_CHECK(cudaFuncSetAttribute(
        residual_pq_probe_chunk_runtime_m_kernel<
            D_SUB, LOCAL_K, WARPS_PER_BLOCK, THREAD_K, PROBES_PER_CHUNK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem));
    residual_pq_probe_chunk_runtime_m_kernel<
        D_SUB, LOCAL_K, WARPS_PER_BLOCK, THREAD_K, PROBES_PER_CHUNK>
        <<<grid, kBlockThreads, shmem, stream>>>(
            d_queries, d_centroids, d_top_nprobe, d_pq_codes, d_cluster_offsets,
            d_cluster_counts, d_codebook, n_query, dim_padded, n_total_clusters, n_probes,
            pq_m, local_k, chunk_count, d_probe_candidate_idx, d_probe_candidate_dist);
}

template <int D_SUB>
inline void launch_residual_pq_probe_chunk_runtime_m(
    int probes_per_chunk,
    dim3 grid,
    size_t shmem,
    cudaStream_t stream,
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const uint8_t* d_pq_codes,
    const long long* d_cluster_offsets,
    const int* d_cluster_counts,
    const float* d_codebook,
    int n_query,
    int dim_padded,
    int n_total_clusters,
    int n_probes,
    int pq_m,
    int local_k,
    int chunk_count,
    int* d_probe_candidate_idx,
    float* d_probe_candidate_dist)
{
    if (probes_per_chunk != 4 || local_k != 32) return;
    launch_probe_chunk_runtime_m_kernel<D_SUB, 32, 2, 8, 4>(
        grid, shmem, stream, d_queries, d_centroids, d_top_nprobe, d_pq_codes,
        d_cluster_offsets, d_cluster_counts, d_codebook, n_query, dim_padded,
        n_total_clusters, n_probes, pq_m, local_k, chunk_count, d_probe_candidate_idx,
        d_probe_candidate_dist);
}

}  // namespace paraselect::search::rpq_kernels

#endif
