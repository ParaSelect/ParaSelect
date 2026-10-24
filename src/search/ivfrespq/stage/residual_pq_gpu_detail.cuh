#ifndef PARASELECT_RESIDUAL_PQ_GPU_DETAIL_CUH
#define PARASELECT_RESIDUAL_PQ_GPU_DETAIL_CUH

#include "search/kernels/topk/warpsort/bitonic.cuh"
#include "search/kernels/topk/fixed_topk.cuh"

#include <cfloat>
#include <cstdint>

namespace paraselect::search::rpq_detail {

constexpr int kResidualPqM = 16;
constexpr int kResidualPqK = 256;
constexpr int kResidualPqLutStride = 257;
constexpr int kResidualPqLutElems = kResidualPqM * kResidualPqLutStride;

__device__ inline uint4 load_pq_code(const uint8_t* __restrict__ pq_codes, size_t row) {
    return *reinterpret_cast<const uint4*>(pq_codes + row * kResidualPqM);
}

using paraselect::search::topk_detail::init_topk_fixed;
using paraselect::search::topk_detail::insert_topk;

template <int THREAD_K>
__device__ __forceinline__ void sort_thread_candidates_across_warp(float* dists, int* idxs) {
    static_assert((THREAD_K & (THREAD_K - 1)) == 0, "THREAD_K must be a power of two");
    paraselect::bitonic::Bitonic<THREAD_K>(true, 32).sort(dists, idxs);
}

template <int LOCAL_K, int THREAD_K>
__device__ __forceinline__ void write_warp_topk_from_sorted_thread_candidates(
    const float* dists,
    const int* idxs,
    float* warp_dist,
    int* warp_idx)
{
    static_assert(LOCAL_K <= 32, "LOCAL_K must fit in one warp row");
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane < LOCAL_K) {
        warp_dist[(size_t)warp_id * LOCAL_K + lane] = dists[0];
        warp_idx[(size_t)warp_id * LOCAL_K + lane] = idxs[0];
    }
}

template <int LOCAL_K, int THREAD_K>
__device__ __forceinline__ void write_warp_topk_from_sorted_thread_candidates_bitonic_pair(
    const float* dists,
    const int* idxs,
    float* warp_dist,
    int* warp_idx)
{
    static_assert(LOCAL_K <= 32, "LOCAL_K must fit in one warp row");
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane < LOCAL_K) {
        const int pos = (warp_id & 1) ? (LOCAL_K - 1 - lane) : lane;
        warp_dist[(size_t)warp_id * LOCAL_K + pos] = dists[0];
        warp_idx[(size_t)warp_id * LOCAL_K + pos] = idxs[0];
    }
}

template <int WARP_OUT_K, int THREAD_K>
__device__ __forceinline__ void write_warp_topk_from_sorted_thread_candidates_limited(
    const float* dists,
    const int* idxs,
    float* warp_dist,
    int* warp_idx)
{
    static_assert(WARP_OUT_K <= 32, "WARP_OUT_K must fit in one warp row");
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane < WARP_OUT_K) {
        const int pos = (warp_id & 1) ? (WARP_OUT_K - 1 - lane) : lane;
        warp_dist[(size_t)warp_id * WARP_OUT_K + pos] = dists[0];
        warp_idx[(size_t)warp_id * WARP_OUT_K + pos] = idxs[0];
    }
}

template <int LOCAL_K, int WARPS_PER_BLOCK, int ITEMS_PER_LANE>
__device__ __forceinline__ void write_block_topk_from_warp_candidates(
    const float* warp_dist,
    const int* warp_idx,
    int local_k,
    int q,
    int n_probes,
    int p,
    float* candidate_dist,
    int* candidate_idx)
{
    static_assert(LOCAL_K <= 32, "LOCAL_K must fit in one warp row");
    static_assert((ITEMS_PER_LANE & (ITEMS_PER_LANE - 1)) == 0,
                  "ITEMS_PER_LANE must be a power of two");
    constexpr int kTotalWarpItems = WARPS_PER_BLOCK * LOCAL_K;
    const int lane = threadIdx.x & 31;
    float merge_dist[ITEMS_PER_LANE];
    int merge_idx[ITEMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const int src = lane + i * 32;
        if (src < kTotalWarpItems) {
            merge_dist[i] = warp_dist[src];
            merge_idx[i] = warp_idx[src];
        } else {
            merge_dist[i] = FLT_MAX;
            merge_idx[i] = -1;
        }
    }
    paraselect::bitonic::Bitonic<ITEMS_PER_LANE>(true, 32).sort(merge_dist, merge_idx);
    if (lane < LOCAL_K) {
        float* gd = candidate_dist + ((size_t)q * n_probes + p) * local_k;
        int* gi = candidate_idx + ((size_t)q * n_probes + p) * local_k;
        gd[lane] = merge_dist[0];
        gi[lane] = merge_idx[0];
    }
}

template <int LOCAL_K, int WARPS_PER_BLOCK, int WARP_OUT_K, int ITEMS_PER_LANE>
__device__ __forceinline__ void write_block_topk_from_limited_warp_candidates_linear(
    const float* warp_dist,
    const int* warp_idx,
    int local_k,
    size_t out_base,
    float* candidate_dist,
    int* candidate_idx)
{
    static_assert(LOCAL_K <= 32, "LOCAL_K must fit in one warp row");
    static_assert(WARP_OUT_K <= 32, "WARP_OUT_K must fit in one warp row");
    static_assert((ITEMS_PER_LANE & (ITEMS_PER_LANE - 1)) == 0,
                  "ITEMS_PER_LANE must be a power of two");
    constexpr int kTotalWarpItems = WARPS_PER_BLOCK * WARP_OUT_K;
    const int lane = threadIdx.x & 31;
    float merge_dist[ITEMS_PER_LANE];
    int merge_idx[ITEMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const int src = lane + i * 32;
        if (src < kTotalWarpItems) {
            merge_dist[i] = warp_dist[src];
            merge_idx[i] = warp_idx[src];
        } else {
            merge_dist[i] = FLT_MAX;
            merge_idx[i] = -1;
        }
    }
    paraselect::bitonic::Bitonic<ITEMS_PER_LANE>(true, 32).sort(merge_dist, merge_idx);
    if (lane < LOCAL_K) {
        candidate_dist[out_base + lane] = merge_dist[0];
        candidate_idx[out_base + lane] = merge_idx[0];
    }
}

template <int LOCAL_K>
__device__ __forceinline__ void write_block_topk_from_2warp32_bitonic_candidates_linear(
    const float* warp_dist,
    const int* warp_idx,
    size_t out_base,
    float* candidate_dist,
    int* candidate_idx)
{
    static_assert(LOCAL_K == 32, "2warp32 bitonic merge is specialized for LOCAL_K=32");
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (warp_id == 0) {
        float merge_dist[2];
        int merge_idx[2];
        merge_dist[0] = warp_dist[lane];
        merge_idx[0] = warp_idx[lane];
        merge_dist[1] = warp_dist[32 + lane];
        merge_idx[1] = warp_idx[32 + lane];
        paraselect::bitonic::Bitonic<2>(true, 32).merge(merge_dist, merge_idx);
        candidate_dist[out_base + lane] = merge_dist[0];
        candidate_idx[out_base + lane] = merge_idx[0];
    }
}

template <int LOCAL_K, int WARPS_PER_BLOCK, int ITEMS_PER_LANE>
__device__ __forceinline__ void write_block_topk_from_warp_candidates_linear(
    const float* warp_dist,
    const int* warp_idx,
    int local_k,
    size_t out_base,
    float* candidate_dist,
    int* candidate_idx)
{
    static_assert(LOCAL_K <= 32, "LOCAL_K must fit in one warp row");
    static_assert((ITEMS_PER_LANE & (ITEMS_PER_LANE - 1)) == 0,
                  "ITEMS_PER_LANE must be a power of two");
    constexpr int kTotalWarpItems = WARPS_PER_BLOCK * LOCAL_K;
    const int lane = threadIdx.x & 31;
    float merge_dist[ITEMS_PER_LANE];
    int merge_idx[ITEMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const int src = lane + i * 32;
        if (src < kTotalWarpItems) {
            merge_dist[i] = warp_dist[src];
            merge_idx[i] = warp_idx[src];
        } else {
            merge_dist[i] = FLT_MAX;
            merge_idx[i] = -1;
        }
    }
    paraselect::bitonic::Bitonic<ITEMS_PER_LANE>(true, 32).sort(merge_dist, merge_idx);
    if (lane < LOCAL_K) {
        candidate_dist[out_base + lane] = merge_dist[0];
        candidate_idx[out_base + lane] = merge_idx[0];
    }
}

template <int PQ_M>
__device__ inline float adc_lut(
    const float* __restrict__ lut,
    const uint8_t* __restrict__ pq_codes,
    size_t row)
{
    const uint8_t* c = pq_codes + row * PQ_M;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
#pragma unroll
    for (int m = 0; m + 3 < PQ_M; m += 4) {
        s0 += lut[(m + 0) * kResidualPqLutStride + __ldg(c + m + 0)];
        s1 += lut[(m + 1) * kResidualPqLutStride + __ldg(c + m + 1)];
        s2 += lut[(m + 2) * kResidualPqLutStride + __ldg(c + m + 2)];
        s3 += lut[(m + 3) * kResidualPqLutStride + __ldg(c + m + 3)];
    }
    float tail = 0.0f;
#pragma unroll
    for (int m = (PQ_M / 4) * 4; m < PQ_M; ++m) {
        tail += lut[m * kResidualPqLutStride + __ldg(c + m)];
    }
    return (s0 + s1) + (s2 + s3) + tail;
}

__device__ inline float adc_lut(const float* __restrict__ lut, uint4 code) {
    const auto* c = reinterpret_cast<const uint8_t*>(&code);
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
#pragma unroll
    for (int m = 0; m < kResidualPqM; m += 4) {
        s0 += lut[(m + 0) * kResidualPqLutStride + c[m + 0]];
        s1 += lut[(m + 1) * kResidualPqLutStride + c[m + 1]];
        s2 += lut[(m + 2) * kResidualPqLutStride + c[m + 2]];
        s3 += lut[(m + 3) * kResidualPqLutStride + c[m + 3]];
    }
    return (s0 + s1) + (s2 + s3);
}

template <>
__device__ inline float adc_lut<16>(
    const float* __restrict__ lut,
    const uint8_t* __restrict__ pq_codes,
    size_t row)
{
    return adc_lut(lut, load_pq_code(pq_codes, row));
}

}  // namespace paraselect::search::rpq_detail

#endif  // PARASELECT_RESIDUAL_PQ_GPU_DETAIL_CUH
