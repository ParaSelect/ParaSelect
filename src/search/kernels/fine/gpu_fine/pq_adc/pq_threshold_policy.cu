#include "search/kernels/fine/gpu_fine/pq_adc/pq_threshold_policy.cuh"

#include "pq_distance_histogram.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <math_constants.h>
#include <stdexcept>

namespace {

struct MinimumBin {
    __device__ int operator()(int a, int b) const { return min(a, b); }
};

template <int Items>
__global__ void pq_histogram_threshold_kernel(
    const unsigned int* histogram, int n_query, int bins, int rank,
    float log_min, float log_step, float* threshold,
    int* status) {
    const int q = blockIdx.x;
    using Scan = cub::BlockScan<std::uint64_t, 256>;
    using Min = cub::BlockReduce<int, 256>;
    __shared__ union {
        typename Scan::TempStorage scan;
        typename Min::TempStorage reduce;
    } temp;
    const std::uint64_t target = static_cast<std::uint64_t>(rank);
    std::uint64_t prefix[Items];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int bin = threadIdx.x * Items + i;
        prefix[i] = bin < bins ? histogram[(size_t)q * bins + bin] : 0;
    }
    Scan(temp.scan).InclusiveSum(prefix, prefix);
    int selected = bins;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int bin = threadIdx.x * Items + i;
        if (bin < bins && prefix[i] >= target) selected = min(selected, bin);
    }
    __syncthreads();
    selected = Min(temp.reduce).Reduce(selected, MinimumBin());
    if (threadIdx.x == 0) {
        // No rank crossing and the overflow bin both require an infinite bound.
        threshold[q] = selected >= bins - 1 ? CUDART_INF_F
            : exp2f(log_min + static_cast<float>(selected + 1) * log_step);
    }
}

__device__ __forceinline__ float positive_or(float value, float fallback) {
    return isfinite(value) && value > 0.0f ? value : fallback;
}

__global__ void pq_threshold_policy_kernel(
    const float* seed_dist,
    const int* seed_idx,
    int n_query,
    int seed_k,
    int seed_rank,
    int reference_rank,
    int policy,
    float multiplier,
    float margin,
    float bucket_edge,
    float bucket_alpha_low,
    float bucket_alpha_high,
    float bucket_edge0,
    float bucket_edge1,
    float bucket_alpha0,
    float bucket_alpha1,
    float bucket_alpha2,
    float ledger_multiplier,
    float ledger_bin_width,
    float ledger_margin,
    float histogram_edge1,
    float histogram_edge2,
    float histogram_edge3,
    float histogram_alpha0,
    float histogram_alpha1,
    float histogram_alpha2,
    float histogram_alpha3,
    float* threshold,
    int* status) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_query) return;
    const int rank = min(max(seed_rank, 1), seed_k);
    const int ref = min(max(reference_rank, 1), seed_k);
    const size_t base = (size_t)q * seed_k;
    const int kth = rank - 1;
    const float value = seed_dist[base + kth];
    if (seed_idx[base + kth] < 0 || !isfinite(value) || value == CUDART_INF_F) {
        threshold[q] = -CUDART_INF_F;
        if (policy == 0) {
            // An incomplete unfiltered seed has no finite Kth-distance
            // upper bound. Scan all probed rows for this query, then let
            // the shared cap-select retain K. This also handles empty
            // seed lists without discarding neighbours in later probes.
            threshold[q] = CUDART_INF_F;
        } else {
            atomicMax(status, 2);
        }
        return;
    }
    const float first = positive_or(seed_dist[base], value);
    const float reference = positive_or(seed_dist[base + ref - 1], value);
    const float gap = fmaxf(value - seed_dist[base + max(kth - 1, 0)], 0.0f);
    const float ratio = reference / fmaxf(value, 1.0e-12f);
    float out = value;
    switch (policy) {
        case 0:
            out = value * fmaxf(multiplier, 1.0f);
            break;
        case 1: {
            float alpha = histogram_alpha3;
            if (ratio <= histogram_edge1) alpha = histogram_alpha0;
            else if (ratio <= histogram_edge2) alpha = histogram_alpha1;
            else if (ratio <= histogram_edge3) alpha = histogram_alpha2;
            out = value * alpha;
            break;
        }
        case 2: {
            float alpha = bucket_alpha2;
            if (ratio <= bucket_edge0) alpha = bucket_alpha0;
            else if (ratio <= bucket_edge1) alpha = bucket_alpha1;
            out = value * alpha;
            break;
        }
        default:
            atomicMax(status, 5);
            out = value;
            break;
    }
    threshold[q] = out;
}

}  // namespace

namespace paraselect::search {

const char* pq_threshold_policy_name(PQThresholdPolicy policy) {
    switch (policy) {
        case PQThresholdPolicy::FixedMultiplier: return "fixed";
        case PQThresholdPolicy::HistogramMargin: return "histogram_margin";
        case PQThresholdPolicy::DistanceHistogram: return "histogram";
        case PQThresholdPolicy::Bucket: return "bucket";
    }
    return "unknown";
}

void pq_threshold_generate_gpu(
    const float* d_seed_dist,
    const int* d_seed_idx,
    int n_query,
    int seed_k,
    const PQAdcThresholdParams& params,
    float* d_threshold,
    int* d_status, cudaStream_t stream) {
    if (!d_seed_dist || !d_seed_idx || !d_threshold || !d_status ||
        n_query <= 0 || seed_k <= 0) {
        throw std::invalid_argument("pq threshold policy received invalid buffers");
    }
    pq_threshold_policy_kernel<<<(n_query + 255) / 256, 256, 0, stream>>>(
        d_seed_dist, d_seed_idx, n_query, seed_k, params.seed_rank,
        params.reference_rank, static_cast<int>(params.policy), params.multiplier,
        params.margin, params.bucket_edge, params.bucket_alpha_low,
        params.bucket_alpha_high, params.bucket_edge0, params.bucket_edge1,
        params.bucket_alpha0, params.bucket_alpha1, params.bucket_alpha2,
        params.ledger_multiplier, params.ledger_bin_width, params.ledger_margin,
        params.histogram_edge1, params.histogram_edge2,
        params.histogram_edge3, params.histogram_alpha0, params.histogram_alpha1,
        params.histogram_alpha2, params.histogram_alpha3, d_threshold, d_status);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

void pq_histogram_threshold_generate_gpu(
    const unsigned int* d_histogram,
    int n_query,
    int histogram_bins,
    int rank,
    float log_min,
    float log_step,
    float* d_threshold,
    int* d_status,
    cudaStream_t stream) {
    if (!d_histogram || !d_threshold || !d_status || n_query <= 0 ||
        histogram_bins <= 0 || histogram_bins > PQDistanceHistogramMaxBins || rank <= 0 ||
        log_step <= 0.0f) {
        throw std::invalid_argument("invalid histogram threshold request");
    }
    if (histogram_bins <= 256) {
        pq_histogram_threshold_kernel<1><<<n_query, 256, 0, stream>>>(
            d_histogram, n_query, histogram_bins, rank,
            log_min, log_step, d_threshold, d_status);
    } else if (histogram_bins <= 512) {
        pq_histogram_threshold_kernel<2><<<n_query, 256, 0, stream>>>(
            d_histogram, n_query, histogram_bins, rank,
            log_min, log_step, d_threshold, d_status);
    } else if (histogram_bins <= 1024) {
        pq_histogram_threshold_kernel<4><<<n_query, 256, 0, stream>>>(
            d_histogram, n_query, histogram_bins, rank,
            log_min, log_step, d_threshold, d_status);
    } else if (histogram_bins <= 2048) {
        pq_histogram_threshold_kernel<8><<<n_query, 256, 0, stream>>>(
            d_histogram, n_query, histogram_bins, rank,
            log_min, log_step, d_threshold, d_status);
    } else if (histogram_bins <= 4096) {
        pq_histogram_threshold_kernel<16><<<n_query, 256, 0, stream>>>(
            d_histogram, n_query, histogram_bins, rank,
            log_min, log_step, d_threshold, d_status);
    } else {
        pq_histogram_threshold_kernel<32><<<n_query, 256, 0, stream>>>(
            d_histogram, n_query, histogram_bins, rank,
            log_min, log_step, d_threshold, d_status);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

}  // namespace paraselect::search
