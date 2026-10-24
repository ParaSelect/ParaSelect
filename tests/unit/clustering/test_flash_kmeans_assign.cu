#include "clustering/flash_kmeans/flash_kmeans_assign.cuh"
#include "common/test_utils.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

__global__ void compute_l2_norm_kernel(
    const float* __restrict__ x,
    int n,
    int dim,
    float* __restrict__ norm2)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= n) return;
    __shared__ float smem[256];
    float sum = 0.0f;
    const float* v = x + (size_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        const float t = v[d];
        sum += t * t;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) norm2[row] = smem[0];
}

__global__ void init_best_kernel(float* best_dist, idx_t* best_idx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_dist[i] = 3.402823e38f;
        best_idx[i] = 0;
    }
}

__global__ void update_best_from_dot_kernel(
    const float* __restrict__ dot,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2,
    int n,
    int k,
    idx_t* __restrict__ best_idx,
    float* __restrict__ best_dist)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xn = xnorm2[i];
    float best = best_dist[i];
    idx_t idx = best_idx[i];
    for (int c = 0; c < k; ++c) {
        const float dist = xn + cnorm2[c] - 2.0f * dot[i + (size_t)c * n];
        if (dist < best) {
            best = dist;
            idx = c;
        }
    }
    best_dist[i] = best;
    best_idx[i] = idx;
}

float run_gemm_baseline(
    cublasHandle_t handle,
    const float* d_data,
    const float* d_centroids,
    int n,
    int k,
    int dim,
    float* d_xnorm2,
    float* d_cnorm2,
    float* d_dot,
    idx_t* d_best_idx,
    float* d_best_dist)
{
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    compute_l2_norm_kernel<<<n, 256>>>(d_data, n, dim, d_xnorm2);
    compute_l2_norm_kernel<<<k, 256>>>(d_centroids, k, dim, d_cnorm2);
    init_best_kernel<<<(n + 255) / 256, 256>>>(d_best_dist, d_best_idx, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        k,
        dim,
        &alpha,
        d_data,
        dim,
        d_centroids,
        dim,
        &beta,
        d_dot,
        n);
    update_best_from_dot_kernel<<<(n + 255) / 256, 256>>>(
        d_dot,
        d_xnorm2,
        d_cnorm2,
        n,
        k,
        d_best_idx,
        d_best_dist);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

void fill_random(std::vector<float>& values) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : values) v = dist(rng);
}

std::vector<int> parse_int_list(const std::string& value) {
    std::vector<int> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

paraselect::clustering::kmeans::FlashKMeansAssignConfig make_flash_config(
    int dim,
    int variant,
    int point_tile_override,
    int centroid_tile_override,
    int cutlass_k_tile_override,
    int cutlass_stages_override,
    int shared_reduce_min_tiles_override,
    int small_reduce_max_tiles_override)
{
    auto config = paraselect::clustering::kmeans::default_flash_kmeans_assign_config(dim);
    config.variant = variant;
    if (variant == 8) {
        config.point_tile = 64;
        config.centroid_tile = 256;
        config.cutlass_k_tile = 16;
        config.cutlass_stages = 2;  //  kmeans.cu
    }
    if (point_tile_override > 0) config.point_tile = point_tile_override;
    if (centroid_tile_override > 0) config.centroid_tile = centroid_tile_override;
    if (cutlass_k_tile_override > 0) config.cutlass_k_tile = cutlass_k_tile_override;
    if (cutlass_stages_override > 0) config.cutlass_stages = cutlass_stages_override;
    if (shared_reduce_min_tiles_override > 0) {
        config.shared_reduce_min_tiles = shared_reduce_min_tiles_override;
    }
    if (small_reduce_max_tiles_override >= 0) {
        config.small_reduce_max_tiles = small_reduce_max_tiles_override;
    }
    return config;
}

void run_correctness_and_speed(
    paraselect::test::TestContext& ctx,
    int n,
    int k,
    int dim,
    bool require_distance_strict,
    bool require_flash_faster,
    int variant = 2,
    int point_tile_override = 0,
    int centroid_tile_override = 0,
    int cutlass_k_tile_override = 0,
    int cutlass_stages_override = 0,
    int shared_reduce_min_tiles_override = 0,
    int small_reduce_max_tiles_override = -1)
{
    std::vector<float> h_data((size_t)n * dim);
    std::vector<float> h_centroids((size_t)k * dim);
    fill_random(h_data);
    fill_random(h_centroids);

    float* d_data = nullptr;
    float* d_centroids = nullptr;
    idx_t* d_flash_idx = nullptr;
    idx_t* d_base_idx = nullptr;
    float* d_flash_dist = nullptr;
    float* d_base_dist = nullptr;
    float* d_base_xnorm2 = nullptr;
    float* d_base_cnorm2 = nullptr;
    float* d_base_dot = nullptr;
    cudaMalloc(&d_data, sizeof(float) * h_data.size());
    cudaMalloc(&d_centroids, sizeof(float) * h_centroids.size());
    cudaMalloc(&d_flash_idx, sizeof(idx_t) * (size_t)n);
    cudaMalloc(&d_base_idx, sizeof(idx_t) * (size_t)n);
    cudaMalloc(&d_flash_dist, sizeof(float) * (size_t)n);
    cudaMalloc(&d_base_dist, sizeof(float) * (size_t)n);
    cudaMalloc(&d_base_xnorm2, sizeof(float) * (size_t)n);
    cudaMalloc(&d_base_cnorm2, sizeof(float) * (size_t)k);
    cudaMalloc(&d_base_dot, sizeof(float) * (size_t)n * (size_t)k);
    cudaMemcpy(d_data, h_data.data(), sizeof(float) * h_data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), sizeof(float) * h_centroids.size(), cudaMemcpyHostToDevice);

    cublasHandle_t handle{};
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    auto config = make_flash_config(
        dim,
        variant,
        point_tile_override,
        centroid_tile_override,
        cutlass_k_tile_override,
        cutlass_stages_override,
        shared_reduce_min_tiles_override,
        small_reduce_max_tiles_override);

    cudaFree(nullptr);
    constexpr int kMeasureRounds = 10;
    paraselect::clustering::kmeans::FlashKMeansAssignWorkspace workspace;
    paraselect::clustering::kmeans::reserve_flash_kmeans_assign_workspace(
        workspace, n, k, config);
    (void)paraselect::clustering::kmeans::flash_kmeans_assign_l2(
        d_data,
        d_centroids,
        n,
        k,
        dim,
        d_flash_idx,
        d_flash_dist,
        config,
        workspace);
    (void)run_gemm_baseline(
        handle,
        d_data,
        d_centroids,
        n,
        k,
        dim,
        d_base_xnorm2,
        d_base_cnorm2,
        d_base_dot,
        d_base_idx,
        d_base_dist);
    cudaDeviceSynchronize();

    paraselect::clustering::kmeans::FlashKMeansAssignResult flash_sum;
    float baseline_sum = 0.0f;
    paraselect::clustering::kmeans::FlashKMeansAssignResult flash;
    float baseline_ms = 0.0f;
    for (int round = 0; round < kMeasureRounds; ++round) {
        flash = paraselect::clustering::kmeans::flash_kmeans_assign_l2(
            d_data,
            d_centroids,
            n,
            k,
            dim,
            d_flash_idx,
            d_flash_dist,
            config,
            workspace);
        baseline_ms = run_gemm_baseline(
            handle,
            d_data,
            d_centroids,
            n,
            k,
            dim,
            d_base_xnorm2,
            d_base_cnorm2,
            d_base_dot,
            d_base_idx,
            d_base_dist);
        flash_sum.ms += flash.ms;
        flash_sum.norm_ms += flash.norm_ms;
        flash_sum.assign_ms += flash.assign_ms;
        flash_sum.init_ms += flash.init_ms;
        flash_sum.cutlass_gemm_ms += flash.cutlass_gemm_ms;
        flash_sum.reduce_ms += flash.reduce_ms;
        flash_sum.partial_reduce_ms += flash.partial_reduce_ms;
        flash_sum.final_reduce_ms += flash.final_reduce_ms;
        baseline_sum += baseline_ms;
    }
    flash.ms = flash_sum.ms / kMeasureRounds;
    flash.norm_ms = flash_sum.norm_ms / kMeasureRounds;
    flash.assign_ms = flash_sum.assign_ms / kMeasureRounds;
    flash.init_ms = flash_sum.init_ms / kMeasureRounds;
    flash.cutlass_gemm_ms = flash_sum.cutlass_gemm_ms / kMeasureRounds;
    flash.reduce_ms = flash_sum.reduce_ms / kMeasureRounds;
    flash.partial_reduce_ms = flash_sum.partial_reduce_ms / kMeasureRounds;
    flash.final_reduce_ms = flash_sum.final_reduce_ms / kMeasureRounds;
    baseline_ms = baseline_sum / kMeasureRounds;

    std::vector<idx_t> h_flash_idx(n);
    std::vector<idx_t> h_base_idx(n);
    std::vector<float> h_flash_dist(n);
    std::vector<float> h_base_dist(n);
    cudaMemcpy(h_flash_idx.data(), d_flash_idx, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_base_idx.data(), d_base_idx, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flash_dist.data(), d_flash_dist, sizeof(float) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_base_dist.data(), d_base_dist, sizeof(float) * (size_t)n, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    int invalid_indices = 0;
    double max_abs_dist_err = 0.0;
    double max_selected_gap = 0.0;
    for (int i = 0; i < n; ++i) {
        if (h_flash_idx[i] < 0 || h_flash_idx[i] >= k) {
            ++invalid_indices;
            max_selected_gap = std::numeric_limits<double>::infinity();
            continue;
        }
        if (h_flash_idx[i] != h_base_idx[i]) {
            ++mismatches;
            double selected_dist = 0.0;
            const float* x = h_data.data() + (size_t)i * dim;
            const float* c = h_centroids.data() + (size_t)h_flash_idx[i] * dim;
            for (int d = 0; d < dim; ++d) {
                const double diff = (double)x[d] - (double)c[d];
                selected_dist += diff * diff;
            }
            max_selected_gap = std::max(
                max_selected_gap,
                selected_dist - (double)h_base_dist[i]);
        }
        max_abs_dist_err = std::max(
            max_abs_dist_err,
            (double)std::fabs(h_flash_dist[i] - h_base_dist[i]));
    }

    const double speedup = baseline_ms / std::max(1e-6f, flash.ms);
    std::cout << "[paraselect:flash_kmeans_assign:summary] n=" << n
              << " k=" << k
              << " dim=" << dim
              << " variant=" << flash.variant
              << " rounds=" << kMeasureRounds
              << " BN=" << flash.point_tile
              << " BK=" << flash.centroid_tile
              << " flash_ms=" << flash.ms
              << " norm_ms=" << flash.norm_ms
              << " assign_ms=" << flash.assign_ms
              << " init_ms=" << flash.init_ms
              << " cutlass_gemm_ms=" << flash.cutlass_gemm_ms
              << " reduce_ms=" << flash.reduce_ms
              << " partial_reduce_ms=" << flash.partial_reduce_ms
              << " final_reduce_ms=" << flash.final_reduce_ms
              << " gemm_baseline_ms=" << baseline_ms
              << " speedup=" << speedup
              << " mismatches=" << mismatches
              << " invalid_indices=" << invalid_indices
              << " max_abs_dist_err=" << max_abs_dist_err
              << " max_selected_gap=" << max_selected_gap
              << std::endl;

    if (variant == 8) {
        ctx.check_eq(invalid_indices, 0, "FlashAssign indices should be in range");
        ctx.check(max_selected_gap < 1e-2, "FlashAssign selected centroid should be near FP32 optimum");
    } else {
        ctx.check_eq(mismatches, 0, "FlashAssign indices should match GEMM baseline");
    }
    if (require_distance_strict) {
        if (variant == 8) {
            ctx.check(max_abs_dist_err < 2e-2, "FlashAssign TF32 distances should be close to GEMM baseline");
        } else {
            ctx.check(max_abs_dist_err < 1e-3, "FlashAssign distances should match GEMM baseline");
        }
    } else {
        if (variant == 8) {
            ctx.check(max_abs_dist_err < 2e-2, "FlashAssign TF32 distances should be close to GEMM baseline");
        } else {
            ctx.check(max_abs_dist_err < 1e-2, "FlashAssign distances should be close to GEMM baseline");
        }
    }
    ctx.check(flash.ms > 0.0f, "FlashAssign timing should be positive");
    ctx.check(baseline_ms > 0.0f, "GEMM baseline timing should be positive");
    if (require_flash_faster) {
        ctx.check(speedup > 1.0, "FlashAssign prototype should be faster than materializing GEMM baseline for this case");
    }

    cublasDestroy(handle);
    paraselect::clustering::kmeans::release_flash_kmeans_assign_workspace(workspace);
    cudaFree(d_base_dot);
    cudaFree(d_base_cnorm2);
    cudaFree(d_base_xnorm2);
    cudaFree(d_base_dist);
    cudaFree(d_flash_dist);
    cudaFree(d_base_idx);
    cudaFree(d_flash_idx);
    cudaFree(d_centroids);
    cudaFree(d_data);
}

void run_flash_only_bench(
    paraselect::test::TestContext& ctx,
    int n,
    int k,
    int dim,
    int variant,
    int rounds,
    int point_tile_override = 0,
    int centroid_tile_override = 0,
    int cutlass_k_tile_override = 0,
    int cutlass_stages_override = 0,
    int shared_reduce_min_tiles_override = 0,
    int small_reduce_max_tiles_override = -1)
{
    std::vector<float> h_data((size_t)n * dim);
    std::vector<float> h_centroids((size_t)k * dim);
    fill_random(h_data);
    fill_random(h_centroids);

    float* d_data = nullptr;
    float* d_centroids = nullptr;
    idx_t* d_flash_idx = nullptr;
    float* d_flash_dist = nullptr;
    cudaMalloc(&d_data, sizeof(float) * h_data.size());
    cudaMalloc(&d_centroids, sizeof(float) * h_centroids.size());
    cudaMalloc(&d_flash_idx, sizeof(idx_t) * (size_t)n);
    cudaMalloc(&d_flash_dist, sizeof(float) * (size_t)n);
    cudaMemcpy(d_data, h_data.data(), sizeof(float) * h_data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), sizeof(float) * h_centroids.size(), cudaMemcpyHostToDevice);

    auto config = make_flash_config(
        dim,
        variant,
        point_tile_override,
        centroid_tile_override,
        cutlass_k_tile_override,
        cutlass_stages_override,
        shared_reduce_min_tiles_override,
        small_reduce_max_tiles_override);

    paraselect::clustering::kmeans::FlashKMeansAssignWorkspace workspace;
    paraselect::clustering::kmeans::reserve_flash_kmeans_assign_workspace(
        workspace, n, k, config);
    (void)paraselect::clustering::kmeans::flash_kmeans_assign_l2(
        d_data,
        d_centroids,
        n,
        k,
        dim,
        d_flash_idx,
        d_flash_dist,
        config,
        workspace);
    cudaDeviceSynchronize();

    paraselect::clustering::kmeans::FlashKMeansAssignResult flash_sum;
    paraselect::clustering::kmeans::FlashKMeansAssignResult flash;
    for (int round = 0; round < rounds; ++round) {
        flash = paraselect::clustering::kmeans::flash_kmeans_assign_l2(
            d_data,
            d_centroids,
            n,
            k,
            dim,
            d_flash_idx,
            d_flash_dist,
            config,
            workspace);
        flash_sum.ms += flash.ms;
        flash_sum.norm_ms += flash.norm_ms;
        flash_sum.assign_ms += flash.assign_ms;
        flash_sum.init_ms += flash.init_ms;
        flash_sum.cutlass_gemm_ms += flash.cutlass_gemm_ms;
        flash_sum.reduce_ms += flash.reduce_ms;
        flash_sum.partial_reduce_ms += flash.partial_reduce_ms;
        flash_sum.final_reduce_ms += flash.final_reduce_ms;
    }
    flash.ms = flash_sum.ms / rounds;
    flash.norm_ms = flash_sum.norm_ms / rounds;
    flash.assign_ms = flash_sum.assign_ms / rounds;
    flash.init_ms = flash_sum.init_ms / rounds;
    flash.cutlass_gemm_ms = flash_sum.cutlass_gemm_ms / rounds;
    flash.reduce_ms = flash_sum.reduce_ms / rounds;
    flash.partial_reduce_ms = flash_sum.partial_reduce_ms / rounds;
    flash.final_reduce_ms = flash_sum.final_reduce_ms / rounds;

    std::vector<idx_t> h_flash_idx(n);
    std::vector<float> h_flash_dist(n);
    cudaMemcpy(h_flash_idx.data(), d_flash_idx, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flash_dist.data(), d_flash_dist, sizeof(float) * (size_t)n, cudaMemcpyDeviceToHost);

    int invalid_indices = 0;
    int nonfinite_dist = 0;
    double min_dist = std::numeric_limits<double>::infinity();
    double max_dist = 0.0;
    for (int i = 0; i < n; ++i) {
        if (h_flash_idx[i] < 0 || h_flash_idx[i] >= k) {
            ++invalid_indices;
        }
        if (!std::isfinite(h_flash_dist[i])) {
            ++nonfinite_dist;
        } else {
            min_dist = std::min(min_dist, (double)h_flash_dist[i]);
            max_dist = std::max(max_dist, (double)h_flash_dist[i]);
        }
    }

    const double pairs = (double)n * (double)k;
    const double gpair_per_s = pairs / std::max(1e-6f, flash.ms) / 1.0e6;
    std::cout << "[paraselect:flash_kmeans_assign:summary] mode=flash_only"
              << " n=" << n
              << " k=" << k
              << " dim=" << dim
              << " variant=" << flash.variant
              << " rounds=" << rounds
              << " BN=" << flash.point_tile
              << " BK=" << flash.centroid_tile
              << " flash_ms=" << flash.ms
              << " norm_ms=" << flash.norm_ms
              << " assign_ms=" << flash.assign_ms
              << " init_ms=" << flash.init_ms
              << " cutlass_gemm_ms=" << flash.cutlass_gemm_ms
              << " reduce_ms=" << flash.reduce_ms
              << " partial_reduce_ms=" << flash.partial_reduce_ms
              << " final_reduce_ms=" << flash.final_reduce_ms
              << " gpair_per_s=" << gpair_per_s
              << " invalid_indices=" << invalid_indices
              << " nonfinite_dist=" << nonfinite_dist
              << " min_dist=" << min_dist
              << " max_dist=" << max_dist
              << std::endl;

    ctx.check_eq(invalid_indices, 0, "FlashOnly indices should be in range");
    ctx.check_eq(nonfinite_dist, 0, "FlashOnly distances should be finite");
    ctx.check(flash.ms > 0.0f, "FlashOnly timing should be positive");

    paraselect::clustering::kmeans::release_flash_kmeans_assign_workspace(workspace);
    cudaFree(d_flash_dist);
    cudaFree(d_flash_idx);
    cudaFree(d_centroids);
    cudaFree(d_data);
}

void run_flash_only_on_device(
    paraselect::test::TestContext& ctx,
    const float* d_data,
    const float* d_centroids,
    idx_t* d_flash_idx,
    float* d_flash_dist,
    int n,
    int k,
    int dim,
    int variant,
    int rounds,
    int point_tile_override,
    int centroid_tile_override,
    int cutlass_k_tile_override,
    int cutlass_stages_override,
    int shared_reduce_min_tiles_override,
    int small_reduce_max_tiles_override,
    paraselect::clustering::kmeans::FlashKMeansAssignWorkspace& workspace)
{
    auto config = make_flash_config(
        dim,
        variant,
        point_tile_override,
        centroid_tile_override,
        cutlass_k_tile_override,
        cutlass_stages_override,
        shared_reduce_min_tiles_override,
        small_reduce_max_tiles_override);
    paraselect::clustering::kmeans::reserve_flash_kmeans_assign_workspace(
        workspace, n, k, config);

    (void)paraselect::clustering::kmeans::flash_kmeans_assign_l2(
        d_data,
        d_centroids,
        n,
        k,
        dim,
        d_flash_idx,
        d_flash_dist,
        config,
        workspace);
    cudaDeviceSynchronize();

    paraselect::clustering::kmeans::FlashKMeansAssignResult flash_sum;
    paraselect::clustering::kmeans::FlashKMeansAssignResult flash;
    for (int round = 0; round < rounds; ++round) {
        flash = paraselect::clustering::kmeans::flash_kmeans_assign_l2(
            d_data,
            d_centroids,
            n,
            k,
            dim,
            d_flash_idx,
            d_flash_dist,
            config,
            workspace);
        flash_sum.ms += flash.ms;
        flash_sum.norm_ms += flash.norm_ms;
        flash_sum.assign_ms += flash.assign_ms;
        flash_sum.init_ms += flash.init_ms;
        flash_sum.cutlass_gemm_ms += flash.cutlass_gemm_ms;
        flash_sum.reduce_ms += flash.reduce_ms;
        flash_sum.partial_reduce_ms += flash.partial_reduce_ms;
        flash_sum.final_reduce_ms += flash.final_reduce_ms;
    }
    flash.ms = flash_sum.ms / rounds;
    flash.norm_ms = flash_sum.norm_ms / rounds;
    flash.assign_ms = flash_sum.assign_ms / rounds;
    flash.init_ms = flash_sum.init_ms / rounds;
    flash.cutlass_gemm_ms = flash_sum.cutlass_gemm_ms / rounds;
    flash.reduce_ms = flash_sum.reduce_ms / rounds;
    flash.partial_reduce_ms = flash_sum.partial_reduce_ms / rounds;
    flash.final_reduce_ms = flash_sum.final_reduce_ms / rounds;

    std::vector<idx_t> h_flash_idx(n);
    std::vector<float> h_flash_dist(n);
    cudaMemcpy(h_flash_idx.data(), d_flash_idx, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flash_dist.data(), d_flash_dist, sizeof(float) * (size_t)n, cudaMemcpyDeviceToHost);

    int invalid_indices = 0;
    int nonfinite_dist = 0;
    double min_dist = std::numeric_limits<double>::infinity();
    double max_dist = 0.0;
    for (int i = 0; i < n; ++i) {
        if (h_flash_idx[i] < 0 || h_flash_idx[i] >= k) ++invalid_indices;
        if (!std::isfinite(h_flash_dist[i])) {
            ++nonfinite_dist;
        } else {
            min_dist = std::min(min_dist, (double)h_flash_dist[i]);
            max_dist = std::max(max_dist, (double)h_flash_dist[i]);
        }
    }

    const double pairs = (double)n * (double)k;
    const double gpair_per_s = pairs / std::max(1e-6f, flash.ms) / 1.0e6;
    std::cout << "[paraselect:flash_kmeans_assign:summary] mode=flash_only"
              << " n=" << n
              << " k=" << k
              << " dim=" << dim
              << " variant=" << flash.variant
              << " rounds=" << rounds
              << " BN=" << flash.point_tile
              << " BK=" << flash.centroid_tile
              << " cutlass_k_tile=" << config.cutlass_k_tile
              << " cutlass_stages=" << config.cutlass_stages
              << " shared_reduce_min_tiles=" << config.shared_reduce_min_tiles
              << " small_reduce_max_tiles=" << config.small_reduce_max_tiles
              << " flash_ms=" << flash.ms
              << " norm_ms=" << flash.norm_ms
              << " assign_ms=" << flash.assign_ms
              << " init_ms=" << flash.init_ms
              << " cutlass_gemm_ms=" << flash.cutlass_gemm_ms
              << " reduce_ms=" << flash.reduce_ms
              << " partial_reduce_ms=" << flash.partial_reduce_ms
              << " final_reduce_ms=" << flash.final_reduce_ms
              << " gpair_per_s=" << gpair_per_s
              << " invalid_indices=" << invalid_indices
              << " nonfinite_dist=" << nonfinite_dist
              << " min_dist=" << min_dist
              << " max_dist=" << max_dist
              << std::endl;

    ctx.check_eq(invalid_indices, 0, "FlashOnly indices should be in range");
    ctx.check_eq(nonfinite_dist, 0, "FlashOnly distances should be finite");
    ctx.check(flash.ms > 0.0f, "FlashOnly timing should be positive");
}

void run_flash_only_sweep(
    paraselect::test::TestContext& ctx,
    int n,
    int k,
    int dim,
    int variant,
    int rounds,
    const std::vector<int>& point_tiles,
    const std::vector<int>& centroid_tiles,
    const std::vector<int>& cutlass_k_tiles,
    const std::vector<int>& cutlass_stages,
    const std::vector<int>& shared_reduce_min_tiles,
    const std::vector<int>& small_reduce_max_tiles)
{
    std::vector<float> h_data((size_t)n * dim);
    std::vector<float> h_centroids((size_t)k * dim);
    fill_random(h_data);
    fill_random(h_centroids);

    float* d_data = nullptr;
    float* d_centroids = nullptr;
    idx_t* d_flash_idx = nullptr;
    float* d_flash_dist = nullptr;
    cudaMalloc(&d_data, sizeof(float) * h_data.size());
    cudaMalloc(&d_centroids, sizeof(float) * h_centroids.size());
    cudaMalloc(&d_flash_idx, sizeof(idx_t) * (size_t)n);
    cudaMalloc(&d_flash_dist, sizeof(float) * (size_t)n);
    cudaMemcpy(d_data, h_data.data(), sizeof(float) * h_data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), sizeof(float) * h_centroids.size(), cudaMemcpyHostToDevice);

    paraselect::clustering::kmeans::FlashKMeansAssignWorkspace workspace;
    int case_id = 0;
    const int total =
        (int)point_tiles.size() *
        (int)centroid_tiles.size() *
        (int)cutlass_k_tiles.size() *
        (int)cutlass_stages.size() *
        (int)shared_reduce_min_tiles.size() *
        (int)small_reduce_max_tiles.size();
    for (int point_tile : point_tiles) {
        for (int centroid_tile : centroid_tiles) {
            for (int cutlass_k_tile : cutlass_k_tiles) {
                for (int stages : cutlass_stages) {
                    for (int shared_min : shared_reduce_min_tiles) {
                        for (int small_max : small_reduce_max_tiles) {
                            ++case_id;
                            std::cout << "[paraselect:flash_kmeans_assign:phase] sweep_case="
                                      << case_id << "/" << total
                                      << " point_tile=" << point_tile
                                      << " centroid_tile=" << centroid_tile
                                      << " cutlass_k_tile=" << cutlass_k_tile
                                      << " cutlass_stages=" << stages
                                      << " shared_reduce_min_tiles=" << shared_min
                                      << " small_reduce_max_tiles=" << small_max
                                      << std::endl;
                            run_flash_only_on_device(
                                ctx,
                                d_data,
                                d_centroids,
                                d_flash_idx,
                                d_flash_dist,
                                n,
                                k,
                                dim,
                                variant,
                                rounds,
                                point_tile,
                                centroid_tile,
                                cutlass_k_tile,
                                stages,
                                shared_min,
                                small_max,
                                workspace);
                        }
                    }
                }
            }
        }
    }

    paraselect::clustering::kmeans::release_flash_kmeans_assign_workspace(workspace);
    cudaFree(d_flash_dist);
    cudaFree(d_flash_idx);
    cudaFree(d_centroids);
    cudaFree(d_data);
}

}  // namespace

int main(int argc, char** argv) {
    paraselect::test::TestContext ctx(
        "flash_kmeans_assign",
        paraselect::test::TestKind::Unit);

    if (argc > 1) {
        const std::string mode = argv[1];
        if (mode == "--bench") {
            if (argc < 5 || argc > 12) {
                std::cerr << "usage: " << argv[0]
                          << " --bench <n> <k> <dim> [variant] [point_tile] [centroid_tile] [cutlass_k_tile] [cutlass_stages] [shared_reduce_min_tiles] [small_reduce_max_tiles]" << std::endl;
                return 2;
            }
            const int n = std::atoi(argv[2]);
            const int k = std::atoi(argv[3]);
            const int dim = std::atoi(argv[4]);
            const int variant = (argc >= 6) ? std::atoi(argv[5]) : 8;
            const int point_tile = (argc >= 7) ? std::atoi(argv[6]) : 0;
            const int centroid_tile = (argc >= 8) ? std::atoi(argv[7]) : 0;
            const int cutlass_k_tile = (argc >= 9) ? std::atoi(argv[8]) : 0;
            const int cutlass_stages = (argc >= 10) ? std::atoi(argv[9]) : 0;
            const int shared_reduce_min_tiles = (argc >= 11) ? std::atoi(argv[10]) : 0;
            const int small_reduce_max_tiles = (argc >= 12) ? std::atoi(argv[11]) : -1;
            ctx.run_case("bench_custom", [&] {
                run_correctness_and_speed(
                    ctx, n, k, dim, false, false, variant,
                    point_tile, centroid_tile, cutlass_k_tile, cutlass_stages,
                    shared_reduce_min_tiles, small_reduce_max_tiles);
            });
            return ctx.exit_code();
        }
        if (mode == "--bench-flash-only") {
            if (argc < 5 || argc > 13) {
                std::cerr << "usage: " << argv[0]
                          << " --bench-flash-only <n> <k> <dim> [variant] [rounds] [point_tile] [centroid_tile] [cutlass_k_tile] [cutlass_stages] [shared_reduce_min_tiles] [small_reduce_max_tiles]" << std::endl;
                return 2;
            }
            const int n = std::atoi(argv[2]);
            const int k = std::atoi(argv[3]);
            const int dim = std::atoi(argv[4]);
            const int variant = (argc >= 6) ? std::atoi(argv[5]) : 8;
            const int rounds = (argc >= 7) ? std::atoi(argv[6]) : 5;
            const int point_tile = (argc >= 8) ? std::atoi(argv[7]) : 0;
            const int centroid_tile = (argc >= 9) ? std::atoi(argv[8]) : 0;
            const int cutlass_k_tile = (argc >= 10) ? std::atoi(argv[9]) : 0;
            const int cutlass_stages = (argc >= 11) ? std::atoi(argv[10]) : 0;
            const int shared_reduce_min_tiles = (argc >= 12) ? std::atoi(argv[11]) : 0;
            const int small_reduce_max_tiles = (argc >= 13) ? std::atoi(argv[12]) : -1;
            ctx.run_case("bench_flash_only", [&] {
                run_flash_only_bench(
                    ctx, n, k, dim, variant, rounds,
                    point_tile, centroid_tile, cutlass_k_tile, cutlass_stages,
                    shared_reduce_min_tiles, small_reduce_max_tiles);
            });
            return ctx.exit_code();
        }
        if (mode == "--sweep-flash-only") {
            if (argc != 13) {
                std::cerr << "usage: " << argv[0]
                          << " --sweep-flash-only <n> <k> <dim> <variant> <rounds> <point_tiles_csv> <centroid_tiles_csv> <cutlass_k_tiles_csv> <cutlass_stages_csv> <shared_reduce_min_tiles_csv> <small_reduce_max_tiles_csv>" << std::endl;
                return 2;
            }
            const int n = std::atoi(argv[2]);
            const int k = std::atoi(argv[3]);
            const int dim = std::atoi(argv[4]);
            const int variant = std::atoi(argv[5]);
            const int rounds = std::atoi(argv[6]);
            const auto point_tiles = parse_int_list(argv[7]);
            const auto centroid_tiles = parse_int_list(argv[8]);
            const auto cutlass_k_tiles = parse_int_list(argv[9]);
            const auto cutlass_stages = parse_int_list(argv[10]);
            const auto shared_reduce_min_tiles = parse_int_list(argv[11]);
            const auto small_reduce_max_tiles = parse_int_list(argv[12]);
            ctx.run_case("sweep_flash_only", [&] {
                run_flash_only_sweep(
                    ctx,
                    n,
                    k,
                    dim,
                    variant,
                    rounds,
                    point_tiles,
                    centroid_tiles,
                    cutlass_k_tiles,
                    cutlass_stages,
                    shared_reduce_min_tiles,
                    small_reduce_max_tiles);
            });
            return ctx.exit_code();
        }
        std::cerr << "unknown argument: " << mode << std::endl;
        std::cerr << "usage: " << argv[0]
                  << " [--bench <n> <k> <dim> [variant] [point_tile] [centroid_tile] [cutlass_k_tile] [cutlass_stages] [shared_reduce_min_tiles] [small_reduce_max_tiles]]"
                  << " [--bench-flash-only <n> <k> <dim> [variant] [rounds] [point_tile] [centroid_tile] [cutlass_k_tile] [cutlass_stages] [shared_reduce_min_tiles] [small_reduce_max_tiles]]"
                  << " [--sweep-flash-only <n> <k> <dim> <variant> <rounds> <point_tiles_csv> <centroid_tiles_csv> <cutlass_k_tiles_csv> <cutlass_stages_csv> <shared_reduce_min_tiles_csv> <small_reduce_max_tiles_csv>]"
                  << std::endl;
        return 2;
    }

    ctx.run_case("small_correctness_and_speed", [&] {
        run_correctness_and_speed(ctx, 2048, 64, 32, true, false, 8);
    });
    ctx.run_case("large_dim128_k1024", [&] {
        run_correctness_and_speed(ctx, 8192, 1024, 128, false, false, 8);
    });
    ctx.run_case("large_dim128_k4096", [&] {
        run_correctness_and_speed(ctx, 8192, 4096, 128, false, false, 8);
    });
    return ctx.exit_code();
}
