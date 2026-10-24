#include "clustering/kmeans/kmeans_multigpu_assign.cuh"

#include "l2norm/l2norm.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

int choose_gpu_count(int requested_max) {
    int n_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&n_gpus);
    if (err != cudaSuccess || n_gpus <= 0) return 1;
    if (requested_max > 0) n_gpus = std::min(n_gpus, requested_max);
    return std::max(1, n_gpus);
}

bool assign_shard_on_gpu(
    int gpu,
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& base_cfg,
    const float* h_data,
    const float* h_centroids,
    idx_t* h_assign_shard,
    int shard_start,
    int shard_n) {
    if (shard_n <= 0) return true;

    cudaSetDevice(gpu);

    const int dim = base_cfg.dim;
    const int k = base_cfg.k;
    const int B = cfg.batch_size > 0 ? cfg.batch_size : (1 << 15);
    const int Ktile = cfg.centroid_tile > 0 ? cfg.centroid_tile : 4096;

    float* d_centroids = nullptr;
    float* d_cnorm2 = nullptr;
    float* d_data_buf = nullptr;
    idx_t* d_assign = nullptr;

    cudaError_t err = cudaMalloc(&d_centroids, sizeof(float) * static_cast<size_t>(k) * dim);
    if (err == cudaSuccess) {
        err = cudaMalloc(&d_cnorm2, sizeof(float) * static_cast<size_t>(k));
    }
    if (err == cudaSuccess) {
        err = cudaMalloc(&d_data_buf, sizeof(float) * static_cast<size_t>(B) * dim);
    }
    if (err == cudaSuccess) {
        err = cudaMalloc(&d_assign, sizeof(idx_t) * static_cast<size_t>(shard_n));
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[gpu_centroid_assignment_multi_device] GPU%d allocation failed: %s\n",
                     gpu, cudaGetErrorString(err));
        if (d_centroids) cudaFree(d_centroids);
        if (d_cnorm2) cudaFree(d_cnorm2);
        if (d_data_buf) cudaFree(d_data_buf);
        if (d_assign) cudaFree(d_assign);
        return false;
    }

    cudaMemcpy(d_centroids,
               h_centroids,
               sizeof(float) * static_cast<size_t>(k) * dim,
               cudaMemcpyHostToDevice);

    StreamEnv env;
    env.allocate(B, Ktile);

    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[gpu_centroid_assignment_multi_device] GPU%d cublasCreate failed\n", gpu);
        env.free();
        cudaFree(d_centroids);
        cudaFree(d_cnorm2);
        cudaFree(d_data_buf);
        cudaFree(d_assign);
        return false;
    }
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    KMeansCase shard_cfg = base_cfg;
    shard_cfg.n = shard_n;
    perform_assignment_only(
        shard_cfg,
        h_data + static_cast<size_t>(shard_start) * dim,
        d_assign,
        d_centroids,
        d_cnorm2,
        env,
        d_data_buf,
        0,
        handle,
        B,
        Ktile,
        shard_n,
        dim,
        k);

    err = cudaMemcpy(h_assign_shard,
                     d_assign,
                     sizeof(idx_t) * static_cast<size_t>(shard_n),
                     cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    env.free();
    cudaFree(d_centroids);
    cudaFree(d_cnorm2);
    cudaFree(d_data_buf);
    cudaFree(d_assign);

    if (err != cudaSuccess) {
        std::fprintf(stderr, "[gpu_centroid_assignment_multi_device] GPU%d assign copy failed: %s\n",
                     gpu, cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace

bool assign_vectors_to_centroids_gpu_multi_device(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    idx_t* d_assign) {
    if (!h_data || !d_centroids || !d_assign || kmeans_cfg.n <= 0 ||
        kmeans_cfg.dim <= 0 || kmeans_cfg.k <= 0) {
        return false;
    }

    const int n = kmeans_cfg.n;
    const int dim = kmeans_cfg.dim;
    const int k = kmeans_cfg.k;
    const int main_device = cfg.device_id;
    const int n_gpus = choose_gpu_count(cfg.max_gpus);

    std::fprintf(stderr,
                 "[gpu_centroid_assignment_multi_device] begin: n=%d dim=%d k=%d gpus=%d\n",
                 n, dim, k, n_gpus);
    std::fflush(stderr);

    cudaSetDevice(main_device);
    std::vector<float> h_centroids(static_cast<size_t>(k) * dim);
    cudaError_t err = cudaMemcpy(h_centroids.data(),
                                 d_centroids,
                                 sizeof(float) * h_centroids.size(),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[gpu_centroid_assignment_multi_device] centroid copy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    std::vector<idx_t> h_assign(static_cast<size_t>(n));
    std::atomic<bool> ok(true);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_gpus));

    for (int g = 0; g < n_gpus; ++g) {
        const int start = static_cast<int>((static_cast<int64_t>(n) * g) / n_gpus);
        const int end = static_cast<int>((static_cast<int64_t>(n) * (g + 1)) / n_gpus);
        workers.emplace_back([&, g, start, end]() {
            const bool shard_ok = assign_shard_on_gpu(
                g,
                cfg,
                kmeans_cfg,
                h_data,
                h_centroids.data(),
                h_assign.data() + start,
                start,
                end - start);
            if (!shard_ok) ok.store(false);
        });
    }

    for (auto& worker : workers) worker.join();
    if (!ok.load()) return false;

    cudaSetDevice(main_device);
    err = cudaMemcpy(d_assign,
                     h_assign.data(),
                     sizeof(idx_t) * static_cast<size_t>(n),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[gpu_centroid_assignment_multi_device] final assign copy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    cudaSetDevice(main_device);
    std::fprintf(stderr, "[gpu_centroid_assignment_multi_device] done\n");
    std::fflush(stderr);
    return true;
}
