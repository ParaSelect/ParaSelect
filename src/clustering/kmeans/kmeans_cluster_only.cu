#include "clustering/kmeans/kmeans_cluster_only.cuh"

#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "clustering/kmeans/kmeans_multigpu_train.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

bool kmeans_fast_ivf_cluster_only(
    const KMeansClusterOnlyConfig& cfg,
    const float* h_data,
    KMeansClusterOnlyResult* out) {
    if (!h_data || !out || cfg.n <= 0 || cfg.dim <= 0 || cfg.k <= 0 || cfg.iters <= 0) {
        return false;
    }
    const int train_n = std::min(cfg.n, cfg.train_samples > 0 ? cfg.train_samples : cfg.n);
    if (train_n < cfg.k) {
        std::fprintf(stderr, "[kmeans_cluster_only] train_n=%d < k=%d\n", train_n, cfg.k);
        return false;
    }

    cudaSetDevice(cfg.device_id);
    CHECK_CUDA_ERRORS;

    KMeansCase base{};
    base.n = cfg.n;
    base.dim = cfg.dim;
    base.k = cfg.k;
    base.iters = cfg.iters;
    base.minibatch_iters = cfg.iters * 4;
    base.seed = cfg.seed;
    base.dist = cfg.dist;
    base.precision = cfg.precision;
    base.use_flash_assign = false;

    float* h_init_centroids = static_cast<float*>(std::malloc(
        static_cast<size_t>(cfg.k) * cfg.dim * sizeof(float)));
    if (!h_init_centroids) return false;
    KMeansCase init_cfg = base;
    init_cfg.n = train_n;
    init_centroids_by_sampling(init_cfg, h_data, h_init_centroids);

    float* d_centroids = nullptr;
    idx_t* d_assign = nullptr;
    cudaError_t err = cudaMalloc(&d_centroids, static_cast<size_t>(cfg.k) * cfg.dim * sizeof(float));
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_centroids, h_init_centroids,
                         static_cast<size_t>(cfg.k) * cfg.dim * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    std::free(h_init_centroids);
    if (err != cudaSuccess) {
        if (d_centroids) cudaFree(d_centroids);
        return false;
    }

    GpuCentroidAssignmentConfig assign_cfg;
    assign_cfg.strategy = KMeansFinalAssignStrategy::Exact;
    assign_cfg.device_id = cfg.device_id;

    KMeansCase train_cfg = base;
    train_cfg.n = train_n;
    float objective = 0.0f;
    bool ok = kmeans_train_centroids_multigpu(
        assign_cfg,
        train_cfg,
        h_data,
        d_centroids,
        &objective);
    if (!ok) {
        cudaFree(d_centroids);
        return false;
    }

    err = cudaMalloc(&d_assign, static_cast<size_t>(cfg.n) * sizeof(idx_t));
    if (err != cudaSuccess) {
        cudaFree(d_centroids);
        return false;
    }

    GpuCentroidAssignmentConfig final_assign_cfg;
    final_assign_cfg.strategy = cfg.final_assign_strategy;
    final_assign_cfg.device_id = cfg.device_id;
    ok = assign_vectors_to_centroids_gpu(final_assign_cfg, base, h_data, d_centroids, d_assign);
    if (!ok) {
        cudaFree(d_assign);
        cudaFree(d_centroids);
        return false;
    }

    ClusterInfo cluster_info{};
    gpu_build_cluster_info(cfg.n, cfg.k, d_assign, &cluster_info, cfg.device_id, 0);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    cudaFree(d_assign);

    out->centroids.resize(static_cast<size_t>(cfg.k) * cfg.dim);
    err = cudaMemcpy(out->centroids.data(), d_centroids,
                     static_cast<size_t>(cfg.k) * cfg.dim * sizeof(float),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_centroids);
    if (err != cudaSuccess) {
        free_cluster_info(&cluster_info, false);
        return false;
    }

    out->counts.assign(cluster_info.counts, cluster_info.counts + cfg.k);
    out->offsets.assign(cluster_info.offsets, cluster_info.offsets + cfg.k + 1);
    out->objective = objective;
    free_cluster_info(&cluster_info, false);
    return true;
}
