#include "clustering/kmeans/kmeans_fast_ivf.cuh"

#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "clustering/kmeans/kmeans_multigpu_train.cuh"
#include "reorder/reorder.cuh"
#include "utils/check_cuda_error.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool assign_full_dataset(
    const ClusteringConfig& cfg,
    const KMeansCase& full_assign_cfg,
    float* h_data,
    float* d_centroids,
    idx_t* d_assign) {
    GpuCentroidAssignmentConfig assign_cfg;
    assign_cfg.strategy = cfg.final_assign_strategy;
    assign_cfg.device_id = cfg.device_id;
    assign_cfg.max_gpus = 1;
    if (cfg.final_assign_strategy == KMeansFinalAssignStrategy::Cagra) {
        assign_cfg.combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
    }
    return assign_vectors_to_centroids_gpu(assign_cfg, full_assign_cfg, h_data, d_centroids, d_assign);
}

}  // namespace

bool run_fast_ivf_kmeans_inplace(
    const ClusteringConfig& cfg,
    const KMeansCase& base_kmeans_cfg,
    float* h_data,
    int n,
    int dim,
    int k,
    float* d_centroids,
    idx_t* reordered_indices,
    ClusterInfo* cluster_info) {
    if (!h_data || !d_centroids || !reordered_indices || !cluster_info || n <= 0 || dim <= 0 || k <= 0) {
        return false;
    }

    const int default_train_n = 50000000;
    const int n_train = std::min(
        n,
        cfg.fast_ivf_train_samples > 0 ? cfg.fast_ivf_train_samples : default_train_n);
    if (n_train < k) {
        fprintf(stderr, "[fast_ivf] train sample count %d is smaller than k=%d\n", n_train, k);
        return false;
    }

    fprintf(stderr,
            "[fast_ivf] train prefix n=%d iters=%d final_assign=%s\n",
            n_train,
            cfg.max_iters,
            kmeans_final_assign_strategy_tag(cfg.final_assign_strategy));
    fflush(stderr);

    KMeansCase train_cfg = base_kmeans_cfg;
    train_cfg.n = n_train;
    train_cfg.iters = cfg.max_iters;
    train_cfg.minibatch_iters = cfg.max_iters * 4;

    GpuCentroidAssignmentConfig train_assign_cfg;
    train_assign_cfg.strategy = KMeansFinalAssignStrategy::Exact;
    train_assign_cfg.device_id = cfg.device_id;
    bool ok = kmeans_train_centroids_multigpu(
        train_assign_cfg,
        train_cfg,
        h_data,
        d_centroids,
        cfg.h_objective);
    if (!ok) return false;

    cudaSetDevice(cfg.device_id);
    CHECK_CUDA_ERRORS;
    idx_t* d_assign = nullptr;
    cudaError_t err = cudaMalloc(&d_assign, sizeof(idx_t) * static_cast<size_t>(n));
    if (err != cudaSuccess) return false;

    KMeansCase full_assign_cfg = base_kmeans_cfg;
    full_assign_cfg.n = n;
    ok = assign_full_dataset(cfg, full_assign_cfg, h_data, d_centroids, d_assign);
    if (!ok) {
        cudaFree(d_assign);
        return false;
    }

    fprintf(stderr, "[kmeans_inplace] build cluster_info begin\n");
    fflush(stderr);
    gpu_build_cluster_info(n, k, d_assign, cluster_info, cfg.device_id, 0);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    if (cfg.cluster_only) {
        cudaFree(d_assign);
        fprintf(stderr, "[kmeans_inplace] cluster-only mode: skip assignment D2H and reorder\n");
        fflush(stderr);
        return true;
    }

    fprintf(stderr, "[kmeans_inplace] build cluster_info done; allocating host assign bytes=%zu\n",
            sizeof(idx_t) * static_cast<size_t>(n));
    fflush(stderr);

    std::vector<idx_t> h_assign(n);
    fprintf(stderr, "[kmeans_inplace] copy final assign D2H begin\n");
    fflush(stderr);
    err = cudaMemcpy(h_assign.data(), d_assign, sizeof(idx_t) * static_cast<size_t>(n), cudaMemcpyDeviceToHost);
    cudaFree(d_assign);
    fprintf(stderr, "[kmeans_inplace] copy final assign D2H done err=%s; d_assign freed\n",
            cudaGetErrorString(err));
    fflush(stderr);
    if (err != cudaSuccess) return false;

    fprintf(stderr, "[kmeans_inplace] backup assign begin bytes=%zu\n",
            sizeof(idx_t) * static_cast<size_t>(n));
    fflush(stderr);
    std::vector<idx_t> h_assign_backup = h_assign;
    fprintf(stderr, "[kmeans_inplace] vector reorder begin n=%d dim=%d k=%d\n", n, dim, k);
    fflush(stderr);
    cpu_cluster_reorder_vectors_inplace(h_data, h_assign.data(), n, dim, k);
    fprintf(stderr, "[kmeans_inplace] vector reorder done; index reorder begin\n");
    fflush(stderr);
    cpu_cluster_reorder_indices_inplace(reordered_indices, h_assign_backup.data(), n, k);
    fprintf(stderr, "[kmeans_inplace] index reorder done\n");
    fflush(stderr);
    return true;
}
