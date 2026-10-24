#include "clustering/kmeans/kmeans_progressive_prefix.cuh"

#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "clustering/kmeans/kmeans_multigpu_train.cuh"
#include "reorder/reorder.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

struct ProgressiveSlice {
    const float* data = nullptr;
    float* owned = nullptr;
    int start = 0;
    int count = 0;
};

int advance_cursor(int cursor, int count, int n) {
    if (n <= 0) return 0;
    return static_cast<int>((static_cast<int64_t>(cursor) + count) % n);
}

ProgressiveSlice make_slice(const float* h_data, int n, int dim, int start, int count) {
    ProgressiveSlice slice;
    slice.start = start;
    slice.count = count;
    if (count <= 0) return slice;

    if (start + count <= n) {
        slice.data = h_data + static_cast<size_t>(start) * dim;
        return slice;
    }

    slice.owned = static_cast<float*>(std::malloc(static_cast<size_t>(count) * dim * sizeof(float)));
    if (!slice.owned) throw std::bad_alloc();

    const int first = n - start;
    std::memcpy(slice.owned,
                h_data + static_cast<size_t>(start) * dim,
                static_cast<size_t>(first) * dim * sizeof(float));
    std::memcpy(slice.owned + static_cast<size_t>(first) * dim,
                h_data,
                static_cast<size_t>(count - first) * dim * sizeof(float));
    slice.data = slice.owned;
    return slice;
}

void release_slice(ProgressiveSlice& slice) {
    if (slice.owned) {
        std::free(slice.owned);
        slice.owned = nullptr;
    }
    slice.data = nullptr;
}

}  // namespace

void initialize_progressive_prefix_centroids(
    const float* h_data,
    int n,
    int dim,
    int k,
    float* h_centroids) {
    if (!h_data || !h_centroids || n <= 0 || dim <= 0 || k <= 0 || k > n) {
        throw std::runtime_error("ProgressivePrefix requires valid data and n_clusters <= n_vectors");
    }
    std::memcpy(h_centroids, h_data, static_cast<size_t>(k) * dim * sizeof(float));
}

bool run_progressive_prefix_kmeans_inplace(
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

    const int default_stage1 = 50000000;
    const int default_stage2 = 200000000;
    const int stages[2] = {
        cfg.progressive_stage1_samples > 0 ? cfg.progressive_stage1_samples : default_stage1,
        cfg.progressive_stage2_samples > 0 ? cfg.progressive_stage2_samples : default_stage2,
    };

    int cursor = k % n;
    for (int si = 0; si < 2; ++si) {
        const int stage_n = std::min(stages[si], n);
        if (stage_n <= 0) continue;

        ProgressiveSlice slice = make_slice(h_data, n, dim, cursor, stage_n);
        fprintf(stderr,
                "[progressive_prefix] stage %d start=%d n=%d next=%d\n",
                si + 1,
                cursor,
                stage_n,
                advance_cursor(cursor, stage_n, n));
        fflush(stderr);

        KMeansCase stage_cfg = base_kmeans_cfg;
        stage_cfg.n = stage_n;
        stage_cfg.iters = 1;
        stage_cfg.minibatch_iters = 4;

        GpuCentroidAssignmentConfig train_assign_cfg;
        train_assign_cfg.strategy = KMeansFinalAssignStrategy::Exact;
        train_assign_cfg.device_id = cfg.device_id;
        const bool ok = kmeans_train_centroids_multigpu(
            train_assign_cfg,
            stage_cfg,
            slice.data,
            d_centroids,
            cfg.h_objective);
        release_slice(slice);
        if (!ok) return false;

        cursor = advance_cursor(cursor, stage_n, n);
    }

    cudaSetDevice(cfg.device_id);
    CHECK_CUDA_ERRORS;
    idx_t* d_assign = nullptr;
    cudaError_t err = cudaMalloc(&d_assign, sizeof(idx_t) * static_cast<size_t>(n));
    if (err != cudaSuccess) return false;

    KMeansCase full_assign_cfg = base_kmeans_cfg;
    full_assign_cfg.n = n;
    GpuCentroidAssignmentConfig assign_cfg;
    assign_cfg.strategy = cfg.final_assign_strategy;
    assign_cfg.device_id = cfg.device_id;
    bool ok = assign_vectors_to_centroids_gpu(assign_cfg, full_assign_cfg, h_data, d_centroids, d_assign);
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
