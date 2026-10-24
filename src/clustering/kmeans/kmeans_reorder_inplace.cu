// kmeans_reorder_inplace.cu
// In-place  K-means reorder
#include "kmeans.cuh"
#include "reorder/reorder.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================
// IVF K-means: In-place version (O(1) extra space for reordering)
// ============================================================

bool ivf_kmeans_inplace(
    const KMeansCase& cfg,
    float* h_data,                 // [n, dim] CPU in/out (in-place reorder)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch,            // true for minibatch, false for Lloyd
    int device_id,
    int batch_size,
    float* h_objective,            // optional output
    idx_t* h_indices                 // [n] CPU (,in-place reorder)
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    //
    if (!h_data || !d_centroids || n <= 0 || dim <= 0 || k <= 0) {
        fprintf(stderr, "[ivf_kmeans_inplace] ERROR: Invalid parameters\n");
        return false;
    }

    // Step 1: , d_assign  GPU
    idx_t* d_assign = nullptr;

    std::vector<idx_t> h_assign(n);

    bool ok = ivf_kmeans_clustering(
        cfg,
        h_data,
        d_centroids,
        h_assign.data(),
        nullptr,
        use_minibatch,
        device_id,
        h_objective,
        &d_assign
    );

    if (!ok) {
        if (d_assign) cudaFree(d_assign);
        return false;
    }

    // Step 2:  ClusterInfo, GPU  d_assign ( perm)
    if (h_cluster_info) {

        gpu_build_cluster_info(n, k, d_assign, h_cluster_info, device_id, 0);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;

    }

    // Step 3:  Cluster (American Flag)
    //  h_assign, cpu_cluster_reorder_vectors_inplace

    std::vector<idx_t> h_assign_backup = h_assign;

    cpu_cluster_reorder_vectors_inplace(h_data, h_assign.data(), n, dim, k);


    // Step 4: ()
    if (h_indices) {

        cpu_cluster_reorder_indices_inplace(h_indices, h_assign_backup.data(), n, k);

    }

    //  d_assign
    if (d_assign) cudaFree(d_assign);

    return true;
}
