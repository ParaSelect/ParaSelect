// kmeans_reorder.cu
// Out-of-place  K-means reorder (,)
#include "kmeans.cuh"
#include "reorder/reorder.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

// ============================================================
// IVF K-means: Clustering only (no reorder)
// ============================================================

bool ivf_kmeans_clustering(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] CPU input
    float* d_centroids,            // [k, dim] GPU (in/out)
    idx_t* h_assign,                  // [n] CPU output (optional)
    int* h_cluster_sizes,           // [k] CPU output (optional)
    bool use_minibatch,            // true for minibatch, false for Lloyd
    int device_id,
    float* h_objective,            // optional output
    idx_t** d_assign_out             // [out]  assign , host
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    //
    if (!h_data || !d_centroids || n <= 0 || dim <= 0 || k <= 0) {
        fprintf(stderr, "[ivf_kmeans_clustering] ERROR: Invalid parameters\n");
        return false;
    }

    cudaSetDevice(device_id);
    CHECK_CUDA_ERRORS;
    // Step 1: Allocate device buffer for assignments
    idx_t* d_assign = nullptr;
    cudaMalloc(&d_assign, sizeof(idx_t) * (size_t)n);
    CHECK_CUDA_ERRORS;
    // Step 2: Run K-means clustering
    float obj = 0.0f;
    if (use_minibatch) {
        gpu_kmeans_minibatch(cfg, h_data, d_assign, d_centroids, &obj);
    } else {
        gpu_kmeans_lloyd(cfg, h_data, d_assign, d_centroids, &obj);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    if (h_objective) {
        *h_objective = obj;
    }

    // Step 3: Copy assignments to host (if requested)
    if (h_assign) {
        cudaMemcpy(h_assign, d_assign, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    }

    // Step 4: Compute cluster sizes (if requested)
    if (h_cluster_sizes) {
        std::vector<idx_t> h_assign_local(n);
        if (!h_assign) {
            cudaMemcpy(h_assign_local.data(), d_assign, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS;
        }
        const idx_t* assign_ptr = h_assign ? h_assign : h_assign_local.data();

        std::memset(h_cluster_sizes, 0, sizeof(int) * (size_t)k);
        for (int i = 0; i < n; ++i) {
            int c = assign_ptr[i];
            if (c >= 0 && c < k) {
                h_cluster_sizes[c]++;
            }
        }
    }

    // Cleanup / output d_assign
    if (d_assign_out) {
        *d_assign_out = d_assign;  //
    } else {
        cudaFree(d_assign);
    }

    return true;
}

// ============================================================
// IVF K-means: Out-of-place version (better performance)
// ============================================================

bool ivf_kmeans(
    const KMeansCase& cfg,
    const float* h_data_in,        // [n, dim] CPU input
    float* h_data_out,             // [n, dim] CPU output (must be pre-allocated)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch,            // true for minibatch, false for Lloyd
    int device_id,
    int batch_size,
    float* h_objective,            // optional output
    const idx_t* h_indices_in,       // [n] CPU ()
    idx_t* h_indices_out,            // [n] CPU ()
    KMeansReorderPolicy reorder_policy
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    //
    if (!h_data_in || !h_data_out || !d_centroids || n <= 0 || dim <= 0) {
        fprintf(stderr, "[ivf_kmeans] ERROR: Invalid parameters\n");
        return false;
    }

    cudaSetDevice(device_id);
    CHECK_CUDA_ERRORS;

    // Step 1: Allocate device buffer for assignments
    idx_t* d_assign = nullptr;
    cudaMalloc(&d_assign, sizeof(idx_t) * (size_t)n);
    CHECK_CUDA_ERRORS;

    // Step 2: Run K-means clustering
    float obj = 0.0f;
    if (use_minibatch) {
        gpu_kmeans_minibatch(cfg, h_data_in, d_assign, d_centroids, &obj);
    } else {
        gpu_kmeans_lloyd(cfg, h_data_in, d_assign, d_centroids, &obj);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    if (h_objective) {
        *h_objective = obj;
    }

    // Step 3: Build permutation on GPU
    std::vector<idx_t> h_perm(n);
    ClusterInfo cluster_info;
    ClusterInfo* info_ptr = h_cluster_info ? h_cluster_info : &cluster_info;

    gpu_build_permutation_by_cluster(n, k, d_assign, h_perm.data(), info_ptr,
                                    device_id, batch_size, 0);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    // Step 4: Out-of-place reorder vectors on CPU
    // Phase3 : [n,dim] ," copy ".
    if (reorder_policy == KMeansReorderPolicy::NumaFirstTouch) {
        std::fprintf(stderr, "[ivf_kmeans] reorder_policy=numa\n");
        cpu_reorder_vectors_by_cluster_ranges_numa(
            h_data_in,
            h_perm.data(),
            h_data_out,
            info_ptr->offsets,
            info_ptr->counts,
            n,
            dim,
            k);
    } else {
        std::fprintf(stderr, "[ivf_kmeans] reorder_policy=standard\n");
        cpu_reorder_vectors_by_permutation(h_data_in, h_perm.data(), h_data_out, n, dim);
    }

    // Step 5: Out-of-place reorder indices on CPU (if provided)
    if (h_indices_in && h_indices_out) {
        cpu_reorder_indices_by_permutation(h_indices_in, h_perm.data(), h_indices_out, n);
    }

    // Cleanup
    cudaFree(d_assign);

    if (!h_cluster_info) {
        free_cluster_info(&cluster_info, false);
    }

    return true;
}
