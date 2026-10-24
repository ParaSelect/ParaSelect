// kmeans_clustering_inplace.cu
// KMeansClusteringInPlace: , n*dim
#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans.cuh"
#include "clustering/kmeans/kmeans_progressive_prefix.cuh"
#include "clustering/kmeans/kmeans_fast_ivf.cuh"
#include "reorder/reorder.cuh"
#include "l2norm/l2norm.cuh"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <vector>

namespace {

void free_inplace_result_buffers(
    idx_t* reordered_indices,
    float* h_centroids,
    ClusterInfo* info
) {
    if (reordered_indices) std::free(reordered_indices);
    if (h_centroids) std::free(h_centroids);
    if (info) {
        if (info->offsets) std::free(info->offsets);
        if (info->counts) std::free(info->counts);
        info->offsets = nullptr;
        info->counts = nullptr;
        info->k = 0;
    }
}

/**
 *
 * @param h_data  [n, dim]
 * @param n
 * @param dim
 * @param n_train
 * @param seed
 * @return  [n_train, dim], free
 */
float* sample_training_data(
    const float* h_data,
    int n,
    int dim,
    int n_train,
    unsigned int seed
) {

    if (n_train >= n) {
        // ,
        float* sampled = (float*)std::malloc((size_t)n * dim * sizeof(float));
        if (!sampled) return nullptr;
        std::memcpy(sampled, h_data, (size_t)n * dim * sizeof(float));

        return sampled;
    }

    //
    float* sampled = (float*)std::malloc((size_t)n_train * dim * sizeof(float));
    if (!sampled) {
        fprintf(stderr, "[sample_training_data] malloc failed: bytes=%zu\n",
                (size_t)n_train * dim * sizeof(float));
        return nullptr;
    }

    //  n .100M  full shuffle  MB
    // ; stride + seed offset .
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> offset_dist(0, std::max(0, n - 1));
    const int offset = offset_dist(rng);
    const int64_t stride = std::max<int64_t>(1, (int64_t)n / std::max(1, n_train));


    //
    for (int i = 0; i < n_train; ++i) {
        int src_idx = (int)(((int64_t)offset + (int64_t)i * stride) % n);
        std::memcpy(
            sampled + (size_t)i * dim,
            h_data + (size_t)src_idx * dim,
            dim * sizeof(float)
        );
    }


    return sampled;
}

/**
 * ()
 * @param cfg KMeans
 * @param h_data  [n, dim]
 * @param d_centroids  [k, dim]
 * @param d_assign : [n]
 * @param device_id GPU  ID
 * @return
 */


bool assign_vectors_to_centroids_gpu_local(
    const KMeansCase& cfg,
    const float* h_data,
    float* d_centroids,
    idx_t* d_assign,
    int device_id
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;


    //  StreamEnv
    const int B = 1 << 15;
    const int Ktile = 4096;
    StreamEnv env;

    env.allocate(B, Ktile);

    //  cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[assign_vectors_to_centroids_gpu_local] cublasCreate failed\n");
        return false;
    }
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);


    //  centroids
    float* d_cnorm2 = nullptr;
    cudaError_t err = cudaMalloc(&d_cnorm2, sizeof(float) * (size_t)k);
    if (err != cudaSuccess) {
        fprintf(stderr, "[assign_vectors_to_centroids_gpu_local] cudaMalloc d_cnorm2 failed: %s\n", cudaGetErrorString(err));
        cublasDestroy(handle);
        env.free();
        return false;
    }
    //
    float* d_data_buf = nullptr;
    err = cudaMalloc(&d_data_buf, sizeof(float) * (size_t)B * dim);
    if (err != cudaSuccess) {
        fprintf(stderr, "[assign_vectors_to_centroids_gpu_local] cudaMalloc d_data_buf failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_cnorm2);
        cublasDestroy(handle);
        env.free();
        return false;
    }

    perform_assignment_only(
        cfg,
        h_data,
        d_assign,
        d_centroids,
        d_cnorm2,
        env,
        d_data_buf,
        0,
        handle,
        B,
        Ktile,
        n,
        dim,
        k
    );

    //
    cudaFree(d_data_buf);
    cudaFree(d_cnorm2);
    cublasDestroy(handle);
    env.free();


    return true;
}

} // namespace

void KMeansClusteringInPlace::fit(float* h_data, int n, int dim, const ClusteringConfig& cfg) {

    if (fitted_) {
        free_inplace_result_buffers(
            result_.reordered_indices,
            h_centroids_,
            &cluster_info_
        );
        result_ = ClusteringResult{};
        cluster_info_ = ClusterInfo{};
        h_centroids_ = nullptr;
        h_data_ = nullptr;
        fitted_ = false;
    }

    // ,
    h_data_ = h_data;

    const int k = cfg.n_clusters;
    const size_t centroid_size = (size_t)k * dim;

    idx_t* reordered_indices = (idx_t*)std::malloc((size_t)n * sizeof(idx_t));
    float* h_centroids = (float*)std::malloc(centroid_size * sizeof(float));
    if (!reordered_indices || !h_centroids) {
        fprintf(stderr, "[KMeansClusteringInPlace::fit] malloc failed: reordered_indices=%p h_centroids=%p\n",
                (void*)reordered_indices, (void*)h_centroids);
        free_inplace_result_buffers(reordered_indices, h_centroids, nullptr);
        throw std::bad_alloc();
    }


    for (int i = 0; i < n; ++i) reordered_indices[i] = i;


    //
    int n_train = n;
    bool use_sampling = false;
    float* h_train_data = h_data;

    if (cfg.train_sample_ratio < 1.0f || cfg.train_max_samples > 0) {
        int n_by_ratio = static_cast<int>(n * cfg.train_sample_ratio);
        if (cfg.train_max_samples > 0) {
            n_train = std::min({n_by_ratio, cfg.train_max_samples, n});
        } else {
            n_train = std::min(n_by_ratio, n);
        }

        if (n_train < n) {
            use_sampling = true;


            h_train_data = sample_training_data(h_data, n, dim, n_train, cfg.seed);
            if (!h_train_data) {
                free_inplace_result_buffers(reordered_indices, h_centroids, nullptr);
                throw std::runtime_error("sample_training_data failed");
            }
        }
    }

    KMeansCase kmeans_cfg;
    kmeans_cfg.n = n_train;  //
    kmeans_cfg.dim = dim;
    kmeans_cfg.k = k;
    kmeans_cfg.iters = cfg.max_iters;
    kmeans_cfg.minibatch_iters = cfg.max_iters * 4;
    kmeans_cfg.seed = cfg.seed;
    kmeans_cfg.dist = cfg.distance_mode;
    kmeans_cfg.precision = cfg.compute_precision;
    kmeans_cfg.use_flash_assign = kmeans_uses_flash_assign(cfg.assign_operator);

    const bool use_progressive_prefix =
        cfg.train_strategy == KMeansTrainStrategy::ProgressivePrefix;
    const bool use_fast_ivf =
        cfg.train_strategy == KMeansTrainStrategy::FastIVF;
    if (use_progressive_prefix) {
        initialize_progressive_prefix_centroids(h_data, n, dim, k, h_centroids);

    } else if (use_fast_ivf) {
        KMeansCase init_cfg = kmeans_cfg;
        init_cfg.n = std::min(n, cfg.fast_ivf_train_samples > 0 ? cfg.fast_ivf_train_samples : 50000000);
        init_centroids_by_sampling(init_cfg, h_data, h_centroids);

    } else {
        init_centroids_by_sampling(kmeans_cfg, h_train_data, h_centroids);

    }

    float* d_centroids = nullptr;
    cudaError_t err = cudaMalloc(&d_centroids, centroid_size * sizeof(float));
    if (err != cudaSuccess) {
        if (use_sampling && h_train_data) std::free(h_train_data);
        free_inplace_result_buffers(reordered_indices, h_centroids, nullptr);
        throw std::runtime_error("cudaMalloc centroids failed");
    }
    cudaMemcpy(d_centroids, h_centroids, centroid_size * sizeof(float), cudaMemcpyHostToDevice);


    cluster_info_.offsets = nullptr;
    cluster_info_.counts = nullptr;
    cluster_info_.k = 0;

    bool ok = false;

    if (use_progressive_prefix) {
        ok = run_progressive_prefix_kmeans_inplace(
            cfg,
            kmeans_cfg,
            h_data,
            n,
            dim,
            k,
            d_centroids,
            reordered_indices,
            &cluster_info_);
        if (!ok) {
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error("run_progressive_prefix_kmeans_inplace failed");
        }
    } else if (use_fast_ivf) {
        ok = run_fast_ivf_kmeans_inplace(
            cfg,
            kmeans_cfg,
            h_data,
            n,
            dim,
            k,
            d_centroids,
            reordered_indices,
            &cluster_info_);
        if (!ok) {
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error("run_fast_ivf_kmeans_inplace failed");
        }
    } else if (use_sampling) {
        // :
        // 1.
        // 2.
        // 3.

        // Step 1:
        idx_t* d_assign_train = nullptr;
        std::vector<idx_t> h_assign_train(n_train);
        ok = ivf_kmeans_clustering(
            kmeans_cfg,
            h_train_data,
            d_centroids,
            h_assign_train.data(),
            nullptr,
            cfg.use_minibatch,
            cfg.device_id,
            cfg.h_objective,
            &d_assign_train
        );


        if (d_assign_train) cudaFree(d_assign_train);
        std::free(h_train_data);  //

        if (!ok) {
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error("ivf_kmeans_clustering (sampling) failed");
        }

        // Step 2:
        idx_t* d_assign = nullptr;
        err = cudaMalloc(&d_assign, sizeof(idx_t) * (size_t)n);
        if (err != cudaSuccess) {
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error(std::string("cudaMalloc d_assign failed: ") + cudaGetErrorString(err));
        }


        kmeans_cfg.n = n;  //
        ok = assign_vectors_to_centroids_gpu_local(kmeans_cfg, h_data, d_centroids, d_assign, cfg.device_id);


        if (!ok) {
            cudaFree(d_assign);
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error("assign_vectors_to_centroids_gpu_local failed");
        }

        // Step 3:  ClusterInfo
        gpu_build_cluster_info(n, k, d_assign, &cluster_info_, cfg.device_id, 0);
        cudaDeviceSynchronize();

        if (cfg.cluster_only) {
            cudaFree(d_assign);

        } else {

        // Step 4:  CPU
        std::vector<idx_t> h_assign(n);

        err = cudaMemcpy(h_assign.data(), d_assign, sizeof(idx_t) * (size_t)n, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_assign);
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error(std::string("cudaMemcpy d_assign failed: ") + cudaGetErrorString(err));
        }
        cudaFree(d_assign);

        //  h_assign, cpu_cluster_reorder_vectors_inplace
        std::vector<idx_t> h_assign_backup = h_assign;


        cpu_cluster_reorder_vectors_inplace(h_data, h_assign.data(), n, dim, k);

        cpu_cluster_reorder_indices_inplace(reordered_indices, h_assign_backup.data(), n, k);

        }

    } else {
        // ()

        ok = ivf_kmeans_inplace(
            kmeans_cfg,
            h_data,
            d_centroids,
            &cluster_info_,
            cfg.use_minibatch,
            cfg.device_id,
            cfg.batch_size,
            cfg.h_objective,
            reordered_indices
        );


        if (!ok) {
            cudaFree(d_centroids);
            free_inplace_result_buffers(reordered_indices, h_centroids, &cluster_info_);
            throw std::runtime_error("ivf_kmeans_inplace failed");
        }
    }

    cudaMemcpy(h_centroids, d_centroids, centroid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_centroids);


    h_centroids_ = h_centroids;

    result_.centroids = h_centroids_;
    result_.assignments = nullptr;
    result_.cluster_sizes = cluster_info_.counts;
    result_.reordered_data = cfg.cluster_only ? nullptr : h_data;
    result_.reordered_indices = reordered_indices;
    result_.cluster_info = &cluster_info_;
    result_.n_vectors = n;
    result_.dim = dim;
    result_.reordered_data_owned = false;
    fitted_ = true;

}

ClusteringResult KMeansClusteringInPlace::get_result() {
    return result_;
}

float* KMeansClusteringInPlace::release_reordered_data() {
    if (!result_.reordered_data) {
        return nullptr;
    }
    float* ptr = h_data_;
    h_data_ = nullptr;
    return ptr;
}

KMeansClusteringInPlace::~KMeansClusteringInPlace() {
    if (!fitted_) return;
    free_inplace_result_buffers(
        result_.reordered_indices,
        h_centroids_,
        &cluster_info_
    );
}
