#include "hierarchical.cuh"
#include "clustering/kmeans/kmeans.cuh"
#include "clustering/utils.cuh"
#include "reorder/reorder.cuh"
#include "l2norm/l2norm.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <queue>
#include <limits>

#include <cublas_v2.h>

namespace {

void free_result_buffers(
    float* reordered_data,
    idx_t* reordered_indices,
    float* h_centroids,
    ClusterInfo* info
) {
    if (reordered_data) std::free(reordered_data);
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

} // namespace

BasicHierarchicalClustering::BasicHierarchicalClustering(int split_k)
    : split_k_(split_k), fitted_(false), h_centroids_(nullptr), h_reordered_data_(nullptr), h_reordered_indices_(nullptr) {
}

BasicHierarchicalClustering::~BasicHierarchicalClustering() {
    release();
}

void BasicHierarchicalClustering::release() {
    free_result_buffers(h_reordered_data_, h_reordered_indices_, h_centroids_, &cluster_info_);
    fitted_ = false;
}

void BasicHierarchicalClustering::fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) {
    if (fitted_) release();

    const int target_c = cfg.n_clusters;
    if (target_c <= 0 || n <= 0 || dim <= 0) {
        throw std::invalid_argument("BasicHierarchicalClustering: invalid parameters");
    }

    // ()
    const float* data_train = h_data;
    const int n_train = n;

    // :1,
    std::vector<int> current_assign(n_train, 0);
    int current_k = 1;

    //
    std::vector<std::vector<float>> centroids;
    std::vector<float> first_centroid(dim, 0.0f);
    for (int i = 0; i < n_train; ++i) {
        for (int d = 0; d < dim; ++d) {
            first_centroid[d] += data_train[(size_t)i * dim + d];
        }
    }
    for (int d = 0; d < dim; ++d) {
        first_centroid[d] /= n_train;
    }
    centroids.push_back(first_centroid);

    //
    while (current_k < target_c) {
        //
        std::vector<int> cluster_sizes(current_k, 0);
        for (int i = 0; i < n_train; ++i) {
            cluster_sizes[current_assign[i]]++;
        }

        int largest_cluster = 0;
        int max_size = cluster_sizes[0];
        for (int c = 1; c < current_k; ++c) {
            if (cluster_sizes[c] > max_size) {
                max_size = cluster_sizes[c];
                largest_cluster = c;
            }
        }

        if (max_size < split_k_) {
            // ,
            break;
        }

        //
        std::vector<int> sub_indices;
        for (int i = 0; i < n_train; ++i) {
            if (current_assign[i] == largest_cluster) {
                sub_indices.push_back(i);
            }
        }

        //
        std::vector<float> sub_data((size_t)sub_indices.size() * dim);
        for (size_t i = 0; i < sub_indices.size(); ++i) {
            std::memcpy(&sub_data[i * dim],
                       &data_train[(size_t)sub_indices[i] * dim],
                       sizeof(float) * dim);
        }

        //  K-means
        int k_to_split = std::min(split_k_, target_c - current_k + 1);
        k_to_split = std::min(k_to_split, (int)sub_indices.size());

        if (k_to_split <= 1) break;

        KMeansCase sub_cfg;
        sub_cfg.n = (int)sub_indices.size();
        sub_cfg.dim = dim;
        sub_cfg.k = k_to_split;
        sub_cfg.iters = cfg.max_iters;
        sub_cfg.seed = cfg.seed + current_k;
        sub_cfg.dist = cfg.distance_mode;
        sub_cfg.precision = cfg.compute_precision;
        sub_cfg.use_flash_assign = kmeans_uses_flash_assign(cfg.assign_operator);

        std::vector<float> sub_centroids((size_t)k_to_split * dim);
        init_centroids_by_sampling(sub_cfg, sub_data.data(), sub_centroids.data());

        float* d_sub_centroids = nullptr;
        cudaMalloc(&d_sub_centroids, sizeof(float) * (size_t)k_to_split * dim);
        cudaMemcpy(d_sub_centroids, sub_centroids.data(), sizeof(float) * (size_t)k_to_split * dim, cudaMemcpyHostToDevice);

        idx_t* d_sub_assign = nullptr;
        cudaMalloc(&d_sub_assign, sizeof(idx_t) * (size_t)sub_indices.size());

        gpu_kmeans_lloyd(sub_cfg, sub_data.data(), d_sub_assign, d_sub_centroids, nullptr);

        std::vector<idx_t> h_sub_assign(sub_indices.size());
        cudaMemcpy(h_sub_assign.data(), d_sub_assign, sizeof(idx_t) * sub_indices.size(), cudaMemcpyDeviceToHost);

        std::vector<float> h_sub_centroids((size_t)k_to_split * dim);
        cudaMemcpy(h_sub_centroids.data(), d_sub_centroids, sizeof(float) * (size_t)k_to_split * dim, cudaMemcpyDeviceToHost);

        cudaFree(d_sub_assign);
        cudaFree(d_sub_centroids);

        // :
        // :largest_cluster ,
        for (size_t i = 0; i < sub_indices.size(); ++i) {
            int new_assign = h_sub_assign[i];
            if (new_assign == 0) {
                current_assign[sub_indices[i]] = largest_cluster;
            } else {
                current_assign[sub_indices[i]] = current_k + new_assign - 1;
            }
        }

        //
        centroids[largest_cluster].assign(h_sub_centroids.begin(), h_sub_centroids.begin() + dim);
        for (int k = 1; k < k_to_split; ++k) {
            std::vector<float> new_centroid(dim);
            for (int d = 0; d < dim; ++d) {
                new_centroid[d] = h_sub_centroids[(size_t)k * dim + d];
            }
            centroids.push_back(new_centroid);
        }

        current_k += k_to_split - 1;
    }

    //
    int final_k = current_k;

    //
    std::vector<int> final_counts(final_k, 0);
    for (int i = 0; i < n; ++i) {
        // ,
        int idx = i % n_train;
        final_counts[current_assign[idx]]++;
    }

    //  offsets
    std::vector<long long> offsets((size_t)final_k + 1u, 0);
    long long accum = 0;
    for (int i = 0; i < final_k; ++i) {
        offsets[i] = accum;
        accum += final_counts[i];
    }
    offsets[(size_t)final_k] = accum;

    //  permutation
    std::vector<idx_t> perm(n);
    {
        std::vector<idx_t> write_ptr(final_k);
        for (int i = 0; i < final_k; ++i) write_ptr[i] = (idx_t)offsets[i];
        for (int i = 0; i < n; ++i) {
            int idx = i % n_train;
            int cid = current_assign[idx];
            perm[write_ptr[cid]++] = i;
        }
    }

    //
    size_t data_size = (size_t)n * dim;
    h_reordered_data_ = (float*)std::malloc(data_size * sizeof(float));
    h_reordered_indices_ = (idx_t*)std::malloc((size_t)n * sizeof(idx_t));
    if (!h_reordered_data_ || !h_reordered_indices_) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, nullptr, nullptr);
        throw std::bad_alloc();
    }

    cpu_reorder_vectors_by_permutation(h_data, perm.data(), h_reordered_data_, n, dim);
    std::memcpy(h_reordered_indices_, perm.data(), sizeof(idx_t) * (size_t)n);

    //
    h_centroids_ = (float*)std::malloc((size_t)final_k * dim * sizeof(float));
    if (!h_centroids_) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, nullptr, nullptr);
        throw std::bad_alloc();
    }
    for (int c = 0; c < final_k; ++c) {
        for (int d = 0; d < dim; ++d) {
            h_centroids_[(size_t)c * dim + d] = centroids[c][d];
        }
    }

    cluster_info_.k = final_k;
    cluster_info_.offsets = (idx_t*)std::malloc(((size_t)final_k + 1u) * sizeof(idx_t));
    cluster_info_.counts = (int*)std::malloc((size_t)final_k * sizeof(int));
    if (!cluster_info_.offsets || !cluster_info_.counts) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, h_centroids_, &cluster_info_);
        throw std::bad_alloc();
    }
    std::memcpy(cluster_info_.offsets, offsets.data(), sizeof(idx_t) * ((size_t)final_k + 1u));
    std::memcpy(cluster_info_.counts, final_counts.data(), sizeof(int) * (size_t)final_k);

    result_.centroids = h_centroids_;
    result_.assignments = nullptr;
    result_.cluster_sizes = cluster_info_.counts;
    result_.reordered_data = h_reordered_data_;
    result_.reordered_indices = h_reordered_indices_;
    result_.cluster_info = &cluster_info_;
    result_.n_vectors = n;
    result_.dim = dim;
    result_.reordered_data_owned = true;  //

    fitted_ = true;
}

ClusteringResult BasicHierarchicalClustering::get_result() {
    return result_;
}

float* BasicHierarchicalClustering::release_reordered_data() {
    if (!fitted_ || !result_.reordered_data_owned) {
        return nullptr;
    }
    float* ptr = h_reordered_data_;
    h_reordered_data_ = nullptr;
    result_.reordered_data = nullptr;
    return ptr;
}
