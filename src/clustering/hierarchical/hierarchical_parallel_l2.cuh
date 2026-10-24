#pragma once

/**
 *  --  K-means
 *
 * - L2 :Grouped Batched K-means( assign/accum/divide ,,
 *    cuBLAS  launch; hierarchical optimize.md)
 * -  L2: gpu_kmeans_lloyd_with_workspace +  workspace()
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans.cuh"
#include "hierarchical_parallel_l2_kernels.cuh"

#include <nvtx3/nvToolsExt.h>

namespace paraselect::hier_parallel_l2 {

/**  GPU Lloyd  K-means ( host) */
struct Level2KMeansTask {
    int l1_cid{0};
    int n_sub{0};
    int k_sub{0};
    /** row-major [n_sub, dim], */
    std::vector<float> sub_rows;
};

namespace detail {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

/** : gpu_kmeans_lloyd_with_workspace( L2 ) */
template <typename ComputeKi>
void run_scheduled_level2_kmeans_workspace_path(
    int K1,
    int dim,
    int s,
    const ClusteringConfig& cfg,
    const std::vector<int>& n_prime,
    const std::vector<std::vector<int>>& cluster_index_in_train,
    const float* h_train_ptr,
    ComputeKi compute_k_i,
    std::vector<int>& out_l2_k_i,
    std::vector<std::vector<float>>& out_l2_flat) {
    out_l2_k_i.assign(static_cast<size_t>(K1), 0);
    out_l2_flat.assign(static_cast<size_t>(K1), {});

    std::vector<Level2KMeansTask> tasks;
    tasks.reserve(static_cast<size_t>(K1));

    for (int cid = 0; cid < K1; ++cid) {
        if (n_prime[cid] == 0) {
            continue;
        }
        if (n_prime[cid] <= s) {
            continue;
        }
        const int k_i = compute_k_i(cid);
        out_l2_k_i[static_cast<size_t>(cid)] = k_i;
        if (k_i <= 1) {
            continue;
        }

        Level2KMeansTask t;
        t.l1_cid = cid;
        t.n_sub = n_prime[cid];
        t.k_sub = k_i;
        t.sub_rows.resize(static_cast<size_t>(t.n_sub) * dim);
        for (int idx = 0; idx < t.n_sub; ++idx) {
            const int ti = cluster_index_in_train[cid][idx];
            std::memcpy(
                t.sub_rows.data() + static_cast<size_t>(idx) * dim,
                h_train_ptr + static_cast<size_t>(ti) * dim,
                sizeof(float) * dim);
        }
        tasks.push_back(std::move(t));
    }

    if (tasks.empty()) {
        return;
    }

    std::sort(
        tasks.begin(),
        tasks.end(),
        [](const Level2KMeansTask& a, const Level2KMeansTask& b) { return a.l1_cid < b.l1_cid; });

    int max_n_sub = 0;
    int max_k_sub = 0;
    for (const auto& t : tasks) {
        max_n_sub = std::max(max_n_sub, t.n_sub);
        max_k_sub = std::max(max_k_sub, t.k_sub);
    }

    KMeansLloydWorkspace* lloyd_ws = kmeans_lloyd_workspace_create(max_n_sub, max_k_sub, dim);
    if (!lloyd_ws) {
        throw std::runtime_error("run_scheduled_level2_kmeans: kmeans_lloyd_workspace_create failed");
    }

    float* d_centroids = nullptr;
    idx_t* d_assign = nullptr;
    cudaError_t ce =
        cudaMalloc(&d_assign, sizeof(idx_t) * static_cast<size_t>(max_n_sub));
    if (ce != cudaSuccess) {
        kmeans_lloyd_workspace_destroy(lloyd_ws);
        throw std::runtime_error("run_scheduled_level2_kmeans: cudaMalloc d_assign failed");
    }
    ce = cudaMalloc(&d_centroids, sizeof(float) * static_cast<size_t>(max_k_sub) * dim);
    if (ce != cudaSuccess) {
        cudaFree(d_assign);
        kmeans_lloyd_workspace_destroy(lloyd_ws);
        throw std::runtime_error("run_scheduled_level2_kmeans: cudaMalloc d_centroids failed");
    }

    std::vector<float> h_centroids2(static_cast<size_t>(max_k_sub) * dim);

    for (const auto& t : tasks) {
        const int cid = t.l1_cid;
        const int k_i = t.k_sub;
        const int n_sub = t.n_sub;

        KMeansCase cfg2;
        cfg2.n = n_sub;
        cfg2.dim = dim;
        cfg2.k = k_i;
        cfg2.iters = cfg.max_iters;
        cfg2.minibatch_iters = cfg.max_iters * 4;
        cfg2.seed = cfg.seed + static_cast<unsigned int>(cid);
        cfg2.dist = cfg.distance_mode;
        cfg2.precision = cfg.compute_precision;
        cfg2.use_flash_assign = kmeans_uses_flash_assign(cfg.assign_operator);

        init_centroids_by_sampling(cfg2, t.sub_rows.data(), h_centroids2.data());

        ce = cudaMemcpy(
            d_centroids,
            h_centroids2.data(),
            sizeof(float) * static_cast<size_t>(k_i) * dim,
            cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            cudaFree(d_assign);
            cudaFree(d_centroids);
            kmeans_lloyd_workspace_destroy(lloyd_ws);
            throw std::runtime_error("run_scheduled_level2_kmeans: H2D centroids failed");
        }

        gpu_kmeans_lloyd_with_workspace(
            lloyd_ws, cfg2, t.sub_rows.data(), d_assign, d_centroids, nullptr);

        out_l2_flat[static_cast<size_t>(cid)].resize(static_cast<size_t>(k_i) * dim);
        ce = cudaMemcpy(
            out_l2_flat[static_cast<size_t>(cid)].data(),
            d_centroids,
            sizeof(float) * static_cast<size_t>(k_i) * dim,
            cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            cudaFree(d_assign);
            cudaFree(d_centroids);
            kmeans_lloyd_workspace_destroy(lloyd_ws);
            throw std::runtime_error("run_scheduled_level2_kmeans: D2H centroids failed");
        }
    }

    cudaFree(d_assign);
    cudaFree(d_centroids);
    kmeans_lloyd_workspace_destroy(lloyd_ws);
}

struct GroupedKMeansChunkBuffer {
    float* d_data = nullptr;
    float* d_centroids = nullptr;
    float* d_new_centroids = nullptr;
    int* d_offsets_data = nullptr;
    int* d_offsets_centroids = nullptr;
    int* d_k_array = nullptr;
    int* d_assign = nullptr;
    int* d_counts = nullptr;

    int max_n = 0;
    int max_k = 0;
    int max_tasks = 0;
    int dim = 0;

    GroupedKMeansChunkBuffer(int max_n_pts, int max_k_pts, int max_task_cnt, int d)
        : max_n(max_n_pts), max_k(max_k_pts), max_tasks(max_task_cnt), dim(d) {
        cuda_check(
            cudaMalloc(&d_data, static_cast<size_t>(max_n) * static_cast<size_t>(dim) * sizeof(float)),
            "GroupedKMeansChunkBuffer d_data");
        cuda_check(
            cudaMalloc(&d_centroids, static_cast<size_t>(max_k) * static_cast<size_t>(dim) * sizeof(float)),
            "GroupedKMeansChunkBuffer d_centroids");
        cuda_check(
            cudaMalloc(&d_new_centroids, static_cast<size_t>(max_k) * static_cast<size_t>(dim) * sizeof(float)),
            "GroupedKMeansChunkBuffer d_new_centroids");
        cuda_check(
            cudaMalloc(&d_offsets_data, static_cast<size_t>(max_tasks + 1) * sizeof(int)),
            "GroupedKMeansChunkBuffer d_offsets_data");
        cuda_check(
            cudaMalloc(&d_offsets_centroids, static_cast<size_t>(max_tasks + 1) * sizeof(int)),
            "GroupedKMeansChunkBuffer d_offsets_centroids");
        cuda_check(cudaMalloc(&d_k_array, static_cast<size_t>(max_tasks) * sizeof(int)), "GroupedKMeansChunkBuffer d_k_array");
        cuda_check(cudaMalloc(&d_assign, static_cast<size_t>(max_n) * sizeof(int)), "GroupedKMeansChunkBuffer d_assign");
        cuda_check(cudaMalloc(&d_counts, static_cast<size_t>(max_k) * sizeof(int)), "GroupedKMeansChunkBuffer d_counts");
    }

    ~GroupedKMeansChunkBuffer() {
        cudaFree(d_data);
        cudaFree(d_centroids);
        cudaFree(d_new_centroids);
        cudaFree(d_offsets_data);
        cudaFree(d_offsets_centroids);
        cudaFree(d_k_array);
        cudaFree(d_assign);
        cudaFree(d_counts);
    }

    GroupedKMeansChunkBuffer(const GroupedKMeansChunkBuffer&) = delete;
    GroupedKMeansChunkBuffer& operator=(const GroupedKMeansChunkBuffer&) = delete;
};

/** L2: + Grouped Lloyd( hierarchical optimize.md) */
template <typename ComputeKi>
void run_scheduled_level2_kmeans_grouped_l2(
    int K1,
    int dim,
    int s,
    const ClusteringConfig& cfg,
    const std::vector<int>& n_prime,
    const std::vector<std::vector<int>>& cluster_index_in_train,
    const float* h_train_ptr,
    ComputeKi compute_k_i,
    std::vector<int>& out_l2_k_i,
    std::vector<std::vector<float>>& out_l2_flat) {
    out_l2_k_i.assign(static_cast<size_t>(K1), 0);
    out_l2_flat.assign(static_cast<size_t>(K1), {});

    // /; OOM( dim )
    const int MAX_CHUNK_N = 1 << 19;       // 524288
    const int MAX_CHUNK_K = 1 << 18;       // 262144 ()
    const int MAX_CHUNK_TASKS = 1 << 16;   // 65536 /

    GroupedKMeansChunkBuffer buffer(MAX_CHUNK_N, MAX_CHUNK_K, MAX_CHUNK_TASKS, dim);
    cudaStream_t stream{};
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate grouped L2");

    std::vector<float> h_data;
    std::vector<float> h_centroids;
    std::vector<int> h_offsets_data;
    std::vector<int> h_offsets_cents;
    std::vector<int> h_k_array;
    std::vector<int> task_cid_map;

    h_data.reserve(static_cast<size_t>(MAX_CHUNK_N) * dim);
    h_centroids.reserve(static_cast<size_t>(MAX_CHUNK_K) * dim);
    h_offsets_data.reserve(static_cast<size_t>(MAX_CHUNK_TASKS) + 1);
    h_offsets_cents.reserve(static_cast<size_t>(MAX_CHUNK_TASKS) + 1);
    h_k_array.reserve(static_cast<size_t>(MAX_CHUNK_TASKS));
    task_cid_map.reserve(static_cast<size_t>(MAX_CHUNK_TASKS));

    auto flush_chunk = [&]() {
        if (h_k_array.empty()) {
            return;
        }

        const int num_tasks = static_cast<int>(h_k_array.size());
        const int total_n = static_cast<int>(h_data.size() / static_cast<size_t>(dim));
        const int total_k = static_cast<int>(h_centroids.size() / static_cast<size_t>(dim));

        h_offsets_data.push_back(total_n);
        h_offsets_cents.push_back(total_k);

        cuda_check(
            cudaMemcpyAsync(
                buffer.d_data,
                h_data.data(),
                h_data.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            "grouped L2 H2D d_data");
        cuda_check(
            cudaMemcpyAsync(
                buffer.d_centroids,
                h_centroids.data(),
                h_centroids.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            "grouped L2 H2D d_centroids");
        cuda_check(
            cudaMemcpyAsync(
                buffer.d_offsets_data,
                h_offsets_data.data(),
                h_offsets_data.size() * sizeof(int),
                cudaMemcpyHostToDevice,
                stream),
            "grouped L2 H2D d_offsets_data");
        cuda_check(
            cudaMemcpyAsync(
                buffer.d_offsets_centroids,
                h_offsets_cents.data(),
                h_offsets_cents.size() * sizeof(int),
                cudaMemcpyHostToDevice,
                stream),
            "grouped L2 H2D d_offsets_centroids");
        cuda_check(
            cudaMemcpyAsync(buffer.d_k_array, h_k_array.data(), h_k_array.size() * sizeof(int), cudaMemcpyHostToDevice, stream),
            "grouped L2 H2D d_k_array");

        for (int iter = 0; iter < cfg.max_iters; ++iter) {
            launch_grouped_kmeans_assign(
                buffer.d_data,
                buffer.d_centroids,
                buffer.d_offsets_data,
                buffer.d_offsets_centroids,
                buffer.d_k_array,
                buffer.d_assign,
                total_n,
                num_tasks,
                dim,
                stream);

            cuda_check(
                cudaMemsetAsync(buffer.d_new_centroids, 0, static_cast<size_t>(total_k) * dim * sizeof(float), stream),
                "memset new_centroids");
            cuda_check(
                cudaMemsetAsync(buffer.d_counts, 0, static_cast<size_t>(total_k) * sizeof(int), stream),
                "memset counts");

            launch_grouped_kmeans_accum(
                buffer.d_data,
                buffer.d_assign,
                buffer.d_offsets_data,
                buffer.d_offsets_centroids,
                buffer.d_new_centroids,
                buffer.d_counts,
                total_n,
                num_tasks,
                dim,
                stream);

            launch_grouped_kmeans_divide(
                buffer.d_new_centroids, buffer.d_centroids, buffer.d_counts, total_k, dim, stream);

            std::swap(buffer.d_centroids, buffer.d_new_centroids);
        }

        cuda_check(cudaStreamSynchronize(stream), "grouped L2 stream sync");
        cuda_check(
            cudaMemcpy(h_centroids.data(), buffer.d_centroids, h_centroids.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "grouped L2 D2H centroids");

        for (int t = 0; t < num_tasks; ++t) {
            const int cid = task_cid_map[static_cast<size_t>(t)];
            const int k_i = h_k_array[static_cast<size_t>(t)];
            const int c_off = h_offsets_cents[static_cast<size_t>(t)];
            out_l2_flat[static_cast<size_t>(cid)].resize(static_cast<size_t>(k_i) * dim);
            std::memcpy(
                out_l2_flat[static_cast<size_t>(cid)].data(),
                h_centroids.data() + static_cast<size_t>(c_off) * dim,
                static_cast<size_t>(k_i) * dim * sizeof(float));
        }

        h_data.clear();
        h_centroids.clear();
        h_offsets_data.clear();
        h_offsets_cents.clear();
        h_k_array.clear();
        task_cid_map.clear();
    };

    nvtxRangePushA("Hier:03_grouped_L2_chunks");

    for (int cid = 0; cid < K1; ++cid) {
        if (n_prime[cid] == 0) {
            continue;
        }
        if (n_prime[cid] <= s) {
            continue;
        }
        const int k_i = compute_k_i(cid);
        out_l2_k_i[static_cast<size_t>(cid)] = k_i;
        if (k_i <= 1) {
            continue;
        }

        const int n_sub = n_prime[cid];

        if (static_cast<int>(h_data.size() / static_cast<size_t>(dim)) + n_sub > MAX_CHUNK_N ||
            static_cast<int>(h_centroids.size() / static_cast<size_t>(dim)) + k_i > MAX_CHUNK_K ||
            static_cast<int>(h_k_array.size()) >= MAX_CHUNK_TASKS) {
            flush_chunk();
        }

        h_offsets_data.push_back(static_cast<int>(h_data.size() / static_cast<size_t>(dim)));
        h_offsets_cents.push_back(static_cast<int>(h_centroids.size() / static_cast<size_t>(dim)));
        h_k_array.push_back(k_i);
        task_cid_map.push_back(cid);

        const size_t data_start = h_data.size();
        h_data.resize(data_start + static_cast<size_t>(n_sub) * dim);
        for (int idx = 0; idx < n_sub; ++idx) {
            const int ti = cluster_index_in_train[cid][idx];
            std::memcpy(
                h_data.data() + data_start + static_cast<size_t>(idx) * dim,
                h_train_ptr + static_cast<size_t>(ti) * dim,
                sizeof(float) * dim);
        }

        const size_t cent_start = h_centroids.size();
        h_centroids.resize(cent_start + static_cast<size_t>(k_i) * dim);
        std::mt19937 rng(cfg.seed + static_cast<unsigned int>(cid));
        std::vector<int> indices(static_cast<size_t>(n_sub));
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (int j = 0; j < k_i; ++j) {
            const int sample_idx = indices[static_cast<size_t>(j)];
            const int ti = cluster_index_in_train[cid][sample_idx];
            std::memcpy(
                h_centroids.data() + cent_start + static_cast<size_t>(j) * dim,
                h_train_ptr + static_cast<size_t>(ti) * dim,
                sizeof(float) * dim);
        }
    }

    flush_chunk();
    nvtxRangePop();
    cudaStreamDestroy(stream);
}

}  // namespace detail

/**
 * @param compute_k_i   n_prime[cid]>s ( max(1,·))
 * @param out_l2_k_i    K1; n_prime>0  n>s  k_i
 * @param out_l2_flat   K1; k_i>1  [k_i*dim]
 */
template <typename ComputeKi>
void run_scheduled_level2_kmeans(
    int K1,
    int dim,
    int s,
    const ClusteringConfig& cfg,
    const std::vector<int>& n_prime,
    const std::vector<std::vector<int>>& cluster_index_in_train,
    const float* h_train_ptr,
    ComputeKi compute_k_i,
    std::vector<int>& out_l2_k_i,
    std::vector<std::vector<float>>& out_l2_flat) {
    if (cfg.distance_mode == L2_DISTANCE) {
        detail::run_scheduled_level2_kmeans_grouped_l2(
            K1,
            dim,
            s,
            cfg,
            n_prime,
            cluster_index_in_train,
            h_train_ptr,
            compute_k_i,
            out_l2_k_i,
            out_l2_flat);
    } else {
        detail::run_scheduled_level2_kmeans_workspace_path(
            K1,
            dim,
            s,
            cfg,
            n_prime,
            cluster_index_in_train,
            h_train_ptr,
            compute_k_i,
            out_l2_k_i,
            out_l2_flat);
    }
}

inline void merge_level2_into_final_centroids(
    int K1,
    int dim,
    int s,
    const std::vector<int>& n_prime,
    const std::vector<int>& l2_k_i,
    const std::vector<std::vector<float>>& l2_flat,
    const float* h_centroids1,
    std::vector<float>& final_centroids,
    std::vector<int>& level1_sub_start,
    std::vector<int>& level1_sub_count) {
    for (int cid = 0; cid < K1; ++cid) {
        if (n_prime[cid] == 0) {
            continue;
        }
        const int cur_start = static_cast<int>(final_centroids.size() / dim);
        level1_sub_start[cid] = cur_start;
        const float* centroid_base = h_centroids1 + static_cast<size_t>(cid) * dim;
        if (n_prime[cid] <= s) {
            final_centroids.insert(final_centroids.end(), centroid_base, centroid_base + dim);
            level1_sub_count[cid] = 1;
            continue;
        }
        const int k_i = l2_k_i[cid];
        if (k_i <= 1) {
            final_centroids.insert(final_centroids.end(), centroid_base, centroid_base + dim);
            level1_sub_count[cid] = 1;
            continue;
        }
        const auto& hc = l2_flat[static_cast<size_t>(cid)];
        if (hc.size() != static_cast<size_t>(k_i) * dim) {
            throw std::runtime_error("merge_level2_into_final_centroids: missing L2 kmeans result");
        }
        final_centroids.insert(final_centroids.end(), hc.begin(), hc.end());
        level1_sub_count[cid] = k_i;
    }
}

}  // namespace paraselect::hier_parallel_l2
