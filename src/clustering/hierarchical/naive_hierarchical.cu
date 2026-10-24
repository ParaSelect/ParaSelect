#include "naive_hierarchical.cuh"
#include "hierarchical_parallel_l2.cuh"
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

#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

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

/**
 *  Kernel(L2 )
 *
 * , d_assign_l1[i],
 *  [sub_start, sub_start+sub_count) .
 *
 * @param d_data         [curB, dim]( batch)
 * @param d_centroids    [final_k, dim]
 * @param d_cnorm2       L2  [final_k]
 * @param d_xnorm2       batch  L2  [curB]
 * @param d_assign_l1    batch  [curB]( [0, K1))
 * @param d_sub_start    ->  [K1]
 * @param d_sub_count    ->  [K1]
 * @param d_assign_out  : batch  [curB]
 * @param curB           batch
 * @param dim
 */
__global__ void kernel_naive_hierarchical_assign_l2(
    const float* __restrict__ d_data,       // [curB, dim]
    const float* __restrict__ d_centroids,  // [final_k, dim]
    const float* __restrict__ d_cnorm2,     // [final_k]
    const float* __restrict__ d_xnorm2,     // [curB]
    const idx_t*   __restrict__ d_assign_l1,  // [curB]
    const int*   __restrict__ d_sub_start,  // [K1]
    const int*   __restrict__ d_sub_count,  // [K1]
    int*         __restrict__ d_assign_out, // [curB]
    int curB, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= curB) return;

    int l1 = d_assign_l1[i];
    int sub_start = d_sub_start[l1];
    int sub_count = d_sub_count[l1];

    const float* xi = d_data + (size_t)i * dim;
    float xi_norm2 = d_xnorm2[i];

    float best_dist = 1e38f;
    int   best_idx  = sub_start;

    for (int j = 0; j < sub_count; ++j) {
        int cj = sub_start + j;
        const float* cj_ptr = d_centroids + (size_t)cj * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += xi[d] * cj_ptr[d];
        }
        // dist2 = ||x||^2 + ||c||^2 - 2*dot
        float dist2 = xi_norm2 + d_cnorm2[cj] - 2.0f * dot;
        if (dist2 < best_dist) {
            best_dist = dist2;
            best_idx  = cj;
        }
    }
    d_assign_out[i] = best_idx;
}

/** Nsight Systems  NVTX ( Hier:xx ) */
struct HierNvtxRange {
    explicit HierNvtxRange(const char* name) { nvtxRangePushA(name); }
    ~HierNvtxRange() { nvtxRangePop(); }
    HierNvtxRange(const HierNvtxRange&) = delete;
    HierNvtxRange& operator=(const HierNvtxRange&) = delete;
};

} // namespace

NaiveHierarchicalClustering::NaiveHierarchicalClustering(float alpha)
    : alpha_(alpha), fitted_(false), h_centroids_(nullptr), h_reordered_data_(nullptr), h_reordered_indices_(nullptr) {
}

NaiveHierarchicalClustering::~NaiveHierarchicalClustering() {
    release();
}

void NaiveHierarchicalClustering::release() {
    if (!fitted_) return;

    free_result_buffers(
        h_reordered_data_,
        h_reordered_indices_,
        h_centroids_,
        &cluster_info_
    );

    result_ = ClusteringResult{};
    cluster_info_ = ClusterInfo{};
    h_centroids_ = nullptr;
    h_reordered_data_ = nullptr;
    h_reordered_indices_ = nullptr;
    fitted_ = false;
}

void NaiveHierarchicalClustering::fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) {
    if (fitted_) {
        release();
    }

    if (!h_data || n <= 0 || dim <= 0 || cfg.n_clusters <= 0) {
        throw std::invalid_argument("NaiveHierarchicalClustering::fit invalid arguments");
    }

    // --- : n_train  ---
    int n_train = std::max(1, (int)std::round(alpha_ * (double)n));
    if (n_train > n) n_train = n;

    // ,
    const float* h_train_ptr = h_data;
    // ---  ---

    int M = cfg.n_clusters;
    int K1 = std::max(2, (int)std::ceil(std::sqrt((double)M)));
    int s = std::max(1, (int)std::ceil((double)n_train / M));

    HierNvtxRange nvtx_fit("Hier:00_fit_total");

    //
    std::vector<float> h_centroids1((size_t)K1 * dim);
    std::vector<int> h_assign1(n_train);
    {
        HierNvtxRange nvtx_l1("Hier:01_level1_kmeans");
        KMeansCase cfg1;
        cfg1.n = n_train;
        cfg1.dim = dim;
        cfg1.k = K1;
        cfg1.iters = cfg.max_iters;
        cfg1.minibatch_iters = cfg.max_iters * 4;
        cfg1.seed = cfg.seed;
        cfg1.dist = cfg.distance_mode;
        cfg1.precision = cfg.compute_precision;
        cfg1.use_flash_assign = kmeans_uses_flash_assign(cfg.assign_operator);
        cfg1.init_policy = KMEANS_INIT_FORCE_RANDOM;

        init_centroids_by_sampling(cfg1, h_train_ptr, h_centroids1.data());

        float* d_centroids1 = nullptr;
        idx_t* d_assign1 = nullptr;
        cudaMalloc(&d_centroids1, sizeof(float) * (size_t)K1 * dim);
        cudaMalloc(&d_assign1, sizeof(idx_t) * (size_t)n_train);
        cudaMemcpy(d_centroids1, h_centroids1.data(), sizeof(float) * (size_t)K1 * dim, cudaMemcpyHostToDevice);

        float obj1 = 0.0f;
        gpu_kmeans_lloyd(cfg1, h_train_ptr, d_assign1, d_centroids1, &obj1);

        cudaMemcpy(h_assign1.data(), d_assign1, sizeof(idx_t) * (size_t)n_train, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_centroids1.data(), d_centroids1, sizeof(float) * (size_t)K1 * dim, cudaMemcpyDeviceToHost);

        cudaFree(d_assign1);
        cudaFree(d_centroids1);
    }

    // :n'_i
    std::vector<int> n_prime(K1, 0);
    std::vector<std::vector<int>> cluster_index_in_train(K1);
    {
        HierNvtxRange nvtx_l1stats("Hier:02_build_l1_cluster_stats");
        for (int i = 0; i < n_train; ++i) {
            int cid = h_assign1[i];
            if (cid < 0 || cid >= K1) {
                throw std::runtime_error("NaiveHierarchicalClustering: invalid cluster id");
            }
            n_prime[cid]++;
            cluster_index_in_train[cid].push_back(i);
        }
    }

    // : -- k_i  cid  M,K1 , n'_i
    //  final_centroids
    std::vector<float> final_centroids;
    final_centroids.reserve((size_t)M * dim * 2);

    // level1_sub_start[cid] / level1_sub_count[cid]: cid
    std::vector<int> level1_sub_start(K1, -1);
    std::vector<int> level1_sub_count(K1, 0);

    {
        HierNvtxRange nvtx_l2("Hier:03_level2_kmeans_parallel");
        std::vector<int> l2_k_i;
        std::vector<std::vector<float>> l2_flat;
        auto naive_k_i = [&](int cid) {
            int base_m = M / K1;
            int rem_m = M % K1;
            int k_i = base_m + (cid < rem_m ? 1 : 0);
            return std::max(1, k_i);
        };
        paraselect::hier_parallel_l2::run_scheduled_level2_kmeans(
            K1,
            dim,
            s,
            cfg,
            n_prime,
            cluster_index_in_train,
            h_train_ptr,
            naive_k_i,
            l2_k_i,
            l2_flat);
        paraselect::hier_parallel_l2::merge_level2_into_final_centroids(
            K1,
            dim,
            s,
            n_prime,
            l2_k_i,
            l2_flat,
            h_centroids1.data(),
            final_centroids,
            level1_sub_start,
            level1_sub_count);
    }

    int final_k = (int)(final_centroids.size() / dim);
    if (final_k <= 0) {
        throw std::runtime_error("NaiveHierarchicalClustering: no final clusters generated");
    }

    // ---------------------------------------------------------------
    // :( O(N*M) )
    //
    //  A: K1 , d_assign_l1
    //         :K1( M)
    //  B:,
    //         :M/K1
    // :O(N*(K1 + M/K1)), O(N*M)
    // ---------------------------------------------------------------

    std::vector<int> h_assign_global(n);

    {
        HierNvtxRange nvtx_p4("Hier:04_gpu_full_n_assign");

        cudaStream_t stream;
        cudaStreamCreate(&stream);

    //  GPU
    float* d_centroids1_gpu = nullptr;
    float* d_centroids_final = nullptr;
    float* d_cnorm2_l1 = nullptr;   //  [K1]
    float* d_cnorm2_final = nullptr; //  [final_k]
    idx_t*   d_assign_l1 = nullptr;   //  [n](, batch )
    int*   d_assign_global = nullptr; //  [n]
    int*   d_sub_start_gpu = nullptr; //  ->  [K1]
    int*   d_sub_count_gpu = nullptr; //  ->  [K1]

    cudaMalloc(&d_centroids1_gpu,  sizeof(float) * (size_t)K1 * dim);
    cudaMalloc(&d_centroids_final, sizeof(float) * (size_t)final_k * dim);
    cudaMalloc(&d_cnorm2_l1,       sizeof(float) * (size_t)K1);
    cudaMalloc(&d_cnorm2_final,    sizeof(float) * (size_t)final_k);
    cudaMalloc(&d_assign_l1,       sizeof(idx_t)   * (size_t)n);
    cudaMalloc(&d_assign_global,   sizeof(int)   * (size_t)n);
    cudaMalloc(&d_sub_start_gpu,   sizeof(int)   * (size_t)K1);
    cudaMalloc(&d_sub_count_gpu,   sizeof(int)   * (size_t)K1);

    cudaMemcpyAsync(d_centroids1_gpu,  h_centroids1.data(),      sizeof(float) * (size_t)K1 * dim,    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_centroids_final, final_centroids.data(),   sizeof(float) * (size_t)final_k * dim, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sub_start_gpu,   level1_sub_start.data(),  sizeof(int)   * (size_t)K1,           cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sub_count_gpu,   level1_sub_count.data(),  sizeof(int)   * (size_t)K1,           cudaMemcpyHostToDevice, stream);

    //
    compute_l2_squared_gpu(d_centroids1_gpu,  d_cnorm2_l1,    K1,      dim, L2NORM_AUTO, stream);
    compute_l2_squared_gpu(d_centroids_final, d_cnorm2_final, final_k, dim, L2NORM_AUTO, stream);
    cudaStreamSynchronize(stream);

    // ---  A:(Block , perform_assignment_only)---
    {
        HierNvtxRange nvtx_p4a("Hier:04a_full_n_l1_assign");
        int B = 1 << 15;
        int Ktile = 4096;
        StreamEnv env_l1;
        env_l1.allocate(B, Ktile);
        float* d_data_buf = nullptr;
        cudaMalloc(&d_data_buf, sizeof(float) * (size_t)B * dim);

        cublasHandle_t cublas_handle;
        cublas_check(cublasCreate(&cublas_handle), "cublasCreate L1");
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode L1");

        KMeansCase cfg_l1;
        cfg_l1.n    = n;
        cfg_l1.dim  = dim;
        cfg_l1.k    = K1;
        cfg_l1.iters = 1;
        cfg_l1.minibatch_iters = 1;
        cfg_l1.seed = cfg.seed;
        cfg_l1.dist = cfg.distance_mode;
        cfg_l1.precision = cfg.compute_precision;
        cfg_l1.use_flash_assign = kmeans_uses_flash_assign(cfg.assign_operator);

        perform_assignment_only(
            cfg_l1,
            h_data,
            d_assign_l1,
            d_centroids1_gpu,
            d_cnorm2_l1,
            env_l1,
            d_data_buf,
            stream,
            cublas_handle,
            B, Ktile, n, dim, K1
        );
        cudaStreamSynchronize(stream);

        env_l1.free();
        cudaFree(d_data_buf);
        cublasDestroy(cublas_handle);
    }

    // ---  B:(Block , kernel)---
    {
        HierNvtxRange nvtx_p4b("Hier:04b_full_n_l2_assign_batches");
        int B = 1 << 15;
        float* d_data_buf  = nullptr;
        float* d_xnorm2_buf = nullptr;
        idx_t*   d_assign_l1_buf = nullptr; //  batch

        cudaMalloc(&d_data_buf,      sizeof(float) * (size_t)B * dim);
        cudaMalloc(&d_xnorm2_buf,    sizeof(float) * (size_t)B);
        cudaMalloc(&d_assign_l1_buf, sizeof(idx_t)   * (size_t)B);

        int threads = 256;
        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);

            //  batch
            cudaMemcpyAsync(d_data_buf,
                            h_data + (size_t)base * dim,
                            sizeof(float) * (size_t)curB * dim,
                            cudaMemcpyHostToDevice, stream);

            //  batch ( A  d_assign_l1)
            cudaMemcpyAsync(d_assign_l1_buf,
                            d_assign_l1 + base,
                            sizeof(idx_t) * (size_t)curB,
                            cudaMemcpyDeviceToDevice, stream);

            //  batch
            {
                compute_l2_squared_gpu(d_data_buf, d_xnorm2_buf, curB, dim, L2NORM_AUTO, stream);
            }

            //
            {
                int blk = (curB + threads - 1) / threads;
                kernel_naive_hierarchical_assign_l2<<<blk, threads, 0, stream>>>(
                    d_data_buf,
                    d_centroids_final,
                    d_cnorm2_final,
                    d_xnorm2_buf,
                    d_assign_l1_buf,
                    d_sub_start_gpu,
                    d_sub_count_gpu,
                    d_assign_global + base,
                    curB, dim
                );
            }
        }
        cudaStreamSynchronize(stream);

        cudaFree(d_data_buf);
        cudaFree(d_xnorm2_buf);
        cudaFree(d_assign_l1_buf);
    }

        cudaMemcpy(h_assign_global.data(), d_assign_global, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost);

    //  GPU
    cudaStreamDestroy(stream);
    cudaFree(d_centroids1_gpu);
    cudaFree(d_centroids_final);
    cudaFree(d_cnorm2_l1);
    cudaFree(d_cnorm2_final);
    cudaFree(d_assign_l1);
    cudaFree(d_assign_global);
    cudaFree(d_sub_start_gpu);
    cudaFree(d_sub_count_gpu);

    }

    //  cluster counts + offsets
    {
        HierNvtxRange nvtx_p5("Hier:05_cpu_finalize_reorder");
        std::vector<int> final_counts(final_k, 0);
    for (int i = 0; i < n; ++i) {
        int cid = h_assign_global[i];
        if (cid < 0 || cid >= final_k) {
            throw std::runtime_error("NaiveHierarchicalClustering: invalid final assignment id");
        }
        final_counts[cid]++;
    }

    //
    {
        int empty_count = 0;
        for (int i = 0; i < final_k; ++i) {
            if (final_counts[i] == 0) ++empty_count;
        }
        fprintf(stderr, "[NaiveHierarchicalClustering] final_k=%d, empty clusters=%d (%.2f%%)\n",
                final_k, empty_count, 100.0f * empty_count / final_k);
    }

    std::vector<long long> offsets((size_t)final_k + 1u, 0);
    long long accum = 0;
    for (int i = 0; i < final_k; ++i) {
        offsets[i] = accum;
        accum += final_counts[i];
    }
    offsets[(size_t)final_k] = accum;

    //  permutation :perm[dst_pos] = src_idx
    std::vector<idx_t> perm(n);
    {
        std::vector<idx_t> write_ptr(final_k);
        for (int i = 0; i < final_k; ++i) write_ptr[i] = (idx_t)offsets[i];
        for (int i = 0; i < n; ++i) {
            int cid = h_assign_global[i];
            perm[write_ptr[cid]++] = i;
        }
    }

    // ( in-place )
    size_t data_size = (size_t)n * dim;
    h_reordered_data_ = (float*)std::malloc(data_size * sizeof(float));
    h_reordered_indices_ = (idx_t*)std::malloc((size_t)n * sizeof(idx_t));
    if (!h_reordered_data_ || !h_reordered_indices_) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, nullptr, nullptr);
        throw std::bad_alloc();
    }

    //
    std::memcpy(h_reordered_data_, h_data, data_size * sizeof(float));

    //  [0, 1, 2, ..., n-1]
    for (int i = 0; i < n; ++i) {
        h_reordered_indices_[i] = i;
    }

    //  in-place
    cpu_reorder_vectors_inplace(h_reordered_data_, perm.data(), n, dim);
    cpu_reorder_indices_inplace((int*)h_reordered_indices_, perm.data(), n);

    //
    h_centroids_ = (float*)std::malloc((size_t)final_k * dim * sizeof(float));
    if (!h_centroids_) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, nullptr, nullptr);
        throw std::bad_alloc();
    }
    std::memcpy(h_centroids_, final_centroids.data(), sizeof(float) * (size_t)final_k * dim);

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
}

ClusteringResult NaiveHierarchicalClustering::get_result() {
    return result_;
}

float* NaiveHierarchicalClustering::release_reordered_data() {
    if (!fitted_ || !result_.reordered_data_owned) {
        return nullptr;
    }
    float* ptr = h_reordered_data_;
    h_reordered_data_ = nullptr;
    result_.reordered_data = nullptr;
    return ptr;
}
