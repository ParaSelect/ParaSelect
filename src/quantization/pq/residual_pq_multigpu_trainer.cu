#include "quantization/pq/residual_pq_trainer.cuh"

#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace paraselect {
namespace quantization {
namespace pq {
namespace {

struct ResidualPQGpuTrainShard {
    int gpu = 0;
    int start = 0;
    int count = 0;
    int batch_size = 0;
    int d_sub = 0;

    float* d_data = nullptr;
    float* d_residual = nullptr;
    float* d_coarse_centroids = nullptr;
    idx_t* d_cluster_offsets = nullptr;
    int* d_cluster_ids = nullptr;
    float* d_centroids = nullptr;
    float* d_accum = nullptr;
    int* d_counts = nullptr;
    float* d_xnorm2 = nullptr;
    float* d_cnorm2 = nullptr;
    float* d_best_dist2 = nullptr;
    float* d_dot = nullptr;
    idx_t* d_assign = nullptr;
    cublasHandle_t handle = nullptr;

    std::vector<float> h_accum;
    std::vector<int> h_counts;
};

void free_shard(ResidualPQGpuTrainShard& shard) {
    cudaSetDevice(shard.gpu);
    if (shard.handle) {
        cublasDestroy(shard.handle);
        shard.handle = nullptr;
    }
    if (shard.d_assign) cudaFree(shard.d_assign);
    if (shard.d_dot) cudaFree(shard.d_dot);
    if (shard.d_best_dist2) cudaFree(shard.d_best_dist2);
    if (shard.d_cnorm2) cudaFree(shard.d_cnorm2);
    if (shard.d_xnorm2) cudaFree(shard.d_xnorm2);
    if (shard.d_counts) cudaFree(shard.d_counts);
    if (shard.d_accum) cudaFree(shard.d_accum);
    if (shard.d_centroids) cudaFree(shard.d_centroids);
    if (shard.d_cluster_ids) cudaFree(shard.d_cluster_ids);
    if (shard.d_cluster_offsets) cudaFree(shard.d_cluster_offsets);
    if (shard.d_coarse_centroids) cudaFree(shard.d_coarse_centroids);
    if (shard.d_residual) cudaFree(shard.d_residual);
    if (shard.d_data) cudaFree(shard.d_data);
    shard = ResidualPQGpuTrainShard{};
}

bool allocate_shard(
    ResidualPQGpuTrainShard& shard,
    const ResidualPQCase& cfg,
    const ResidualPQInput& input,
    const float* h_centroids)
{
    cudaSetDevice(shard.gpu);
    const int d_sub = shard.d_sub;
    const int B = shard.batch_size;
    cudaError_t err = cudaMalloc(&shard.d_data, sizeof(float) * (size_t)B * (size_t)cfg.dim);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_residual, sizeof(float) * (size_t)B * (size_t)cfg.dim);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_coarse_centroids, sizeof(float) * (size_t)cfg.nlist * (size_t)cfg.dim);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_cluster_offsets, sizeof(idx_t) * (size_t)(cfg.nlist + 1));
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_cluster_ids, sizeof(int) * (size_t)B);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_centroids, sizeof(float) * (size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_accum, sizeof(float) * (size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_counts, sizeof(int) * (size_t)cfg.m * (size_t)cfg.k);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_xnorm2, sizeof(float) * (size_t)B);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_cnorm2, sizeof(float) * (size_t)cfg.k);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_best_dist2, sizeof(float) * (size_t)B);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_dot, sizeof(float) * (size_t)B * (size_t)cfg.k);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_assign, sizeof(idx_t) * (size_t)B);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] GPU%d allocation failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        free_shard(shard);
        return false;
    }

    err = cudaMemcpy(shard.d_coarse_centroids,
                     input.coarse_centroids,
                     sizeof(float) * (size_t)cfg.nlist * (size_t)cfg.dim,
                     cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        err = cudaMemcpy(shard.d_cluster_offsets,
                         input.cluster_offsets,
                         sizeof(idx_t) * (size_t)(cfg.nlist + 1),
                         cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(shard.d_centroids,
                         h_centroids,
                         sizeof(float) * (size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub,
                         cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] GPU%d initial copy failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        free_shard(shard);
        return false;
    }

    cublasStatus_t st = cublasCreate(&shard.handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] GPU%d cublasCreate failed\n", shard.gpu);
        free_shard(shard);
        return false;
    }
    cublasSetMathMode(shard.handle, CUBLAS_TF32_TENSOR_OP_MATH);

    shard.h_accum.assign((size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub, 0.0f);
    shard.h_counts.assign((size_t)cfg.m * (size_t)cfg.k, 0);
    return true;
}

bool run_shard_iteration(
    ResidualPQGpuTrainShard& shard,
    const ResidualPQCase& cfg,
    const ResidualPQInput& input,
    const float* h_centroids)
{
    if (shard.count <= 0) return true;
    cudaSetDevice(shard.gpu);

    const int d_sub = shard.d_sub;
    const int B = shard.batch_size;
    const int threads = 256;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaError_t err = cudaMemcpy(shard.d_centroids,
                                 h_centroids,
                                 sizeof(float) * (size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub,
                                 cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        err = cudaMemset(shard.d_accum, 0, sizeof(float) * (size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub);
    }
    if (err == cudaSuccess) {
        err = cudaMemset(shard.d_counts, 0, sizeof(int) * (size_t)cfg.m * (size_t)cfg.k);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] GPU%d iteration init failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        return false;
    }

    for (int local_base = 0; local_base < shard.count; local_base += B) {
        const int curB = std::min(B, shard.count - local_base);
        const int global_base = shard.start + local_base;
        err = cudaMemcpy(shard.d_data,
                         input.data + (size_t)global_base * (size_t)cfg.dim,
                         sizeof(float) * (size_t)curB * (size_t)cfg.dim,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) break;

        detail::build_cluster_ids_from_offsets_kernel<<<(curB + threads - 1) / threads, threads>>>(
            shard.d_cluster_offsets,
            cfg.nlist,
            global_base,
            curB,
            shard.d_cluster_ids);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;

        detail::build_full_residual_kernel<<<(curB * cfg.dim + threads - 1) / threads, threads>>>(
            shard.d_data,
            shard.d_coarse_centroids,
            shard.d_cluster_ids,
            curB,
            cfg.dim,
            shard.d_residual);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;

        for (int subspace = 0; subspace < cfg.m; ++subspace) {
            const int offset = subspace * d_sub;
            float* d_centroids_m = shard.d_centroids + (size_t)subspace * (size_t)cfg.k * (size_t)d_sub;
            float* d_accum_m = shard.d_accum + (size_t)subspace * (size_t)cfg.k * (size_t)d_sub;
            int* d_counts_m = shard.d_counts + (size_t)subspace * (size_t)cfg.k;

            detail::compute_strided_l2_norm_kernel<<<curB, 256>>>(
                shard.d_residual, curB, cfg.dim, offset, d_sub, shard.d_xnorm2);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;

            detail::compute_centroid_l2_norm_kernel<<<cfg.k, 256>>>(
                d_centroids_m, cfg.k, d_sub, shard.d_cnorm2);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;

            detail::init_best_kernel<<<(curB + threads - 1) / threads, threads>>>(
                shard.d_best_dist2, shard.d_assign, curB);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;

            cublasStatus_t st = cublasSgemm(
                shard.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                curB,
                cfg.k,
                d_sub,
                &alpha,
                shard.d_residual + offset,
                cfg.dim,
                d_centroids_m,
                d_sub,
                &beta,
                shard.d_dot,
                curB);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] GPU%d cublasSgemm failed: %d\n",
                             shard.gpu, (int)st);
                return false;
            }

            detail::update_best_from_dot_kernel<<<(curB + threads - 1) / threads, threads>>>(
                shard.d_dot,
                shard.d_xnorm2,
                shard.d_cnorm2,
                curB,
                cfg.k,
                shard.d_assign,
                shard.d_best_dist2);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;

            detail::accum_strided_subspace_kernel<<<(curB * d_sub + threads - 1) / threads, threads>>>(
                shard.d_residual,
                shard.d_assign,
                curB,
                cfg.dim,
                offset,
                d_sub,
                cfg.k,
                d_accum_m,
                d_counts_m);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;
        }
        if (err != cudaSuccess) break;
    }

    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(shard.h_accum.data(),
                         shard.d_accum,
                         sizeof(float) * shard.h_accum.size(),
                         cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(shard.h_counts.data(),
                         shard.d_counts,
                         sizeof(int) * shard.h_counts.size(),
                         cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] GPU%d iteration failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        return false;
    }
    return true;
}

void update_centroids_host(
    const std::vector<float>& accum,
    const std::vector<int>& counts,
    const ResidualPQCase& cfg,
    int d_sub,
    std::vector<float>& centroids)
{
    for (int subspace = 0; subspace < cfg.m; ++subspace) {
        for (int c = 0; c < cfg.k; ++c) {
            const size_t count_idx = (size_t)subspace * (size_t)cfg.k + (size_t)c;
            const int cnt = counts[count_idx];
            if (cnt <= 0) continue;
            const float inv = 1.0f / (float)cnt;
            const size_t base = ((size_t)subspace * (size_t)cfg.k + (size_t)c) * (size_t)d_sub;
            for (int d = 0; d < d_sub; ++d) {
                centroids[base + (size_t)d] = accum[base + (size_t)d] * inv;
            }
        }
    }
}

}  // namespace

void train_residual_pq_codebook_multigpu(
    const ResidualPQCase& cfg,
    const ResidualPQInput& input,
    float* codebook)
{
    const int d_sub = cfg.dim / cfg.m;
    const int batch_size = 1 << 15;
    const int n_gpus = detail::choose_pq_gpu_count(cfg.max_gpus);
    if (n_gpus <= 1) {
        throw std::runtime_error("ResidualPQTrainerGPU multi-GPU requested but only one GPU is available");
    }

    std::fprintf(stderr,
                 "[ResidualPQTrainerGPU:multi] fit begin n=%d dim=%d nlist=%d M=%d K=%d d_sub=%d iters=%d batch_size=%d gpus=%d\n",
                 cfg.n, cfg.dim, cfg.nlist, cfg.m, cfg.k, d_sub, cfg.iters, batch_size, n_gpus);
    std::fflush(stderr);

    std::vector<float> h_centroids((size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub);
    for (int subspace = 0; subspace < cfg.m; ++subspace) {
        std::mt19937 rng((unsigned)(42 + subspace));
        std::uniform_int_distribution<int> dist(0, cfg.n - 1);
        for (int k = 0; k < cfg.k; ++k) {
            const int ridx = dist(rng);
            int cid = (int)(std::upper_bound(
                input.cluster_offsets,
                input.cluster_offsets + cfg.nlist + 1,
                (idx_t)ridx) - input.cluster_offsets) - 1;
            if (cid < 0) cid = 0;
            if (cid >= cfg.nlist) cid = cfg.nlist - 1;
            const float* vec = input.data + (size_t)ridx * (size_t)cfg.dim + (size_t)subspace * (size_t)d_sub;
            const float* cen = input.coarse_centroids + (size_t)cid * (size_t)cfg.dim + (size_t)subspace * (size_t)d_sub;
            float* dst = h_centroids.data() + ((size_t)subspace * (size_t)cfg.k + (size_t)k) * (size_t)d_sub;
            for (int d = 0; d < d_sub; ++d) dst[d] = vec[d] - cen[d];
        }
    }

    std::vector<ResidualPQGpuTrainShard> shards((size_t)n_gpus);
    bool allocated = true;
    for (int g = 0; g < n_gpus; ++g) {
        auto& shard = shards[(size_t)g];
        shard.gpu = g;
        shard.start = (int)(((int64_t)cfg.n * g) / n_gpus);
        const int end = (int)(((int64_t)cfg.n * (g + 1)) / n_gpus);
        shard.count = end - shard.start;
        shard.batch_size = batch_size;
        shard.d_sub = d_sub;
        allocated = allocate_shard(shard, cfg, input, h_centroids.data()) && allocated;
    }
    if (!allocated) {
        for (auto& shard : shards) free_shard(shard);
        throw std::runtime_error("ResidualPQTrainerGPU multi-GPU allocation failed");
    }

    std::vector<float> total_accum((size_t)cfg.m * (size_t)cfg.k * (size_t)d_sub, 0.0f);
    std::vector<int> total_counts((size_t)cfg.m * (size_t)cfg.k, 0);

    bool ok = true;
    std::mutex worker_mutex;
    std::condition_variable worker_cv;
    std::condition_variable done_cv;
    int requested_iter = -1;
    int completed_workers = 0;
    bool stop_workers = false;
    std::atomic<bool> iter_ok(true);

    std::vector<std::thread> workers;
    workers.reserve((size_t)n_gpus);
    for (int g = 0; g < n_gpus; ++g) {
        workers.emplace_back([&, g]() {
            int last_seen_iter = -1;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(worker_mutex);
                    worker_cv.wait(lock, [&]() { return stop_workers || requested_iter != last_seen_iter; });
                    if (stop_workers) return;
                    last_seen_iter = requested_iter;
                }

                bool shard_ok = run_shard_iteration(shards[(size_t)g], cfg, input, h_centroids.data());
                if (!shard_ok) iter_ok.store(false);

                {
                    std::lock_guard<std::mutex> lock(worker_mutex);
                    completed_workers += 1;
                }
                done_cv.notify_one();
            }
        });
    }

    for (int it = 0; it < cfg.iters && ok; ++it) {
        const auto iter_t0 = std::chrono::high_resolution_clock::now();
        iter_ok.store(true);
        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            completed_workers = 0;
            requested_iter = it;
        }
        worker_cv.notify_all();

        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            done_cv.wait(lock, [&]() { return completed_workers == n_gpus; });
        }
        if (!iter_ok.load()) {
            ok = false;
            break;
        }

        std::fill(total_accum.begin(), total_accum.end(), 0.0f);
        std::fill(total_counts.begin(), total_counts.end(), 0);
        for (const auto& shard : shards) {
            for (size_t i = 0; i < total_accum.size(); ++i) total_accum[i] += shard.h_accum[i];
            for (size_t i = 0; i < total_counts.size(); ++i) total_counts[i] += shard.h_counts[i];
        }
        update_centroids_host(total_accum, total_counts, cfg, d_sub, h_centroids);

        const auto iter_t1 = std::chrono::high_resolution_clock::now();
        const double iter_ms = std::chrono::duration<double, std::milli>(iter_t1 - iter_t0).count();
        std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] iter %d/%d done ms=%.3f\n",
                     it + 1, cfg.iters, iter_ms);
        std::fflush(stderr);
    }

    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        stop_workers = true;
    }
    worker_cv.notify_all();
    for (auto& worker : workers) worker.join();

    for (auto& shard : shards) free_shard(shard);
    cudaSetDevice(cfg.device_id);

    if (!ok) {
        throw std::runtime_error("ResidualPQTrainerGPU multi-GPU iteration failed");
    }
    std::copy(h_centroids.begin(), h_centroids.end(), codebook);
    std::fprintf(stderr, "[ResidualPQTrainerGPU:multi] fit end\n");
    std::fflush(stderr);
}

}  // namespace pq
}  // namespace quantization
}  // namespace paraselect
