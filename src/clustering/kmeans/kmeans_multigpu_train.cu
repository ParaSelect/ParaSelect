#include "clustering/kmeans/kmeans_multigpu_train.cuh"

#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

void kmeans_update_centroids_host(
    const std::vector<float>& accum,
    const std::vector<int>& counts,
    int k,
    int dim,
    std::vector<float>& centroids) {
    for (int c = 0; c < k; ++c) {
        const int cnt = counts[static_cast<size_t>(c)];
        if (cnt <= 0) continue;
        const float inv = 1.0f / static_cast<float>(cnt);
        const float* src = accum.data() + static_cast<size_t>(c) * dim;
        float* dst = centroids.data() + static_cast<size_t>(c) * dim;
        for (int d = 0; d < dim; ++d) dst[d] = src[d] * inv;
    }
}

cudaError_t kmeans_accumulate_host_vectors(
    const float* data, int n, int dim, int batch, const idx_t* assignments,
    float* data_buffer, float* sums, int* counts) {
    for (int base = 0; base < n; base += batch) {
        const int rows = std::min(batch, n - base);
        auto error = cudaMemcpy(data_buffer, data + static_cast<size_t>(base)*dim,
            static_cast<size_t>(rows)*dim*sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) return error;
        kernel_accum_from_assign<<<(rows+255)/256,256>>>(
            data_buffer, rows, dim, assignments+base, sums, counts);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

namespace {

int choose_gpu_count(int requested_max) {
    int n_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&n_gpus);
    if (err != cudaSuccess || n_gpus <= 0) return 1;
    if (requested_max > 0) n_gpus = std::min(n_gpus, requested_max);
    return std::max(1, n_gpus);
}

struct MultiGpuTrainShard {
    int gpu = 0;
    int start = 0;
    int count = 0;
    int dim = 0;
    int k = 0;
    int batch_size = 0;
    int centroid_tile = 0;

    float* d_centroids = nullptr;
    float* d_cnorm2 = nullptr;
    float* d_data_buf = nullptr;
    float* d_accum = nullptr;
    int* d_counts = nullptr;
    idx_t* d_assign = nullptr;
    float* d_objective_sum = nullptr;
    StreamEnv env;
    cublasHandle_t handle = nullptr;

    std::vector<float> h_accum;
    std::vector<int> h_counts;
    double h_objective = 0.0;
};

void free_shard(MultiGpuTrainShard& shard) {
    cudaSetDevice(shard.gpu);
    if (shard.handle) {
        cublasDestroy(shard.handle);
        shard.handle = nullptr;
    }
    shard.env.free();
    if (shard.d_centroids) cudaFree(shard.d_centroids);
    if (shard.d_cnorm2) cudaFree(shard.d_cnorm2);
    if (shard.d_data_buf) cudaFree(shard.d_data_buf);
    if (shard.d_accum) cudaFree(shard.d_accum);
    if (shard.d_counts) cudaFree(shard.d_counts);
    if (shard.d_assign) cudaFree(shard.d_assign);
    if (shard.d_objective_sum) cudaFree(shard.d_objective_sum);
    shard.d_centroids = nullptr;
    shard.d_cnorm2 = nullptr;
    shard.d_data_buf = nullptr;
    shard.d_accum = nullptr;
    shard.d_counts = nullptr;
    shard.d_assign = nullptr;
    shard.d_objective_sum = nullptr;
}

bool allocate_shard(MultiGpuTrainShard& shard, bool collect_objective) {
    cudaSetDevice(shard.gpu);
    cudaError_t err = cudaMalloc(&shard.d_centroids, sizeof(float) * static_cast<size_t>(shard.k) * shard.dim);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_cnorm2, sizeof(float) * static_cast<size_t>(shard.k));
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_data_buf, sizeof(float) * static_cast<size_t>(shard.batch_size) * shard.dim);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_accum, sizeof(float) * static_cast<size_t>(shard.k) * shard.dim);
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_counts, sizeof(int) * static_cast<size_t>(shard.k));
    if (err == cudaSuccess) err = cudaMalloc(&shard.d_assign, sizeof(idx_t) * static_cast<size_t>(shard.count));
    if (collect_objective && err == cudaSuccess) err = cudaMalloc(&shard.d_objective_sum, sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[kmeans_multigpu_train] GPU%d allocation failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        free_shard(shard);
        return false;
    }

    shard.env.allocate(shard.batch_size, shard.centroid_tile);
    cublasStatus_t st = cublasCreate(&shard.handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[kmeans_multigpu_train] GPU%d cublasCreate failed\n", shard.gpu);
        free_shard(shard);
        return false;
    }
    cublasSetMathMode(shard.handle, CUBLAS_TF32_TENSOR_OP_MATH);

    shard.h_accum.assign(static_cast<size_t>(shard.k) * shard.dim, 0.0f);
    shard.h_counts.assign(static_cast<size_t>(shard.k), 0);
    return true;
}

bool run_shard_iteration(
    MultiGpuTrainShard& shard,
    const KMeansCase& base_cfg,
    const float* h_data,
    const float* h_centroids,
    bool collect_objective) {
    if (shard.count <= 0) return true;
    cudaSetDevice(shard.gpu);

    const int dim = base_cfg.dim;
    const int k = base_cfg.k;
    const int B = shard.batch_size;

    cudaError_t err = cudaMemcpy(shard.d_centroids,
                                 h_centroids,
                                 sizeof(float) * static_cast<size_t>(k) * dim,
                                 cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(shard.d_accum, 0, sizeof(float) * static_cast<size_t>(k) * dim);
    if (err == cudaSuccess) err = cudaMemset(shard.d_counts, 0, sizeof(int) * static_cast<size_t>(k));
    if (collect_objective && err == cudaSuccess) err = cudaMemset(shard.d_objective_sum, 0, sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[kmeans_multigpu_train] GPU%d iteration init failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        return false;
    }

    KMeansCase shard_cfg = base_cfg;
    shard_cfg.n = shard.count;
    perform_assignment_only(
        shard_cfg,
        h_data + static_cast<size_t>(shard.start) * dim,
        shard.d_assign,
        shard.d_centroids,
        shard.d_cnorm2,
        shard.env,
        shard.d_data_buf,
        0,
        shard.handle,
        B,
        shard.centroid_tile,
        shard.count,
        dim,
        k,
        collect_objective ? shard.d_objective_sum : nullptr);

    err = kmeans_accumulate_host_vectors(
        h_data + static_cast<size_t>(shard.start) * dim, shard.count, dim, B,
        shard.d_assign, shard.d_data_buf, shard.d_accum, shard.d_counts);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(shard.h_accum.data(),
                         shard.d_accum,
                         sizeof(float) * static_cast<size_t>(k) * dim,
                         cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(shard.h_counts.data(),
                         shard.d_counts,
                         sizeof(int) * static_cast<size_t>(k),
                         cudaMemcpyDeviceToHost);
    }
    shard.h_objective = 0.0;
    if (collect_objective && err == cudaSuccess) {
        float obj = 0.0f;
        err = cudaMemcpy(&obj, shard.d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost);
        shard.h_objective = static_cast<double>(obj);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[kmeans_multigpu_train] GPU%d iteration failed: %s\n",
                     shard.gpu, cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace

bool kmeans_train_centroids_multigpu(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    float* d_centroids,
    float* h_objective) {
    if (!h_data || !d_centroids || kmeans_cfg.n <= 0 || kmeans_cfg.dim <= 0 || kmeans_cfg.k <= 0) {
        return false;
    }
    if (kmeans_cfg.use_flash_assign) {
        std::fprintf(stderr,
                     "[kmeans_multigpu_train] flash assign is not supported in multi-GPU training yet\n");
        return false;
    }

    const int n = kmeans_cfg.n;
    const int dim = kmeans_cfg.dim;
    const int k = kmeans_cfg.k;
    const int batch_size = cfg.batch_size > 0 ? cfg.batch_size : (1 << 15);
    const int centroid_tile = cfg.centroid_tile > 0 ? cfg.centroid_tile : 4096;
    const int n_gpus = choose_gpu_count(cfg.max_gpus);

    std::fprintf(stderr,
                 "[kmeans_multigpu_train] begin: n=%d dim=%d k=%d iters=%d gpus=%d\n",
                 n, dim, k, kmeans_cfg.iters, n_gpus);
    std::fflush(stderr);

    cudaSetDevice(cfg.device_id);
    std::vector<float> h_centroids(static_cast<size_t>(k) * dim);
    cudaError_t err = cudaMemcpy(h_centroids.data(),
                                 d_centroids,
                                 sizeof(float) * h_centroids.size(),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[kmeans_multigpu_train] initial centroid copy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    std::vector<MultiGpuTrainShard> shards(static_cast<size_t>(n_gpus));
    bool allocated = true;
    for (int g = 0; g < n_gpus; ++g) {
        auto& shard = shards[static_cast<size_t>(g)];
        shard.gpu = g;
        shard.start = static_cast<int>((static_cast<int64_t>(n) * g) / n_gpus);
        const int end = static_cast<int>((static_cast<int64_t>(n) * (g + 1)) / n_gpus);
        shard.count = end - shard.start;
        shard.dim = dim;
        shard.k = k;
        shard.batch_size = batch_size;
        shard.centroid_tile = centroid_tile;
        allocated = allocate_shard(shard, h_objective != nullptr) && allocated;
    }
    if (!allocated) {
        for (auto& shard : shards) free_shard(shard);
        return false;
    }

    std::vector<float> total_accum(static_cast<size_t>(k) * dim, 0.0f);
    std::vector<int> total_counts(static_cast<size_t>(k), 0);

    bool ok = true;
    std::mutex worker_mutex;
    std::condition_variable worker_cv;
    std::condition_variable done_cv;
    int requested_iter = -1;
    int completed_workers = 0;
    bool stop_workers = false;
    std::atomic<bool> iter_ok(true);

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_gpus));
    for (int g = 0; g < n_gpus; ++g) {
        workers.emplace_back([&, g]() {
            int last_seen_iter = -1;
            while (true) {
                int current_iter = -1;
                {
                    std::unique_lock<std::mutex> lock(worker_mutex);
                    worker_cv.wait(lock, [&]() {
                        return stop_workers || requested_iter != last_seen_iter;
                    });
                    if (stop_workers) return;
                    current_iter = requested_iter;
                    last_seen_iter = current_iter;
                }

                bool shard_ok = run_shard_iteration(
                    shards[static_cast<size_t>(g)],
                    kmeans_cfg,
                    h_data,
                    h_centroids.data(),
                    h_objective != nullptr);
                if (!shard_ok) iter_ok.store(false);

                {
                    std::lock_guard<std::mutex> lock(worker_mutex);
                    completed_workers += 1;
                }
                done_cv.notify_one();
            }
        });
    }

    for (int it = 0; it < kmeans_cfg.iters && ok; ++it) {
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
        double objective = 0.0;
        for (const auto& shard : shards) {
            for (size_t i = 0; i < total_accum.size(); ++i) total_accum[i] += shard.h_accum[i];
            for (size_t i = 0; i < total_counts.size(); ++i) total_counts[i] += shard.h_counts[i];
            objective += shard.h_objective;
        }

        kmeans_update_centroids_host(total_accum, total_counts, k, dim, h_centroids);
        if (h_objective) *h_objective = static_cast<float>(objective);
        std::fprintf(stderr, "[kmeans_multigpu_train] iter %d/%d done obj=%g\n",
                     it + 1, kmeans_cfg.iters, objective);
        std::fflush(stderr);
    }

    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        stop_workers = true;
    }
    worker_cv.notify_all();
    for (auto& worker : workers) worker.join();

    if (ok) {
        cudaSetDevice(cfg.device_id);
        err = cudaMemcpy(d_centroids,
                         h_centroids.data(),
                         sizeof(float) * h_centroids.size(),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[kmeans_multigpu_train] final centroid copy failed: %s\n",
                         cudaGetErrorString(err));
            ok = false;
        }
    }

    for (auto& shard : shards) free_shard(shard);
    cudaSetDevice(cfg.device_id);

    if (ok) {
        std::fprintf(stderr, "[kmeans_multigpu_train] done\n");
        std::fflush(stderr);
    }
    return ok;
}
