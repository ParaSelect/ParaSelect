#include "clustering/kmeans/kmeans_cagra_train.cuh"

#include "clustering/kmeans/kmeans_final_assign.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <type_traits>
#include <unordered_set>

namespace {

template <typename T>
class DeviceAllocation {
public:
    ~DeviceAllocation() {
        if (data_) cudaFree(data_);
    }

    bool allocate(size_t count) {
        return cudaMalloc(&data_, count * sizeof(T)) == cudaSuccess;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

template <typename T>
__global__ void kernel_accum_by_assign(
    const T* __restrict__ data,
    const idx_t* __restrict__ assign,
    float* __restrict__ accum,
    int* __restrict__ counts,
    int n,
    int dim) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * dim;
    if (value >= total) return;
    const int row = value / dim;
    const int col = value - row * dim;
    const idx_t cluster = assign[row];
    atomicAdd(accum + static_cast<size_t>(cluster) * dim + col,
              static_cast<float>(data[value]));
    if (col == 0) atomicAdd(counts + cluster, 1);
}

__global__ void kernel_update_cagra_centroids(
    float* __restrict__ centroids,
    const float* __restrict__ accum,
    const int* __restrict__ counts,
    int k,
    int dim) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = k * dim;
    if (value >= total) return;
    const int cluster = value / dim;
    const int count = counts[cluster];
    if (count > 0) centroids[value] = accum[value] / static_cast<float>(count);
}

template <typename T>
std::vector<float> sample_centroids(
    const KMeansCagraTrainConfig& cfg,
    const T* data) {
    std::mt19937 rng(cfg.seed);
    std::uniform_int_distribution<int> row_dist(0, cfg.n - 1);
    std::unordered_set<int> selected;
    selected.reserve(static_cast<size_t>(cfg.k) * 2);
    std::vector<float> centroids(static_cast<size_t>(cfg.k) * cfg.dim);
    for (int cluster = 0; cluster < cfg.k; ++cluster) {
        int row = row_dist(rng);
        while (!selected.insert(row).second) row = row_dist(rng);
        const T* source = data + static_cast<size_t>(row) * cfg.dim;
        float* destination = centroids.data() + static_cast<size_t>(cluster) * cfg.dim;
        for (int d = 0; d < cfg.dim; ++d) {
            destination[d] = static_cast<float>(source[d]);
        }
    }
    return centroids;
}

GpuCentroidAssignmentConfig make_assignment_config(
    const KMeansCagraTrainConfig& cfg) {
    GpuCentroidAssignmentConfig out;
    out.strategy = KMeansFinalAssignStrategy::Cagra;
    out.combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
    out.device_id = cfg.device_id;
    out.max_gpus = 1;
    out.batch_size = cfg.params.train_batch;
    out.cagra_graph_degree = cfg.params.graph_degree;
    out.cagra_intermediate_graph_degree = cfg.params.intermediate_graph_degree;
    out.cagra_topk_centroids = cfg.params.topk_centroids;
    out.cagra_itopk_size = cfg.params.itopk_size;
    out.cagra_search_width = cfg.params.search_width;
    out.cagra_max_iterations = cfg.params.max_iterations;
    out.cagra_team_size = cfg.params.team_size;
    out.cagra_thread_block_size = cfg.params.thread_block_size;
    out.cagra_index_scale = cfg.params.index_scale;
    out.cagra_build_algo = cfg.params.build_algo;
    return out;
}

}  // namespace

template <typename T>
bool kmeans_train_centroids_cagra_impl(
    const KMeansCagraTrainConfig& cfg,
    const T* h_data,
    std::vector<float>* centroids) {
    if (!h_data || !centroids || cfg.n <= 0 || cfg.dim <= 0 || cfg.k <= 0 ||
        cfg.k > cfg.n || cfg.iters <= 0 || cfg.params.train_batch <= 0) {
        return false;
    }
    if (cudaSetDevice(cfg.device_id) != cudaSuccess) return false;

    const size_t data_values = static_cast<size_t>(cfg.n) * cfg.dim;
    const size_t centroid_values = static_cast<size_t>(cfg.k) * cfg.dim;
    DeviceAllocation<T> d_data;
    DeviceAllocation<float> d_centroids;
    DeviceAllocation<float> d_accum;
    DeviceAllocation<int> d_counts;
    DeviceAllocation<idx_t> d_assign;
    if (!d_data.allocate(data_values) || !d_centroids.allocate(centroid_values) ||
        !d_accum.allocate(centroid_values) || !d_counts.allocate(cfg.k) ||
        !d_assign.allocate(cfg.params.train_batch)) {
        return false;
    }
    if (cudaMemcpy(d_data.data(), h_data, data_values * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    std::vector<float> initial = sample_centroids(cfg, h_data);
    if (cudaMemcpy(d_centroids.data(), initial.data(),
                   centroid_values * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    const GpuCentroidAssignmentConfig assign_cfg = make_assignment_config(cfg);
    for (int iteration = 0; iteration < cfg.iters; ++iteration) {
        const auto iteration_begin = std::chrono::steady_clock::now();
        GpuCentroidAssignmentCagraWorkspace workspace;
        if (!gpu_centroid_assignment_cagra_workspace_build(
                workspace, assign_cfg, d_centroids.data(), cfg.k, cfg.dim,
                cfg.params.train_batch)) {
            gpu_centroid_assignment_cagra_workspace_release(workspace);
            return false;
        }
        const auto build_end = std::chrono::steady_clock::now();
        const double build_seconds =
            std::chrono::duration<double>(build_end - iteration_begin).count();
        std::fprintf(stderr, "[kmeans_cagra_train] iter %d/%d build_s=%.3f\n",
                     iteration + 1, cfg.iters, build_seconds);
        std::fflush(stderr);
        if (cudaMemset(d_accum.data(), 0, centroid_values * sizeof(float)) != cudaSuccess ||
            cudaMemset(d_counts.data(), 0, static_cast<size_t>(cfg.k) * sizeof(int)) !=
                cudaSuccess) {
            gpu_centroid_assignment_cagra_workspace_release(workspace);
            return false;
        }

        bool iteration_ok = true;
        for (int begin = 0; begin < cfg.n; begin += cfg.params.train_batch) {
            const int count = std::min(cfg.params.train_batch, cfg.n - begin);
            KMeansCase shape{};
            shape.n = count;
            shape.dim = cfg.dim;
            shape.k = cfg.k;
            shape.dist = L2_DISTANCE;
            const T* batch = d_data.data() + static_cast<size_t>(begin) * cfg.dim;
            const bool assigned = [&] {
                if constexpr (std::is_same_v<T, uint8_t>) {
                    return assign_device_u8_vectors_to_centroids_gpu_cagra_with_workspace(
                        assign_cfg, shape, batch, d_centroids.data(),
                        workspace, d_assign.data());
                } else {
                    return assign_device_vectors_to_centroids_gpu_cagra_with_workspace(
                        assign_cfg, shape, batch, d_centroids.data(),
                        workspace, d_assign.data());
                }
            }();
            if (!assigned) {
                iteration_ok = false;
                break;
            }
            const int values = count * cfg.dim;
            kernel_accum_by_assign<<<(values + 255) / 256, 256>>>(
                batch,
                d_assign.data(), d_accum.data(), d_counts.data(), count, cfg.dim);
            if (cudaGetLastError() != cudaSuccess) {
                iteration_ok = false;
                break;
            }
        }
        const auto assign_end = std::chrono::steady_clock::now();
        if (iteration_ok) {
            const int values = cfg.k * cfg.dim;
            kernel_update_cagra_centroids<<<(values + 255) / 256, 256>>>(
                d_centroids.data(), d_accum.data(), d_counts.data(), cfg.k, cfg.dim);
            iteration_ok = cudaGetLastError() == cudaSuccess &&
                           cudaDeviceSynchronize() == cudaSuccess;
        }
        gpu_centroid_assignment_cagra_workspace_release(workspace);
        if (!iteration_ok) return false;
        const auto update_end = std::chrono::steady_clock::now();
        const auto seconds_since = [](auto begin, auto end) {
            return std::chrono::duration<double>(end - begin).count();
        };
        std::fprintf(
            stderr,
            "[kmeans_cagra_train] iter %d/%d build_s=%.3f assign_s=%.3f update_s=%.3f total_s=%.3f\n",
            iteration + 1, cfg.iters,
            build_seconds,
            seconds_since(build_end, assign_end),
            seconds_since(assign_end, update_end),
            seconds_since(iteration_begin, update_end));
        std::fflush(stderr);
    }

    centroids->resize(centroid_values);
    return cudaMemcpy(centroids->data(), d_centroids.data(),
                      centroid_values * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool kmeans_train_centroids_cagra_u8(
    const KMeansCagraTrainConfig& cfg,
    const uint8_t* h_data,
    std::vector<float>* centroids) {
    return kmeans_train_centroids_cagra_impl(cfg, h_data, centroids);
}

bool kmeans_train_centroids_cagra_fp32(
    const KMeansCagraTrainConfig& cfg,
    const float* h_data,
    std::vector<float>* centroids) {
    return kmeans_train_centroids_cagra_impl(cfg, h_data, centroids);
}
