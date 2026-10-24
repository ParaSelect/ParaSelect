#include "clustering/kmeans/kmeans_final_assign.cuh"

#include "clustering/kmeans/kmeans_multigpu_assign.cuh"
#include "l2norm/l2norm.cuh"
#include "utils/check_cuda_error.cuh"

#if PARASELECT_ENABLE_CUVS
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_view.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

namespace {

int normalized_batch_size(const GpuCentroidAssignmentConfig& cfg) {
    return cfg.batch_size > 0 ? cfg.batch_size : (1 << 15);
}

int normalized_centroid_tile(const GpuCentroidAssignmentConfig& cfg) {
    return cfg.centroid_tile > 0 ? cfg.centroid_tile : 4096;
}

#if PARASELECT_ENABLE_CUVS
struct CagraWorkspaceImpl {
    int device_id = -1;
    int max_batch = 0;
    int k = 0;
    int dim = 0;
    int topk = 0;
    int graph_degree = 0;
    int intermediate_graph_degree = 0;
    float index_scale = 0.0f;
    KMeansCagraBuildAlgo build_algo = KMeansCagraBuildAlgo::Auto;
    raft::resources res;
    const float* source_centroids = nullptr;  // Identity only; content changes require explicit rebuild.
    float* d_scaled_centroids = nullptr;
    uint8_t* d_query_u8 = nullptr;
    float* d_query_raw = nullptr;
    float* d_query_scaled = nullptr;
    uint32_t* d_neighbors = nullptr;
    float* d_distances = nullptr;
    std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> index;
};

CagraWorkspaceImpl* cagra_impl(GpuCentroidAssignmentCagraWorkspace& ws) {
    return static_cast<CagraWorkspaceImpl*>(ws.impl);
}

bool cagra_index_matches(
    const CagraWorkspaceImpl& impl,
    const GpuCentroidAssignmentConfig& cfg,
    int k,
    int dim) {
    const int topk = std::max(1, std::min(cfg.cagra_topk_centroids, k));
    return impl.index && impl.device_id == cfg.device_id && impl.k == k && impl.dim == dim &&
           impl.topk == topk && impl.graph_degree == cfg.cagra_graph_degree &&
           impl.intermediate_graph_degree == cfg.cagra_intermediate_graph_degree &&
           impl.index_scale == cfg.cagra_index_scale &&
           impl.build_algo == cfg.cagra_build_algo;
}

bool cagra_reserve_query_buffers(CagraWorkspaceImpl& impl, int max_batch) {
    if (impl.max_batch >= max_batch && impl.d_query_u8 && impl.d_query_raw &&
        impl.d_query_scaled && impl.d_neighbors && impl.d_distances) {
        return true;
    }

    uint8_t* d_query_u8 = nullptr;
    float* d_query_raw = nullptr;
    float* d_query_scaled = nullptr;
    uint32_t* d_neighbors = nullptr;
    float* d_distances = nullptr;
    const size_t query_values = static_cast<size_t>(max_batch) * impl.dim;
    const size_t result_values = static_cast<size_t>(max_batch) * impl.topk;
    cudaError_t err = cudaMalloc(&d_query_u8, sizeof(uint8_t) * query_values);
    if (err == cudaSuccess) err = cudaMalloc(&d_query_raw, sizeof(float) * query_values);
    if (err == cudaSuccess) err = cudaMalloc(&d_query_scaled, sizeof(float) * query_values);
    if (err == cudaSuccess) err = cudaMalloc(&d_neighbors, sizeof(uint32_t) * result_values);
    if (err == cudaSuccess) err = cudaMalloc(&d_distances, sizeof(float) * result_values);
    if (err != cudaSuccess) {
        if (d_distances) cudaFree(d_distances);
        if (d_neighbors) cudaFree(d_neighbors);
        if (d_query_scaled) cudaFree(d_query_scaled);
        if (d_query_raw) cudaFree(d_query_raw);
        if (d_query_u8) cudaFree(d_query_u8);
        return false;
    }

    if (impl.d_distances) cudaFree(impl.d_distances);
    if (impl.d_neighbors) cudaFree(impl.d_neighbors);
    if (impl.d_query_scaled) cudaFree(impl.d_query_scaled);
    if (impl.d_query_raw) cudaFree(impl.d_query_raw);
    if (impl.d_query_u8) cudaFree(impl.d_query_u8);
    impl.d_query_u8 = d_query_u8;
    impl.d_query_raw = d_query_raw;
    impl.d_query_scaled = d_query_scaled;
    impl.d_neighbors = d_neighbors;
    impl.d_distances = d_distances;
    impl.max_batch = max_batch;
    return true;
}
#endif

bool assign_vectors_to_centroids_gpu_single_device(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    idx_t* d_assign) {
    GpuCentroidAssignmentWorkspace ws;
    if (!gpu_centroid_assignment_workspace_prepare(
            ws, cfg, normalized_batch_size(cfg), kmeans_cfg.k, kmeans_cfg.dim)) {
        gpu_centroid_assignment_workspace_release(ws);
        return false;
    }

    perform_assignment_only(
        kmeans_cfg,
        h_data,
        d_assign,
        d_centroids,
        ws.d_cnorm2,
        ws.env,
        ws.d_data_buf,
        0,
        ws.handle,
        normalized_batch_size(cfg),
        normalized_centroid_tile(cfg),
        kmeans_cfg.n,
        kmeans_cfg.dim,
        kmeans_cfg.k);

    const cudaError_t err = cudaGetLastError();
    gpu_centroid_assignment_workspace_release(ws);
    return err == cudaSuccess;
}

#if PARASELECT_ENABLE_CUVS
__global__ void kernel_scale_copy(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n,
                                  float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale;
}

__global__ void kernel_u8_to_float_scaled(const uint8_t* __restrict__ in,
                                          float* __restrict__ raw,
                                          float* __restrict__ scaled,
                                          int n,
                                          float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = static_cast<float>(in[i]);
        raw[i] = value;
        scaled[i] = value * scale;
    }
}

__global__ void kernel_select_cagra_candidates(
    const float* __restrict__ query,
    const float* __restrict__ centroids,
    const uint32_t* __restrict__ candidates,
    idx_t* __restrict__ assign,
    int n,
    int dim,
    int topk) {
    const int q = blockIdx.x;
    if (q >= n) return;

    __shared__ float partial[256];
    const float* qv = query + static_cast<size_t>(q) * dim;
    float best_dist = INFINITY;
    idx_t best_idx = 0;

    for (int r = 0; r < topk; ++r) {
        const uint32_t cid = candidates[static_cast<size_t>(q) * topk + r];
        const float* cv = centroids + static_cast<size_t>(cid) * dim;
        float sum = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            const float diff = qv[d] - cv[d];
            sum += diff * diff;
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0 && partial[0] < best_dist) {
            best_dist = partial[0];
            best_idx = static_cast<idx_t>(cid);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) assign[q] = best_idx;
}
#endif

int visible_gpu_count() {
    int n_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&n_gpus);
    if (err != cudaSuccess) return 0;
    return n_gpus;
}

#if PARASELECT_ENABLE_CUVS
bool assign_vectors_to_centroids_gpu_cagra_topk_select(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    idx_t* d_assign) {
    GpuCentroidAssignmentCagraWorkspace ws;
    const bool ok = gpu_centroid_assignment_cagra_workspace_build(
                        ws, cfg, d_centroids, kmeans_cfg.k, kmeans_cfg.dim, normalized_batch_size(cfg)) &&
                    assign_vectors_to_centroids_gpu_cagra_with_workspace(
                        cfg, kmeans_cfg, h_data, d_centroids, ws, d_assign);
    gpu_centroid_assignment_cagra_workspace_release(ws);
    return ok;
}
#endif

} // namespace

#if PARASELECT_ENABLE_CUVS
bool gpu_centroid_assignment_cagra_workspace_build(
    GpuCentroidAssignmentCagraWorkspace& ws,
    const GpuCentroidAssignmentConfig& cfg,
    const float* d_centroids,
    int k,
    int dim,
    int max_batch) {
    if (!d_centroids || k <= 0 || dim <= 0 || max_batch <= 0) return false;

    gpu_centroid_assignment_cagra_workspace_release(ws);
    cudaSetDevice(cfg.device_id);
    CHECK_CUDA_ERRORS;

    auto* impl = new CagraWorkspaceImpl();
    ws.impl = impl;
    impl->source_centroids = d_centroids;
    impl->device_id = cfg.device_id;
    impl->k = k;
    impl->dim = dim;
    impl->topk = std::max(1, std::min(cfg.cagra_topk_centroids, k));
    impl->graph_degree = cfg.cagra_graph_degree;
    impl->intermediate_graph_degree = cfg.cagra_intermediate_graph_degree;
    impl->index_scale = cfg.cagra_index_scale;
    impl->build_algo = cfg.cagra_build_algo;

    cudaError_t err = cudaMalloc(
        &impl->d_scaled_centroids, sizeof(float) * static_cast<size_t>(k) * dim);
    if (err != cudaSuccess || !cagra_reserve_query_buffers(*impl, max_batch)) {
        gpu_centroid_assignment_cagra_workspace_release(ws);
        return false;
    }

    try {
        cudaStream_t stream = 0;
        raft::resource::set_cuda_stream(impl->res, rmm::cuda_stream_view(stream));

        const int threads = 256;
        const int total = k * dim;
        const int blocks = (total + threads - 1) / threads;
        kernel_scale_copy<<<blocks, threads>>>(
            d_centroids, impl->d_scaled_centroids, total, cfg.cagra_index_scale);
        CUDA_LAUNCH_CHECK("kernel_scale_copy centroids");

        auto centroid_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            impl->d_scaled_centroids, static_cast<int64_t>(k), static_cast<int64_t>(dim));
        cuvs::neighbors::cagra::index_params index_params;
        index_params.metric = cuvs::distance::DistanceType::L2Expanded;
        index_params.graph_degree = static_cast<size_t>(std::max(1, cfg.cagra_graph_degree));
        index_params.intermediate_graph_degree = static_cast<size_t>(
            std::max(cfg.cagra_graph_degree, cfg.cagra_intermediate_graph_degree));
        KMeansCagraBuildAlgo build_algo = cfg.cagra_build_algo;
        if (build_algo == KMeansCagraBuildAlgo::Auto) {
            build_algo = k <= 65536
                ? KMeansCagraBuildAlgo::NnDescent
                : KMeansCagraBuildAlgo::IvfPq;
        }
        if (build_algo == KMeansCagraBuildAlgo::NnDescent) {
            cuvs::neighbors::cagra::graph_build_params::nn_descent_params
                graph_params((size_t)index_params.intermediate_graph_degree,
                             index_params.metric);
            graph_params.intermediate_graph_degree = std::max<size_t>(
                graph_params.graph_degree * 3 / 2,
                (size_t)index_params.intermediate_graph_degree);
            graph_params.return_distances = false;
            index_params.graph_build_params = graph_params;
        } else {
            index_params.graph_build_params =
                cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
                    raft::matrix_extent<int64_t>(
                        static_cast<int64_t>(k), static_cast<int64_t>(dim)),
                    index_params.metric);
        }
        auto built = cuvs::neighbors::cagra::build(impl->res, index_params, centroid_view);
        impl->index = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(std::move(built));
        raft::resource::sync_stream(impl->res);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[cagra_build] %s\n", error.what());
        gpu_centroid_assignment_cagra_workspace_release(ws);
        return false;
    }
    return true;
}

void gpu_centroid_assignment_cagra_workspace_release(GpuCentroidAssignmentCagraWorkspace& ws) {
    auto* impl = cagra_impl(ws);
    if (!impl) return;
    impl->index.reset();
    if (impl->d_distances) cudaFree(impl->d_distances);
    if (impl->d_neighbors) cudaFree(impl->d_neighbors);
    if (impl->d_query_scaled) cudaFree(impl->d_query_scaled);
    if (impl->d_query_raw) cudaFree(impl->d_query_raw);
    if (impl->d_query_u8) cudaFree(impl->d_query_u8);
    if (impl->d_scaled_centroids) cudaFree(impl->d_scaled_centroids);
    delete impl;
    ws.impl = nullptr;
}

namespace {

bool assign_vectors_to_centroids_gpu_cagra_impl(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const void* h_data,
    bool input_u8,
    bool input_device,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign) {
    if (!h_data || !d_centroids || !d_assign || kmeans_cfg.n <= 0 ||
        kmeans_cfg.dim <= 0 || kmeans_cfg.k <= 0) {
        return false;
    }
    auto* impl = cagra_impl(ws);
    if (!impl || !cagra_index_matches(*impl, cfg, kmeans_cfg.k, kmeans_cfg.dim) ||
        impl->source_centroids != d_centroids || impl->max_batch < normalized_batch_size(cfg)) {
        std::fprintf(stderr, "[cagra_assignment] build workspace explicitly before search; centers/config/capacity must match\n");
        return false;
    }

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size = static_cast<size_t>(std::max(cfg.cagra_itopk_size, impl->topk));
    search_params.search_width = static_cast<size_t>(std::max(1, cfg.cagra_search_width));
    search_params.max_iterations = static_cast<size_t>(std::max(0, cfg.cagra_max_iterations));
    search_params.team_size = static_cast<size_t>(std::max(0, cfg.cagra_team_size));
    search_params.thread_block_size = static_cast<size_t>(std::max(0, cfg.cagra_thread_block_size));
    search_params.algo = cuvs::neighbors::cagra::search_algo::AUTO;

    const int threads = 256;
    const int batch_size = normalized_batch_size(cfg);
    for (int base = 0; base < kmeans_cfg.n; base += batch_size) {
        const int curB = std::min(batch_size, kmeans_cfg.n - base);
        const size_t cur_values = static_cast<size_t>(curB) * kmeans_cfg.dim;
        const int blocks = (static_cast<int>(cur_values) + threads - 1) / threads;
        if (input_u8) {
            const auto* input = static_cast<const uint8_t*>(h_data) +
                                static_cast<size_t>(base) * kmeans_cfg.dim;
            const uint8_t* query_u8 = input;
            if (!input_device) {
                CUDA_CHECK(cudaMemcpy(impl->d_query_u8, input,
                                      sizeof(uint8_t) * cur_values,
                                      cudaMemcpyHostToDevice));
                query_u8 = impl->d_query_u8;
            }
            kernel_u8_to_float_scaled<<<blocks, threads>>>(
                query_u8, impl->d_query_raw, impl->d_query_scaled,
                static_cast<int>(cur_values), cfg.cagra_index_scale);
            CUDA_LAUNCH_CHECK("kernel_u8_to_float_scaled queries");
        } else {
            const auto* input = static_cast<const float*>(h_data) +
                                static_cast<size_t>(base) * kmeans_cfg.dim;
            CUDA_CHECK(cudaMemcpy(impl->d_query_raw, input,
                                  sizeof(float) * cur_values,
                                  input_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
            kernel_scale_copy<<<blocks, threads>>>(
                impl->d_query_raw, impl->d_query_scaled,
                static_cast<int>(cur_values), cfg.cagra_index_scale);
            CUDA_LAUNCH_CHECK("kernel_scale_copy queries");
        }

        auto query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            impl->d_query_scaled, static_cast<int64_t>(curB), static_cast<int64_t>(kmeans_cfg.dim));
        auto neigh_view = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
            impl->d_neighbors, static_cast<int64_t>(curB), static_cast<int64_t>(impl->topk));
        auto dist_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
            impl->d_distances, static_cast<int64_t>(curB), static_cast<int64_t>(impl->topk));
        try {
            cuvs::neighbors::cagra::search(impl->res, search_params, *impl->index, query_view, neigh_view, dist_view);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[cagra_search] %s\n", error.what());
            return false;
        }
        kernel_select_cagra_candidates<<<curB, 256>>>(
            impl->d_query_raw, d_centroids, impl->d_neighbors, d_assign + base, curB, kmeans_cfg.dim, impl->topk);
        CUDA_LAUNCH_CHECK("kernel_select_cagra_candidates");
    }
    try {
        raft::resource::sync_stream(impl->res);
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace

bool assign_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign) {
    return assign_vectors_to_centroids_gpu_cagra_impl(
        cfg, kmeans_cfg, h_data, false, false, d_centroids, ws, d_assign);
}

bool assign_u8_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const uint8_t* h_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign) {
    return assign_vectors_to_centroids_gpu_cagra_impl(
        cfg, kmeans_cfg, h_data, true, false, d_centroids, ws, d_assign);
}

bool assign_device_u8_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const uint8_t* d_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign) {
    return assign_vectors_to_centroids_gpu_cagra_impl(
        cfg, kmeans_cfg, d_data, true, true, d_centroids, ws, d_assign);
}

bool assign_device_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* d_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign) {
    return assign_vectors_to_centroids_gpu_cagra_impl(
        cfg, kmeans_cfg, d_data, false, true, d_centroids, ws, d_assign);
}
#else
bool gpu_centroid_assignment_cagra_workspace_build(
    GpuCentroidAssignmentCagraWorkspace&, const GpuCentroidAssignmentConfig&,
    const float*, int, int, int) {
    std::fprintf(stderr, "[gpu_centroid_assignment] CAGRA backend was not built\n");
    return false;
}

void gpu_centroid_assignment_cagra_workspace_release(
    GpuCentroidAssignmentCagraWorkspace& ws) {
    ws.impl = nullptr;
}

bool assign_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig&, const KMeansCase&, const float*,
    const float*, GpuCentroidAssignmentCagraWorkspace&, idx_t*) {
    std::fprintf(stderr, "[gpu_centroid_assignment] CAGRA backend was not built\n");
    return false;
}

bool assign_u8_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig&, const KMeansCase&, const uint8_t*,
    const float*, GpuCentroidAssignmentCagraWorkspace&, idx_t*) {
    std::fprintf(stderr, "[gpu_centroid_assignment] CAGRA backend was not built\n");
    return false;
}

bool assign_device_u8_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig&, const KMeansCase&, const uint8_t*,
    const float*, GpuCentroidAssignmentCagraWorkspace&, idx_t*) {
    std::fprintf(stderr, "[gpu_centroid_assignment] CAGRA backend was not built\n");
    return false;
}

bool assign_device_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig&, const KMeansCase&, const float*,
    const float*, GpuCentroidAssignmentCagraWorkspace&, idx_t*) {
    std::fprintf(stderr, "[gpu_centroid_assignment] CAGRA backend was not built\n");
    return false;
}

#endif

static bool reserve_assignment_output(
    GpuCentroidAssignmentWorkspace& ws, int rows) {
    if (rows <= 0) return false;
    if (ws.d_assign && ws.assign_capacity >= rows) return true;
    idx_t* replacement = nullptr;
    const cudaError_t err = cudaMalloc(
        &replacement, sizeof(idx_t) * static_cast<size_t>(rows));
    if (err != cudaSuccess) return false;
    if (ws.d_assign) cudaFree(ws.d_assign);
    ws.d_assign = replacement;
    ws.assign_capacity = rows;
    return true;
}

bool gpu_centroid_assignment_workspace_prepare(
    GpuCentroidAssignmentWorkspace& ws,
    const GpuCentroidAssignmentConfig& cfg,
    int max_batch,
    int k,
    int dim) {
    if (max_batch <= 0 || k <= 0 || dim <= 0) return false;
    const int B = max_batch;
    const int Ktile = normalized_centroid_tile(cfg);
    const bool reusable = ws.handle && ws.d_cnorm2 && ws.d_data_buf && ws.d_assign &&
                          ws.device_id == cfg.device_id && ws.max_batch >= B && ws.assign_capacity >= B && ws.max_k >= k &&
                          ws.dim == dim && ws.centroid_tile == Ktile;
    if (reusable) return true;

    gpu_centroid_assignment_workspace_release(ws);
    cudaSetDevice(cfg.device_id);
    CHECK_CUDA_ERRORS;

    ws.device_id = cfg.device_id;
    ws.max_batch = B;
    ws.max_k = k;
    ws.dim = dim;
    ws.centroid_tile = Ktile;
    ws.env.allocate(B, Ktile);

    cublasStatus_t status = cublasCreate(&ws.handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        gpu_centroid_assignment_workspace_release(ws);
        return false;
    }
    cublasSetMathMode(ws.handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cudaError_t err = cudaMalloc(&ws.d_cnorm2, sizeof(float) * static_cast<size_t>(k));
    if (err == cudaSuccess) {
        err = cudaMalloc(&ws.d_data_buf, sizeof(float) * static_cast<size_t>(B) * dim);
    }
    if (err == cudaSuccess) {
        err = cudaMalloc(&ws.d_assign, sizeof(idx_t) * static_cast<size_t>(B));
        if (err == cudaSuccess) ws.assign_capacity = B;
    }
    if (err != cudaSuccess) {
        gpu_centroid_assignment_workspace_release(ws);
        return false;
    }
    return true;
}

void gpu_centroid_assignment_workspace_release(GpuCentroidAssignmentWorkspace& ws) {
    if (ws.d_assign) cudaFree(ws.d_assign);
    if (ws.d_data_buf) cudaFree(ws.d_data_buf);
    if (ws.d_cnorm2) cudaFree(ws.d_cnorm2);
    if (ws.handle) cublasDestroy(ws.handle);
    ws.env.free();
    ws = GpuCentroidAssignmentWorkspace{};
}

bool assign_vectors_to_centroids_gpu_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    GpuCentroidAssignmentWorkspace& ws) {
    if (!h_data || !d_centroids || kmeans_cfg.n <= 0 ||
        kmeans_cfg.dim <= 0 || kmeans_cfg.k <= 0) {
        return false;
    }
    if (cfg.strategy != KMeansFinalAssignStrategy::Exact ||
        cfg.combo != GpuCentroidAssignmentCombo::ExactGemm) {
        std::fprintf(stderr,
                     "[gpu_centroid_assignment] workspace path supports exact_gemm only\n");
        return false;
    }

    const int B = normalized_batch_size(cfg);
    if (!gpu_centroid_assignment_workspace_prepare(
            ws, cfg, B, kmeans_cfg.k, kmeans_cfg.dim)) {
        return false;
    }
    if (!reserve_assignment_output(ws, kmeans_cfg.n)) {
        return false;
    }

    perform_assignment_only(
        kmeans_cfg,
        h_data,
        ws.d_assign,
        d_centroids,
        ws.d_cnorm2,
        ws.env,
        ws.d_data_buf,
        0,
        ws.handle,
        ws.max_batch,
        ws.centroid_tile,
        kmeans_cfg.n,
        kmeans_cfg.dim,
        kmeans_cfg.k);
    return cudaGetLastError() == cudaSuccess;
}

bool assign_vectors_to_centroids_gpu(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    idx_t* d_assign) {
    if (!h_data || !d_centroids || !d_assign || kmeans_cfg.n <= 0 ||
        kmeans_cfg.dim <= 0 || kmeans_cfg.k <= 0) {
        return false;
    }
    GpuCentroidAssignmentCombo combo = cfg.combo;
    if (cfg.strategy == KMeansFinalAssignStrategy::Cagra) {
        combo = GpuCentroidAssignmentCombo::CagraTopkSelect;
    } else if (cfg.strategy != KMeansFinalAssignStrategy::Exact) {
        std::fprintf(stderr,
                     "[gpu_centroid_assignment] unsupported final assign strategy\n");
        return false;
    }

    if (combo == GpuCentroidAssignmentCombo::CagraTopkSelect) {
#if PARASELECT_ENABLE_CUVS
        return assign_vectors_to_centroids_gpu_cagra_topk_select(
            cfg, kmeans_cfg, h_data, d_centroids, d_assign);
#else
        std::fprintf(stderr, "[gpu_centroid_assignment] CAGRA backend was not built\n");
        return false;
#endif
    }

    const int n_gpus = visible_gpu_count();
    if (n_gpus > 1 && cfg.max_gpus != 1) {
        return assign_vectors_to_centroids_gpu_multi_device(cfg, kmeans_cfg, h_data, d_centroids, d_assign);
    }

    std::fprintf(stderr,
                 "[gpu_centroid_assignment] single GPU exact assign: n=%d dim=%d k=%d\n",
                 kmeans_cfg.n, kmeans_cfg.dim, kmeans_cfg.k);
    std::fflush(stderr);
    return assign_vectors_to_centroids_gpu_single_device(cfg, kmeans_cfg, h_data, d_centroids, d_assign);
}
