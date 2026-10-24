#include "quantization/pca/pca_storage.h"
#include "search/ivf_common/ivf_pca_refine.h"

#include "search/kernels/topk/reducer/topk_reducer.cuh"
#include "utils/check_cuda_error.cuh"

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace paraselect::search {
namespace {

__global__ void project_pca_queries_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ projection,
    const float* __restrict__ bias,
    int batch_n,
    int source_dim,
    int dim,
    float* __restrict__ output) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= batch_n * dim) return;
    const int query = position / dim;
    const int output_dim = position - query * dim;
    const float* input = queries + static_cast<size_t>(query) * source_dim;
    const float* row = projection + static_cast<size_t>(output_dim) * source_dim;
    float value = bias[output_dim];
    for (int d = 0; d < source_dim; ++d) value += row[d] * input[d];
    output[position] = value;
}

__global__ void pca_candidate_distance_kernel(
    const float* __restrict__ projected_queries,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scales,
    const int* __restrict__ candidates,
    int batch_n,
    int candidate_stride,
    int candidate_count,
    int count,
    int dim,
    float* __restrict__ distances,
    int* __restrict__ compact_candidates,
    int* __restrict__ valid_count,
    int* __restrict__ invalid) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= batch_n * candidate_count) return;
    const int query = position / candidate_count;
    const int slot = position - query * candidate_count;
    const int id = candidates[static_cast<size_t>(query) * candidate_stride + slot];
    compact_candidates[position] = id;
    if (id == -1) {
        distances[position] = FLT_MAX;
        return;
    }
    if (id < 0 || id >= count) {
        distances[position] = FLT_MAX;
        atomicExch(invalid, 1);
        return;
    }
    atomicAdd(valid_count, 1);
    const float* projected_query = projected_queries + static_cast<size_t>(query) * dim;
    const int8_t* vector = vectors + static_cast<size_t>(id) * dim;
    float distance = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float value = static_cast<float>(vector[d]) * scales[d];
        const float delta = projected_query[d] - value;
        distance += delta * delta;
    }
    distances[position] = distance;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

struct IVFPCAGpuRefiner::Impl {
    Impl(
        const paraselect::quantization::PCAStorageView& value,
        const int8_t* requested_device_vectors,
        int requested_batch,
        int requested_candidates,
        int requested_output)
        : storage(value),
          device_vectors(requested_device_vectors),
          max_batch(requested_batch),
          max_candidates(requested_candidates),
          max_output(requested_output) {
        if (!device_vectors || !storage.projection || !storage.bias ||
            !storage.scales || storage.count <= 0 || storage.count > INT_MAX ||
            storage.source_dim <= 0 || storage.dim <= 0 ||
            storage.dim > storage.source_dim || max_batch <= 0 ||
            max_candidates <= 0 || max_output <= 0 || max_output > max_candidates) {
            throw std::invalid_argument("invalid GPU IVF PCA refiner configuration");
        }
        const size_t projection_values =
            static_cast<size_t>(storage.dim) * storage.source_dim;
        try {
            check_cuda(
                cudaMalloc(&d_projection, projection_values * sizeof(float)),
                "allocate GPU PCA projection");
            check_cuda(
                cudaMalloc(&d_bias, static_cast<size_t>(storage.dim) * sizeof(float)),
                "allocate GPU PCA bias");
            check_cuda(
                cudaMalloc(&d_scales, static_cast<size_t>(storage.dim) * sizeof(float)),
                "allocate GPU PCA scales");
            check_cuda(
                cudaMalloc(
                    &d_projected_queries,
                    static_cast<size_t>(max_batch) * storage.dim * sizeof(float)),
                "allocate GPU PCA query workspace");
            check_cuda(
                cudaMalloc(
                    &d_distances,
                    static_cast<size_t>(max_batch) * max_candidates * sizeof(float)),
                "allocate GPU PCA distance workspace");
            check_cuda(
                cudaMalloc(
                    &d_compact_candidates,
                    static_cast<size_t>(max_batch) * max_candidates * sizeof(int)),
                "allocate GPU PCA candidate workspace");
            check_cuda(
                cudaMalloc(
                    &d_output_distances,
                    static_cast<size_t>(max_batch) * max_output * sizeof(float)),
                "allocate GPU PCA output workspace");
            check_cuda(cudaMalloc(&d_valid_count, sizeof(int)), "allocate GPU PCA counter");
            check_cuda(cudaMalloc(&d_invalid, sizeof(int)), "allocate GPU PCA status");
            check_cuda(
                cudaMemcpy(
                    d_projection,
                    storage.projection,
                    projection_values * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy GPU PCA projection");
            check_cuda(
                cudaMemcpy(
                    d_bias,
                    storage.bias,
                    static_cast<size_t>(storage.dim) * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy GPU PCA bias");
            check_cuda(
                cudaMemcpy(
                    d_scales,
                    storage.scales,
                    static_cast<size_t>(storage.dim) * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy GPU PCA scales");

            topk_args.n_rows = max_batch;
            topk_args.n_sources = 1;
            topk_args.local_k = max_candidates;
            topk_args.global_k = max_output;
            topk_args.thread_topk = 8;
            workspace_bytes = topk::candidate_topk_workspace_bytes(
                topk_args, topk::TopKReducerBackend::kCubBlock);
            if (workspace_bytes > 0) {
                check_cuda(
                    cudaMalloc(&d_topk_workspace, workspace_bytes),
                    "allocate GPU PCA Top-K workspace");
            }
            check_cuda(cudaEventCreate(&transform_begin), "create GPU PCA event");
            check_cuda(cudaEventCreate(&transform_end), "create GPU PCA event");
            check_cuda(cudaEventCreate(&distance_end), "create GPU PCA event");
            check_cuda(cudaEventCreate(&select_end), "create GPU PCA event");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        if (transform_begin) cudaEventDestroy(transform_begin);
        if (transform_end) cudaEventDestroy(transform_end);
        if (distance_end) cudaEventDestroy(distance_end);
        if (select_end) cudaEventDestroy(select_end);
        if (d_projection) cudaFree(d_projection);
        if (d_bias) cudaFree(d_bias);
        if (d_scales) cudaFree(d_scales);
        if (d_projected_queries) cudaFree(d_projected_queries);
        if (d_distances) cudaFree(d_distances);
        if (d_compact_candidates) cudaFree(d_compact_candidates);
        if (d_output_distances) cudaFree(d_output_distances);
        if (d_topk_workspace) cudaFree(d_topk_workspace);
        if (d_valid_count) cudaFree(d_valid_count);
        if (d_invalid) cudaFree(d_invalid);
        transform_begin = nullptr;
        transform_end = nullptr;
        distance_end = nullptr;
        select_end = nullptr;
        d_projection = nullptr;
        d_bias = nullptr;
        d_scales = nullptr;
        d_projected_queries = nullptr;
        d_distances = nullptr;
        d_compact_candidates = nullptr;
        d_output_distances = nullptr;
        d_topk_workspace = nullptr;
        d_valid_count = nullptr;
        d_invalid = nullptr;
    }

    paraselect::quantization::PCAStorageView storage;
    const int8_t* device_vectors = nullptr;
    int max_batch = 0;
    int max_candidates = 0;
    int max_output = 0;
    float* d_projection = nullptr;
    float* d_bias = nullptr;
    float* d_scales = nullptr;
    float* d_projected_queries = nullptr;
    float* d_distances = nullptr;
    int* d_compact_candidates = nullptr;
    float* d_output_distances = nullptr;
    void* d_topk_workspace = nullptr;
    int* d_valid_count = nullptr;
    int* d_invalid = nullptr;
    size_t workspace_bytes = 0;
    topk::CandidateTopKArgs topk_args;
    cudaEvent_t transform_begin = nullptr;
    cudaEvent_t transform_end = nullptr;
    cudaEvent_t distance_end = nullptr;
    cudaEvent_t select_end = nullptr;
};

IVFPCAGpuRefiner::IVFPCAGpuRefiner(
    const paraselect::quantization::PCAStorageView& storage,
    const int8_t* device_vectors,
    int max_batch,
    int max_candidates,
    int max_output)
    : impl_(std::make_unique<Impl>(
          storage, device_vectors, max_batch, max_candidates, max_output)) {}

IVFPCAGpuRefiner::~IVFPCAGpuRefiner() = default;

void IVFPCAGpuRefiner::run(
    const float* device_queries,
    const int* device_candidates,
    int batch_n,
    int candidate_stride,
    int candidate_count,
    int output_count,
    int* device_output,
    IVFPCARefineStats* stats,
    cudaStream_t stream) {
    if (!device_queries || !device_candidates || !device_output || batch_n <= 0 ||
        batch_n > impl_->max_batch || candidate_stride <= 0 ||
        candidate_stride > impl_->max_candidates || candidate_count <= 0 ||
        candidate_count > candidate_stride || output_count <= 0 ||
        output_count > candidate_count || output_count > impl_->max_output) {
        throw std::invalid_argument("invalid GPU IVF PCA refinement request");
    }

    check_cuda(cudaMemsetAsync(impl_->d_valid_count, 0, sizeof(int), stream),
               "clear GPU PCA counter");
    check_cuda(cudaMemsetAsync(impl_->d_invalid, 0, sizeof(int), stream),
               "clear GPU PCA status");
    check_cuda(cudaEventRecord(impl_->transform_begin, stream),
               "record GPU PCA transform start");
    constexpr int kThreads = 256;
    const int projected_values = batch_n * impl_->storage.dim;
    project_pca_queries_kernel<<<
        (projected_values + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        device_queries,
        impl_->d_projection,
        impl_->d_bias,
        batch_n,
        impl_->storage.source_dim,
        impl_->storage.dim,
        impl_->d_projected_queries);
    CUDA_LAUNCH_CHECK("project_pca_queries_kernel");
    check_cuda(cudaEventRecord(impl_->transform_end, stream),
               "record GPU PCA transform end");

    const int candidate_values = batch_n * candidate_count;
    pca_candidate_distance_kernel<<<
        (candidate_values + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        impl_->d_projected_queries,
        impl_->device_vectors,
        impl_->d_scales,
        device_candidates,
        batch_n,
        candidate_stride,
        candidate_count,
        static_cast<int>(impl_->storage.count),
        impl_->storage.dim,
        impl_->d_distances,
        impl_->d_compact_candidates,
        impl_->d_valid_count,
        impl_->d_invalid);
    CUDA_LAUNCH_CHECK("pca_candidate_distance_kernel");
    check_cuda(cudaEventRecord(impl_->distance_end, stream),
               "record GPU PCA distance end");

    topk::CandidateTopKArgs args;
    args.candidate_idx = impl_->d_compact_candidates;
    args.candidate_dist = impl_->d_distances;
    args.n_rows = batch_n;
    args.n_sources = 1;
    args.local_k = candidate_count;
    args.global_k = output_count;
    args.thread_topk = 8;
    args.out_idx = device_output;
    args.out_dist = impl_->d_output_distances;
    args.workspace = impl_->d_topk_workspace;
    args.workspace_bytes = impl_->workspace_bytes;
    args.stream = stream;
    check_cuda(
        topk::candidate_topk_reduce(args, topk::TopKReducerBackend::kCubBlock),
        "reduce GPU PCA candidates");
    check_cuda(cudaEventRecord(impl_->select_end, stream),
               "record GPU PCA selection end");

    int invalid = 0;
    int valid_count = 0;
    check_cuda(
        cudaMemcpyAsync(
            &invalid, impl_->d_invalid, sizeof(int), cudaMemcpyDeviceToHost, stream),
        "copy GPU PCA status");
    check_cuda(
        cudaMemcpyAsync(
            &valid_count,
            impl_->d_valid_count,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream),
        "copy GPU PCA counter");
    check_cuda(cudaEventSynchronize(impl_->select_end), "synchronize GPU PCA refinement");
    if (invalid != 0) throw std::out_of_range("GPU IVF PCA candidate is outside storage");

    if (stats) {
        float transform_ms = 0.0f;
        float distance_ms = 0.0f;
        float reduce_ms = 0.0f;
        float select_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime(
                &transform_ms, impl_->transform_begin, impl_->transform_end),
            "measure GPU PCA transform");
        check_cuda(
            cudaEventElapsedTime(&select_ms, impl_->transform_end, impl_->select_end),
            "measure GPU PCA selection");
        check_cuda(
            cudaEventElapsedTime(&distance_ms, impl_->transform_end, impl_->distance_end),
            "measure GPU PCA distance");
        check_cuda(
            cudaEventElapsedTime(&reduce_ms, impl_->distance_end, impl_->select_end),
            "measure GPU PCA reduction");
        stats->query_transform_ms = transform_ms;
        stats->select_ms = select_ms;
        stats->distance_ms = distance_ms;
        stats->reduce_ms = reduce_ms;
        stats->avg_input_candidates = static_cast<double>(valid_count) / batch_n;
        stats->threads = 0;
    }
}

}  // namespace paraselect::search
