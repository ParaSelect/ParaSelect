#include "transform/pca_transform.cuh"

#include "transform/orthogonal_projection_transform.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace paraselect::transform {
namespace {

__global__ void accumulate_pca_mean_kernel(
    const float* input,
    int rows,
    int dim,
    double* sums) {
    const int d = blockIdx.x;
    extern __shared__ double mean_shared[];
    double value = 0.0;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        value += static_cast<double>(input[static_cast<size_t>(row) * dim + d]);
    }
    mean_shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mean_shared[threadIdx.x] += mean_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[d] += mean_shared[0];
}

__global__ void finalize_pca_mean_kernel(
    const double* sums,
    int dim,
    double inverse_count,
    float* mean) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) mean[d] = static_cast<float>(sums[d] * inverse_count);
}

__global__ void center_pca_input_kernel(
    float* input,
    const float* mean,
    size_t values,
    int dim) {
    const size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position < values) input[position] -= mean[position % static_cast<size_t>(dim)];
}

__global__ void add_pca_bias_kernel(
    float* projected,
    const float* bias,
    size_t values,
    int dim) {
    const size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position < values) projected[position] += bias[position % static_cast<size_t>(dim)];
}

__global__ void calibrate_pca_scales_kernel(
    const float* projected,
    const float* bias,
    size_t rows,
    int dim,
    float* scales) {
    const int d = blockIdx.x;
    extern __shared__ float scale_shared[];
    float maximum = 0.0f;
    for (size_t row = threadIdx.x; row < rows; row += blockDim.x) {
        maximum = fmaxf(
            maximum,
            fabsf(projected[row * static_cast<size_t>(dim) + d] + bias[d]));
    }
    scale_shared[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scale_shared[threadIdx.x] = fmaxf(
                scale_shared[threadIdx.x], scale_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float scale = scale_shared[0] / 127.0f;
        scales[d] = scale > 0.0f && isfinite(scale) ? scale : 1.0f;
    }
}

__global__ void quantize_pca_kernel(
    const float* projected,
    const float* bias,
    const float* scales,
    size_t values,
    int dim,
    int8_t* output,
    unsigned long long* clipped) {
    const size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position >= values) return;
    const int d = static_cast<int>(position % static_cast<size_t>(dim));
    const long rounded = lrintf((projected[position] + bias[d]) / scales[d]);
    if (rounded < -127 || rounded > 127) atomicAdd(clipped, 1ULL);
    output[position] = static_cast<int8_t>(max(-127L, min(127L, rounded)));
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void validate_shape(size_t count, int source_dim, int dim) {
    if (count == 0 || source_dim <= 0 || dim <= 0 || dim > source_dim ||
        count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid PCA transform shape");
    }
}

}  // namespace

PCATransform fit_pca_transform(
    const float* vectors,
    size_t count,
    int source_dim,
    int dim) {
    validate_shape(count, source_dim, dim);
    if (!vectors) throw std::invalid_argument("PCA training vectors are null");

    constexpr size_t kChunkRows = 1U << 17;
    const size_t chunk_rows = std::min(count, kChunkRows);
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    float* d_chunk = nullptr;
    double* d_sums = nullptr;
    float* d_mean = nullptr;
    float* d_covariance = nullptr;
    float* d_eigenvalues = nullptr;
    float* d_work = nullptr;
    int* d_info = nullptr;

    auto release = [&] {
        if (d_chunk) cudaFree(d_chunk);
        if (d_sums) cudaFree(d_sums);
        if (d_mean) cudaFree(d_mean);
        if (d_covariance) cudaFree(d_covariance);
        if (d_eigenvalues) cudaFree(d_eigenvalues);
        if (d_work) cudaFree(d_work);
        if (d_info) cudaFree(d_info);
        if (blas) cublasDestroy(blas);
        if (solver) cusolverDnDestroy(solver);
    };

    PCATransform transform;
    transform.source_dim = source_dim;
    transform.dim = dim;
    try {
        const cublasStatus_t blas_status = cublasCreate(&blas);
        if (blas_status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "failed to create PCA cuBLAS handle: status=" +
                std::to_string(static_cast<int>(blas_status)));
        }
        const cusolverStatus_t solver_status = cusolverDnCreate(&solver);
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(
                "failed to create PCA cuSOLVER handle: status=" +
                std::to_string(static_cast<int>(solver_status)));
        }
        check_cuda(
            cudaMalloc(
                &d_chunk,
                chunk_rows * static_cast<size_t>(source_dim) * sizeof(float)),
            "allocate PCA input workspace");
        check_cuda(
            cudaMalloc(&d_sums, static_cast<size_t>(source_dim) * sizeof(double)),
            "allocate PCA mean accumulator");
        check_cuda(
            cudaMalloc(&d_mean, static_cast<size_t>(source_dim) * sizeof(float)),
            "allocate PCA mean");
        check_cuda(
            cudaMemset(d_sums, 0, static_cast<size_t>(source_dim) * sizeof(double)),
            "clear PCA mean accumulator");
        check_cuda(
            cudaMalloc(
                &d_covariance,
                static_cast<size_t>(source_dim) * source_dim * sizeof(float)),
            "allocate PCA covariance");
        check_cuda(
            cudaMalloc(&d_eigenvalues, static_cast<size_t>(source_dim) * sizeof(float)),
            "allocate PCA eigenvalues");
        check_cuda(cudaMalloc(&d_info, sizeof(int)), "allocate PCA solver status");

        constexpr int kThreads = 256;
        for (size_t begin = 0; begin < count; begin += chunk_rows) {
            const int rows = static_cast<int>(std::min(chunk_rows, count - begin));
            check_cuda(
                cudaMemcpy(
                    d_chunk,
                    vectors + begin * static_cast<size_t>(source_dim),
                    static_cast<size_t>(rows) * source_dim * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy PCA mean input chunk");
            accumulate_pca_mean_kernel<<<
                source_dim, kThreads, kThreads * sizeof(double)>>>(
                d_chunk, rows, source_dim, d_sums);
            check_cuda(cudaGetLastError(), "launch PCA mean accumulation");
        }
        finalize_pca_mean_kernel<<<
            (source_dim + kThreads - 1) / kThreads, kThreads>>>(
            d_sums, source_dim, 1.0 / static_cast<double>(count), d_mean);
        check_cuda(cudaGetLastError(), "launch PCA mean finalization");

        const float alpha = 1.0f / static_cast<float>(count);
        for (size_t begin = 0; begin < count; begin += chunk_rows) {
            const int rows = static_cast<int>(std::min(chunk_rows, count - begin));
            check_cuda(
                cudaMemcpy(
                    d_chunk,
                    vectors + begin * static_cast<size_t>(source_dim),
                    static_cast<size_t>(rows) * source_dim * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy PCA training chunk");
            const size_t values = static_cast<size_t>(rows) * source_dim;
            center_pca_input_kernel<<<
                static_cast<unsigned>((values + kThreads - 1) / kThreads), kThreads>>>(
                d_chunk, d_mean, values, source_dim);
            check_cuda(cudaGetLastError(), "launch PCA centering");
            const float beta = begin == 0 ? 0.0f : 1.0f;
            if (cublasSgemm(
                    blas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    source_dim,
                    source_dim,
                    rows,
                    &alpha,
                    d_chunk,
                    source_dim,
                    d_chunk,
                    source_dim,
                    &beta,
                    d_covariance,
                    source_dim) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("PCA covariance GEMM failed");
            }
        }

        int work_size = 0;
        if (cusolverDnSsyevd_bufferSize(
                solver,
                CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER,
                source_dim,
                d_covariance,
                source_dim,
                d_eigenvalues,
                &work_size) != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("PCA eigensolver workspace query failed");
        }
        check_cuda(
            cudaMalloc(&d_work, static_cast<size_t>(work_size) * sizeof(float)),
            "allocate PCA eigensolver workspace");
        if (cusolverDnSsyevd(
                solver,
                CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER,
                source_dim,
                d_covariance,
                source_dim,
                d_eigenvalues,
                d_work,
                work_size,
                d_info) != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("PCA eigensolver launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "synchronize PCA eigensolver");
        int info = 0;
        check_cuda(
            cudaMemcpy(&info, d_info, sizeof(info), cudaMemcpyDeviceToHost),
            "copy PCA solver status");
        if (info != 0) throw std::runtime_error("PCA eigensolver did not converge");

        std::vector<float> eigenvectors(
            static_cast<size_t>(source_dim) * source_dim);
        std::vector<float> mean(static_cast<size_t>(source_dim));
        check_cuda(
            cudaMemcpy(
                eigenvectors.data(),
                d_covariance,
                eigenvectors.size() * sizeof(float),
                cudaMemcpyDeviceToHost),
            "copy PCA eigenvectors");
        check_cuda(
            cudaMemcpy(
                mean.data(), d_mean, mean.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "copy PCA mean");
        transform.projection.resize(static_cast<size_t>(dim) * source_dim);
        transform.bias.resize(static_cast<size_t>(dim));
        for (int output_dim = 0; output_dim < dim; ++output_dim) {
            const int eigenvector = source_dim - 1 - output_dim;
            double projected_mean = 0.0;
            for (int input_dim = 0; input_dim < source_dim; ++input_dim) {
                const float value = eigenvectors[
                    static_cast<size_t>(eigenvector) * source_dim + input_dim];
                transform.projection[
                    static_cast<size_t>(output_dim) * source_dim + input_dim] = value;
                projected_mean += static_cast<double>(value) * mean[input_dim];
            }
            transform.bias[output_dim] = static_cast<float>(-projected_mean);
        }
    } catch (...) {
        release();
        throw;
    }
    release();
    return transform;
}

struct PCAProjector::Impl {
    explicit Impl(const PCATransform& value)
        : transform(value) {
        validate_shape(1, transform.source_dim, transform.dim);
        if (transform.projection.size() !=
                static_cast<size_t>(transform.source_dim) * transform.dim ||
            transform.bias.size() != static_cast<size_t>(transform.dim)) {
            throw std::invalid_argument("invalid PCA projector model");
        }
        try {
            if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("failed to create PCA projector handle");
            }
            check_cuda(
                cudaMalloc(&d_projection, transform.projection.size() * sizeof(float)),
                "allocate PCA projection");
            check_cuda(
                cudaMemcpy(
                    d_projection,
                    transform.projection.data(),
                    transform.projection.size() * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy PCA projection");
        } catch (...) {
            if (d_projection) cudaFree(d_projection);
            if (blas) cublasDestroy(blas);
            d_projection = nullptr;
            blas = nullptr;
            throw;
        }
    }

    ~Impl() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_quantized) cudaFree(d_quantized);
        if (d_bias) cudaFree(d_bias);
        if (d_scales) cudaFree(d_scales);
        if (d_clipped) cudaFree(d_clipped);
        if (d_projection) cudaFree(d_projection);
        if (blas) cublasDestroy(blas);
    }

    void reserve(size_t rows) {
        if (rows <= capacity) return;
        if (d_input) check_cuda(cudaFree(d_input), "release PCA input workspace");
        if (d_output) check_cuda(cudaFree(d_output), "release PCA output workspace");
        if (d_quantized) {
            check_cuda(cudaFree(d_quantized), "release quantized PCA workspace");
        }
        d_input = nullptr;
        d_output = nullptr;
        d_quantized = nullptr;
        check_cuda(
            cudaMalloc(
                &d_input,
                rows * static_cast<size_t>(transform.source_dim) * sizeof(float)),
            "allocate PCA input workspace");
        check_cuda(
            cudaMalloc(
                &d_output,
                rows * static_cast<size_t>(transform.dim) * sizeof(float)),
            "allocate PCA output workspace");
        check_cuda(
            cudaMalloc(
                &d_quantized,
                rows * static_cast<size_t>(transform.dim) * sizeof(int8_t)),
            "allocate quantized PCA workspace");
        capacity = rows;
    }

    void ensure_quantization_workspace() {
        if (d_bias) return;
        check_cuda(
            cudaMalloc(&d_bias, static_cast<size_t>(transform.dim) * sizeof(float)),
            "allocate PCA bias");
        check_cuda(
            cudaMalloc(&d_scales, static_cast<size_t>(transform.dim) * sizeof(float)),
            "allocate PCA quantization scales");
        check_cuda(cudaMalloc(&d_clipped, sizeof(unsigned long long)),
                   "allocate PCA clip counter");
        check_cuda(
            cudaMemcpy(
                d_bias,
                transform.bias.data(),
                static_cast<size_t>(transform.dim) * sizeof(float),
                cudaMemcpyHostToDevice),
            "copy PCA bias");
    }

    PCATransform transform;
    cublasHandle_t blas = nullptr;
    float* d_projection = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;
    int8_t* d_quantized = nullptr;
    float* d_bias = nullptr;
    float* d_scales = nullptr;
    unsigned long long* d_clipped = nullptr;
    size_t capacity = 0;
};

PCAProjector::PCAProjector(const PCATransform& transform)
    : impl_(std::make_unique<Impl>(transform)) {}

PCAProjector::~PCAProjector() = default;

void PCAProjector::apply(const float* input, size_t count, float* output) {
    if (count == 0) return;
    if (!input || !output || count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid PCA projection request");
    }
    impl_->reserve(count);
    check_cuda(
        cudaMemcpy(
            impl_->d_input,
            input,
            count * static_cast<size_t>(impl_->transform.source_dim) * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy PCA projection input");
    orthogonal_projection_apply_device_gemm(
        impl_->blas,
        impl_->d_input,
        impl_->d_projection,
        impl_->d_output,
        count,
        impl_->transform.source_dim,
        impl_->transform.dim,
        nullptr);
    const size_t values = count * static_cast<size_t>(impl_->transform.dim);
    constexpr int kThreads = 256;
    impl_->ensure_quantization_workspace();
    add_pca_bias_kernel<<<
        static_cast<unsigned>((values + kThreads - 1) / kThreads), kThreads>>>(
        impl_->d_output, impl_->d_bias, values, impl_->transform.dim);
    check_cuda(cudaGetLastError(), "launch PCA bias addition");
    check_cuda(
        cudaMemcpy(
            output,
            impl_->d_output,
            count * static_cast<size_t>(impl_->transform.dim) * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy PCA projection output");
}

void PCAProjector::calibrate_scales(
    const float* input,
    size_t count,
    float* scales) {
    if (!input || !scales || count == 0 ||
        count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid PCA scale calibration request");
    }
    impl_->reserve(count);
    impl_->ensure_quantization_workspace();
    check_cuda(
        cudaMemcpy(
            impl_->d_input,
            input,
            count * static_cast<size_t>(impl_->transform.source_dim) * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy PCA calibration input");
    orthogonal_projection_apply_device_gemm(
        impl_->blas,
        impl_->d_input,
        impl_->d_projection,
        impl_->d_output,
        count,
        impl_->transform.source_dim,
        impl_->transform.dim,
        nullptr);
    constexpr int kThreads = 256;
    calibrate_pca_scales_kernel<<<
        impl_->transform.dim, kThreads, kThreads * sizeof(float)>>>(
        impl_->d_output,
        impl_->d_bias,
        count,
        impl_->transform.dim,
        impl_->d_scales);
    check_cuda(cudaGetLastError(), "launch PCA scale calibration");
    check_cuda(
        cudaMemcpy(
            scales,
            impl_->d_scales,
            static_cast<size_t>(impl_->transform.dim) * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy PCA quantization scales");
}

uint64_t PCAProjector::apply_quantized(
    const float* input,
    size_t count,
    const float* scales,
    int8_t* output) {
    if (count == 0) return 0;
    if (!input || !scales || !output ||
        count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid quantized PCA projection request");
    }
    for (int d = 0; d < impl_->transform.dim; ++d) {
        if (!(scales[d] > 0.0f) || !std::isfinite(scales[d])) {
            throw std::invalid_argument("invalid PCA quantization scale");
        }
    }
    impl_->reserve(count);
    impl_->ensure_quantization_workspace();
    check_cuda(
        cudaMemcpy(
            impl_->d_scales,
            scales,
            static_cast<size_t>(impl_->transform.dim) * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy PCA quantization scales");
    check_cuda(cudaMemset(impl_->d_clipped, 0, sizeof(unsigned long long)),
               "clear PCA clip counter");
    check_cuda(
        cudaMemcpy(
            impl_->d_input,
            input,
            count * static_cast<size_t>(impl_->transform.source_dim) * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy quantized PCA input");
    orthogonal_projection_apply_device_gemm(
        impl_->blas,
        impl_->d_input,
        impl_->d_projection,
        impl_->d_output,
        count,
        impl_->transform.source_dim,
        impl_->transform.dim,
        nullptr);
    const size_t values = count * static_cast<size_t>(impl_->transform.dim);
    constexpr int kThreads = 256;
    quantize_pca_kernel<<<
        static_cast<unsigned>((values + kThreads - 1) / kThreads), kThreads>>>(
        impl_->d_output,
        impl_->d_bias,
        impl_->d_scales,
        values,
        impl_->transform.dim,
        impl_->d_quantized,
        impl_->d_clipped);
    check_cuda(cudaGetLastError(), "launch PCA quantization");
    check_cuda(
        cudaMemcpy(output, impl_->d_quantized, values * sizeof(int8_t),
                   cudaMemcpyDeviceToHost),
        "copy quantized PCA output");
    unsigned long long clipped = 0;
    check_cuda(
        cudaMemcpy(
            &clipped,
            impl_->d_clipped,
            sizeof(clipped),
            cudaMemcpyDeviceToHost),
        "copy PCA clip counter");
    return static_cast<uint64_t>(clipped);
}

}  // namespace paraselect::transform
