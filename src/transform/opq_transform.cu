#include "transform/opq_transform.cuh"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace paraselect::transform {

OPQTransform::OPQTransform(const OPQTransformSpec& spec) : spec_(spec) {
    validate_spec();
    set_identity();
}

OPQTransform::OPQTransform(int dim) : OPQTransform(OPQTransformSpec{dim}) {}

OPQTransform::~OPQTransform() {
    release();
}

void OPQTransform::validate_spec() const {
    if (spec_.dim <= 0) throw std::invalid_argument("OPQTransform: dim must be positive");
}

void OPQTransform::set_matrix(const std::vector<float>& matrix_rowmajor_out_in) {
    validate_spec();
    if (matrix_rowmajor_out_in.size() != (size_t)spec_.dim * (size_t)spec_.dim) {
        throw std::invalid_argument("OPQTransform: invalid matrix size");
    }
    h_matrix_ = matrix_rowmajor_out_in;
    if (prepared_) {
        const size_t bytes = h_matrix_.size() * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_matrix_, h_matrix_.data(), bytes, cudaMemcpyHostToDevice));
    }
}

void OPQTransform::set_identity() {
    validate_spec();
    h_matrix_.assign((size_t)spec_.dim * (size_t)spec_.dim, 0.0f);
    for (int i = 0; i < spec_.dim; ++i) h_matrix_[(size_t)i * spec_.dim + i] = 1.0f;
    if (prepared_) {
        const size_t bytes = h_matrix_.size() * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_matrix_, h_matrix_.data(), bytes, cudaMemcpyHostToDevice));
    }
}

void OPQTransform::prepare() {
    if (prepared_) return;
    validate_spec();
    if (h_matrix_.empty()) set_identity();
    const size_t bytes = h_matrix_.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_matrix_, bytes));
    CUDA_CHECK(cudaMemcpy(d_matrix_, h_matrix_.data(), bytes, cudaMemcpyHostToDevice));
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaFree(d_matrix_));
        d_matrix_ = nullptr;
        throw std::runtime_error("OPQTransform: cublasCreate failed");
    }
    prepared_ = true;
}

void OPQTransform::release() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (d_matrix_) {
        cudaFree(d_matrix_);
        d_matrix_ = nullptr;
    }
    prepared_ = false;
}

void OPQTransform::apply_device(const float* d_input, float* d_output, size_t n, cudaStream_t stream) const {
    if (!prepared_ || !d_matrix_ || !cublas_handle_) {
        throw std::runtime_error("OPQTransform: prepare() must be called before apply_device");
    }
    opq_apply_device_gemm(cublas_handle_, d_input, d_matrix_, d_output, n, spec_.dim, stream);
}

void OPQTransform::apply_host_to_host(const float* h_input, float* h_output, size_t n, cudaStream_t stream) const {
    if (!prepared_) throw std::runtime_error("OPQTransform: prepare() must be called before apply_host_to_host");
    if (n == 0) return;
    if (!h_input || !h_output) throw std::invalid_argument("OPQTransform::apply_host_to_host: null pointer");

    size_t chunk_rows = 1 << 20;
    if (const char* env = std::getenv("PARASELECT_OPQ_TRANSFORM_CHUNK_ROWS")) {
        if (*env) chunk_rows = std::max<size_t>(1, std::strtoull(env, nullptr, 10));
    }
    chunk_rows = std::min(chunk_rows, n);

    float* d_input = nullptr;
    float* d_output = nullptr;
    const size_t chunk_bytes = chunk_rows * (size_t)spec_.dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, chunk_bytes));
    try {
        for (size_t start = 0; start < n; start += chunk_rows) {
            const size_t rows = std::min(chunk_rows, n - start);
            const size_t bytes = rows * (size_t)spec_.dim * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(
                d_input,
                h_input + start * (size_t)spec_.dim,
                bytes,
                cudaMemcpyHostToDevice,
                stream));
            apply_device(d_input, d_output, rows, stream);
            CUDA_CHECK(cudaMemcpyAsync(
                h_output + start * (size_t)spec_.dim,
                d_output,
                bytes,
                cudaMemcpyDeviceToHost,
                stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw;
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void opq_apply_device_gemm(
    cublasHandle_t handle,
    const float* d_input,
    const float* d_matrix_rowmajor_out_in,
    float* d_output,
    size_t n,
    int dim,
    cudaStream_t stream)
{
    if (n == 0) return;
    if (!handle || !d_input || !d_matrix_rowmajor_out_in || !d_output) {
        throw std::invalid_argument("opq_apply_device_gemm: null pointer");
    }
    if (dim <= 0) throw std::invalid_argument("opq_apply_device_gemm: invalid dim");
    if (n > (size_t)std::numeric_limits<int>::max()) {
        throw std::invalid_argument("opq_apply_device_gemm: n exceeds cublas int range");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetStream(handle, stream);
    const cublasStatus_t st = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        dim,
        (int)n,
        dim,
        &alpha,
        d_matrix_rowmajor_out_in,
        dim,
        d_input,
        dim,
        &beta,
        d_output,
        dim);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("opq_apply_device_gemm: cublasSgemm failed");
    }
    CUDA_LAUNCH_CHECK("opq_apply_device_gemm cublasSgemm");
}

}  // namespace paraselect::transform
