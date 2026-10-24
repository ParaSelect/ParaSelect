#include "transform/opq_trainer.cuh"

#include "quantization/pq/pq_trainer.cuh"
#include "quantization/pq/pq_encoder.cuh"
#include "transform/opq_transform.cuh"

#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

namespace paraselect::transform {
namespace {

double opq_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void opq_log(const char* stage, int iter, double start_ms = 0.0) {
    if (start_ms > 0.0) {
        std::fprintf(stderr, "[OPQTrain] iter=%d stage=%s done ms=%.3f\n", iter + 1, stage, opq_now_ms() - start_ms);
    } else {
        std::fprintf(stderr, "[OPQTrain] iter=%d stage=%s begin\n", iter + 1, stage);
    }
    std::fflush(stderr);
}

void validate_spec(const float* base, const OPQTrainSpec& spec) {
    if (!base) throw std::invalid_argument("train_opq_transform: null base");
    if (spec.n <= 0) throw std::invalid_argument("train_opq_transform: n must be positive");
    if (spec.dim <= 0) throw std::invalid_argument("train_opq_transform: dim must be positive");
    if (spec.pq_m <= 0) throw std::invalid_argument("train_opq_transform: pq_m must be positive");
    if (spec.dim % spec.pq_m != 0) throw std::invalid_argument("train_opq_transform: dim must be divisible by pq_m");
    if (spec.pq_k <= 0 || spec.pq_k > 256) throw std::invalid_argument("train_opq_transform: pq_k must be in [1, 256]");
    if (spec.n < spec.pq_k) throw std::invalid_argument("train_opq_transform: n must be >= pq_k");
    if (spec.opq_iters <= 0) throw std::invalid_argument("train_opq_transform: opq_iters must be positive");
    if (spec.pq_iters <= 0) throw std::invalid_argument("train_opq_transform: pq_iters must be positive");
}

std::vector<float> identity_matrix(int dim) {
    std::vector<float> r((size_t)dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) r[(size_t)i * dim + i] = 1.0f;
    return r;
}

std::vector<float> compute_cross_covariance_xt_y(const float* x, const float* y, int n, int dim) {
    if (!x || !y) throw std::invalid_argument("compute_cross_covariance_xt_y: null pointer");
    if (n <= 0 || dim <= 0) throw std::invalid_argument("compute_cross_covariance_xt_y: invalid shape");

    size_t chunk_rows = 1 << 20;
    if (const char* env = std::getenv("PARASELECT_OPQ_COV_CHUNK_ROWS")) {
        if (*env) chunk_rows = std::max<size_t>(1, std::strtoull(env, nullptr, 10));
    }
    chunk_rows = std::min(chunk_rows, (size_t)n);

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("OPQ trainer: cublasCreate failed for cross covariance");
    }

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_c = nullptr;
    const size_t chunk_elems = chunk_rows * (size_t)dim;
    const size_t matrix_elems = (size_t)dim * dim;
    std::vector<float> c_colmajor(matrix_elems, 0.0f);
    try {
        CUDA_CHECK(cudaMalloc(&d_x, chunk_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, chunk_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c, matrix_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_c, 0, matrix_elems * sizeof(float)));

        const float one = 1.0f;
        for (size_t start = 0; start < (size_t)n; start += chunk_rows) {
            const size_t rows = std::min(chunk_rows, (size_t)n - start);
            const size_t bytes = rows * (size_t)dim * sizeof(float);
            CUDA_CHECK(cudaMemcpy(d_x, x + start * (size_t)dim, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_y, y + start * (size_t)dim, bytes, cudaMemcpyHostToDevice));
            const float beta = (start == 0) ? 0.0f : 1.0f;
            const cublasStatus_t st = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                dim,
                dim,
                (int)rows,
                &one,
                d_x,
                dim,
                d_y,
                dim,
                &beta,
                d_c,
                dim);
            if (st != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("OPQ trainer: cross covariance cublasSgemm failed");
            }
        }
        CUDA_CHECK(cudaMemcpy(c_colmajor.data(), d_c, matrix_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_c);
        cublasDestroy(handle);
    } catch (...) {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_c) cudaFree(d_c);
        cublasDestroy(handle);
        throw;
    }

    std::vector<float> a((size_t)dim * dim);
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            a[(size_t)r * dim + c] = c_colmajor[(size_t)r + (size_t)c * dim];
        }
    }
    return a;
}

std::vector<float> transpose_rowmajor(const std::vector<float>& a, int rows, int cols) {
    std::vector<float> out((size_t)rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) out[(size_t)c * rows + r] = a[(size_t)r * cols + c];
    }
    return out;
}

std::vector<float> matmul_rowmajor(const std::vector<float>& a, const std::vector<float>& b, int m, int k, int n) {
    std::vector<float> c((size_t)m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int kk = 0; kk < k; ++kk) {
            const float av = a[(size_t)i * k + kk];
            const float* brow = b.data() + (size_t)kk * n;
            float* crow = c.data() + (size_t)i * n;
            for (int j = 0; j < n; ++j) crow[j] += av * brow[j];
        }
    }
    return c;
}

std::vector<float> procrustes_rotation_rowmajor_out_in(const std::vector<float>& xt_y, int dim) {
    cusolverDnHandle_t solver = nullptr;
    if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("OPQ trainer: cusolverDnCreate failed");
    }

    float* d_a = nullptr;
    float* d_s = nullptr;
    float* d_u = nullptr;
    float* d_vt = nullptr;
    float* d_work = nullptr;
    int* d_info = nullptr;
    const size_t matrix_elems = (size_t)dim * dim;

    std::vector<float> a_colmajor(matrix_elems);
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) a_colmajor[(size_t)c * dim + r] = xt_y[(size_t)r * dim + c];
    }

    int lwork = 0;
    try {
        CUDA_CHECK(cudaMalloc(&d_a, matrix_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_s, (size_t)dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_u, matrix_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vt, matrix_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_a, a_colmajor.data(), matrix_elems * sizeof(float), cudaMemcpyHostToDevice));

        cusolverStatus_t st = cusolverDnSgesvd_bufferSize(solver, dim, dim, &lwork);
        if (st != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("OPQ trainer: SVD bufferSize failed");
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(float)));

        signed char jobu = 'A';
        signed char jobvt = 'A';
        st = cusolverDnSgesvd(
            solver,
            jobu,
            jobvt,
            dim,
            dim,
            d_a,
            dim,
            d_s,
            d_u,
            dim,
            d_vt,
            dim,
            d_work,
            lwork,
            nullptr,
            d_info);
        if (st != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("OPQ trainer: SVD failed");
        CUDA_CHECK(cudaDeviceSynchronize());

        int info = 0;
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) throw std::runtime_error("OPQ trainer: SVD did not converge");

        std::vector<float> u_colmajor(matrix_elems);
        std::vector<float> vt_colmajor(matrix_elems);
        CUDA_CHECK(cudaMemcpy(u_colmajor.data(), d_u, matrix_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vt_colmajor.data(), d_vt, matrix_elems * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> u_rowmajor(matrix_elems);
        std::vector<float> vt_rowmajor(matrix_elems);
        for (int r = 0; r < dim; ++r) {
            for (int c = 0; c < dim; ++c) {
                u_rowmajor[(size_t)r * dim + c] = u_colmajor[(size_t)c * dim + r];
                vt_rowmajor[(size_t)r * dim + c] = vt_colmajor[(size_t)c * dim + r];
            }
        }

        // minimize ||X * R^T - Yhat|| with A = X^T Yhat, SVD(A)=U*S*V^T.
        // The transform stores R, so R^T = U*V^T and R = V*U^T.
        const std::vector<float> v_rowmajor = transpose_rowmajor(vt_rowmajor, dim, dim);
        const std::vector<float> u_t_rowmajor = transpose_rowmajor(u_rowmajor, dim, dim);
        std::vector<float> rotation = matmul_rowmajor(v_rowmajor, u_t_rowmajor, dim, dim, dim);

        cudaFree(d_a);
        cudaFree(d_s);
        cudaFree(d_u);
        cudaFree(d_vt);
        cudaFree(d_work);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        return rotation;
    } catch (...) {
        if (d_a) cudaFree(d_a);
        if (d_s) cudaFree(d_s);
        if (d_u) cudaFree(d_u);
        if (d_vt) cudaFree(d_vt);
        if (d_work) cudaFree(d_work);
        if (d_info) cudaFree(d_info);
        cusolverDnDestroy(solver);
        throw;
    }
}

}  // namespace

void opq_encode_plain_pq(
    const float* data,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    const float* codebooks,
    uint8_t* codes)
{
    if (!data || !codebooks || !codes) throw std::invalid_argument("opq_encode_plain_pq: null pointer");
    if (n <= 0 || dim <= 0 || pq_m <= 0 || dim % pq_m != 0 || pq_k <= 0 || pq_k > 256) {
        throw std::invalid_argument("opq_encode_plain_pq: invalid shape");
    }
    const int sub_dim = dim / pq_m;
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < pq_m; ++m) {
            const float* x = data + (size_t)i * dim + (size_t)m * sub_dim;
            const float* cb = codebooks + (size_t)m * pq_k * sub_dim;
            int best = 0;
            float best_sq = std::numeric_limits<float>::infinity();
            for (int k = 0; k < pq_k; ++k) {
                const float* c = cb + (size_t)k * sub_dim;
                float sq = 0.0f;
                for (int d = 0; d < sub_dim; ++d) {
                    const float diff = x[d] - c[d];
                    sq += diff * diff;
                }
                if (sq < best_sq) {
                    best_sq = sq;
                    best = k;
                }
            }
            codes[(size_t)i * pq_m + m] = (uint8_t)best;
        }
    }
}

void opq_decode_plain_pq(
    const uint8_t* codes,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    const float* codebooks,
    float* reconstructed)
{
    if (!codes || !codebooks || !reconstructed) throw std::invalid_argument("opq_decode_plain_pq: null pointer");
    if (n <= 0 || dim <= 0 || pq_m <= 0 || dim % pq_m != 0 || pq_k <= 0 || pq_k > 256) {
        throw std::invalid_argument("opq_decode_plain_pq: invalid shape");
    }
    const int sub_dim = dim / pq_m;
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < pq_m; ++m) {
            const int code = (int)codes[(size_t)i * pq_m + m];
            const float* src = codebooks + ((size_t)m * pq_k + code) * sub_dim;
            float* dst = reconstructed + (size_t)i * dim + (size_t)m * sub_dim;
            std::memcpy(dst, src, (size_t)sub_dim * sizeof(float));
        }
    }
}



void opq_encode_plain_pq_gpu(
    const float* data,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    const float* codebooks,
    uint8_t* codes,
    size_t chunk_rows)
{
    if (!data || !codebooks || !codes) throw std::invalid_argument("opq_encode_plain_pq_gpu: null pointer");
    if (n <= 0 || dim <= 0 || pq_m <= 0 || dim % pq_m != 0 || pq_k <= 0 || pq_k > 256) {
        throw std::invalid_argument("opq_encode_plain_pq_gpu: invalid shape");
    }
    if (chunk_rows == 0) chunk_rows = 1 << 20;
    const size_t max_rows = std::min(chunk_rows, static_cast<size_t>(n));
    quantization::pq::PQEncoderGPU encoder(codebooks, dim, pq_m, pq_k, max_rows);
    for (size_t start = 0; start < static_cast<size_t>(n); start += max_rows) {
        const int rows = static_cast<int>(std::min(max_rows, static_cast<size_t>(n) - start));
        encoder.encode(data + start * dim, rows, codes + start * pq_m);
    }
}

float opq_reconstruction_objective(const float* data, const float* reconstructed, int n, int dim) {
    if (!data || !reconstructed) throw std::invalid_argument("opq_reconstruction_objective: null pointer");
    if (n <= 0 || dim <= 0) throw std::invalid_argument("opq_reconstruction_objective: invalid shape");
    double sum = 0.0;
    const size_t total = (size_t)n * dim;
    for (size_t i = 0; i < total; ++i) {
        const double diff = (double)data[i] - (double)reconstructed[i];
        sum += diff * diff;
    }
    return (float)(sum / (double)n);
}

OPQTrainResult train_opq_transform(const float* base_rowmajor, const OPQTrainSpec& spec) {
    validate_spec(base_rowmajor, spec);

    OPQTrainResult result;
    result.rotation_rowmajor_out_in = identity_matrix(spec.dim);
    std::vector<float> rotated((size_t)spec.n * spec.dim);
    result.codes.resize((size_t)spec.n * spec.pq_m);
    result.reconstructed_rotated.resize((size_t)spec.n * spec.dim);
    result.objective_by_iter.reserve((size_t)spec.opq_iters);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    try {
        OPQTransform transform(spec.dim);
        transform.prepare();
        std::fprintf(stderr,
                     "[OPQTrain] begin n=%d dim=%d M=%d K=%d opq_iters=%d pq_iters=%d\n",
                     spec.n, spec.dim, spec.pq_m, spec.pq_k, spec.opq_iters, spec.pq_iters);
        std::fflush(stderr);
        for (int iter = 0; iter < spec.opq_iters; ++iter) {
            double t = 0.0;

            opq_log("rotate", iter);
            t = opq_now_ms();
            transform.set_matrix(result.rotation_rowmajor_out_in);
            transform.apply_host_to_host(base_rowmajor, rotated.data(), (size_t)spec.n, stream);
            opq_log("rotate", iter, t);

            ivf::PlainPQTrainConfig pq_cfg;
            pq_cfg.n = spec.n;
            pq_cfg.dim = spec.dim;
            pq_cfg.n_subspace = spec.pq_m;
            pq_cfg.n_codes = spec.pq_k;
            pq_cfg.iters = spec.pq_iters;
            pq_cfg.seed = spec.seed + (unsigned)iter * 1000003u;

            opq_log("pq_train", iter);
            t = opq_now_ms();
            ivf::train_plain_pq_codebooks_gpu(rotated.data(), pq_cfg, result.codebooks);
            opq_log("pq_train", iter, t);

            opq_log("pq_encode_gpu", iter);
            t = opq_now_ms();
            opq_encode_plain_pq_gpu(
                rotated.data(),
                spec.n,
                spec.dim,
                spec.pq_m,
                spec.pq_k,
                result.codebooks.data(),
                result.codes.data());
            opq_log("pq_encode_gpu", iter, t);

            opq_log("pq_decode_cpu", iter);
            t = opq_now_ms();
            opq_decode_plain_pq(
                result.codes.data(),
                spec.n,
                spec.dim,
                spec.pq_m,
                spec.pq_k,
                result.codebooks.data(),
                result.reconstructed_rotated.data());
            opq_log("pq_decode_cpu", iter, t);

            opq_log("objective_cpu", iter);
            t = opq_now_ms();
            result.objective_by_iter.push_back(opq_reconstruction_objective(
                rotated.data(), result.reconstructed_rotated.data(), spec.n, spec.dim));
            opq_log("objective_cpu", iter, t);

            std::fprintf(stderr,
                         "[OPQTrain] iter=%d objective=%g\n",
                         iter + 1,
                         result.objective_by_iter.back());
            std::fflush(stderr);

            if (iter + 1 == spec.opq_iters) break;
            opq_log("cross_cov_cpu", iter);
            t = opq_now_ms();
            const std::vector<float> xt_y = compute_cross_covariance_xt_y(
                base_rowmajor, result.reconstructed_rotated.data(), spec.n, spec.dim);
            opq_log("cross_cov_cpu", iter, t);

            opq_log("procrustes_svd", iter);
            t = opq_now_ms();
            result.rotation_rowmajor_out_in = procrustes_rotation_rowmajor_out_in(xt_y, spec.dim);
            opq_log("procrustes_svd", iter, t);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }

    return result;
}

}  // namespace paraselect::transform
