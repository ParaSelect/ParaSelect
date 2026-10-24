#include "common/test_utils.cuh"

#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

template <int BM, int BN, int BK>
__global__ void wmma_gemm_nt_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int m,
    int n,
    int k,
    float* __restrict__ c)
{
    using namespace nvcuda;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;
    constexpr int WARPS_M = BM / WMMA_M;
    constexpr int WARPS_N = BN / WMMA_N;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    if (warp_id >= WARPS_M * WARPS_N) return;

    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id - warp_m * WARPS_N;
    const int row_base = blockIdx.y * BM + warp_m * WMMA_M;
    const int col_base = blockIdx.x * BN + warp_n * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int kb = 0; kb < k; kb += WMMA_K) {
        const float* a_ptr = a + (size_t)row_base * k + kb;
        const float* b_ptr = b + (size_t)col_base * k + kb;
        wmma::load_matrix_sync(a_frag, a_ptr, k);
        wmma::load_matrix_sync(b_frag, b_ptr, k);
#pragma unroll
        for (int e = 0; e < a_frag.num_elements; ++e) {
            a_frag.x[e] = wmma::__float_to_tf32(a_frag.x[e]);
        }
#pragma unroll
        for (int e = 0; e < b_frag.num_elements; ++e) {
            b_frag.x[e] = wmma::__float_to_tf32(b_frag.x[e]);
        }
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(c + (size_t)row_base * n + col_base,
                            acc_frag,
                            n,
                            wmma::mem_row_major);
}

float time_cublas_sgemm_nt(
    cublasHandle_t handle,
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c)
{
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_a,
        k,
        d_b,
        k,
        &beta,
        d_c,
        m);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <int BM, int BN, int BK>
float time_wmma_gemm_nt(
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c)
{
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const dim3 block((BM / 16) * (BN / 16) * 32);
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    cudaEventRecord(start);
    wmma_gemm_nt_kernel<BM, BN, BK><<<grid, block>>>(d_a, d_b, m, n, k, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

void fill_random(std::vector<float>& v) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

}  // namespace

int main(int argc, char** argv) {
    int m = 65536;
    int n = 1024;
    int k = 128;
    int rounds = 20;
    if (argc >= 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }
    if (argc >= 5) rounds = std::atoi(argv[4]);

    paraselect::test::TestContext ctx(
        "wmma_gemm_vs_cublas",
        paraselect::test::TestKind::Benchmark);

    if ((m % 128) != 0 || (n % 64) != 0 || (k % 8) != 0) {
        std::cerr << "requires m%128==0 n%64==0 k%8==0" << std::endl;
        return 2;
    }

    std::vector<float> h_a((size_t)m * k);
    std::vector<float> h_b((size_t)n * k);
    fill_random(h_a);
    fill_random(h_b);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c_wmma = nullptr;
    float* d_c_cublas = nullptr;
    cudaMalloc(&d_a, sizeof(float) * h_a.size());
    cudaMalloc(&d_b, sizeof(float) * h_b.size());
    cudaMalloc(&d_c_wmma, sizeof(float) * (size_t)m * n);
    cudaMalloc(&d_c_cublas, sizeof(float) * (size_t)m * n);
    cudaMemcpy(d_a, h_a.data(), sizeof(float) * h_a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), sizeof(float) * h_b.size(), cudaMemcpyHostToDevice);

    cublasHandle_t handle{};
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    (void)time_wmma_gemm_nt<128, 64, 8>(d_a, d_b, m, n, k, d_c_wmma);
    (void)time_cublas_sgemm_nt(handle, d_a, d_b, m, n, k, d_c_cublas);

    float wmma_total = 0.0f;
    float cublas_total = 0.0f;
    for (int i = 0; i < rounds; ++i) {
        wmma_total += time_wmma_gemm_nt<128, 64, 8>(d_a, d_b, m, n, k, d_c_wmma);
        cublas_total += time_cublas_sgemm_nt(handle, d_a, d_b, m, n, k, d_c_cublas);
    }
    const float wmma_ms = wmma_total / rounds;
    const float cublas_ms = cublas_total / rounds;

    std::vector<float> h_wmma(4096);
    std::vector<float> h_cublas(4096);
    cudaMemcpy(h_wmma.data(), d_c_wmma, sizeof(float) * h_wmma.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cublas.data(), d_c_cublas, sizeof(float) * h_cublas.size(), cudaMemcpyDeviceToHost);
    double max_abs_err = 0.0;
    for (size_t i = 0; i < h_wmma.size(); ++i) {
        max_abs_err = std::max(max_abs_err, (double)std::fabs(h_wmma[i] - h_cublas[i]));
    }

    const double ops = 2.0 * (double)m * (double)n * (double)k;
    std::cout << "[wmma_gemm_vs_cublas] m=" << m
              << " n=" << n
              << " k=" << k
              << " rounds=" << rounds
              << " wmma_ms=" << wmma_ms
              << " cublas_ms=" << cublas_ms
              << " speedup=" << cublas_ms / std::max(1e-6f, wmma_ms)
              << " wmma_tflops=" << ops / (wmma_ms * 1e9)
              << " cublas_tflops=" << ops / (cublas_ms * 1e9)
              << " max_abs_err_sample=" << max_abs_err
              << std::endl;

    ctx.check(max_abs_err < 2e-2, "WMMA TF32 sample should be close to cuBLAS");

    cublasDestroy(handle);
    cudaFree(d_c_cublas);
    cudaFree(d_c_wmma);
    cudaFree(d_b);
    cudaFree(d_a);
    return ctx.exit_code();
}
