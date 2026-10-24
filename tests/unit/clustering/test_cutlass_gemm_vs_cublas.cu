#include "common/test_utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

using ElementOutput = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

template <int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM = 64,
          int WarpN = 64,
          int WarpK = ThreadblockK,
          int Stages = 3,
          typename LayoutOutput = cutlass::layout::RowMajor,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using CutlassGemmBase = cutlass::gemm::device::Gemm<
    float,
    LayoutA,
    float,
    LayoutB,
    ElementOutput,
    LayoutOutput,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        float,
        float>,
    ThreadblockSwizzle,
    Stages>;

struct CutlassTiming {
    const char* name = "";
    float ms = 0.0f;
    bool supported = false;
    cutlass::Status status = cutlass::Status::kSuccess;
    cudaError_t cuda_status = cudaSuccess;
    size_t shared_storage_bytes = 0;
};

template <typename Gemm>
size_t cutlass_shared_storage_bytes() {
    return sizeof(typename Gemm::GemmKernel::SharedStorage);
}

void fill_random(std::vector<float>& v) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(2);
    }
}

void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << what << ": cublasStatus=" << static_cast<int>(status) << std::endl;
        std::exit(2);
    }
}

void check_cutlass(cutlass::Status status, const char* what) {
    if (status != cutlass::Status::kSuccess) {
        std::cerr << what << ": " << cutlassGetStatusString(status) << std::endl;
        std::exit(2);
    }
}

float time_cublas_rowmajor_abt(
    cublasHandle_t handle,
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c,
    int rounds)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < rounds; ++i) {
        // Row-major C[m,n] = A[m,k] * B[n,k]^T has the same memory layout as
        // column-major C_col[n,m] = B[n,k] * A[m,k]^T.
        check_cublas(cublasSgemm(
                         handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         n,
                         m,
                         k,
                         &alpha,
                         d_b,
                         k,
                         d_a,
                         k,
                         &beta,
                         d_c,
                         n),
                     "cublasSgemm");
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    return ms / rounds;
}

template <typename Gemm>
CutlassTiming time_cutlass_rowmajor_abt(
    const char* name,
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c,
    int rounds)
{
    Gemm gemm;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmCoord(m, n, k),
        {d_a, k},
        {d_b, k},
        {d_c, n},
        {d_c, n},
        {1.0f, 0.0f});

    const cutlass::Status can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        return {name, 0.0f, false, can_implement, cudaSuccess, cutlass_shared_storage_bytes<Gemm>()};
    }
    const cutlass::Status init_status = gemm.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
        return {name, 0.0f, false, init_status, cudaGetLastError(), cutlass_shared_storage_bytes<Gemm>()};
    }

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < rounds; ++i) {
        const cutlass::Status launch_status = gemm();
        if (launch_status != cutlass::Status::kSuccess) {
            cudaError_t cuda_status = cudaGetLastError();
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return {name, 0.0f, false, launch_status, cuda_status, cutlass_shared_storage_bytes<Gemm>()};
        }
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    return {name, ms / rounds, true, cutlass::Status::kSuccess, cudaSuccess, cutlass_shared_storage_bytes<Gemm>()};
}

template <typename Gemm>
CutlassTiming run_cutlass_variant(
    const char* name,
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c,
    int rounds)
{
    CutlassTiming warmup = time_cutlass_rowmajor_abt<Gemm>(
        name, d_a, d_b, m, n, k, d_c, 3);
    if (!warmup.supported) return warmup;
    return time_cutlass_rowmajor_abt<Gemm>(
        name, d_a, d_b, m, n, k, d_c, rounds);
}

template <typename Gemm>
CutlassTiming time_cutlass_transposed_problem_abt(
    const char* name,
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c,
    int rounds)
{
    Gemm gemm;
    typename Gemm::Arguments args(
        // Same row-major C[m,n] memory as column-major C_col[n,m].
        cutlass::gemm::GemmCoord(n, m, k),
        {d_b, k},
        {d_a, k},
        {d_c, n},
        {d_c, n},
        {1.0f, 0.0f});

    const cutlass::Status can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        return {name, 0.0f, false, can_implement, cudaSuccess, cutlass_shared_storage_bytes<Gemm>()};
    }
    const cutlass::Status init_status = gemm.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
        return {name, 0.0f, false, init_status, cudaGetLastError(), cutlass_shared_storage_bytes<Gemm>()};
    }

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < rounds; ++i) {
        const cutlass::Status launch_status = gemm();
        if (launch_status != cutlass::Status::kSuccess) {
            cudaError_t cuda_status = cudaGetLastError();
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return {name, 0.0f, false, launch_status, cuda_status, cutlass_shared_storage_bytes<Gemm>()};
        }
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    return {name, ms / rounds, true, cutlass::Status::kSuccess, cudaSuccess, cutlass_shared_storage_bytes<Gemm>()};
}

template <typename Gemm>
CutlassTiming run_cutlass_transposed_variant(
    const char* name,
    const float* d_a,
    const float* d_b,
    int m,
    int n,
    int k,
    float* d_c,
    int rounds)
{
    CutlassTiming warmup = time_cutlass_transposed_problem_abt<Gemm>(
        name, d_a, d_b, m, n, k, d_c, 3);
    if (!warmup.supported) return warmup;
    return time_cutlass_transposed_problem_abt<Gemm>(
        name, d_a, d_b, m, n, k, d_c, rounds);
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
    rounds = std::max(rounds, 1);

    paraselect::test::TestContext ctx(
        "cutlass_gemm_vs_cublas",
        paraselect::test::TestKind::Benchmark);

    std::vector<float> h_a((size_t)m * k);
    std::vector<float> h_b((size_t)n * k);
    fill_random(h_a);
    fill_random(h_b);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c_cutlass = nullptr;
    float* d_c_cublas = nullptr;
    check_cuda(cudaMalloc(&d_a, sizeof(float) * h_a.size()), "cudaMalloc(d_a)");
    check_cuda(cudaMalloc(&d_b, sizeof(float) * h_b.size()), "cudaMalloc(d_b)");
    check_cuda(cudaMalloc(&d_c_cutlass, sizeof(float) * (size_t)m * n), "cudaMalloc(d_c_cutlass)");
    check_cuda(cudaMalloc(&d_c_cublas, sizeof(float) * (size_t)m * n), "cudaMalloc(d_c_cublas)");
    check_cuda(cudaMemcpy(d_a, h_a.data(), sizeof(float) * h_a.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy(d_a)");
    check_cuda(cudaMemcpy(d_b, h_b.data(), sizeof(float) * h_b.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy(d_b)");

    cublasHandle_t handle{};
    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode");

    int max_dynamic_smem = 0;
    check_cuda(cudaDeviceGetAttribute(
                   &max_dynamic_smem,
                   cudaDevAttrMaxSharedMemoryPerBlockOptin,
                   0),
               "cudaDeviceGetAttribute(max opt-in shared memory)");

    (void)time_cublas_rowmajor_abt(handle, d_a, d_b, m, n, k, d_c_cublas, 3);

    using GemmF32_128x128x16 = CutlassGemmBase<128, 128, 16, 64, 64, 16, 4>;
    using GemmF32_128x128x32 = CutlassGemmBase<128, 128, 32>;
    using GemmF32_128x256x32 = CutlassGemmBase<128, 256, 32>;
    using GemmF32_256x128x32 = CutlassGemmBase<256, 128, 32>;
    using GemmF32_64x256x32 = CutlassGemmBase<64, 256, 32>;
    using GemmF32_256x64x32 = CutlassGemmBase<256, 64, 32>;
    using GemmF32_128x256x32s2 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2>;
    using GemmF32_256x128x32s2 = CutlassGemmBase<256, 128, 32, 64, 64, 32, 2>;
    using GemmF32_64x256x32s2 = CutlassGemmBase<64, 256, 32, 64, 64, 32, 2>;
    using GemmF32_256x64x32s2 = CutlassGemmBase<256, 64, 32, 64, 64, 32, 2>;
    using GemmF32_128x256x16 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4>;
    using GemmF32_256x128x16 = CutlassGemmBase<256, 128, 16, 64, 64, 16, 4>;
    using GemmT_128x128x16 = CutlassGemmBase<128, 128, 16, 64, 64, 16, 4, cutlass::layout::ColumnMajor>;
    using GemmT_128x256x16 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::ColumnMajor>;
    using GemmT_256x128x16 = CutlassGemmBase<256, 128, 16, 64, 64, 16, 4, cutlass::layout::ColumnMajor>;
    using GemmT_128x256x32s2 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::ColumnMajor>;
    using GemmT_256x128x32s2 = CutlassGemmBase<256, 128, 32, 64, 64, 32, 2, cutlass::layout::ColumnMajor>;
    using Id2 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>;
    using Id4 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
    using Id8 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
    using GemmF32_128x256x16_id2 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::RowMajor, Id2>;
    using GemmF32_128x256x16_id4 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::RowMajor, Id4>;
    using GemmF32_128x256x16_id8 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::RowMajor, Id8>;
    using GemmF32_128x256x32s2_id2 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::RowMajor, Id2>;
    using GemmF32_128x256x32s2_id4 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::RowMajor, Id4>;
    using GemmF32_128x256x32s2_id8 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::RowMajor, Id8>;
    using GemmT_128x256x16_id2 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::ColumnMajor, Id2>;
    using GemmT_128x256x16_id4 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::ColumnMajor, Id4>;
    using GemmT_128x256x16_id8 = CutlassGemmBase<128, 256, 16, 64, 64, 16, 4, cutlass::layout::ColumnMajor, Id8>;
    using GemmT_128x256x32s2_id2 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::ColumnMajor, Id2>;
    using GemmT_128x256x32s2_id4 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::ColumnMajor, Id4>;
    using GemmT_128x256x32s2_id8 = CutlassGemmBase<128, 256, 32, 64, 64, 32, 2, cutlass::layout::ColumnMajor, Id8>;

    std::vector<CutlassTiming> timings;
    timings.push_back(run_cutlass_variant<GemmF32_128x128x16>(
        "f32_128x128x16_s4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x128x32>(
        "f32_128x128x32_s3", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x32>(
        "f32_128x256x32_s3", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_256x128x32>(
        "f32_256x128x32_s3", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_64x256x32>(
        "f32_64x256x32_s3", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_256x64x32>(
        "f32_256x64x32_s3", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x32s2>(
        "f32_128x256x32_s2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_256x128x32s2>(
        "f32_256x128x32_s2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_64x256x32s2>(
        "f32_64x256x32_s2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_256x64x32s2>(
        "f32_256x64x32_s2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x16>(
        "f32_128x256x16_s4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_256x128x16>(
        "f32_256x128x16_s4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x128x16>(
        "transpose_problem_f32_128x128x16_s4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x16>(
        "transpose_problem_f32_128x256x16_s4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_256x128x16>(
        "transpose_problem_f32_256x128x16_s4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x32s2>(
        "transpose_problem_f32_128x256x32_s2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_256x128x32s2>(
        "transpose_problem_f32_256x128x32_s2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x16_id2>(
        "f32_128x256x16_s4_id2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x16_id4>(
        "f32_128x256x16_s4_id4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x16_id8>(
        "f32_128x256x16_s4_id8", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x32s2_id2>(
        "f32_128x256x32_s2_id2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x32s2_id4>(
        "f32_128x256x32_s2_id4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_variant<GemmF32_128x256x32s2_id8>(
        "f32_128x256x32_s2_id8", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x16_id2>(
        "transpose_problem_f32_128x256x16_s4_id2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x16_id4>(
        "transpose_problem_f32_128x256x16_s4_id4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x16_id8>(
        "transpose_problem_f32_128x256x16_s4_id8", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x32s2_id2>(
        "transpose_problem_f32_128x256x32_s2_id2", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x32s2_id4>(
        "transpose_problem_f32_128x256x32_s2_id4", d_a, d_b, m, n, k, d_c_cutlass, rounds));
    timings.push_back(run_cutlass_transposed_variant<GemmT_128x256x32s2_id8>(
        "transpose_problem_f32_128x256x32_s2_id8", d_a, d_b, m, n, k, d_c_cutlass, rounds));

    CutlassTiming best;
    for (const CutlassTiming& timing : timings) {
        if (!timing.supported) {
            std::cout << "[cutlass_gemm_variant] name=" << timing.name
                      << " supported=0"
                      << " smem_bytes=" << timing.shared_storage_bytes
                      << " max_optin_smem=" << max_dynamic_smem
                      << " cutlass_status=" << cutlassGetStatusString(timing.status)
                      << " cuda_status=" << cudaGetErrorString(timing.cuda_status)
                      << std::endl;
            continue;
        }
        if (!best.supported || timing.ms < best.ms) best = timing;
        std::cout << "[cutlass_gemm_variant] name=" << timing.name
                  << " supported=1"
                  << " ms=" << timing.ms
                  << " smem_bytes=" << timing.shared_storage_bytes
                  << " max_optin_smem=" << max_dynamic_smem
                  << std::endl;
    }
    ctx.check(best.supported, "At least one CUTLASS GEMM variant should be supported");
    if (!best.supported) return ctx.exit_code();

    // Regenerate the output with the winning variant so the accuracy check below
    // matches the reported best timing. Dispatch only the candidates we emit.
    if (std::strcmp(best.name, "f32_128x128x16_s4") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x128x16>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x128x32_s3") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x128x32>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x32_s3") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x32>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_256x128x32_s3") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_256x128x32>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_64x256x32_s3") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_64x256x32>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_256x64x32_s3") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_256x64x32>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x32_s2") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x32s2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_256x128x32_s2") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_256x128x32s2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_64x256x32_s2") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_64x256x32s2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_256x64x32_s2") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_256x64x32s2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x16_s4") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x16>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_256x128x16_s4") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_256x128x16>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x128x16_s4") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x128x16>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x16_s4") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x16>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_256x128x16_s4") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_256x128x16>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x32_s2") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x32s2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_256x128x32_s2") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_256x128x32s2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x16_s4_id2") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x16_id2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x16_s4_id4") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x16_id4>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x16_s4_id8") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x16_id8>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x32_s2_id2") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x32s2_id2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x32_s2_id4") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x32s2_id4>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "f32_128x256x32_s2_id8") == 0) {
        (void)time_cutlass_rowmajor_abt<GemmF32_128x256x32s2_id8>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x16_s4_id2") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x16_id2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x16_s4_id4") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x16_id4>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x16_s4_id8") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x16_id8>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x32_s2_id2") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x32s2_id2>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x32_s2_id4") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x32s2_id4>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    } else if (std::strcmp(best.name, "transpose_problem_f32_128x256x32_s2_id8") == 0) {
        (void)time_cutlass_transposed_problem_abt<GemmT_128x256x32s2_id8>(
            best.name, d_a, d_b, m, n, k, d_c_cutlass, 1);
    }

    const float cublas_ms = time_cublas_rowmajor_abt(handle, d_a, d_b, m, n, k, d_c_cublas, rounds);

    const size_t sample = std::min<size_t>((size_t)m * n, 1 << 20);
    std::vector<float> h_cutlass(sample);
    std::vector<float> h_cublas(sample);
    check_cuda(cudaMemcpy(h_cutlass.data(), d_c_cutlass, sizeof(float) * sample, cudaMemcpyDeviceToHost),
               "cudaMemcpy(h_cutlass)");
    check_cuda(cudaMemcpy(h_cublas.data(), d_c_cublas, sizeof(float) * sample, cudaMemcpyDeviceToHost),
               "cudaMemcpy(h_cublas)");

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (size_t i = 0; i < sample; ++i) {
        const double abs_err = std::fabs((double)h_cutlass[i] - (double)h_cublas[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, abs_err / std::max(1e-6, std::fabs((double)h_cublas[i])));
    }

    const double ops = 2.0 * (double)m * (double)n * (double)k;
    std::cout << "[cutlass_gemm_vs_cublas] m=" << m
              << " n=" << n
              << " k=" << k
              << " rounds=" << rounds
              << " best_variant=" << best.name
              << " cutlass_ms=" << best.ms
              << " cublas_ms=" << cublas_ms
              << " cutlass_vs_cublas=" << cublas_ms / std::max(1e-6f, best.ms)
              << " cutlass_tflops=" << ops / (best.ms * 1e9)
              << " cublas_tflops=" << ops / (cublas_ms * 1e9)
              << " max_abs_err_sample=" << max_abs_err
              << " max_rel_err_sample=" << max_rel_err
              << std::endl;

    ctx.check(max_abs_err < 2e-2, "CUTLASS TF32 sample should match cuBLAS TF32");

    check_cublas(cublasDestroy(handle), "cublasDestroy");
    check_cuda(cudaFree(d_c_cublas), "cudaFree(d_c_cublas)");
    check_cuda(cudaFree(d_c_cutlass), "cudaFree(d_c_cutlass)");
    check_cuda(cudaFree(d_b), "cudaFree(d_b)");
    check_cuda(cudaFree(d_a), "cudaFree(d_a)");
    return ctx.exit_code();
}
