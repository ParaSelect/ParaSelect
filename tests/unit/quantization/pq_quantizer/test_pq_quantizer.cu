/**
 * PQ :CPU  GPU  case, --cpu  --gpu , test_linear_fp16.
 */
#include "src/utils/pch.h"
#include "src/quantization/pq/pq_quantizer.cuh"
#include "common/test_utils.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

/* PQ , fp16 , */
#define PQ_TOLERANCE_ABS 2.5f
#define PQ_TOLERANCE_REL 1.0f

static bool compare_pq_ok(float a, float b) {
    float abs_err = std::abs(a - b);
    float sum = std::abs(a) + std::abs(b);
    if (sum < 1e-8f) return abs_err < PQ_TOLERANCE_ABS;
    float rel_err = 2.f * abs_err / sum;
    return abs_err < PQ_TOLERANCE_ABS || rel_err < PQ_TOLERANCE_REL;
}

static bool run_case_gpu(int n, int dim, int n_subspace, int n_codes, unsigned seed) {
    using namespace ivf;
    if (dim % n_subspace != 0 || n_codes > 256) return true;
    const size_t size = (size_t)n * dim;
    std::vector<float> h_data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (size_t i = 0; i < size; ++i) h_data[i] = dist(gen);

    const int code_bytes = n * n_subspace;
    float* d_data = nullptr;
    void* d_codes = nullptr;
    float* d_decoded = nullptr;
    if (cudaMalloc(&d_data, size * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_codes, (size_t)code_bytes) != cudaSuccess) { cudaFree(d_data); return false; }
    if (cudaMalloc(&d_decoded, size * sizeof(float)) != cudaSuccess) { cudaFree(d_data); cudaFree(d_codes); return false; }

    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    PQQuantizerGPU quant(n_subspace, n_codes);
    quant.train(h_data.data(), n, dim);
    quant.encode(d_data, n, dim, d_codes);
    cudaDeviceSynchronize();
    quant.decode(d_codes, n, dim, d_decoded);
    cudaDeviceSynchronize();

    std::vector<float> h_decoded(size);
    cudaMemcpy(h_decoded.data(), d_decoded, size * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    float max_err = 0.f;
    for (size_t i = 0; i < size; ++i) {
        if (!compare_pq_ok(h_data[i], h_decoded[i])) pass = false;
        max_err = std::max(max_err, std::abs(h_data[i] - h_decoded[i]));
    }

    if (!QUIET) {
        COUT_VAL("[PQ GPU] n=", n, " dim=", dim, " M=", n_subspace, " K=", n_codes,
                 " code_bytes=", quant.code_size_bytes(n, dim), " max_err=", max_err, pass ? " PASS" : " FAIL");
    }

    cudaFree(d_data);
    cudaFree(d_codes);
    cudaFree(d_decoded);
    return pass;
}

static bool run_case_cpu(int n, int dim, int n_subspace, int n_codes, unsigned seed) {
    using namespace ivf;
    if (dim % n_subspace != 0 || n_codes > 256) return true;
    const size_t size = (size_t)n * dim;
    std::vector<float> h_data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (size_t i = 0; i < size; ++i) h_data[i] = dist(gen);

    const int code_bytes = n * n_subspace;
    std::vector<uint8_t> h_codes((size_t)code_bytes);
    std::vector<float> h_decoded(size);

    PQQuantizerCPU quant(n_subspace, n_codes);
    quant.train(h_data.data(), n, dim);
    quant.encode(h_data.data(), n, dim, h_codes.data());
    quant.decode(h_codes.data(), n, dim, h_decoded.data());

    bool pass = true;
    float max_err = 0.f;
    for (size_t i = 0; i < size; ++i) {
        if (!compare_pq_ok(h_data[i], h_decoded[i])) pass = false;
        max_err = std::max(max_err, std::abs(h_data[i] - h_decoded[i]));
    }

    if (!QUIET) {
        COUT_VAL("[PQ CPU] n=", n, " dim=", dim, " M=", n_subspace, " K=", n_codes,
                 " code_bytes=", quant.code_size_bytes(n, dim), " max_err=", max_err, pass ? " PASS" : " FAIL");
    }

    return pass;
}

template <typename Func>
static bool run_timed_case(paraselect::test::TestContext& ctx, const std::string& name, Func&& func) {
    auto start = std::chrono::steady_clock::now();
    bool ok = func();
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    ctx.add_metric(name + ".ms", elapsed_ms);
    ctx.check(ok, "PQ roundtrip within tolerance");
    return ok;
}

static void run_all_cases(paraselect::test::TestContext& ctx, bool use_cpu, bool perf_mode) {
    auto run = [&](int n, int dim, int M, int K, unsigned s) {
        std::string name = std::string(use_cpu ? "cpu" : "gpu") +
            "_n" + std::to_string(n) +
            "_d" + std::to_string(dim) +
            "_m" + std::to_string(M) +
            "_k" + std::to_string(K);
        ctx.run_case(name, [&] {
            run_timed_case(ctx, name, [&] {
                return use_cpu ? run_case_cpu(n, dim, M, K, s) : run_case_gpu(n, dim, M, K, s);
            });
        });
    };
    if (perf_mode) {
        run(8192, 64, 8, 256, 2026u);
        run(4096, 128, 16, 256, 2027u);
        return;
    }

    run(200, 64, 8, 256, 42u);
    run(100, 128, 16, 256, 123u);
    run(50, 32, 4, 128, 0u);
    run(300, 96, 12, 256, 999u);
    run(80, 64, 8, 64, 7u);
}

int main(int argc, char** argv) {
    bool use_cpu = false;
    bool perf_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_cpu = true;
        else if (arg == "--gpu") use_cpu = false;
        else if (arg == "--perf") perf_mode = true;
    }

    std::string backend = use_cpu ? "CPU" : "GPU";
    paraselect::test::TestContext ctx(
        "pq_quantizer_" + backend,
        perf_mode ? paraselect::test::TestKind::Benchmark : paraselect::test::TestKind::Unit);
    ctx.add_note("backend=" + backend);
    ctx.add_note(std::string("perf_mode=") + (perf_mode ? "true" : "false"));
    run_all_cases(ctx, use_cpu, perf_mode);
    return ctx.exit_code();
}
