/**
 * fp16 :CPU  GPU  case, --cpu  --gpu .
 */
#include "src/utils/pch.h"
#include <cuda_fp16.h>
#include "src/quantization/linear/linear_fp16.cuh"
#include "common/test_utils.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define FP16_TOLERANCE_ABS 5e-2f
#define FP16_TOLERANCE_REL 5e-2f

static bool compare_fp16_ok(float a, float b) {
    float abs_err = std::abs(a - b);
    float sum = std::abs(a) + std::abs(b);
    if (sum < 1e-8f) return abs_err < FP16_TOLERANCE_ABS;
    float rel_err = 2.f * abs_err / sum;
    return abs_err < FP16_TOLERANCE_ABS || rel_err < FP16_TOLERANCE_REL;
}

static bool run_case_gpu(int n, int dim, float scale, float zero_point, unsigned seed) {
    using namespace ivf;
    const size_t size = (size_t)n * dim;
    std::vector<float> h_data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (size_t i = 0; i < size; ++i) h_data[i] = dist(gen);

    float* d_data = nullptr;
    void* d_codes = nullptr;
    float* d_decoded = nullptr;
    if (cudaMalloc(&d_data, size * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_codes, (size_t)n * dim * sizeof(__half)) != cudaSuccess) { cudaFree(d_data); return false; }
    if (cudaMalloc(&d_decoded, size * sizeof(float)) != cudaSuccess) { cudaFree(d_data); cudaFree(d_codes); return false; }

    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    LinearQuantizerFP16GPU quant(scale, zero_point);
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
        if (!compare_fp16_ok(h_data[i], h_decoded[i])) pass = false;
        max_err = std::max(max_err, std::abs(h_data[i] - h_decoded[i]));
    }

    if (!QUIET) {
        COUT_VAL("[linear_fp16 GPU] n=", n, " dim=", dim, " scale=", scale, " zp=", zero_point,
                 " code_bytes=", quant.code_size_bytes(n, dim), " max_err=", max_err, pass ? " PASS" : " FAIL");
    }

    cudaFree(d_data);
    cudaFree(d_codes);
    cudaFree(d_decoded);
    return pass;
}

static bool run_case_cpu(int n, int dim, float scale, float zero_point, unsigned seed) {
    using namespace ivf;
    const size_t size = (size_t)n * dim;
    std::vector<float> h_data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (size_t i = 0; i < size; ++i) h_data[i] = dist(gen);

    const int code_bytes = n * dim * (int)sizeof(uint16_t);
    std::vector<uint8_t> h_codes((size_t)code_bytes);
    std::vector<float> h_decoded(size);

    LinearQuantizerFP16CPU quant(scale, zero_point);
    quant.train(h_data.data(), n, dim);
    quant.encode(h_data.data(), n, dim, h_codes.data());
    quant.decode(h_codes.data(), n, dim, h_decoded.data());

    bool pass = true;
    float max_err = 0.f;
    for (size_t i = 0; i < size; ++i) {
        if (!compare_fp16_ok(h_data[i], h_decoded[i])) pass = false;
        max_err = std::max(max_err, std::abs(h_data[i] - h_decoded[i]));
    }

    if (!QUIET) {
        COUT_VAL("[linear_fp16 CPU] n=", n, " dim=", dim, " scale=", scale, " zp=", zero_point,
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
    ctx.check(ok, "linear_fp16 roundtrip within tolerance");
    return ok;
}

static void run_all_cases(paraselect::test::TestContext& ctx, bool use_cpu, bool perf_mode) {
    auto run = [&](int n, int d, float sc, float zp, unsigned s) {
        std::string name = std::string(use_cpu ? "cpu" : "gpu") +
            "_n" + std::to_string(n) +
            "_d" + std::to_string(d);
        ctx.run_case(name, [&] {
            run_timed_case(ctx, name, [&] {
                return use_cpu ? run_case_cpu(n, d, sc, zp, s) : run_case_gpu(n, d, sc, zp, s);
            });
        });
    };
    if (perf_mode) {
        run(65536, 128, 1.f, 0.f, 2026u);
        run(32768, 256, 0.5f, 0.f, 2027u);
        return;
    }

    run(100, 64, 1.f, 0.f, 42u);
    run(10, 128, 1.f, 0.f, 123u);
    run(1, 256, 1.f, 0.f, 0u);
    run(1000, 32, 1.f, 0.f, 999u);
    run(50, 64, 0.5f, 0.f, 7u);
    run(20, 16, 1.f, -1.f, 3u);
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
        "linear_fp16_" + backend,
        perf_mode ? paraselect::test::TestKind::Benchmark : paraselect::test::TestKind::Unit);
    ctx.add_note("backend=" + backend);
    ctx.add_note(std::string("perf_mode=") + (perf_mode ? "true" : "false"));
    run_all_cases(ctx, use_cpu, perf_mode);
    return ctx.exit_code();
}
