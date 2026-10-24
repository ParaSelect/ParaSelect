#include "common/test_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>

int main() {
    paraselect::test::TestContext ctx("print_cuda", paraselect::test::TestKind::Smoke);
    const int x = 3;
    const int y = 4;
    const size_t n = static_cast<size_t>(x) * y;
    float* test = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* roundtrip = static_cast<float*>(std::malloc(n * sizeof(float)));
    for (size_t i = 0; i < n; ++i) {
        test[i] = static_cast<float>(i + 1);
    }
    float* d_test = nullptr;
    cudaMalloc(&d_test, n * sizeof(float));
    CHECK_CUDA_ERRORS;
    cudaMemcpy(d_test, test, n * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    COUT_ENDL("test print cuda 2d", d_test, x, y);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    cudaMemcpy(roundtrip, d_test, n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    ctx.run_case("cuda_malloc_memcpy_roundtrip", [&] {
        for (size_t i = 0; i < n; ++i) {
            ctx.check_near(roundtrip[i], test[i], 1e-6, "CUDA roundtrip value");
        }
    });
    ctx.add_metric("roundtrip_values", static_cast<double>(n));

    cudaFree(d_test);
    std::free(roundtrip);
    std::free(test);
    return ctx.exit_code();
}
