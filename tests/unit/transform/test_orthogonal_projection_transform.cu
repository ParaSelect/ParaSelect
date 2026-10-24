#include "transform/orthogonal_projection_transform.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require_true(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

void require_close(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) {
        std::cerr << msg << ": got=" << a << " expected=" << b << " diff=" << std::fabs(a - b) << std::endl;
        throw std::runtime_error(msg);
    }
}

std::vector<float> host_reference(
    const std::vector<float>& x,
    const std::vector<float>& a,
    int n,
    int d_in,
    int d_out)
{
    std::vector<float> y((size_t)n * d_out, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int od = 0; od < d_out; ++od) {
            float acc = 0.0f;
            for (int id = 0; id < d_in; ++id) {
                acc += a[(size_t)od * d_in + id] * x[(size_t)i * d_in + id];
            }
            y[(size_t)i * d_out + od] = acc;
        }
    }
    return y;
}

void test_invalid_shape() {
    bool threw = false;
    try {
        paraselect::transform::OrthogonalProjectionTransform bad({0, 8, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require_true(threw, "invalid d_in should throw");
}

void test_seed_determinism() {
    paraselect::transform::OrthogonalProjectionTransform a({8, 4, 123});
    paraselect::transform::OrthogonalProjectionTransform b({8, 4, 123});
    paraselect::transform::OrthogonalProjectionTransform c({8, 4, 456});
    a.prepare();
    b.prepare();
    c.prepare();
    require_true(a.host_matrix() == b.host_matrix(), "same seed should generate same matrix");
    require_true(a.host_matrix() != c.host_matrix(), "different seed should generate different matrix");
}

void test_gpu_matches_host(int n) {
    constexpr int d_in = 7;
    constexpr int d_out = 5;
    paraselect::transform::OrthogonalProjectionTransform proj({d_in, d_out, 999});
    proj.prepare();

    std::vector<float> x((size_t)n * d_in);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float)((int)(i % 11) - 5) * 0.25f;
    std::vector<float> y((size_t)n * d_out, 0.0f);
    const std::vector<float> ref = host_reference(x, proj.host_matrix(), n, d_in, d_out);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    proj.apply_host_to_host(x.data(), y.data(), (size_t)n, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));

    for (size_t i = 0; i < y.size(); ++i) {
        require_close(y[i], ref[i], 1e-4f, "gpu output should match host reference");
    }
}

void test_rows_are_orthogonal() {
    constexpr int d_in = 8;
    constexpr int d_out = 4;
    paraselect::transform::OrthogonalProjectionTransform proj({d_in, d_out, 777});
    proj.prepare();
    const auto& a = proj.host_matrix();
    const float expected_norm2 = (float)d_in / (float)d_out;
    for (int r = 0; r < d_out; ++r) {
        float norm2 = 0.0f;
        for (int c = 0; c < d_in; ++c) norm2 += a[(size_t)r * d_in + c] * a[(size_t)r * d_in + c];
        require_close(norm2, expected_norm2, 1e-4f, "projection row norm should match JL scale");
        for (int prev = 0; prev < r; ++prev) {
            float dot = 0.0f;
            for (int c = 0; c < d_in; ++c) dot += a[(size_t)r * d_in + c] * a[(size_t)prev * d_in + c];
            require_close(dot, 0.0f, 1e-4f, "projection rows should be orthogonal");
        }
    }
}

}  // namespace

int main() {
    try {
        test_invalid_shape();
        test_seed_determinism();
        test_rows_are_orthogonal();
        test_gpu_matches_host(1);
        test_gpu_matches_host(3);
        test_gpu_matches_host(32);
    } catch (const std::exception& e) {
        std::cerr << "[test_orthogonal_projection_transform] failed: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "[test_orthogonal_projection_transform] passed" << std::endl;
    return 0;
}
