#include "transform/opq_trainer.cuh"
#include "transform/opq_transform.cuh"

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
    int dim)
{
    std::vector<float> y((size_t)n * dim, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int od = 0; od < dim; ++od) {
            float acc = 0.0f;
            for (int id = 0; id < dim; ++id) {
                acc += a[(size_t)od * dim + id] * x[(size_t)i * dim + id];
            }
            y[(size_t)i * dim + od] = acc;
        }
    }
    return y;
}

void test_identity_and_custom_transform() {
    constexpr int n = 3;
    constexpr int dim = 4;
    std::vector<float> x = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.0f, 8.0f,
        0.0f, -3.0f, 7.0f, 1.0f,
    };

    paraselect::transform::OPQTransform opq(dim);
    opq.prepare();
    std::vector<float> y((size_t)n * dim, 0.0f);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    opq.apply_host_to_host(x.data(), y.data(), n, stream);
    for (size_t i = 0; i < x.size(); ++i) require_close(y[i], x[i], 1e-5f, "identity OPQ should match input");

    std::vector<float> r = {
        0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0,
    };
    opq.set_matrix(r);
    opq.apply_host_to_host(x.data(), y.data(), n, stream);
    const auto ref = host_reference(x, r, n, dim);
    for (size_t i = 0; i < y.size(); ++i) require_close(y[i], ref[i], 1e-5f, "custom OPQ should match host");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_fixed_residual_adc_identity() {
    // Non-symmetric rotation catches an [in,out]/[out,in] transpose error.
    const std::vector<float> rotation = {
        .6f, -.8f, 0, 0, .8f, .6f, 0, 0,
        0, 0, 0, -1, 0, 0, 1, 0};
    const std::vector<float> input = {
        3, 5, 7, 11,   // query
        2, -1, 4, 6,   // frozen centroid
        .5f, 2, -3, 1 // reconstructed residual before rotation
    };
    paraselect::transform::OPQTransform transform(4);
    transform.set_matrix(rotation);
    transform.prepare();
    std::vector<float> rotated(12);
    transform.apply_host_to_host(input.data(), rotated.data(), 3, nullptr);
    const auto expected = host_reference(input, rotation, 3, 4);
    for (size_t i = 0; i < expected.size(); ++i)
        require_close(rotated[i], expected[i], 2e-5f, "residual OPQ orientation");
    float dqc = 0, row_correction = 0, dot = 0, direct = 0;
    for (int d = 0; d < 4; ++d) {
        const float residual = rotated[8 + d];
        const float delta = input[d] - input[4 + d];
        dqc += delta * delta;
        row_correction += residual * (residual + 2 * rotated[4 + d]);
        dot += rotated[d] * residual;
        direct += (delta - input[8 + d]) * (delta - input[8 + d]);
    }
    require_close(dqc + row_correction - 2 * dot, direct, 1e-4f, "rotated residual ADC preserves L2");
}

void test_plain_pq_encode_decode() {
    constexpr int n = 4;
    constexpr int dim = 4;
    constexpr int pq_m = 2;
    constexpr int pq_k = 2;
    std::vector<float> x = {
        0, 0, 10, 10,
        1, 1, 11, 11,
        8, 8, 20, 20,
        9, 9, 21, 21,
    };
    std::vector<float> codebooks = {
        0, 0, 9, 9,
        10, 10, 21, 21,
    };
    std::vector<uint8_t> codes((size_t)n * pq_m);
    std::vector<float> decoded((size_t)n * dim);
    paraselect::transform::opq_encode_plain_pq(x.data(), n, dim, pq_m, pq_k, codebooks.data(), codes.data());
    paraselect::transform::opq_decode_plain_pq(codes.data(), n, dim, pq_m, pq_k, codebooks.data(), decoded.data());
    require_true(codes[0] == 0 && codes[1] == 0, "first vector codes");
    require_true(codes[6] == 1 && codes[7] == 1, "last vector codes");
    require_true(std::isfinite(paraselect::transform::opq_reconstruction_objective(x.data(), decoded.data(), n, dim)),
                 "objective should be finite");
}

void test_opq_trainer_smoke() {
    constexpr int n = 64;
    constexpr int dim = 8;
    std::vector<float> x((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            x[(size_t)i * dim + d] = (float)((i * 17 + d * 13) % 31) * 0.125f + (float)d;
        }
    }

    paraselect::transform::OPQTrainSpec spec;
    spec.n = n;
    spec.dim = dim;
    spec.pq_m = 2;
    spec.pq_k = 8;
    spec.opq_iters = 2;
    spec.pq_iters = 3;
    spec.seed = 7;
    auto result = paraselect::transform::train_opq_transform(x.data(), spec);

    require_true(result.rotation_rowmajor_out_in.size() == (size_t)dim * dim, "rotation size");
    require_true(result.codebooks.size() == (size_t)spec.pq_m * spec.pq_k * (dim / spec.pq_m), "codebook size");
    require_true(result.codes.size() == (size_t)n * spec.pq_m, "codes size");
    require_true(result.objective_by_iter.size() == (size_t)spec.opq_iters, "objective count");
    for (float v : result.objective_by_iter) require_true(std::isfinite(v), "objective finite");

    const auto& r = result.rotation_rowmajor_out_in;
    for (int i = 0; i < dim; ++i) {
        float norm = 0.0f;
        for (int d = 0; d < dim; ++d) norm += r[(size_t)i * dim + d] * r[(size_t)i * dim + d];
        require_close(norm, 1.0f, 2e-3f, "OPQ rotation row norm");
        for (int j = 0; j < i; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += r[(size_t)i * dim + d] * r[(size_t)j * dim + d];
            require_close(dot, 0.0f, 2e-3f, "OPQ rotation row dot");
        }
    }
}

}  // namespace

int main() {
    try {
        test_identity_and_custom_transform();
        test_fixed_residual_adc_identity();
        test_plain_pq_encode_decode();
        test_opq_trainer_smoke();
    } catch (const std::exception& e) {
        std::cerr << "[test_opq_transform] failed: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "[test_opq_transform] passed" << std::endl;
    return 0;
}
