#include "transform/pca_transform.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(float actual, float expected, float tolerance, const char* message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << message << ": actual=" << actual
                  << " expected=" << expected << std::endl;
        throw std::runtime_error(message);
    }
}

void test_gpu_pca_pipeline() {
    constexpr int kCount = 1024;
    constexpr int kSourceDim = 4;
    constexpr int kDim = 2;
    std::vector<float> input(static_cast<size_t>(kCount) * kSourceDim);
    for (int row = 0; row < kCount; ++row) {
        const float primary = static_cast<float>((row % 97) - 48);
        const float secondary = static_cast<float>(((row * 17) % 53) - 26);
        float* vector = input.data() + static_cast<size_t>(row) * kSourceDim;
        vector[0] = primary;
        vector[1] = 2.0f * primary + 0.01f * secondary;
        vector[2] = secondary;
        vector[3] = 7.0f + 0.1f * secondary;
    }

    const auto model = paraselect::transform::fit_pca_transform(
        input.data(), kCount, kSourceDim, kDim);
    require(model.source_dim == kSourceDim && model.dim == kDim,
            "PCA model shape mismatch");
    for (int row = 0; row < kDim; ++row) {
        float norm = 0.0f;
        for (int d = 0; d < kSourceDim; ++d) {
            const float value = model.projection[static_cast<size_t>(row) * kSourceDim + d];
            norm += value * value;
        }
        require_close(norm, 1.0f, 2e-4f, "PCA row is not normalized");
    }
    float dot = 0.0f;
    for (int d = 0; d < kSourceDim; ++d) {
        dot += model.projection[d] * model.projection[kSourceDim + d];
    }
    require_close(dot, 0.0f, 2e-4f, "PCA rows are not orthogonal");

    paraselect::transform::PCAProjector projector(model);
    std::vector<float> projected(static_cast<size_t>(kCount) * kDim);
    projector.apply(input.data(), kCount, projected.data());
    for (int row = 0; row < 8; ++row) {
        for (int output_dim = 0; output_dim < kDim; ++output_dim) {
            float reference = model.bias[output_dim];
            for (int source_dim = 0; source_dim < kSourceDim; ++source_dim) {
                reference +=
                    model.projection[static_cast<size_t>(output_dim) * kSourceDim + source_dim] *
                    input[static_cast<size_t>(row) * kSourceDim + source_dim];
            }
            require_close(
                projected[static_cast<size_t>(row) * kDim + output_dim],
                reference,
                1e-3f,
                "GPU PCA projection differs from host reference");
        }
    }
    for (int d = 0; d < kDim; ++d) {
        double mean = 0.0;
        for (int row = 0; row < kCount; ++row) {
            mean += projected[static_cast<size_t>(row) * kDim + d];
        }
        require_close(
            static_cast<float>(mean / kCount), 0.0f, 2e-4f,
            "PCA output is not centered");
    }

    std::vector<float> scales(kDim);
    projector.calibrate_scales(input.data(), kCount, scales.data());
    std::vector<int8_t> quantized(static_cast<size_t>(kCount) * kDim);
    const uint64_t clipped = projector.apply_quantized(
        input.data(), kCount, scales.data(), quantized.data());
    require(clipped == 0, "calibration input should not clip");
    for (size_t position = 0; position < quantized.size(); ++position) {
        const int d = static_cast<int>(position % kDim);
        require(scales[d] > 0.0f && std::isfinite(scales[d]),
                "PCA scale is invalid");
        require_close(
            static_cast<float>(quantized[position]) * scales[d],
            projected[position],
            scales[d] * 0.51f + 1e-5f,
            "PCA int8 quantization error is too large");
    }
}

}  // namespace

int main() {
    try {
        test_gpu_pca_pipeline();
    } catch (const std::exception& error) {
        std::cerr << "[test_pca_transform] failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "[test_pca_transform] passed" << std::endl;
    return 0;
}
