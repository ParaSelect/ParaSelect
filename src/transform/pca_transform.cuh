#ifndef PARASELECT_TRANSFORM_PCA_TRANSFORM_CUH
#define PARASELECT_TRANSFORM_PCA_TRANSFORM_CUH

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace paraselect::transform {

struct PCATransform {
    int source_dim = 0;
    int dim = 0;
    std::vector<float> projection;
    std::vector<float> bias;
};

PCATransform fit_pca_transform(
    const float* vectors,
    size_t count,
    int source_dim,
    int dim);

class PCAProjector {
public:
    explicit PCAProjector(const PCATransform& transform);
    ~PCAProjector();
    PCAProjector(const PCAProjector&) = delete;
    PCAProjector& operator=(const PCAProjector&) = delete;

    void apply(const float* input, size_t count, float* output);
    void calibrate_scales(const float* input, size_t count, float* scales);
    uint64_t apply_quantized(
        const float* input,
        size_t count,
        const float* scales,
        int8_t* output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace paraselect::transform

#endif  // PARASELECT_TRANSFORM_PCA_TRANSFORM_CUH
