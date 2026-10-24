#pragma once
#include "search/ivf_common/quantized/residual_codec.h"

namespace paraselect::search::turboquant {
// Encoding assets only: no IVF search, threshold, pool, or reranking state.
class ResidualEncoding final : public ResidualCodec {
public:
    explicit ResidualEncoding(const ResidualCodecBuildInput& input);
    ~ResidualEncoding() override;
    ResidualEncoding(const ResidualEncoding&) = delete;
    ResidualEncoding& operator=(const ResidualEncoding&) = delete;
    SignResidualDeviceView prepare_queries(
        const float* queries, int count, cudaStream_t stream) override;
private:
    void release() noexcept;
    int dim_;
    int max_batch_;
    std::uint32_t* codes_ = nullptr;
    float* scales_ = nullptr;
    float* rotated_centroids_ = nullptr;
    float* rotated_queries_ = nullptr;
};
} // namespace paraselect::search::turboquant
