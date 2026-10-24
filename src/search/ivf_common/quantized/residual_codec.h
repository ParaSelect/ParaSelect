#pragma once
#include "residual_codec_view.h"
#include <cuda_runtime_api.h>
#include <memory>

namespace paraselect::search {

// Resolve once during search-context preparation, never in a device loop.
ResidualCodecKind residual_codec_from_environment();

struct ResidualCodecBuildInput {
    const std::uint8_t* vectors;
    const float* centroids;
    const long long* offsets;
    int rows;
    int lists;
    int dim;
    int max_batch;
    // Optional cached host rows (packed codes + per-row scales). When both are
    // set the codec uploads them instead of re-encoding the database.
    const std::uint32_t* cached_codes = nullptr;
    const float* cached_scales = nullptr;
};

// Host-only boundary. Online preparation is included in query timing.
// The established PQ path owns its existing LUT workspace; alternative codecs
// own only their encoding assets and online query workspace through this API.
class ResidualCodec {
public:
    virtual ~ResidualCodec() = default;
    virtual SignResidualDeviceView prepare_queries(
        const float* queries, int count, cudaStream_t stream) = 0;
};
std::unique_ptr<ResidualCodec> make_residual_codec(
    ResidualCodecKind kind, const ResidualCodecBuildInput& input);
} // namespace paraselect::search
