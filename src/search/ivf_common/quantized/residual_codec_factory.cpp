#include "residual_codec.h"
#include "quantization/turboquant/residual_encoding.cuh"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace paraselect::search {
ResidualCodecKind residual_codec_from_environment() {
    const char* value = std::getenv("PARASELECT_RESIDUAL_CODEC");
    if (!value || !*value || std::strcmp(value, "respq") == 0) return ResidualCodecKind::PQ;
    if (std::strcmp(value, "turboquant") == 0) return ResidualCodecKind::Sign1Bit;
    throw std::invalid_argument("PARASELECT_RESIDUAL_CODEC must be respq or turboquant");
}
std::unique_ptr<ResidualCodec> make_residual_codec(
    ResidualCodecKind kind, const ResidualCodecBuildInput& input) {
    switch (kind) {
    case ResidualCodecKind::PQ: return nullptr;
    case ResidualCodecKind::Sign1Bit:
        return std::make_unique<turboquant::ResidualEncoding>(input);
    }
    throw std::invalid_argument("unknown residual codec");
}
} // namespace paraselect::search
