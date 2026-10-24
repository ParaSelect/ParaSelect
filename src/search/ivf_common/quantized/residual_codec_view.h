#pragma once
#include <cstdint>

namespace paraselect::search {
enum class ResidualCodecKind { PQ, Sign1Bit };

// Trivially copyable device operands for sign + per-row-scale reconstruction.
// No ownership, CUDA implementation, policy, or environment parsing here.
struct SignResidualDeviceView {
    const std::uint32_t* codes = nullptr;
    const float* scales = nullptr;
    const float* queries = nullptr;
    const float* centroids = nullptr;
};
} // namespace paraselect::search
