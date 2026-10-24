#pragma once

#include <cstdint>

namespace paraselect::row_correction {

inline constexpr char kU16Magic[8] = {
    0x49, 0x53, 0x52, 0x43, 0x31, 0x36, 0x00, 0x00};
inline constexpr std::uint32_t kU16Version = 2;

struct U16Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t count;
    float scale;
    float offset;
};

static_assert(sizeof(U16Header) == 32);

// Non-owning descriptor. The pointers may address host or device memory;
// ownership and address space are determined by the containing component.
struct StorageView {
    const float* fp32 = nullptr;
    const std::uint16_t* u16 = nullptr;
    float scale = 0.0f;
    float offset = 0.0f;
    std::int64_t count = 0;

    bool empty() const noexcept { return fp32 == nullptr && u16 == nullptr; }
    bool enabled() const noexcept { return (fp32 != nullptr) != (u16 != nullptr); }
};

inline float value(const float* centroid, const float* reconstruction, int dim) {
    float reconstruction_norm = 0.0f;
    float centroid_dot_reconstruction = 0.0f;
    for (int d = 0; d < dim; ++d) {
        reconstruction_norm += reconstruction[d] * reconstruction[d];
        centroid_dot_reconstruction += centroid[d] * reconstruction[d];
    }
    return reconstruction_norm + 2.0f * centroid_dot_reconstruction;
}

}  // namespace paraselect::row_correction
