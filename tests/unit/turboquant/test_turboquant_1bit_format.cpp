#include "quantization/turboquant/turboquant_1bit_format.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace tq = paraselect::search::turboquant;

static bool close(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-5f;
}

int main() {
    std::vector<float> values(70, -1.0f);
    for (int i : {0, 1, 31, 32, 63, 69}) values[i] = 1.0f;

    const std::vector<std::uint32_t> packed = tq::pack_signs(values.data(), values.size());
    assert(packed.size() == 3);
    assert(packed[0] == ((1u << 0) | (1u << 1) | (1u << 31)));
    assert(packed[1] == ((1u << 0) | (1u << 31)));
    assert(packed[2] == (1u << 5));

    std::vector<float> other(70, -1.0f);
    for (int i : {0, 2, 31, 33, 63, 68}) other[i] = 1.0f;
    const std::vector<std::uint32_t> other_packed =
        tq::pack_signs(other.data(), other.size());
    assert(tq::hamming_distance(packed.data(), other_packed.data(), values.size()) == 6);

    const float centroid = 0.25f;
    float scalar_distance = 0.0f;
    for (std::size_t d = 0; d < values.size(); ++d) {
        const float reconstructed = values[d] > 0.0f ? centroid : -centroid;
        const float delta = other[d] - reconstructed;
        scalar_distance += delta * delta;
    }
    assert(close(
        tq::mse_distance(other.data(), packed.data(), values.size(), centroid),
        scalar_distance));

    const float binary_distance = tq::binary_l2_distance(
        packed.data(), other_packed.data(), values.size(), 2.0f, 3.0f);
    const float expected_dot = 2.0f * 3.0f * (70.0f - 2.0f * 6.0f) / 70.0f;
    assert(close(binary_distance, 4.0f + 9.0f - 2.0f * expected_dot));
    return 0;
}
