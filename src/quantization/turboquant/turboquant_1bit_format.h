#ifndef PARASELECT_TURBOQUANT_1BIT_FORMAT_H
#define PARASELECT_TURBOQUANT_1BIT_FORMAT_H

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace paraselect::search::turboquant {

inline constexpr int kBitsPerWord = 32;

inline std::size_t packed_word_count(std::size_t dim) {
    return (dim + kBitsPerWord - 1) / kBitsPerWord;
}

inline std::vector<std::uint32_t> pack_signs(const float* values, std::size_t dim) {
    if (!values && dim != 0) throw std::invalid_argument("pack_signs: null input");
    std::vector<std::uint32_t> words(packed_word_count(dim), 0);
    for (std::size_t d = 0; d < dim; ++d) {
        if (values[d] >= 0.0f) words[d / kBitsPerWord] |= 1u << (d % kBitsPerWord);
    }
    return words;
}

inline int hamming_distance(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    std::size_t dim) {
    if ((!lhs || !rhs) && dim != 0) {
        throw std::invalid_argument("hamming_distance: null input");
    }
    const std::size_t full_words = dim / kBitsPerWord;
    int distance = 0;
    for (std::size_t word = 0; word < full_words; ++word) {
        distance += __builtin_popcount(lhs[word] ^ rhs[word]);
    }
    const int tail_bits = static_cast<int>(dim % kBitsPerWord);
    if (tail_bits != 0) {
        const std::uint32_t mask = (1u << tail_bits) - 1u;
        distance += __builtin_popcount((lhs[full_words] ^ rhs[full_words]) & mask);
    }
    return distance;
}

inline float mse_distance(
    const float* query,
    const std::uint32_t* database_code,
    std::size_t dim,
    float centroid) {
    if ((!query || !database_code) && dim != 0) {
        throw std::invalid_argument("mse_distance: null input");
    }
    float distance = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const bool positive = (database_code[d / kBitsPerWord] >> (d % kBitsPerWord)) & 1u;
        const float reconstructed = positive ? centroid : -centroid;
        const float delta = query[d] - reconstructed;
        distance += delta * delta;
    }
    return distance;
}

inline float scaled_sign_l2_distance(
    const float* query,
    const std::uint32_t* database_code,
    std::size_t dim,
    float database_norm) {
    if ((!query || !database_code) && dim != 0) {
        throw std::invalid_argument("scaled_sign_l2_distance: null input");
    }
    float query_norm_sq = 0.0f;
    float signed_dot = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const bool positive = (database_code[d / kBitsPerWord] >> (d % kBitsPerWord)) & 1u;
        query_norm_sq += query[d] * query[d];
        signed_dot += positive ? query[d] : -query[d];
    }
    const float scale = database_norm / std::sqrt(static_cast<float>(dim));
    return query_norm_sq + database_norm * database_norm - 2.0f * scale * signed_dot;
}

inline float sign_scale_l2_distance(
    const float* query,
    const std::uint32_t* database_code,
    std::size_t dim,
    float reconstruction_scale) {
    if ((!query || !database_code) && dim != 0) {
        throw std::invalid_argument("sign_scale_l2_distance: null input");
    }
    float query_norm_sq = 0.0f;
    float signed_dot = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const bool positive = (database_code[d / kBitsPerWord] >> (d % kBitsPerWord)) & 1u;
        query_norm_sq += query[d] * query[d];
        signed_dot += positive ? query[d] : -query[d];
    }
    return query_norm_sq + static_cast<float>(dim) * reconstruction_scale * reconstruction_scale -
           2.0f * reconstruction_scale * signed_dot;
}

inline float binary_l2_distance(
    const std::uint32_t* query_code,
    const std::uint32_t* database_code,
    std::size_t dim,
    float query_norm,
    float database_norm) {
    if (dim == 0) return 0.0f;
    const int hamming = hamming_distance(query_code, database_code, dim);
    const float signed_cosine =
        (static_cast<float>(dim) - 2.0f * static_cast<float>(hamming)) /
        static_cast<float>(dim);
    return query_norm * query_norm + database_norm * database_norm -
           2.0f * query_norm * database_norm * signed_cosine;
}

}  // namespace paraselect::search::turboquant

#endif  // PARASELECT_TURBOQUANT_1BIT_FORMAT_H
