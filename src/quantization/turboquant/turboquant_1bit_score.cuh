#ifndef PARASELECT_TURBOQUANT_1BIT_SCORE_CUH
#define PARASELECT_TURBOQUANT_1BIT_SCORE_CUH

#include "quantization/turboquant/turboquant_1bit_scan.cuh"

namespace paraselect::search::turboquant::detail {

__device__ __forceinline__ bool pair_greater(
    float lhs_dist,
    int lhs_idx,
    float rhs_dist,
    int rhs_idx) {
    return lhs_dist > rhs_dist || (lhs_dist == rhs_dist && lhs_idx > rhs_idx);
}

template <TurboQuant1BitScoreMode Mode, int FixedWordsPerCode = 0>
__device__ __forceinline__ float score_row(
    const TurboQuant1BitScanInput input,
    const float* shared_query,
    int query_state,
    int row,
    float reconstruction_centroid) {
    const std::uint32_t* database_code =
        input.packed_database_codes + static_cast<std::size_t>(row) * input.words_per_code;

    if constexpr (Mode == TurboQuant1BitScoreMode::BinaryQuery) {
        const std::uint32_t* query_code =
            reinterpret_cast<const std::uint32_t*>(shared_query);
        int hamming = 0;
        if constexpr (FixedWordsPerCode == 4) {
            hamming = __popc(query_code[0] ^ database_code[0]) +
                __popc(query_code[1] ^ database_code[1]) +
                __popc(query_code[2] ^ database_code[2]) +
                __popc(query_code[3] ^ database_code[3]);
        } else {
            for (int word = 0; word < input.words_per_code; ++word) {
                std::uint32_t different = query_code[word] ^ database_code[word];
                if (word + 1 == input.words_per_code && (input.dim & 31) != 0) {
                    different &= (1u << (input.dim & 31)) - 1u;
                }
                hamming += __popc(different);
            }
        }
        const float query_norm = input.query_norms[query_state];
        const float database_norm = input.database_row_parameters[row];
        const float signed_cosine =
            (static_cast<float>(input.dim) - 2.0f * static_cast<float>(hamming)) /
            static_cast<float>(input.dim);
        return query_norm * query_norm + database_norm * database_norm -
               2.0f * query_norm * database_norm * signed_cosine;
    } else {
        float signed_dot = 0.0f;
        for (int word = 0; word < input.words_per_code; ++word) {
            const std::uint32_t signs = database_code[word];
#pragma unroll
            for (int bit = 0; bit < 32; ++bit) {
                const int d = word * 32 + bit;
                if (d < input.dim) {
                    std::uint32_t query_bits = __float_as_uint(shared_query[d]);
                    query_bits ^= ((signs >> bit) & 1u) ? 0u : 0x80000000u;
                    signed_dot += __uint_as_float(query_bits);
                }
            }
        }
        const float reconstruction_scale = input.database_row_parameters[row];
        return input.query_norm_sq[query_state] +
               static_cast<float>(input.dim) * reconstruction_scale * reconstruction_scale -
               2.0f * reconstruction_scale * signed_dot;
    }
}

}  // namespace paraselect::search::turboquant::detail

#endif  // PARASELECT_TURBOQUANT_1BIT_SCORE_CUH
