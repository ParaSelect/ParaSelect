#include "quantization/turboquant/turboquant_1bit_scan.cuh"
#include "quantization/turboquant/turboquant_1bit_score.cuh"

#include "utils/check_cuda_error.cuh"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace paraselect::search::turboquant {
namespace {

constexpr int kBlockThreads = kTurboQuantLocalTopK;

template <TurboQuant1BitScoreMode Mode, int FixedWordsPerCode = 0>
__global__ void turboquant_1bit_scan_top256_kernel(
    TurboQuant1BitScanInput input,
    float reconstruction_centroid,
    int* __restrict__ output_indices,
    float* __restrict__ output_distances) {
    extern __shared__ unsigned char shared_bytes[];
    float* shared_distances = reinterpret_cast<float*>(shared_bytes);
    int* shared_indices = reinterpret_cast<int*>(shared_distances + kTurboQuantLocalTopK);
    float* shared_query = reinterpret_cast<float*>(shared_indices + kTurboQuantLocalTopK);

    const int chunk_id = blockIdx.x;
    const int lane = threadIdx.x;
    if (chunk_id >= input.n_chunks) return;
    const TurboQuantChunkDesc chunk = input.chunks[chunk_id];

    if constexpr (Mode == TurboQuant1BitScoreMode::MseFloatQuery) {
        const float* query =
            input.rotated_queries + static_cast<std::size_t>(chunk.query_state) * input.dim;
        for (int d = lane; d < input.dim; d += blockDim.x) shared_query[d] = query[d];
    } else {
        const std::uint32_t* query = input.packed_query_codes +
            static_cast<std::size_t>(chunk.query_state) * input.words_per_code;
        auto* shared_code = reinterpret_cast<std::uint32_t*>(shared_query);
        for (int word = lane; word < input.words_per_code; word += blockDim.x) {
            shared_code[word] = query[word];
        }
    }
    __syncthreads();

    float best_distance = FLT_MAX;
    int best_index = -1;
    const long long base = input.cluster_offsets[chunk.cluster] + chunk.start;
    for (int item = lane; item < chunk.length; item += blockDim.x) {
        const int row = static_cast<int>(base + item);
        const float distance = detail::score_row<Mode, FixedWordsPerCode>(
            input, shared_query, chunk.query_state, row, reconstruction_centroid);
        if (detail::pair_greater(best_distance, best_index, distance, row)) {
            best_distance = distance;
            best_index = row;
        }
    }
    shared_distances[lane] = best_distance;
    shared_indices[lane] = best_index;
    __syncthreads();

    for (int width = 2; width <= kTurboQuantLocalTopK; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int other = lane ^ stride;
            if (other > lane) {
                const bool ascending = (lane & width) == 0;
                const float lhs_distance = shared_distances[lane];
                const float rhs_distance = shared_distances[other];
                const int lhs_index = shared_indices[lane];
                const int rhs_index = shared_indices[other];
                const bool lhs_greater = detail::pair_greater(
                    lhs_distance, lhs_index, rhs_distance, rhs_index);
                const bool lhs_less = detail::pair_greater(
                    rhs_distance, rhs_index, lhs_distance, lhs_index);
                if ((ascending && lhs_greater) || (!ascending && lhs_less)) {
                    shared_distances[lane] = rhs_distance;
                    shared_indices[lane] = rhs_index;
                    shared_distances[other] = lhs_distance;
                    shared_indices[other] = lhs_index;
                }
            }
            __syncthreads();
        }
    }

    const std::size_t output =
        static_cast<std::size_t>(chunk_id) * kTurboQuantLocalTopK + lane;
    output_distances[output] = shared_distances[lane];
    output_indices[output] = shared_indices[lane];
}

static void validate_scan_input(
    const TurboQuant1BitScanInput& input,
    TurboQuant1BitScoreMode mode,
    const int* output_indices,
    const float* output_distances) {
    if (!input.packed_database_codes || !input.cluster_offsets || !input.chunks ||
        !output_indices || !output_distances || input.n_chunks <= 0 || input.dim <= 0 ||
        input.words_per_code != (input.dim + 31) / 32 || input.vector_chunk_size <= 0 ||
        input.vector_chunk_size > kTurboQuantLocalTopK) {
        throw std::invalid_argument("turboquant_1bit_scan_top256_gpu: invalid common input");
    }
    if (mode == TurboQuant1BitScoreMode::MseFloatQuery &&
        (!input.rotated_queries || !input.query_norm_sq || !input.database_row_parameters)) {
        throw std::invalid_argument("turboquant_1bit_scan_top256_gpu: missing MSE query input");
    }
    if (mode == TurboQuant1BitScoreMode::BinaryQuery &&
        (!input.packed_query_codes || !input.query_norms || !input.database_row_parameters)) {
        throw std::invalid_argument("turboquant_1bit_scan_top256_gpu: missing binary query input");
    }
}

}  // namespace

void turboquant_1bit_scan_top256_gpu(
    const TurboQuant1BitScanInput& input,
    TurboQuant1BitScoreMode mode,
    float reconstruction_centroid,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream) {
    validate_scan_input(input, mode, output_indices, output_distances);
    const std::size_t query_bytes =
        mode == TurboQuant1BitScoreMode::MseFloatQuery
            ? static_cast<std::size_t>(input.dim) * sizeof(float)
            : static_cast<std::size_t>(input.words_per_code) * sizeof(std::uint32_t);
    const std::size_t shared_bytes =
        kTurboQuantLocalTopK * (sizeof(float) + sizeof(int)) + query_bytes;
    if (mode == TurboQuant1BitScoreMode::MseFloatQuery) {
        turboquant_1bit_scan_top256_kernel<TurboQuant1BitScoreMode::MseFloatQuery>
            <<<input.n_chunks, kBlockThreads, shared_bytes, stream>>>(
                input, reconstruction_centroid, output_indices, output_distances);
    } else {
        if (input.words_per_code == 4 && input.dim == 128) {
            turboquant_1bit_scan_top256_kernel<TurboQuant1BitScoreMode::BinaryQuery, 4>
                <<<input.n_chunks, kBlockThreads, shared_bytes, stream>>>(
                    input, reconstruction_centroid, output_indices, output_distances);
        } else {
            turboquant_1bit_scan_top256_kernel<TurboQuant1BitScoreMode::BinaryQuery>
                <<<input.n_chunks, kBlockThreads, shared_bytes, stream>>>(
                    input, reconstruction_centroid, output_indices, output_distances);
        }
    }
    CUDA_LAUNCH_CHECK("turboquant_1bit_scan_top256_kernel");
}

}  // namespace paraselect::search::turboquant
