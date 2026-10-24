#include "quantization/turboquant/turboquant_1bit_encode.cuh"

#include "utils/check_cuda_error.cuh"

#include <cmath>
#include <stdexcept>

namespace paraselect::search::turboquant {
namespace {

__device__ __forceinline__ std::uint32_t mix_bits(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__device__ __forceinline__ int row_cluster(
    int row, const long long* offsets, int n_clusters) {
    int lo = 0;
    int hi = n_clusters;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (offsets[mid] <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

template <bool StoreL1Scale = false>
__device__ __forceinline__ void encode_rotated_row(
    int row,
    int dim,
    int padded_dim,
    std::uint32_t rotation_seed,
    float original,
    TurboQuant1BitEncodeOutput output,
    float* values) {
    const int lane = threadIdx.x;
    if (lane < padded_dim) {
        const bool negative =
            (mix_bits(rotation_seed ^ static_cast<std::uint32_t>(lane)) & 1u) != 0;
        values[lane] = negative ? -original : original;
    }
    __syncthreads();

    for (int width = 1; width < padded_dim; width <<= 1) {
        if (lane < padded_dim / 2) {
            const int group = lane / width;
            const int offset = lane - group * width;
            const int left = group * (width << 1) + offset;
            const int right = left + width;
            const float lhs = values[left];
            const float rhs = values[right];
            values[left] = lhs + rhs;
            values[right] = lhs - rhs;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(static_cast<float>(padded_dim));
    float rotated = 0.0f;
    if (lane < padded_dim) {
        rotated = values[lane] * scale;
        values[lane] = rotated;
        if (output.rotated_vectors) {
            output.rotated_vectors[static_cast<std::size_t>(row) * padded_dim + lane] = rotated;
        }
    }
    __syncthreads();

    const unsigned signs = __ballot_sync(
        __activemask(), lane < padded_dim && values[lane] >= 0.0f);
    if (output.packed_codes && (lane & 31) == 0 && lane < padded_dim) {
        output.packed_codes[
            static_cast<std::size_t>(row) * (padded_dim / 32) + lane / 32] = signs;
    }

    float statistic = StoreL1Scale
        ? (lane < padded_dim ? fabsf(rotated) : 0.0f)
        : (lane < dim ? original * original : 0.0f);
    for (int delta = 16; delta > 0; delta >>= 1) {
        statistic += __shfl_down_sync(0xffffffffu, statistic, delta);
    }
    __shared__ float warp_sums[8];
    if ((lane & 31) == 0) warp_sums[lane / 32] = statistic;
    __syncthreads();
    if (lane < 32) {
        float sum = lane < padded_dim / 32 ? warp_sums[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        }
        if (lane == 0) {
            if (output.norm_sq) output.norm_sq[row] = sum;
            if (output.norms) {
                output.norms[row] = StoreL1Scale
                    ? sum / static_cast<float>(padded_dim)
                    : sqrtf(sum);
            }
        }
    }
}

__global__ void turboquant_1bit_encode_kernel(
    TurboQuant1BitEncodeInput input,
    TurboQuant1BitEncodeOutput output) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= input.n_vectors) return;

    float original = 0.0f;
    if (lane < input.padded_dim) {
        original = lane < input.dim
            ? input.vectors[static_cast<std::size_t>(row) * input.dim + lane]
            : 0.0f;
        const bool negative = (mix_bits(input.rotation_seed ^ static_cast<std::uint32_t>(lane)) & 1u) != 0;
        values[lane] = negative ? -original : original;
    }
    __syncthreads();

    for (int width = 1; width < input.padded_dim; width <<= 1) {
        if (lane < input.padded_dim / 2) {
            const int group = lane / width;
            const int offset = lane - group * width;
            const int left = group * (width << 1) + offset;
            const int right = left + width;
            const float lhs = values[left];
            const float rhs = values[right];
            values[left] = lhs + rhs;
            values[right] = lhs - rhs;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(static_cast<float>(input.padded_dim));
    if (lane < input.padded_dim) {
        const float rotated = values[lane] * scale;
        values[lane] = rotated;
        if (output.rotated_vectors) {
            output.rotated_vectors[
                static_cast<std::size_t>(row) * input.padded_dim + lane] = rotated;
        }
    }
    __syncthreads();

    const unsigned active = __activemask();
    const unsigned signs = __ballot_sync(
        active, lane < input.padded_dim && values[lane] >= 0.0f);
    if (output.packed_codes && (lane & 31) == 0 && lane < input.padded_dim) {
        output.packed_codes[
            static_cast<std::size_t>(row) * (input.padded_dim / 32) + lane / 32] = signs;
    }

    float square = lane < input.dim ? original * original : 0.0f;
    for (int delta = 16; delta > 0; delta >>= 1) {
        square += __shfl_down_sync(0xffffffffu, square, delta);
    }
    __shared__ float warp_sums[8];
    if ((lane & 31) == 0) warp_sums[lane / 32] = square;
    __syncthreads();
    if (lane < 32) {
        float sum = lane < input.padded_dim / 32 ? warp_sums[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        }
        if (lane == 0) {
            if (output.norm_sq) output.norm_sq[row] = sum;
            if (output.norms) output.norms[row] = sqrtf(sum);
        }
    }
}

__global__ void turboquant_1bit_encode_residual_l1_kernel(
    TurboQuant1BitResidualEncodeInput input,
    TurboQuant1BitEncodeOutput output) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    if (row >= input.n_vectors) return;
    const int lane = threadIdx.x;
    const float original = lane < input.dim
        ? input.residuals[static_cast<std::size_t>(row) * input.dim + lane]
        : 0.0f;
    encode_rotated_row<true>(
        row, input.dim, input.padded_dim, input.rotation_seed,
        original, output, values);
}

__global__ void ivf_res_turboquant_database_encode_u8_kernel(
    IVFResTurboQuantDatabaseEncodeInput input,
    TurboQuant1BitEncodeOutput output) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    if (row >= input.n_vectors) return;
    const int global_row = row + input.row_offset;
    const int cluster = row_cluster(global_row, input.cluster_offsets, input.n_clusters);
    const int lane = threadIdx.x;
    const float original = lane < input.dim
        ? static_cast<float>(
              input.vectors_u8[static_cast<std::size_t>(row) * input.dim + lane]) -
              input.centroids[static_cast<std::size_t>(cluster) * input.dim + lane]
        : 0.0f;
    encode_rotated_row<true>(
        global_row, input.dim, input.padded_dim, input.rotation_seed,
        original, output, values);
}

__global__ void ivf_res_turboquant_query_encode_kernel(
    IVFResTurboQuantQueryEncodeInput input,
    TurboQuant1BitEncodeOutput output) {
    extern __shared__ float values[];
    const int state = blockIdx.x;
    const int total = input.n_queries * input.n_probes;
    if (state >= total) return;
    const int query = state / input.n_probes;
    const long long cluster = input.probe_ids[state];
    const int lane = threadIdx.x;
    const float original = lane < input.dim
        ? input.queries[static_cast<std::size_t>(query) * input.dim + lane] -
              input.centroids[static_cast<std::size_t>(cluster) * input.dim + lane]
        : 0.0f;
    encode_rotated_row<false>(
        state, input.dim, input.padded_dim, input.rotation_seed,
        original, output, values);
}

static void validate_shape(int dim, int padded_dim) {
    const bool padded_power_of_two =
        padded_dim > 0 && (padded_dim & (padded_dim - 1)) == 0;
    if (dim <= 0 || dim > padded_dim || !padded_power_of_two || padded_dim > 256 ||
        (padded_dim & 31) != 0) {
        throw std::invalid_argument("TurboQuant encode: invalid dimensions");
    }
}

}  // namespace

void turboquant_1bit_encode_gpu(
    const TurboQuant1BitEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream) {
    const bool padded_power_of_two =
        input.padded_dim > 0 && (input.padded_dim & (input.padded_dim - 1)) == 0;
    if (!input.vectors || (!output.packed_codes && !output.rotated_vectors) || input.n_vectors <= 0 || input.dim <= 0 ||
        input.dim > input.padded_dim || !padded_power_of_two || input.padded_dim > 256 ||
        (input.padded_dim & 31) != 0) {
        throw std::invalid_argument("turboquant_1bit_encode_gpu: invalid input");
    }
    turboquant_1bit_encode_kernel<<<
        input.n_vectors, input.padded_dim, input.padded_dim * sizeof(float), stream>>>(
            input, output);
    CUDA_LAUNCH_CHECK("turboquant_1bit_encode_kernel");
}

void turboquant_1bit_encode_residual_l1_gpu(
    const TurboQuant1BitResidualEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream) {
    validate_shape(input.dim, input.padded_dim);
    if (!input.residuals || !output.packed_codes || !output.norms ||
        input.n_vectors <= 0) {
        throw std::invalid_argument("turboquant_1bit_encode_residual_l1_gpu: invalid input");
    }
    turboquant_1bit_encode_residual_l1_kernel<<<
        input.n_vectors, input.padded_dim, input.padded_dim * sizeof(float), stream>>>(
            input, output);
    CUDA_LAUNCH_CHECK("turboquant_1bit_encode_residual_l1_kernel");
}

void ivf_res_turboquant_encode_database_u8_gpu(
    const IVFResTurboQuantDatabaseEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream) {
    validate_shape(input.dim, input.padded_dim);
    if (!input.vectors_u8 || !input.centroids || !input.cluster_offsets ||
        !output.packed_codes || input.n_vectors <= 0 || input.n_clusters <= 0) {
        throw std::invalid_argument("ivf_res_turboquant_encode_database_u8_gpu: invalid input");
    }
    ivf_res_turboquant_database_encode_u8_kernel<<<
        input.n_vectors, input.padded_dim, input.padded_dim * sizeof(float), stream>>>(
            input, output);
    CUDA_LAUNCH_CHECK("ivf_res_turboquant_database_encode_u8_kernel");
}

void ivf_res_turboquant_encode_queries_gpu(
    const IVFResTurboQuantQueryEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream) {
    validate_shape(input.dim, input.padded_dim);
    if (!input.queries || !input.centroids || !input.probe_ids ||
        (!output.rotated_vectors && !output.packed_codes) ||
        input.n_queries <= 0 || input.n_probes <= 0) {
        throw std::invalid_argument("ivf_res_turboquant_encode_queries_gpu: invalid input");
    }
    const int states = input.n_queries * input.n_probes;
    ivf_res_turboquant_query_encode_kernel<<<
        states, input.padded_dim, input.padded_dim * sizeof(float), stream>>>(input, output);
    CUDA_LAUNCH_CHECK("ivf_res_turboquant_query_encode_kernel");
}

}  // namespace paraselect::search::turboquant
