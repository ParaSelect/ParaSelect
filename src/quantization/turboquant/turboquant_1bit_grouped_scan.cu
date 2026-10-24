#include "quantization/turboquant/turboquant_1bit_scan.cuh"
#include "quantization/turboquant/turboquant_1bit_score.cuh"

#include "utils/check_cuda_error.cuh"

#include <cfloat>
#include <cstddef>
#include <stdexcept>

namespace paraselect::search::turboquant {
namespace {

constexpr int kBlockThreads = 256;
constexpr int kDefaultGlobalTopK = 512;

__device__ __forceinline__ bool pair_less_equal(
    float lhs_dist, int lhs_idx, float rhs_dist, int rhs_idx) {
    return !detail::pair_greater(lhs_dist, lhs_idx, rhs_dist, rhs_idx);
}

template <int GlobalTopK>
__device__ __forceinline__ int merge_partition(
    const float* a_dist,
    const int* a_idx,
    const float* b_dist,
    const int* b_idx,
    int rank) {
    int lo = max(0, rank - GlobalTopK);
    int hi = min(rank, GlobalTopK);
    while (lo < hi) {
        const int i = (lo + hi) >> 1;
        const int j = rank - i;
        if (j > 0 && i < GlobalTopK &&
            detail::pair_greater(
                b_dist[j - 1], b_idx[j - 1], a_dist[i], a_idx[i])) {
            lo = i + 1;
        } else {
            hi = i;
        }
    }
    return lo;
}

template <int GlobalTopK>
__global__ void turboquant_merge_query_groups_topk_kernel(
    const int* __restrict__ group_indices, const float* __restrict__ group_distances,
    const int* __restrict__ query_group_starts, const int* __restrict__ query_group_counts,
    int n_queries, int* __restrict__ output_indices, float* __restrict__ output_distances,
    int* __restrict__ scratch_indices, float* __restrict__ scratch_distances) {
    const int query = blockIdx.x; if (query >= n_queries) return;
    const int start = query_group_starts[query]; const int count = query_group_counts[query];
    __shared__ float local_dist[GlobalTopK <= 2048 ? GlobalTopK * 2 : 1];
    __shared__ int local_idx[GlobalTopK <= 2048 ? GlobalTopK * 2 : 1];
    float* current_dist = output_distances + static_cast<std::size_t>(query) * GlobalTopK;
    int* current_idx = output_indices + static_cast<std::size_t>(query) * GlobalTopK;
    float* next_dist = scratch_distances ? scratch_distances + static_cast<std::size_t>(query) * GlobalTopK : nullptr;
    int* next_idx = scratch_indices ? scratch_indices + static_cast<std::size_t>(query) * GlobalTopK : nullptr;
    if constexpr (GlobalTopK <= 2048) {
        current_dist = local_dist;
        next_dist = local_dist + GlobalTopK;
        current_idx = local_idx;
        next_idx = local_idx + GlobalTopK;
    }
    for (int k = threadIdx.x; k < GlobalTopK; k += blockDim.x) {
        if (count > 0) { current_dist[k] = group_distances[static_cast<std::size_t>(start) * GlobalTopK + k]; current_idx[k] = group_indices[static_cast<std::size_t>(start) * GlobalTopK + k]; }
        else { current_dist[k] = FLT_MAX; current_idx[k] = -1; }
    }
    __syncthreads();
    for (int source = 1; source < count; ++source) {
        const float* b_dist = group_distances + static_cast<std::size_t>(start + source) * GlobalTopK;
        const int* b_idx = group_indices + static_cast<std::size_t>(start + source) * GlobalTopK;
        for (int rank = threadIdx.x; rank < GlobalTopK; rank += blockDim.x) {
            const int i = merge_partition<GlobalTopK>(current_dist, current_idx, b_dist, b_idx, rank);
            const int j = rank - i;
            const bool take_a = i < GlobalTopK && (j >= GlobalTopK || pair_less_equal(current_dist[i], current_idx[i], b_dist[j], b_idx[j]));
            next_dist[rank] = take_a ? current_dist[i] : b_dist[j]; next_idx[rank] = take_a ? current_idx[i] : b_idx[j];
        }
        __syncthreads();
        float* td = current_dist; current_dist = next_dist; next_dist = td;
        int* ti = current_idx; current_idx = next_idx; next_idx = ti;
        __syncthreads();
    }
    if (current_dist != output_distances + static_cast<std::size_t>(query) * GlobalTopK) {
        for (int k = threadIdx.x; k < GlobalTopK; k += blockDim.x) {
            output_distances[static_cast<std::size_t>(query) * GlobalTopK + k] = current_dist[k];
            output_indices[static_cast<std::size_t>(query) * GlobalTopK + k] = current_idx[k];
        }
    }
}

template <TurboQuant1BitScoreMode Mode, int GlobalTopK, int FixedWordsPerCode = 0>
__global__ void turboquant_1bit_grouped_scan_topk_kernel(
    TurboQuant1BitScanInput input,
    const TurboQuantGroupDesc* __restrict__ groups,
    int n_groups,
    int group_chunks,
    float reconstruction_centroid,
    int* __restrict__ output_indices,
    float* __restrict__ output_distances,
    float* __restrict__ workspace_distances,
    int* __restrict__ workspace_indices) {
    extern __shared__ unsigned char shared_bytes[];
    const int group_id = blockIdx.x;
    const int lane = threadIdx.x;
    if (group_id >= n_groups) return;
    float* shared_query = reinterpret_cast<float*>(shared_bytes);
    float* shared_distances = workspace_distances ? workspace_distances + static_cast<std::size_t>(group_id) * group_chunks * kTurboQuantLocalTopK : nullptr;
    int* shared_indices = workspace_indices ? workspace_indices + static_cast<std::size_t>(group_id) * group_chunks * kTurboQuantLocalTopK : nullptr;

    if constexpr (GlobalTopK <= 2048) {
        shared_distances = reinterpret_cast<float*>(shared_bytes);
        shared_indices = reinterpret_cast<int*>(shared_distances + group_chunks * kTurboQuantLocalTopK);
        shared_query = reinterpret_cast<float*>(shared_indices + group_chunks * kTurboQuantLocalTopK);
    }
    const TurboQuantGroupDesc group = groups[group_id];

    for (int source = 0; source < group.source_count; ++source) {
        const TurboQuantChunkDesc chunk = input.chunks[group.source_start + source];
        if constexpr (Mode == TurboQuant1BitScoreMode::MseFloatQuery) {
            const float* query = input.rotated_queries +
                static_cast<std::size_t>(chunk.query_state) * input.dim;
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
        const int output = source * kTurboQuantLocalTopK + lane;
        shared_distances[output] = best_distance;
        shared_indices[output] = best_index;
        __syncthreads();
    }

    const int merge_items = group_chunks * kTurboQuantLocalTopK;
    for (int item = lane + group.source_count * kTurboQuantLocalTopK;
         item < merge_items;
         item += blockDim.x) {
        shared_distances[item] = FLT_MAX;
        shared_indices[item] = -1;
    }
    __syncthreads();

    for (int width = 2; width <= merge_items; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int item = lane; item < merge_items; item += blockDim.x) {
                const int other = item ^ stride;
                if (other > item) {
                    const bool ascending = (item & width) == 0;
                    const float lhs_distance = shared_distances[item];
                    const float rhs_distance = shared_distances[other];
                    const int lhs_index = shared_indices[item];
                    const int rhs_index = shared_indices[other];
                    const bool lhs_greater = detail::pair_greater(
                        lhs_distance, lhs_index, rhs_distance, rhs_index);
                    const bool lhs_less = detail::pair_greater(
                        rhs_distance, rhs_index, lhs_distance, lhs_index);
                    if ((ascending && lhs_greater) || (!ascending && lhs_less)) {
                        shared_distances[item] = rhs_distance;
                        shared_indices[item] = rhs_index;
                        shared_distances[other] = lhs_distance;
                        shared_indices[other] = lhs_index;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int k = lane; k < GlobalTopK; k += blockDim.x) {
        const std::size_t output = static_cast<std::size_t>(group_id) * GlobalTopK + k;
        output_distances[output] = shared_distances[k];
        output_indices[output] = shared_indices[k];
    }
}

static void validate_grouped_input(
    const TurboQuant1BitScanInput& input,
    const TurboQuantGroupDesc* groups,
    int n_groups,
    int group_chunks,
    TurboQuant1BitScoreMode mode,
    const int* output_indices,
    const float* output_distances) {
    const bool valid_group_chunks =
        group_chunks == 2 || group_chunks == 4 || group_chunks == 8 ||
        group_chunks == 16 || group_chunks == 32 || group_chunks == 64;
    if (!input.packed_database_codes || !input.cluster_offsets || !input.chunks || !groups ||
        !output_indices || !output_distances || input.n_chunks <= 0 || n_groups <= 0 ||
        input.dim <= 0 || input.words_per_code != (input.dim + 31) / 32 ||
        input.vector_chunk_size <= 0 || !valid_group_chunks) {
        throw std::invalid_argument("turboquant_1bit_grouped_scan_top512_gpu: invalid common input");
    }
    if (mode == TurboQuant1BitScoreMode::MseFloatQuery &&
        (!input.rotated_queries || !input.query_norm_sq || !input.database_row_parameters)) {
        throw std::invalid_argument("turboquant_1bit_grouped_scan_top512_gpu: missing MSE input");
    }
    if (mode == TurboQuant1BitScoreMode::BinaryQuery &&
        (!input.packed_query_codes || !input.query_norms || !input.database_row_parameters)) {
        throw std::invalid_argument("turboquant_1bit_grouped_scan_top512_gpu: missing binary input");
    }
}

}  // namespace

template <int GlobalTopK>
static void launch_grouped_scan_topk(
    const TurboQuant1BitScanInput& input,
    const TurboQuantGroupDesc* groups,
    int n_groups,
    int group_chunks,
    TurboQuant1BitScoreMode mode,
    float reconstruction_centroid,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream) {
    validate_grouped_input(
        input, groups, n_groups, group_chunks, mode, output_indices, output_distances);
    const std::size_t query_bytes =
        mode == TurboQuant1BitScoreMode::MseFloatQuery
            ? static_cast<std::size_t>(input.dim) * sizeof(float)
            : static_cast<std::size_t>(input.words_per_code) * sizeof(std::uint32_t);
    const std::size_t shared_bytes = query_bytes + (GlobalTopK <= 2048 ? static_cast<std::size_t>(group_chunks) * kTurboQuantLocalTopK * (sizeof(float) + sizeof(int)) : 0);
    const std::size_t workspace_items = static_cast<std::size_t>(n_groups) * group_chunks * kTurboQuantLocalTopK;
    float* workspace_distances = nullptr;
    int* workspace_indices = nullptr;
    if constexpr (GlobalTopK > 2048) {
    if (cudaMalloc(&workspace_distances, workspace_items * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&workspace_indices, workspace_items * sizeof(int)) != cudaSuccess) {
        if (workspace_distances) cudaFree(workspace_distances);
        if (workspace_indices) cudaFree(workspace_indices);
        throw std::runtime_error("failed to allocate TurboQuant scan workspace");
    }
    }
    if (mode == TurboQuant1BitScoreMode::MseFloatQuery) {
        turboquant_1bit_grouped_scan_topk_kernel<TurboQuant1BitScoreMode::MseFloatQuery, GlobalTopK>
            <<<n_groups, kBlockThreads, shared_bytes, stream>>>(
                input, groups, n_groups, group_chunks, reconstruction_centroid,
                output_indices, output_distances, workspace_distances, workspace_indices);
    } else {
        if (input.words_per_code == 4 && input.dim == 128) {
            turboquant_1bit_grouped_scan_topk_kernel<TurboQuant1BitScoreMode::BinaryQuery, GlobalTopK, 4>
                <<<n_groups, kBlockThreads, shared_bytes, stream>>>(
                    input, groups, n_groups, group_chunks, reconstruction_centroid,
                    output_indices, output_distances, workspace_distances, workspace_indices);
        } else {
            turboquant_1bit_grouped_scan_topk_kernel<TurboQuant1BitScoreMode::BinaryQuery, GlobalTopK>
                <<<n_groups, kBlockThreads, shared_bytes, stream>>>(
                    input, groups, n_groups, group_chunks, reconstruction_centroid,
                    output_indices, output_distances, workspace_distances, workspace_indices);
        }
    }
    CUDA_LAUNCH_CHECK("turboquant_1bit_grouped_scan_topk_kernel");
    if (workspace_distances) cudaFree(workspace_distances);
    if (workspace_indices) cudaFree(workspace_indices);
}

template <int GlobalTopK>
static void launch_merge_query_groups_topk(
    const int* group_indices,
    const float* group_distances,
    const int* query_group_starts,
    const int* query_group_counts,
    int n_queries,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream) {
    if (!group_indices || !group_distances || !query_group_starts ||
        !query_group_counts || !output_indices || !output_distances || n_queries <= 0) {
        throw std::invalid_argument(
            "turboquant_merge_query_groups_top512_gpu: invalid input");
    }
    int* scratch_indices = nullptr; float* scratch_distances = nullptr;
    const std::size_t scratch_items = static_cast<std::size_t>(n_queries) * GlobalTopK;
    if constexpr (GlobalTopK > 2048) {
    if (cudaMalloc(&scratch_indices, scratch_items * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&scratch_distances, scratch_items * sizeof(float)) != cudaSuccess) {
        if (scratch_indices) cudaFree(scratch_indices); if (scratch_distances) cudaFree(scratch_distances);
        throw std::runtime_error("failed to allocate TurboQuant merge scratch");
    }
    }
    turboquant_merge_query_groups_topk_kernel<GlobalTopK><<<n_queries, kBlockThreads, 0, stream>>>(
        group_indices, group_distances, query_group_starts, query_group_counts,
        n_queries, output_indices, output_distances, scratch_indices, scratch_distances);
    CUDA_LAUNCH_CHECK("turboquant_merge_query_groups_topk_kernel");
    if (scratch_indices) cudaFree(scratch_indices);
    if (scratch_distances) cudaFree(scratch_distances);
}

void turboquant_1bit_grouped_scan_topk_gpu(
    const TurboQuant1BitScanInput& input,
    const TurboQuantGroupDesc* groups,
    int n_groups,
    int group_chunks,
    TurboQuant1BitScoreMode mode,
    float reconstruction_centroid,
    int topk,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream) {
    validate_grouped_input(input, groups, n_groups, group_chunks, mode,
                           output_indices, output_distances);
    switch (topk) {
        case 256: launch_grouped_scan_topk<256>(input, groups, n_groups, group_chunks, mode,
                                                reconstruction_centroid, output_indices, output_distances, stream); break;
        case 512: launch_grouped_scan_topk<512>(input, groups, n_groups, group_chunks, mode,
                                                reconstruction_centroid, output_indices, output_distances, stream); break;
        case 1024: launch_grouped_scan_topk<1024>(input, groups, n_groups, group_chunks, mode,
                                                  reconstruction_centroid, output_indices, output_distances, stream); break;
        case 2048: launch_grouped_scan_topk<2048>(input, groups, n_groups, group_chunks, mode,
                                                  reconstruction_centroid, output_indices, output_distances, stream); break;
        case 4096: launch_grouped_scan_topk<4096>(input, groups, n_groups, group_chunks, mode,
                                                  reconstruction_centroid, output_indices, output_distances, stream); break;
        case 8192: launch_grouped_scan_topk<8192>(input, groups, n_groups, group_chunks, mode,
                                                  reconstruction_centroid, output_indices, output_distances, stream); break;
        case 16384: launch_grouped_scan_topk<16384>(input, groups, n_groups, group_chunks, mode,
                                                    reconstruction_centroid, output_indices, output_distances, stream); break;
        default: throw std::invalid_argument("TurboQuant topk must be one of 256, 512, 1024, 2048, 4096, 8192, 16384");
    }
}

void turboquant_merge_query_groups_topk_gpu(
    const int* group_indices, const float* group_distances,
    const int* query_group_starts, const int* query_group_counts,
    int n_queries, int topk, int* output_indices, float* output_distances,
    cudaStream_t stream) {
    if (topk != 256 && topk != 512 && topk != 1024 && topk != 2048 &&
        topk != 4096 && topk != 8192 && topk != 16384)
        throw std::invalid_argument("TurboQuant topk must be one of 256, 512, 1024, 2048, 4096, 8192, 16384");
    if (topk == 256) launch_merge_query_groups_topk<256>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
    else if (topk == 512) launch_merge_query_groups_topk<512>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
    else if (topk == 1024) launch_merge_query_groups_topk<1024>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
    else if (topk == 2048) launch_merge_query_groups_topk<2048>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
    else if (topk == 4096) launch_merge_query_groups_topk<4096>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
    else if (topk == 8192) launch_merge_query_groups_topk<8192>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
    else launch_merge_query_groups_topk<16384>(group_indices, group_distances, query_group_starts, query_group_counts, n_queries, output_indices, output_distances, stream);
}

void turboquant_1bit_grouped_scan_top512_gpu(
    const TurboQuant1BitScanInput& input, const TurboQuantGroupDesc* groups,
    int n_groups, int group_chunks, TurboQuant1BitScoreMode mode,
    float reconstruction_centroid, int* output_indices, float* output_distances,
    cudaStream_t stream) {
    turboquant_1bit_grouped_scan_topk_gpu(input, groups, n_groups, group_chunks, mode,
                                          reconstruction_centroid, kDefaultGlobalTopK,
                                          output_indices, output_distances, stream);
}

void turboquant_merge_query_groups_top512_gpu(
    const int* group_indices, const float* group_distances,
    const int* query_group_starts, const int* query_group_counts,
    int n_queries, int* output_indices, float* output_distances,
    cudaStream_t stream) {
    turboquant_merge_query_groups_topk_gpu(group_indices, group_distances,
                                            query_group_starts, query_group_counts,
                                            n_queries, kDefaultGlobalTopK,
                                            output_indices, output_distances, stream);
}

}  // namespace paraselect::search::turboquant
