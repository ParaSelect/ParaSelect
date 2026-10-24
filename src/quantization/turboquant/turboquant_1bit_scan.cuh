#ifndef PARASELECT_TURBOQUANT_1BIT_SCAN_CUH
#define PARASELECT_TURBOQUANT_1BIT_SCAN_CUH

#include <cuda_runtime.h>

#include <cstdint>

namespace paraselect::search::turboquant {

enum class TurboQuant1BitScoreMode {
    MseFloatQuery,
    BinaryQuery,
};

struct TurboQuantChunkDesc {
    int query;
    int cluster;
    int query_state;
    int start;
    int length;
};

struct TurboQuantGroupDesc {
    int source_start;
    int source_count;
};

struct TurboQuant1BitScanInput {
    const float* rotated_queries = nullptr;
    const std::uint32_t* packed_query_codes = nullptr;
    const float* query_norms = nullptr;
    const float* query_norm_sq = nullptr;
    const std::uint32_t* packed_database_codes = nullptr;
    // L2 norms for BinaryQuery; reconstruction scales for MseFloatQuery.
    const float* database_row_parameters = nullptr;
    const long long* cluster_offsets = nullptr;
    const TurboQuantChunkDesc* chunks = nullptr;
    int n_chunks = 0;
    int dim = 0;
    int words_per_code = 0;
    int vector_chunk_size = 0;
};

inline constexpr int kTurboQuantLocalTopK = 256;

void turboquant_1bit_scan_top256_gpu(
    const TurboQuant1BitScanInput& input,
    TurboQuant1BitScoreMode mode,
    float reconstruction_centroid,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream);

void turboquant_1bit_grouped_scan_top512_gpu(
    const TurboQuant1BitScanInput& input,
    const TurboQuantGroupDesc* groups,
    int n_groups,
    int group_chunks,
    TurboQuant1BitScoreMode mode,
    float reconstruction_centroid,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream);

// Runtime candidate width. Supported widths are 256, 512, 1024 and 2048;
// the implementation dispatches to explicit CUDA top-k instantiations.
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
    cudaStream_t stream);

// Merge the sorted top-512 emitted by each chunk group into one sorted
// top-512 per query. query_group_starts/counts describe contiguous groups.
void turboquant_merge_query_groups_top512_gpu(
    const int* group_indices,
    const float* group_distances,
    const int* query_group_starts,
    const int* query_group_counts,
    int n_queries,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream);

void turboquant_merge_query_groups_topk_gpu(
    const int* group_indices,
    const float* group_distances,
    const int* query_group_starts,
    const int* query_group_counts,
    int n_queries,
    int topk,
    int* output_indices,
    float* output_distances,
    cudaStream_t stream);

}  // namespace paraselect::search::turboquant

#endif  // PARASELECT_TURBOQUANT_1BIT_SCAN_CUH
