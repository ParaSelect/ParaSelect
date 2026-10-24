#ifndef PARASELECT_TURBOQUANT_1BIT_ENCODE_CUH
#define PARASELECT_TURBOQUANT_1BIT_ENCODE_CUH

#include <cuda_runtime.h>

#include <cstdint>

namespace paraselect::search::turboquant {

struct TurboQuant1BitEncodeInput {
    const float* vectors = nullptr;
    int n_vectors = 0;
    int dim = 0;
    int padded_dim = 0;
    std::uint32_t rotation_seed = 0;
};

struct TurboQuant1BitEncodeOutput {
    float* rotated_vectors = nullptr;
    std::uint32_t* packed_codes = nullptr;
    float* norms = nullptr;
    float* norm_sq = nullptr;
};

// Offline residual encoding: input rows are already residual vectors
// (vector - centroid), so only rotation/packing runs. Per-row output norms are
// the L1 reconstruction scales used by the scan.
struct TurboQuant1BitResidualEncodeInput {
    const float* residuals = nullptr;
    int n_vectors = 0;
    int dim = 0;
    int padded_dim = 0;
    std::uint32_t rotation_seed = 0;
};

void turboquant_1bit_encode_residual_l1_gpu(
    const TurboQuant1BitResidualEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream);

// Offline IVFResTurboQuant encoding. Rows must use the same list-contiguous
// order described by cluster_offsets.
struct IVFResTurboQuantDatabaseEncodeInput {
    const std::uint8_t* vectors_u8 = nullptr;
    const float* centroids = nullptr;
    const long long* cluster_offsets = nullptr;
    int n_vectors = 0;
    int n_clusters = 0;
    int dim = 0;
    int padded_dim = 0;
    std::uint32_t rotation_seed = 0;
    // Global row index of vectors_u8[0]; lets callers encode the database in
    // row chunks while cluster_offsets and outputs stay in global row order.
    int row_offset = 0;
};

struct IVFResTurboQuantQueryEncodeInput {
    const float* queries = nullptr;
    const float* centroids = nullptr;
    const long long* probe_ids = nullptr;
    int n_queries = 0;
    int n_probes = 0;
    int dim = 0;
    int padded_dim = 0;
    std::uint32_t rotation_seed = 0;
};

void turboquant_1bit_encode_gpu(
    const TurboQuant1BitEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream);

void ivf_res_turboquant_encode_database_u8_gpu(
    const IVFResTurboQuantDatabaseEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream);

void ivf_res_turboquant_encode_queries_gpu(
    const IVFResTurboQuantQueryEncodeInput& input,
    const TurboQuant1BitEncodeOutput& output,
    cudaStream_t stream);

}  // namespace paraselect::search::turboquant

#endif  // PARASELECT_TURBOQUANT_1BIT_ENCODE_CUH
