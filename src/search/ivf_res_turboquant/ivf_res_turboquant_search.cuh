#ifndef PARASELECT_IVF_RES_TURBOQUANT_SEARCH_CUH
#define PARASELECT_IVF_RES_TURBOQUANT_SEARCH_CUH

#include "quantization/turboquant/turboquant_1bit_scan.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace paraselect::search::ivf_res_turboquant {

inline constexpr const char* kMethodName = "IVFResTurboQuant";
inline constexpr int kCandidateCount = 512;
inline constexpr int kMaxCandidateCount = 16384;

struct SearchTiming {
    double total_ms = 0.0;
    float query_encode_ms = 0.0f;
    float grouped_scan_ms = 0.0f;
    float query_merge_ms = 0.0f;
};

class Index {
public:
    Index() = default;
    ~Index();
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    void build_u8(
        const std::uint8_t* host_vectors,
        const float* host_centroids,
        const long long* host_cluster_offsets,
        int n_vectors,
        int n_clusters,
        int dim,
        std::uint32_t rotation_seed = 17u);
    void build_cached_u32(
        const std::uint32_t* host_codes,
        const float* host_scales,
        const float* host_centroids,
        const long long* host_cluster_offsets,
        int n_vectors,
        int n_clusters,
        int dim);
    void release();

    int dim() const { return dim_; }
    int n_clusters() const { return n_clusters_; }
    int n_vectors() const { return n_vectors_; }
    std::size_t bytes_per_vector() const {
        return static_cast<std::size_t>(words_per_code_) * sizeof(std::uint32_t) + sizeof(float);
    }

private:
    friend class SearchWorkspace;
    int n_vectors_ = 0;
    int n_clusters_ = 0;
    int dim_ = 0;
    int padded_dim_ = 0;
    int words_per_code_ = 0;
    std::uint32_t rotation_seed_ = 17u;
    float* d_centroids_ = nullptr;
    long long* d_cluster_offsets_ = nullptr;
    std::uint32_t* d_codes_ = nullptr;
    float* d_reconstruction_scales_ = nullptr;
};

class SearchWorkspace {
public:
    SearchWorkspace() = default;
    ~SearchWorkspace();
    SearchWorkspace(const SearchWorkspace&) = delete;
    SearchWorkspace& operator=(const SearchWorkspace&) = delete;

    void search_candidates(
        const Index& index,
        const float* host_queries,
        const long long* host_probe_ids,
        const long long* host_cluster_offsets,
        int n_queries,
        int n_probes,
        int* host_candidate_ids,
        int candidate_count = kCandidateCount,
        SearchTiming* timing = nullptr,
        turboquant::TurboQuant1BitScoreMode score_mode =
            turboquant::TurboQuant1BitScoreMode::MseFloatQuery,
        bool exact_selection = false);
    void release();

private:
    void reserve(const Index& index, int n_queries, int n_probes, int n_chunks, int n_groups,
                 int candidate_count);

    int query_capacity_ = 0;
    int probe_capacity_ = 0;
    int chunk_capacity_ = 0;
    int group_capacity_ = 0;
    std::size_t group_items_capacity_ = 0;
    float* d_queries_ = nullptr;
    long long* d_probe_ids_ = nullptr;
    float* d_rotated_query_states_ = nullptr;
    float* d_query_norm_sq_ = nullptr;
    std::uint32_t* d_packed_query_states_ = nullptr;
    float* d_query_norms_ = nullptr;
    turboquant::TurboQuantChunkDesc* d_chunks_ = nullptr;
    turboquant::TurboQuantGroupDesc* d_groups_ = nullptr;
    int* d_query_group_starts_ = nullptr;
    int* d_query_group_counts_ = nullptr;
    int* d_group_indices_ = nullptr;
    float* d_group_distances_ = nullptr;
    int* d_candidate_ids_ = nullptr;
    float* d_candidate_distances_ = nullptr;
    cudaEvent_t encode_done_ = nullptr;
    cudaEvent_t scan_done_ = nullptr;
    cudaEvent_t merge_done_ = nullptr;
};

}  // namespace paraselect::search::ivf_res_turboquant

#endif  // PARASELECT_IVF_RES_TURBOQUANT_SEARCH_CUH
