#include "search/ivf_res_turboquant/ivf_res_turboquant_search.cuh"

#include "quantization/turboquant/turboquant_1bit_encode.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace paraselect::search::ivf_res_turboquant {
namespace {

constexpr int kVectorChunkSize = 2048;
constexpr int kGroupChunks = 8;

template <typename T>
void replace_allocation(T** pointer, std::size_t count) {
    if (*pointer) CUDA_CHECK(cudaFree(*pointer));
    CUDA_CHECK(cudaMalloc(pointer, count * sizeof(T)));
}

int padded_dimension(int dim) {
    int padded = 32;
    while (padded < dim) padded <<= 1;
    if (padded > 256) throw std::invalid_argument("IVFResTurboQuant supports dimensions <= 256");
    return padded;
}

}  // namespace

Index::~Index() { release(); }

void Index::release() {
    if (d_centroids_) cudaFree(d_centroids_);
    if (d_cluster_offsets_) cudaFree(d_cluster_offsets_);
    if (d_codes_) cudaFree(d_codes_);
    if (d_reconstruction_scales_) cudaFree(d_reconstruction_scales_);
    d_centroids_ = nullptr;
    d_cluster_offsets_ = nullptr;
    d_codes_ = nullptr;
    d_reconstruction_scales_ = nullptr;
    n_vectors_ = n_clusters_ = dim_ = padded_dim_ = words_per_code_ = 0;
}

void Index::build_u8(
    const std::uint8_t* host_vectors,
    const float* host_centroids,
    const long long* host_cluster_offsets,
    int n_vectors,
    int n_clusters,
    int dim,
    std::uint32_t rotation_seed) {
    if (!host_vectors || !host_centroids || !host_cluster_offsets ||
        n_vectors <= 0 || n_clusters <= 0 || dim <= 0) {
        throw std::invalid_argument("IVFResTurboQuant index build received invalid input");
    }
    release();
    n_vectors_ = n_vectors;
    n_clusters_ = n_clusters;
    dim_ = dim;
    padded_dim_ = padded_dimension(dim);
    words_per_code_ = padded_dim_ / 32;
    rotation_seed_ = rotation_seed;

    std::uint8_t* d_vectors = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&d_vectors, static_cast<std::size_t>(n_vectors_) * dim_));
        replace_allocation(&d_centroids_, static_cast<std::size_t>(n_clusters_) * dim_);
        replace_allocation(&d_cluster_offsets_, static_cast<std::size_t>(n_clusters_) + 1);
        replace_allocation(&d_codes_, static_cast<std::size_t>(n_vectors_) * words_per_code_);
        replace_allocation(&d_reconstruction_scales_, n_vectors_);
        CUDA_CHECK(cudaMemcpy(
            d_vectors, host_vectors, static_cast<std::size_t>(n_vectors_) * dim_,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_centroids_, host_centroids,
            static_cast<std::size_t>(n_clusters_) * dim_ * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cluster_offsets_, host_cluster_offsets,
            (static_cast<std::size_t>(n_clusters_) + 1) * sizeof(long long),
            cudaMemcpyHostToDevice));
        turboquant::ivf_res_turboquant_encode_database_u8_gpu(
            {d_vectors, d_centroids_, d_cluster_offsets_, n_vectors_, n_clusters_,
             dim_, padded_dim_, rotation_seed_},
            {nullptr, d_codes_, d_reconstruction_scales_, nullptr}, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_vectors));
    } catch (...) {
        if (d_vectors) cudaFree(d_vectors);
        release();
        throw;
    }
}

void Index::build_cached_u32(
    const std::uint32_t* host_codes,
    const float* host_scales,
    const float* host_centroids,
    const long long* host_cluster_offsets,
    int n_vectors,
    int n_clusters,
    int dim) {
    if (!host_codes || !host_scales || !host_centroids || !host_cluster_offsets ||
        n_vectors <= 0 || n_clusters <= 0 || dim <= 0 || dim % 32 != 0) {
        throw std::invalid_argument("IVFResTurboQuant cached build received invalid input");
    }
    release();
    n_vectors_ = n_vectors;
    n_clusters_ = n_clusters;
    dim_ = dim;
    padded_dim_ = padded_dimension(dim);
    words_per_code_ = padded_dim_ / 32;
    rotation_seed_ = 17u;
    try {
        replace_allocation(&d_centroids_, static_cast<std::size_t>(n_clusters_) * dim_);
        replace_allocation(&d_cluster_offsets_, static_cast<std::size_t>(n_clusters_) + 1);
        replace_allocation(&d_codes_, static_cast<std::size_t>(n_vectors_) * words_per_code_);
        replace_allocation(&d_reconstruction_scales_, static_cast<std::size_t>(n_vectors_));
        CUDA_CHECK(cudaMemcpy(
            d_centroids_, host_centroids,
            static_cast<std::size_t>(n_clusters_) * dim_ * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cluster_offsets_, host_cluster_offsets,
            (static_cast<std::size_t>(n_clusters_) + 1) * sizeof(long long),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_codes_, host_codes,
            static_cast<std::size_t>(n_vectors_) * words_per_code_ * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_reconstruction_scales_, host_scales,
            static_cast<std::size_t>(n_vectors_) * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        release();
        throw;
    }
}

SearchWorkspace::~SearchWorkspace() { release(); }

void SearchWorkspace::release() {
    if (d_queries_) cudaFree(d_queries_);
    if (d_probe_ids_) cudaFree(d_probe_ids_);
    if (d_rotated_query_states_) cudaFree(d_rotated_query_states_);
    if (d_query_norm_sq_) cudaFree(d_query_norm_sq_);
    if (d_packed_query_states_) cudaFree(d_packed_query_states_);
    if (d_query_norms_) cudaFree(d_query_norms_);
    if (d_chunks_) cudaFree(d_chunks_);
    if (d_groups_) cudaFree(d_groups_);
    if (d_query_group_starts_) cudaFree(d_query_group_starts_);
    if (d_query_group_counts_) cudaFree(d_query_group_counts_);
    if (d_group_indices_) cudaFree(d_group_indices_);
    if (d_group_distances_) cudaFree(d_group_distances_);
    if (d_candidate_ids_) cudaFree(d_candidate_ids_);
    if (d_candidate_distances_) cudaFree(d_candidate_distances_);
    if (encode_done_) cudaEventDestroy(encode_done_);
    if (scan_done_) cudaEventDestroy(scan_done_);
    if (merge_done_) cudaEventDestroy(merge_done_);
    query_capacity_ = probe_capacity_ = chunk_capacity_ = group_capacity_ = 0;
    group_items_capacity_ = 0;
    d_queries_ = nullptr;
    d_probe_ids_ = nullptr;
    d_rotated_query_states_ = nullptr;
    d_query_norm_sq_ = nullptr;
    d_packed_query_states_ = nullptr;
    d_query_norms_ = nullptr;
    d_chunks_ = nullptr;
    d_groups_ = nullptr;
    d_query_group_starts_ = nullptr;
    d_query_group_counts_ = nullptr;
    d_group_indices_ = nullptr;
    d_group_distances_ = nullptr;
    d_candidate_ids_ = nullptr;
    d_candidate_distances_ = nullptr;
    encode_done_ = scan_done_ = merge_done_ = nullptr;
}

void SearchWorkspace::reserve(
    const Index& index, int n_queries, int n_probes, int n_chunks, int n_groups,
    int candidate_count) {
    const int states = n_queries * n_probes;
    const bool query_grew = n_queries > query_capacity_;
    if (query_grew) {
        replace_allocation(&d_queries_, static_cast<std::size_t>(n_queries) * index.dim_);
        replace_allocation(&d_query_group_starts_, n_queries);
        replace_allocation(&d_query_group_counts_, n_queries);
        query_capacity_ = n_queries;
    }
    if (query_grew) {
        replace_allocation(
            &d_candidate_ids_, static_cast<std::size_t>(query_capacity_) * kMaxCandidateCount);
        replace_allocation(
            &d_candidate_distances_, static_cast<std::size_t>(query_capacity_) * kMaxCandidateCount);
    }
    if (states > probe_capacity_) {
        replace_allocation(&d_probe_ids_, states);
        replace_allocation(
            &d_rotated_query_states_, static_cast<std::size_t>(states) * index.padded_dim_);
        replace_allocation(&d_query_norm_sq_, states);
        replace_allocation(&d_query_norms_, states);
        replace_allocation(&d_packed_query_states_,
            static_cast<std::size_t>(states) * index.words_per_code_);
        probe_capacity_ = states;
    }
    if (n_chunks > chunk_capacity_) {
        replace_allocation(&d_chunks_, n_chunks);
        chunk_capacity_ = n_chunks;
    }
    if (n_groups > group_capacity_) {
        replace_allocation(&d_groups_, n_groups);
        group_capacity_ = n_groups;
    }
    const std::size_t group_items = static_cast<std::size_t>(n_groups) * candidate_count;
    if (group_items > group_items_capacity_) {
        replace_allocation(&d_group_indices_, group_items);
        replace_allocation(&d_group_distances_, group_items);
        group_items_capacity_ = group_items;
    }
    if (!encode_done_) {
        CUDA_CHECK(cudaEventCreate(&encode_done_));
        CUDA_CHECK(cudaEventCreate(&scan_done_));
        CUDA_CHECK(cudaEventCreate(&merge_done_));
    }
}

void SearchWorkspace::search_candidates(
    const Index& index,
    const float* host_queries,
    const long long* host_probe_ids,
    const long long* host_cluster_offsets,
    int n_queries,
    int n_probes,
    int* host_candidate_ids,
    int candidate_count,
    SearchTiming* timing,
    turboquant::TurboQuant1BitScoreMode score_mode,
    bool exact_selection) {
    if (!host_queries || !host_probe_ids || !host_cluster_offsets || !host_candidate_ids ||
        n_queries <= 0 || n_probes <= 0 || candidate_count <= 0 ||
        candidate_count > kMaxCandidateCount || index.n_vectors_ <= 0) {
        throw std::invalid_argument("IVFResTurboQuant search received invalid input");
    }
    const int vector_chunk_size = exact_selection ? 256 : kVectorChunkSize;
    const auto total_begin = std::chrono::steady_clock::now();
    std::vector<turboquant::TurboQuantChunkDesc> chunks;
    std::vector<turboquant::TurboQuantGroupDesc> groups;
    std::vector<int> query_group_starts(n_queries);
    std::vector<int> query_group_counts(n_queries);
    for (int query = 0; query < n_queries; ++query) {
        const int query_chunk_start = static_cast<int>(chunks.size());
        for (int probe = 0; probe < n_probes; ++probe) {
            const long long cluster_ll = host_probe_ids[static_cast<std::size_t>(query) * n_probes + probe];
            if (cluster_ll < 0 || cluster_ll >= index.n_clusters_) continue;
            const int cluster = static_cast<int>(cluster_ll);
            const long long count_ll =
                host_cluster_offsets[cluster + 1] - host_cluster_offsets[cluster];
            if (count_ll < 0 || count_ll > index.n_vectors_) {
                throw std::runtime_error("IVFResTurboQuant list offsets are invalid");
            }
            const int count = static_cast<int>(count_ll);
            for (int start = 0; start < count; start += vector_chunk_size) {
                chunks.push_back({
                    query, cluster, query * n_probes + probe, start,
                    std::min(vector_chunk_size, count - start)});
            }
        }
        const int query_chunks = static_cast<int>(chunks.size()) - query_chunk_start;
        query_group_starts[query] = static_cast<int>(groups.size());
        const int group_chunks = std::max(kGroupChunks,
            (candidate_count + 255) / 256);
        for (int first = 0; first < query_chunks; first += group_chunks) {
            groups.push_back({
                query_chunk_start + first, std::min(group_chunks, query_chunks - first)});
        }
        query_group_counts[query] =
            static_cast<int>(groups.size()) - query_group_starts[query];
    }
    if (chunks.empty() || groups.empty()) {
        throw std::runtime_error("IVFResTurboQuant search generated no scan work");
    }
    reserve(index, n_queries, n_probes, static_cast<int>(chunks.size()),
            static_cast<int>(groups.size()), candidate_count);

    CUDA_CHECK(cudaMemcpy(
        d_queries_, host_queries,
        static_cast<std::size_t>(n_queries) * index.dim_ * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_probe_ids_, host_probe_ids,
        static_cast<std::size_t>(n_queries) * n_probes * sizeof(long long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_chunks_, chunks.data(), chunks.size() * sizeof(chunks.front()), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_groups_, groups.data(), groups.size() * sizeof(groups.front()), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_query_group_starts_, query_group_starts.data(),
        query_group_starts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_query_group_counts_, query_group_counts.data(),
        query_group_counts.size() * sizeof(int), cudaMemcpyHostToDevice));

    turboquant::ivf_res_turboquant_encode_queries_gpu(
        {d_queries_, index.d_centroids_, d_probe_ids_, n_queries, n_probes,
         index.dim_, index.padded_dim_, index.rotation_seed_},
        {score_mode == turboquant::TurboQuant1BitScoreMode::MseFloatQuery
             ? d_rotated_query_states_ : nullptr,
         score_mode == turboquant::TurboQuant1BitScoreMode::BinaryQuery
             ? d_packed_query_states_ : nullptr,
         score_mode == turboquant::TurboQuant1BitScoreMode::BinaryQuery
             ? d_query_norms_ : nullptr,
         score_mode == turboquant::TurboQuant1BitScoreMode::MseFloatQuery
             ? d_query_norm_sq_ : nullptr}, nullptr);
    CUDA_CHECK(cudaEventRecord(encode_done_));
    turboquant::TurboQuant1BitScanInput scan{
        d_rotated_query_states_, d_packed_query_states_, d_query_norms_, d_query_norm_sq_,
        index.d_codes_, index.d_reconstruction_scales_, index.d_cluster_offsets_, d_chunks_,
        static_cast<int>(chunks.size()), index.padded_dim_, index.words_per_code_,
        vector_chunk_size};
    const int group_chunks = std::max(kGroupChunks, (candidate_count + 255) / 256);
    turboquant::turboquant_1bit_grouped_scan_topk_gpu(
        scan, d_groups_, static_cast<int>(groups.size()), group_chunks,
        score_mode, 0.0f, candidate_count,
        d_group_indices_, d_group_distances_, nullptr);
    CUDA_CHECK(cudaEventRecord(scan_done_));
    turboquant::turboquant_merge_query_groups_topk_gpu(
        d_group_indices_, d_group_distances_, d_query_group_starts_, d_query_group_counts_,
        n_queries, candidate_count, d_candidate_ids_, d_candidate_distances_, nullptr);
    CUDA_CHECK(cudaEventRecord(merge_done_));
    CUDA_CHECK(cudaMemcpy(
        host_candidate_ids, d_candidate_ids_,
        static_cast<std::size_t>(n_queries) * candidate_count * sizeof(int),
        cudaMemcpyDeviceToHost));
    if (timing) {
        timing->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_begin).count();
        CUDA_CHECK(cudaEventElapsedTime(&timing->grouped_scan_ms, encode_done_, scan_done_));
        CUDA_CHECK(cudaEventElapsedTime(&timing->query_merge_ms, scan_done_, merge_done_));
        // The first event follows synchronous H2D metadata copies; encode time is
        // reported by subtraction from the complete GPU section in the driver.
        timing->query_encode_ms = 0.0f;
    }
}

}  // namespace paraselect::search::ivf_res_turboquant
