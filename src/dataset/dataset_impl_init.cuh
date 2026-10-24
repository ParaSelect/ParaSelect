#ifndef PARASELECT_DATASET_IMPL_INIT_CUH
#define PARASELECT_DATASET_IMPL_INIT_CUH

#include "dataset_rowmajor.cuh"
#include "assignment_materializer.cuh"
#include "dataset/cache/cluster_cache.h"
#include "clustering/kmeans/kmeans_cache_recipe.h"
#include "common/dataset_spec.h"
#include "cpu/utils/cpu_array_utils.h"
#include "reorder/reorder.cuh"
#include "cpu/utils/numa_page_stats.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <malloc.h>
#include <stdexcept>

namespace paraselect {
namespace dataset {
namespace detail {

inline bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

} // namespace detail
} // namespace dataset
} // namespace paraselect

inline void ClusterDataset::init_with_kmeans(
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    float* h_objective,
    float* h_data,
    int kmeans_iters,
    bool use_minibatch,
    DistanceType distance_mode,
    unsigned int seed,
    int batch_size,
    int device_id,
    paraselect::ownership::BufferOwner input_owner,
    float train_sample_ratio,
    int train_max_samples,
    KMeansTrainStrategy train_strategy,
    int progressive_stage1_samples,
    int progressive_stage2_samples,
    int fast_ivf_train_samples,
    KMeansFinalAssignStrategy final_assign_strategy,
    bool cluster_only,
    const std::string& cache_dataset_name,
    const std::string& cache_build_recipe,
    bool use_cache,
    KMeansReorderPolicy reorder_policy,
    const std::string& raw_base_path)
{
    if (is_valid()) release();
    const bool disable_cache_env =
        paraselect::dataset::detail::env_flag_enabled("PARASELECT_DISABLE_CLUSTER_CACHE");
    const bool effective_use_cache = use_cache && !disable_cache_env;
    if (use_cache && disable_cache_env) {
        std::cerr << "[ClusterCache] disabled by PARASELECT_DISABLE_CLUSTER_CACHE" << std::endl;
    }

    std::string resolved_build_recipe;
    if (effective_use_cache && !cache_dataset_name.empty()) {
        resolved_build_recipe = cache_build_recipe.empty()
            ? make_kmeans_cache_recipe(make_kmeans_spec(
                use_minibatch,
                kmeans_iters,
                train_sample_ratio,
                train_max_samples,
                KMeansAssignOperator::Kmeans,
                KMeansComputePrecision::Tf32,
                train_strategy,
                progressive_stage1_samples,
                progressive_stage2_samples,
                fast_ivf_train_samples,
                final_assign_strategy,
                cluster_only))
            : cache_build_recipe;
        paraselect::dataset::ClusterCache cache;
        auto result = cache.try_load(cache_dataset_name,
                                     paraselect::dataset::ClusterCacheSpec{
                                         n_total_vectors, n_clusters, vector_dim,
                                         resolved_build_recipe});
        if (result.hit &&
            (result.request.n_vectors != n_total_vectors ||
             result.request.dim != vector_dim ||
             result.request.n_lists != n_clusters ||
             result.reordered_indices.size() != (size_t)n_total_vectors ||
             result.cluster_counts.size() != (size_t)n_clusters ||
             result.cluster_offsets.size() != (size_t)n_clusters + 1u ||
             result.centroids.size() != (size_t)n_clusters * (size_t)vector_dim)) {
            std::cerr << "[ClusterCache] shape mismatch, ignore cache: " << result.entry_dir
                      << " request_n=" << result.request.n_vectors
                      << " request_n=" << n_total_vectors
                      << " request_dim=" << result.request.dim
                      << " request_dim=" << vector_dim
                      << " indices=" << result.reordered_indices.size()
                      << " counts=" << result.cluster_counts.size()
                      << " offsets=" << result.cluster_offsets.size()
                      << " centroids=" << result.centroids.size()
                      << std::endl;
            result.hit = false;
        }
        if (result.hit) {
            if (reorder_policy == KMeansReorderPolicy::NumaFirstTouch) {
                std::cerr << "[ClusterDataset] cache materialization reorder_policy=numa file" << std::endl;
                paraselect::ownership::free_raw(h_data, input_owner);
                h_data = nullptr;
                paraselect::dataset::ReorderedFp32MaterializeResult materialized = paraselect::dataset::materialize_reordered_from_file_numa(
                    raw_base_path,
                    result.reordered_indices.data(),
                    result.cluster_offsets.data(),
                    result.cluster_counts.data(),
                    n_total_vectors,
                    vector_dim,
                    n_clusters);
                init_from_cluster_metadata(
                    result.reordered_indices.data(),
                    result.centroids.data(),
                    result.cluster_offsets.data(),
                    result.cluster_counts.data(),
                    n_total_vectors,
                    vector_dim,
                    n_clusters);
                set_reordered_fp32_owned(materialized.data, materialized.owner, true, true);
                materialized.data = nullptr;
            } else {
                if (!h_data) {
                    throw std::runtime_error("init_with_kmeans: cache hit but h_data is null");
                }
                std::cerr << "[ClusterDataset] cache materialization reorder_policy=standard" << std::endl;
                cpu_reorder_vectors_inplace(h_data, result.reordered_indices.data(), n_total_vectors, vector_dim);
                init_from_cluster_metadata(
                    result.reordered_indices.data(),
                    result.centroids.data(),
                    result.cluster_offsets.data(),
                    result.cluster_counts.data(),
                    n_total_vectors,
                    vector_dim,
                    n_clusters);
                set_reordered_fp32_owned(h_data, input_owner, false, false);
            }
            if (h_objective) *h_objective = 0.0f;
            return;
        }
    }

    this->n_total_vectors = n_total_vectors;
    this->vector_dim = vector_dim;
    size_t data_size = (size_t)n_total_vectors * vector_dim;
    float* h_data_ptr = nullptr;
    bool need_free = false;
    if (!h_data) {
        need_free = true;
        h_data_ptr = (float*)memalign(64, data_size * sizeof(float));
        init_array_multithreaded(h_data_ptr, data_size, seed, -1.0f, 1.0f);
    } else {
        h_data_ptr = h_data;
    }

    ClusteringConfig cfg;
    cfg.n_clusters = n_clusters;
    cfg.max_iters = kmeans_iters;
    cfg.distance_mode = distance_mode;
    cfg.seed = seed;
    cfg.use_minibatch = use_minibatch;
    cfg.batch_size = batch_size;
    cfg.device_id = device_id;
    cfg.train_strategy = (train_strategy == KMeansTrainStrategy::FullLloyd)
        ? infer_kmeans_train_strategy(use_minibatch, train_sample_ratio, train_max_samples)
        : train_strategy;
    cfg.assign_operator = KMeansAssignOperator::Kmeans;
    cfg.compute_precision = KMeansComputePrecision::Tf32;
    cfg.h_objective = h_objective;
    cfg.train_sample_ratio = train_sample_ratio;
    cfg.train_max_samples = train_max_samples;
    cfg.progressive_stage1_samples = progressive_stage1_samples;
    cfg.progressive_stage2_samples = progressive_stage2_samples;
    cfg.fast_ivf_train_samples = fast_ivf_train_samples;
    cfg.final_assign_strategy = final_assign_strategy;
    cfg.reorder_policy = reorder_policy;
    cfg.cluster_only = cluster_only;

    KMeansClusteringInPlace kmeans_inplace;
    ClusteringResult res;
    float* transferred_data = nullptr;
    paraselect::ownership::BufferOwner transferred_owner =
        need_free ? paraselect::ownership::BufferOwner::StdMalloc : input_owner;
    bool numa_file_materialized = false;

    if (reorder_policy == KMeansReorderPolicy::NumaFirstTouch && !cluster_only) {
        std::cerr << "[ClusterDataset] build materialization reorder_policy=numa file" << std::endl;
        if (!h_data_ptr) {
            throw std::runtime_error("init_with_kmeans: NUMA build requires h_data for assignment");
        }
        kmeans_inplace.fit(h_data_ptr, n_total_vectors, vector_dim, cfg);
        res = kmeans_inplace.get_result();
        float* raw_to_free = kmeans_inplace.release_reordered_data();
        paraselect::ownership::free_raw(raw_to_free, transferred_owner);
        h_data_ptr = nullptr;
        h_data = nullptr;
        need_free = false;

        paraselect::dataset::ReorderedFp32MaterializeResult materialized = paraselect::dataset::materialize_reordered_from_file_numa(
            raw_base_path,
            res.reordered_indices,
            res.cluster_info ? res.cluster_info->offsets : nullptr,
            res.cluster_info ? res.cluster_info->counts : nullptr,
            n_total_vectors,
            vector_dim,
            res.cluster_info ? res.cluster_info->k : n_clusters);
        init_from_cluster_metadata(
            res.reordered_indices,
            res.centroids,
            res.cluster_info ? res.cluster_info->offsets : nullptr,
            res.cluster_info ? res.cluster_info->counts : nullptr,
            n_total_vectors,
            vector_dim,
            res.cluster_info ? res.cluster_info->k : n_clusters);
        set_reordered_fp32_owned(materialized.data, materialized.owner, true, true);
        materialized.data = nullptr;
        numa_file_materialized = true;
    } else {
        std::cerr << "[ClusterDataset] build materialization reorder_policy=standard" << std::endl;
        kmeans_inplace.fit(h_data_ptr, n_total_vectors, vector_dim, cfg);
        res = kmeans_inplace.get_result();
        transferred_data = kmeans_inplace.release_reordered_data();
    }

    if (cfg.actual_n_clusters && res.cluster_info) {
        *cfg.actual_n_clusters = res.cluster_info->k;
    }

    if (cluster_only) {
        n_total_vectors = res.n_vectors;
        vector_dim = res.dim;
        const int actual_k = res.cluster_info ? res.cluster_info->k : 0;
        auto meta = paraselect::dataset::copy_cluster_metadata_without_reordered_indices(
            res.centroids,
            res.cluster_info ? res.cluster_info->offsets : nullptr,
            res.cluster_info ? res.cluster_info->counts : nullptr,
            vector_dim,
            actual_k);
        cluster_info.k = actual_k;
        cluster_info.offsets = meta.cluster_offsets;
        cluster_info.counts = meta.cluster_counts;
        centroids = meta.centroids;
        reordered_indices = nullptr;
        reordered_data = nullptr;
        reordered_data_owner_ = paraselect::ownership::BufferOwner::None;
    } else if (!numa_file_materialized) {
        if (!transferred_data) {
            throw std::runtime_error("init_with_kmeans: clusterer did not return transferable reordered_data");
        }
        init_from_cluster_metadata(
            res.reordered_indices,
            res.centroids,
            res.cluster_info ? res.cluster_info->offsets : nullptr,
            res.cluster_info ? res.cluster_info->counts : nullptr,
            res.n_vectors,
            res.dim,
            res.cluster_info ? res.cluster_info->k : n_clusters);
        set_reordered_fp32_owned(transferred_data, transferred_owner, false, false);
    }

    if (effective_use_cache && !cache_dataset_name.empty() && !cluster_only) {
        paraselect::dataset::ClusterCache cache;
        if (resolved_build_recipe.empty()) {
            resolved_build_recipe = cache_build_recipe.empty()
                ? make_kmeans_cache_recipe(make_kmeans_spec(
                    use_minibatch,
                    kmeans_iters,
                    train_sample_ratio,
                    train_max_samples,
                    KMeansAssignOperator::Kmeans,
                    KMeansComputePrecision::Tf32,
                    train_strategy,
                    progressive_stage1_samples,
                    progressive_stage2_samples,
                    fast_ivf_train_samples,
                    final_assign_strategy,
                    cluster_only))
                : cache_build_recipe;
        }
        cache.save(
            cache_dataset_name,
            paraselect::dataset::ClusterCacheSpec{
                this->n_total_vectors, n_clusters, this->vector_dim, resolved_build_recipe},
            centroids,
            reordered_indices,
            cluster_info.counts,
            cluster_info.offsets,
            cluster_info.k);
    }

}

inline void ClusterDataset::init_from_cluster_metadata(
    const idx_t* h_reordered_indices,
    const float* h_centroids,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int n_total_vectors_,
    int vector_dim_,
    int n_clusters_)
{
    release();
    owns_cluster_metadata = true;
    n_total_vectors = n_total_vectors_;
    vector_dim = vector_dim_;

    auto meta = paraselect::dataset::copy_cluster_metadata(
        h_reordered_indices, h_centroids, h_cluster_offsets, h_cluster_counts,
        n_total_vectors_, vector_dim_, n_clusters_);
    cluster_info.k = n_clusters_;
    cluster_info.offsets = meta.cluster_offsets;
    cluster_info.counts = meta.cluster_counts;
    centroids = meta.centroids;
    reordered_indices = meta.reordered_indices;

    reordered_data = nullptr;
    reordered_data_u8 = nullptr;
    reordered_data_owner_ = paraselect::ownership::BufferOwner::None;
}

inline void ClusterDataset::attach_cluster_metadata(
    const idx_t* h_reordered_indices,
    const float* h_centroids,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int n_total_vectors_,
    int vector_dim_,
    int n_clusters_)
{
    if (!h_reordered_indices || !h_centroids || !h_cluster_offsets || !h_cluster_counts ||
        n_total_vectors_ <= 0 || vector_dim_ <= 0 || n_clusters_ <= 0) {
        throw std::invalid_argument("attach_cluster_metadata: invalid arguments");
    }
    release();
    reordered_indices = const_cast<idx_t*>(h_reordered_indices);
    centroids = const_cast<float*>(h_centroids);
    cluster_info.offsets = const_cast<idx_t*>(h_cluster_offsets);
    cluster_info.counts = const_cast<int*>(h_cluster_counts);
    cluster_info.k = n_clusters_;
    n_total_vectors = n_total_vectors_;
    vector_dim = vector_dim_;
    owns_cluster_metadata = false;
}

inline void ClusterDataset::set_reordered_fp32_owned(
    float* h_reordered_data,
    paraselect::ownership::BufferOwner data_owner,
    bool numa_first_touch_row_major,
    bool print_numa_distribution)
{
    if (!h_reordered_data) {
        throw std::runtime_error("set_reordered_fp32_owned requires data");
    }
    if (n_total_vectors <= 0 || vector_dim <= 0 || cluster_info.k <= 0) {
        throw std::runtime_error("set_reordered_fp32_owned requires initialized cluster metadata");
    }
    if (reordered_data) {
        paraselect::ownership::free_raw(reordered_data, reordered_data_owner_);
        reordered_data = nullptr;
        reordered_data_owner_ = paraselect::ownership::BufferOwner::None;
    }
    if (reordered_data_u8) {
        paraselect::ownership::free_raw(reordered_data_u8, row_major_storage.u8_owner);
        reordered_data_u8 = nullptr;
        row_major_storage.u8_owner = paraselect::ownership::BufferOwner::None;
    }
    aosoa_storage.release();
    gpu_resident_interleaved.release();

    paraselect::ownership::BufferHandle owned_fp32(
        h_reordered_data,
        (size_t)n_total_vectors * (size_t)vector_dim * sizeof(float),
        data_owner);
    paraselect::ownership::take_to(owned_fp32, reordered_data, reordered_data_owner_);
    if (print_numa_distribution) {
        paraselect::numa::print_page_distribution(
            "reordered_data_fp32", reordered_data,
            (size_t)n_total_vectors * (size_t)vector_dim * sizeof(float));
    }

    _compute_vector_l2_norm(reordered_data, numa_first_touch_row_major);
}

inline void ClusterDataset::set_reordered_u8_owned(
    uint8_t* h_reordered_data_u8,
    paraselect::ownership::BufferOwner data_owner,
    bool print_numa_distribution)
{
    if (!h_reordered_data_u8) {
        throw std::runtime_error("set_reordered_u8_owned requires data");
    }
    if (n_total_vectors <= 0 || vector_dim <= 0 || cluster_info.k <= 0) {
        throw std::runtime_error("set_reordered_u8_owned requires initialized cluster metadata");
    }
    if (reordered_data) {
        paraselect::ownership::free_raw(reordered_data, reordered_data_owner_);
        reordered_data = nullptr;
        reordered_data_owner_ = paraselect::ownership::BufferOwner::None;
    }
    if (reordered_data_u8) {
        paraselect::ownership::free_raw(reordered_data_u8, row_major_storage.u8_owner);
        reordered_data_u8 = nullptr;
        row_major_storage.u8_owner = paraselect::ownership::BufferOwner::None;
    }
    aosoa_storage.release();
    gpu_resident_interleaved.release();
    if (vector_l2_norm) {
        std::free(vector_l2_norm);
        vector_l2_norm = nullptr;
    }

    paraselect::ownership::BufferHandle owned_u8(
        h_reordered_data_u8,
        (size_t)n_total_vectors * (size_t)vector_dim * sizeof(uint8_t),
        data_owner);
    paraselect::ownership::take_to(owned_u8, reordered_data_u8, row_major_storage.u8_owner);
    if (print_numa_distribution) {
        paraselect::numa::print_page_distribution(
            "reordered_data_u8", reordered_data_u8,
            (size_t)n_total_vectors * (size_t)vector_dim * sizeof(uint8_t));
    }
}

inline void ClusterDataset::set_aosoa_owned(
    float* h_aosoa_data,
    paraselect::ownership::BufferOwner data_owner,
    long long* h_aosoa_offsets,
    paraselect::ownership::BufferOwner offsets_owner,
    int tile,
    bool print_numa_distribution)
{
    if (!h_aosoa_data || !h_aosoa_offsets || tile <= 0) {
        throw std::runtime_error("set_aosoa_owned requires data/offsets/tile");
    }
    if (n_total_vectors <= 0 || vector_dim <= 0 || cluster_info.k <= 0) {
        throw std::runtime_error("set_aosoa_owned requires initialized cluster metadata");
    }
    paraselect::dataset::release_rowmajor_storage(&row_major_storage);
    gpu_resident_interleaved.release();
    if (vector_l2_norm) {
        std::free(vector_l2_norm);
        vector_l2_norm = nullptr;
    }
    aosoa_storage.release();
    const size_t data_bytes = (size_t)h_aosoa_offsets[cluster_info.k] * sizeof(float);
    const size_t offsets_bytes = (size_t)(cluster_info.k + 1) * sizeof(long long);
    paraselect::ownership::BufferHandle data_handle(h_aosoa_data, data_bytes, data_owner);
    paraselect::ownership::BufferHandle offsets_handle(h_aosoa_offsets, offsets_bytes, offsets_owner);
    paraselect::ownership::take_to(data_handle, aosoa_storage.data, aosoa_storage.data_owner);
    paraselect::ownership::take_to(offsets_handle, aosoa_storage.offsets, aosoa_storage.offsets_owner);
    aosoa_storage.tile = tile;
    if (print_numa_distribution) {
        paraselect::numa::print_page_distribution("aosoa_data", aosoa_storage.data, data_bytes);
    }
}

inline void ClusterDataset::set_gpu_resident_interleaved_owned(
    paraselect::dataset::GpuResidentInterleavedStorage& storage)
{
    if (!storage.is_valid()) {
        throw std::runtime_error("set_gpu_resident_interleaved_owned requires valid storage");
    }
    if (n_total_vectors <= 0 || vector_dim <= 0 || cluster_info.k <= 0) {
        storage.release();
        throw std::runtime_error("set_gpu_resident_interleaved_owned requires initialized cluster metadata");
    }
    paraselect::dataset::release_rowmajor_storage(&row_major_storage);
    aosoa_storage.release();
    if (vector_l2_norm) {
        std::free(vector_l2_norm);
        vector_l2_norm = nullptr;
    }
    gpu_resident_interleaved.release();
    gpu_resident_interleaved = storage;
    storage.d_vectors = nullptr;
    storage.d_vectors_u8 = nullptr;
    storage.d_norms = nullptr;
    storage.d_reordered_indices = nullptr;
}


#endif
