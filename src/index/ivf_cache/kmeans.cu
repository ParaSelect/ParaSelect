#include "index/ivf_cache.h"
#include "clustering/kmeans/kmeans_cache_recipe.h"
#include "index/ivf_cache/cache_policy.h"

#include "index/ivf_cache/cache_file.h"
#include "index/ivf_cache/kmeans.h"
#include "index/ivf_cache/source_vectors.h"

#include "clustering/kmeans/kmeans_cagra_train.cuh"
#include "clustering/kmeans/kmeans_cluster_only.cuh"
#include "clustering/kmeans/kmeans_final_assign.cuh"
#include "dataset/cache/cache_fs.h"
#include "dataset/dataset.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace paraselect::ivf_cache {
namespace detail {

void write_cluster_meta_atomic(
    const std::string& entry_dir,
    const std::string& name,
    const std::string& recipe,
    int nlist,
    int n_vectors,
    int dim) {
    const std::string path = entry_dir + "/meta.json";
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp);
    if (!out) throw std::runtime_error("failed to create cluster cache metadata");
    const nlohmann::json metadata{{"dataset_name",name},{"n_lists",nlist},
        {"build",nlohmann::json::parse(recipe)},{"n_vectors",n_vectors},
        {"n_dim",dim},{"n_clusters_actual",nlist}};
    out << metadata.dump(2) << std::endl;
    out.close();
    if (!out || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to commit cluster cache metadata");
    }
}


bool valid_cluster_cache(
    const dataset::ClusterCache::CacheResult& cache,
    size_t n,
    int dim,
    int nlist) {
    return cache.hit && cache.request.n_vectors == static_cast<int>(n) &&
           cache.request.dim == dim && cache.cluster_counts.size() == static_cast<size_t>(nlist) &&
           cache.reordered_indices.size() == n &&
           cache.centroids.size() == static_cast<size_t>(nlist) * dim &&
           cache.cluster_counts.size() == static_cast<size_t>(nlist) &&
           cache.cluster_offsets.size() == static_cast<size_t>(nlist) + 1u;
}

dataset::ClusterCache::CacheResult load_required_cluster_cache(
    dataset::ClusterCache& cache,
    const std::string& name,
    int nlist,
    const std::string& recipe,
    size_t n,
    int dim) {
    auto result = cache.try_load(name, dataset::ClusterCacheSpec{
        static_cast<int>(n), nlist, dim, recipe});
    if (!valid_cluster_cache(result, n, dim, nlist)) {
        throw std::runtime_error("KMeans cache generation did not produce a compatible cache");
    }
    return result;
}

void build_prefix_kmeans_cache_chunked(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& spec,
    dataset::ClusterCache& cache,
    const std::string& name,
    const std::string& recipe) {
    const bool exact_path = spec.train_strategy == KMeansTrainStrategy::FastIVF &&
        spec.final_assign_strategy == KMeansFinalAssignStrategy::Exact;
    const bool cagra_path = spec.train_strategy == KMeansTrainStrategy::CagraLloyd &&
        spec.final_assign_strategy == KMeansFinalAssignStrategy::Cagra;
    if (!exact_path && !cagra_path) {
        throw std::invalid_argument("unsupported chunked prefix KMeans strategy");
    }
    if (cagra_path && spec.cagra_train_samples <= 0) {
        throw std::invalid_argument("CAGRA Lloyd requires a positive training prefix size");
    }

    const int requested_train_n = cagra_path
        ? spec.cagra_train_samples
        : spec.fast_ivf_train_samples;
    const size_t configured_train_n = requested_train_n > 0
        ? static_cast<size_t>(requested_train_n)
        : 50000000ULL;
    const size_t train_n = std::min(n_vectors, configured_train_n);
    if (train_n < static_cast<size_t>(nlist)) {
        throw std::invalid_argument("KMeans training prefix is smaller than nlist");
    }

    std::cerr << "[ivf_cache:kmeans] train prefix n=" << train_n
              << " full_assign_n=" << n_vectors << std::endl;
    KMeansClusterOnlyResult trained;
    if (cagra_path) {
        KMeansCagraTrainConfig train_cfg;
        train_cfg.n = static_cast<int>(train_n);
        train_cfg.dim = source.spec().dim;
        train_cfg.k = nlist;
        train_cfg.iters = spec.kmeans_iters;
        train_cfg.device_id = 0;
        train_cfg.params = resolve_kmeans_cagra_profile(spec.cagra_profile);
        bool trained_ok = false;
        if (source.spec().dtype == dataset::VectorDType::U8) {
            std::unique_ptr<uint8_t, decltype(&std::free)> train(
                load_u8_prefix(source, train_n), &std::free);
            trained_ok = kmeans_train_centroids_cagra_u8(
                train_cfg, train.get(), &trained.centroids);
        } else {
            std::unique_ptr<float, decltype(&std::free)> train(
                load_fp32_prefix(source, train_n), &std::free);
            trained_ok = kmeans_train_centroids_cagra_fp32(
                train_cfg, train.get(), &trained.centroids);
        }
        if (!trained_ok) {
            throw std::runtime_error("CAGRA Lloyd centroid training failed");
        }
    } else {
        std::unique_ptr<float, decltype(&std::free)> train(
            load_fp32_prefix(source, train_n), &std::free);
        KMeansClusterOnlyConfig train_cfg;
        train_cfg.n = static_cast<int>(train_n);
        train_cfg.dim = source.spec().dim;
        train_cfg.k = nlist;
        train_cfg.iters = spec.kmeans_iters;
        train_cfg.train_samples = static_cast<int>(train_n);
        train_cfg.device_id = 0;
        train_cfg.dist = source.spec().metric;
        train_cfg.precision = spec.precision;
        train_cfg.final_assign_strategy = KMeansFinalAssignStrategy::Exact;
        if (!kmeans_fast_ivf_cluster_only(train_cfg, train.get(), &trained)) {
            throw std::runtime_error("FastIVF centroid training failed");
        }
    }

    const std::string entry_dir = cache.get_cache_dir() + "/.building-" +
        std::to_string(static_cast<long long>(::getpid()));
    if (!dataset::cache_fs::ensure_dir_exists(entry_dir)) {
        throw std::runtime_error("failed to create cluster cache directory: " + entry_dir);
    }
    TemporaryMapping<uint32_t> assignment_workspace(
        entry_dir, "ivf-assignment", n_vectors);

    float* d_centroids = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_centroids, trained.centroids.size() * sizeof(float)));
    try {
        CUDA_CHECK(cudaMemcpy(
            d_centroids, trained.centroids.data(),
            trained.centroids.size() * sizeof(float), cudaMemcpyHostToDevice));
    } catch (...) {
        cudaFree(d_centroids);
        throw;
    }

    constexpr size_t kAssignChunk = 1 << 20;
    GpuCentroidAssignmentConfig assign_cfg;
    assign_cfg.strategy = spec.final_assign_strategy;
    assign_cfg.combo = cagra_path
        ? GpuCentroidAssignmentCombo::CagraTopkSelect
        : GpuCentroidAssignmentCombo::ExactGemm;
    assign_cfg.device_id = 0;
    assign_cfg.max_gpus = 1;
    if (cagra_path) {
        const KMeansCagraParams params = resolve_kmeans_cagra_profile(spec.cagra_profile);
        assign_cfg.batch_size = params.final_batch;
        assign_cfg.cagra_graph_degree = params.graph_degree;
        assign_cfg.cagra_intermediate_graph_degree = params.intermediate_graph_degree;
        assign_cfg.cagra_topk_centroids = params.topk_centroids;
        assign_cfg.cagra_itopk_size = params.itopk_size;
        assign_cfg.cagra_search_width = params.search_width;
        assign_cfg.cagra_max_iterations = params.max_iterations;
        assign_cfg.cagra_team_size = params.team_size;
        assign_cfg.cagra_thread_block_size = params.thread_block_size;
        assign_cfg.cagra_index_scale = params.index_scale;
        assign_cfg.cagra_build_algo = params.build_algo;
    }
    GpuCentroidAssignmentWorkspace exact_workspace;
    GpuCentroidAssignmentCagraWorkspace cagra_workspace;
    idx_t* d_cagra_assign = nullptr;
    auto release_assignment = [&]() {
        gpu_centroid_assignment_workspace_release(exact_workspace);
        gpu_centroid_assignment_cagra_workspace_release(cagra_workspace);
        if (d_cagra_assign) cudaFree(d_cagra_assign);
        cudaFree(d_centroids);
    };
    if (cagra_path) {
        if (!gpu_centroid_assignment_cagra_workspace_build(
                cagra_workspace, assign_cfg, d_centroids, nlist,
                source.spec().dim, assign_cfg.batch_size)) {
            release_assignment();
            throw std::runtime_error("failed to build final CAGRA assignment graph");
        }
        const cudaError_t err = cudaMalloc(
            &d_cagra_assign, kAssignChunk * sizeof(idx_t));
        if (err != cudaSuccess) {
            release_assignment();
            throw std::runtime_error("failed to allocate final CAGRA assignments");
        }
    }
    uint32_t* assignments = assignment_workspace.data();
    std::vector<int> counts(static_cast<size_t>(nlist), 0);
    std::vector<float> converted;
    std::vector<uint8_t> native_u8;
    std::vector<float> native_fp32;
    std::vector<idx_t> chunk_assign;
    try {
        for (size_t begin = 0; begin < n_vectors; begin += kAssignChunk) {
            const size_t count = std::min(kAssignChunk, n_vectors - begin);
            dataset::VectorView view{};
            view.dtype = source.spec().dtype;
            view.n = count;
            view.dim = source.spec().dim;
            if (view.dtype == dataset::VectorDType::U8) {
                native_u8.resize(count * static_cast<size_t>(view.dim));
                if (source.read_base_range(begin, count, native_u8.data()) != count) {
                    throw std::runtime_error("short read during chunked FastIVF assignment");
                }
                view.data = native_u8.data();
            } else {
                native_fp32.resize(count * static_cast<size_t>(view.dim));
                if (source.read_base_range(begin, count, native_fp32.data()) != count) {
                    throw std::runtime_error("short read during chunked FastIVF assignment");
                }
                view.data = native_fp32.data();
            }
            KMeansCase shape{};
            shape.n = static_cast<int>(count);
            shape.dim = source.spec().dim;
            shape.k = nlist;
            shape.dist = source.spec().metric;
            idx_t* d_chunk_assign = nullptr;
            bool assigned = false;
            if (cagra_path) {
                d_chunk_assign = d_cagra_assign;
                assigned = view.dtype == dataset::VectorDType::U8
                    ? assign_u8_vectors_to_centroids_gpu_cagra_with_workspace(
                          assign_cfg, shape, view.u8(), d_centroids,
                          cagra_workspace, d_chunk_assign)
                    : assign_vectors_to_centroids_gpu_cagra_with_workspace(
                          assign_cfg, shape, view.fp32(), d_centroids,
                          cagra_workspace, d_chunk_assign);
            } else {
                const float* vectors = as_fp32(view, converted);
                assigned = assign_vectors_to_centroids_gpu_with_workspace(
                    assign_cfg, shape, vectors, d_centroids, exact_workspace);
                d_chunk_assign = exact_workspace.d_assign;
            }
            if (!assigned) {
                throw std::runtime_error("chunked KMeans final assignment failed");
            }
            chunk_assign.resize(count);
            CUDA_CHECK(cudaMemcpy(
                chunk_assign.data(), d_chunk_assign,
                count * sizeof(idx_t), cudaMemcpyDeviceToHost));
            for (size_t row = 0; row < count; ++row) {
                const idx_t cluster = chunk_assign[row];
                if (cluster < 0 || cluster >= nlist) {
                    throw std::runtime_error("KMeans produced an invalid assignment");
                }
                assignments[begin + row] = static_cast<uint32_t>(cluster);
                ++counts[static_cast<size_t>(cluster)];
            }
            std::cerr << "[ivf_cache:kmeans] full assignment "
                      << (begin + count) << "/" << n_vectors << std::endl;
        }
    } catch (...) {
        release_assignment();
        throw;
    }
    release_assignment();

    std::vector<idx_t> offsets(static_cast<size_t>(nlist) + 1u, 0);
    for (int cluster = 0; cluster < nlist; ++cluster) {
        offsets[static_cast<size_t>(cluster) + 1u] =
            offsets[static_cast<size_t>(cluster)] + counts[static_cast<size_t>(cluster)];
    }
    std::vector<idx_t> cursors(offsets.begin(), offsets.end() - 1);
    WritableBinaryMapping<idx_t> reordered_file(
        entry_dir + "/reordered_indices.ibin",
        static_cast<uint32_t>(n_vectors), 1);
    idx_t* reordered_indices = reordered_file.data();
    for (size_t source_id = 0; source_id < n_vectors; ++source_id) {
        const uint32_t cluster = assignments[source_id];
        reordered_indices[static_cast<size_t>(cursors[cluster]++)] =
            static_cast<idx_t>(source_id);
    }
    reordered_file.commit();

    write_binary_atomic(
        entry_dir + "/centroids.fbin",
        static_cast<uint32_t>(nlist), static_cast<uint32_t>(source.spec().dim),
        trained.centroids.data());
    write_binary_atomic(
        entry_dir + "/cluster_counts.ibin",
        static_cast<uint32_t>(nlist), 1, counts.data());
    write_binary_atomic(
        entry_dir + "/cluster_offsets.ibin",
        static_cast<uint32_t>(nlist + 1), 1, offsets.data());
    write_cluster_meta_atomic(
        entry_dir, name, recipe, nlist, static_cast<int>(n_vectors), source.spec().dim);
    cache.publish_staged(entry_dir, name, dataset::ClusterCacheSpec{
        static_cast<int>(n_vectors), nlist, source.spec().dim, recipe});
}

}  // namespace detail

using detail::as_fp32;
using detail::build_prefix_kmeans_cache_chunked;
using detail::kIoChunk;
using detail::load_fp32_prefix;
using detail::load_required_cluster_cache;
using detail::valid_cluster_cache;

dataset::ClusterCache::CacheResult ensure_kmeans(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& spec) {
    if (nlist <= 0 || n_vectors < static_cast<size_t>(nlist) ||
        spec.assign_operator != KMeansAssignOperator::Kmeans ||
        spec.precision != KMeansComputePrecision::Tf32 || spec.cluster_only) {
        throw std::invalid_argument("unsupported static KMeans cache request");
    }
    dataset::ClusterCache cache;
    if (cache.get_cache_dir().empty()) {
        throw std::runtime_error("IVF_CLUSTER_CACHE_DIR is not configured");
    }
    const std::string name = dataset::dataset_key_name(source.spec());
    const std::string recipe = make_kmeans_cache_recipe(spec);
    auto existing = cache.try_load(name, dataset::ClusterCacheSpec{
        static_cast<int>(n_vectors), nlist, source.spec().dim, recipe});
    if (valid_cluster_cache(existing, n_vectors, source.spec().dim, nlist)) return existing;
    require_build_policy("cluster:" + name);

    const bool chunked_prefix =
        spec.train_strategy == KMeansTrainStrategy::CagraLloyd ||
        (spec.train_strategy == KMeansTrainStrategy::FastIVF &&
         n_vectors > static_cast<size_t>(std::max(0, spec.fast_ivf_train_samples)));
    if (chunked_prefix) {
        build_prefix_kmeans_cache_chunked(
            source, n_vectors, nlist, spec, cache, name, recipe);
        return load_required_cluster_cache(
            cache, name, nlist, recipe, n_vectors, source.spec().dim);
    }

    float* vectors = load_fp32_prefix(source, n_vectors);
    ClusterDataset generated;
    float objective = 0.0f;
    generated.init_with_kmeans(
        static_cast<int>(n_vectors), source.spec().dim, nlist, &objective, vectors,
        spec.kmeans_iters, spec.train_strategy == KMeansTrainStrategy::MiniBatch,
        source.spec().metric, 1234, kIoChunk, 0,
        ownership::BufferOwner::StdMalloc, spec.train_sample_ratio,
        spec.train_max_samples, spec.train_strategy, spec.progressive_stage1_samples,
        spec.progressive_stage2_samples, spec.fast_ivf_train_samples,
        spec.final_assign_strategy, false, name, recipe, true, spec.reorder_policy,
        source.base_path());
    generated.release();
    return load_required_cluster_cache(
        cache, name, nlist, recipe, n_vectors, source.spec().dim);
}

dataset::ClusterCache::MappedResult ensure_kmeans_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& spec) {
    const bool cached_full = spec.train_strategy == KMeansTrainStrategy::FullLloyd;
    if (nlist <= 0 || n_vectors < static_cast<size_t>(nlist) ||
        n_vectors > static_cast<size_t>(INT_MAX) ||
        (!cached_full && spec.train_strategy != KMeansTrainStrategy::FastIVF &&
         spec.train_strategy != KMeansTrainStrategy::CagraLloyd) ||
        spec.assign_operator != KMeansAssignOperator::Kmeans ||
        spec.precision != KMeansComputePrecision::Tf32 || spec.cluster_only ||
        (spec.train_strategy == KMeansTrainStrategy::FastIVF &&
         spec.final_assign_strategy != KMeansFinalAssignStrategy::Exact) ||
        (spec.train_strategy == KMeansTrainStrategy::CagraLloyd &&
         spec.final_assign_strategy != KMeansFinalAssignStrategy::Cagra)) {
        throw std::invalid_argument("unsupported mapped KMeans cache request");
    }
    dataset::ClusterCache cache;
    if (cache.get_cache_dir().empty()) {
        throw std::runtime_error("IVF_CLUSTER_CACHE_DIR is not configured");
    }
    const std::string name = dataset::dataset_key_name(source.spec());
    const std::string recipe = make_kmeans_cache_recipe(spec);
    auto mapped = cache.try_map(name, dataset::ClusterCacheSpec{
        static_cast<int>(n_vectors), nlist, source.spec().dim, recipe});
    if (mapped.hit && mapped.request.n_vectors == static_cast<int>(n_vectors) &&
        mapped.request.dim == source.spec().dim) {
        return mapped;
    }
    // Only reached under an explicit build policy; the formal "require" policy
    // stops above, so a query job never starts a full-Lloyd training implicitly.
    require_build_policy("cluster:" + name);
    if (cached_full) {
        // Full Lloyd is not a chunked prefix recipe: publish it through the
        // in-memory builder (same recipe and manifest), then map the result.
        ensure_kmeans(source, n_vectors, nlist, spec);
    } else {
        build_prefix_kmeans_cache_chunked(
            source, n_vectors, nlist, spec, cache, name, recipe);
    }
    mapped = cache.try_map(name, dataset::ClusterCacheSpec{
        static_cast<int>(n_vectors), nlist, source.spec().dim, recipe});
    if (!mapped.hit) throw std::runtime_error("manifest cache generation failed for cluster asset");
    if (!mapped.hit || mapped.request.n_vectors != static_cast<int>(n_vectors) ||
        mapped.request.dim != source.spec().dim) {
        throw std::runtime_error("mapped KMeans cache generation failed");
    }
    return mapped;
}

}  // namespace paraselect::ivf_cache
