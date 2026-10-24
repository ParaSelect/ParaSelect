#include "index/ivf_cache.h"
#include "index/ivf_cache/cache_policy.h"

#include "index/ivf_cache/kmeans.h"
#include "index/ivf_cache/source_vectors.h"

#include "dataset/cache/plain_pq_cache.h"
#include "quantization/pq/pq_trainer.cuh"
#include "transform/opq_trainer.cuh"
#include "transform/opq_transform.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace paraselect::ivf_cache {
namespace {

using detail::as_fp32;
using detail::valid_cluster_cache;

template <bool Mapped = false, typename ClusterView, typename SourceIdAt>
auto ensure_plain_pq_cache(
    const dataset::DatasetSource& source,
    const std::string& dataset_name,
    const ClusterView& clusters,
    const dataset::PQCacheRequest& spec,
    const PlainPQParams& params,
    const std::string& cache_dir,
    size_t source_begin,
    size_t source_end,
    bool dense_source_span,
    bool source_ids_are_prefix,
    const std::vector<float>* rotation,
    unsigned seed,
    SourceIdAt source_id_at)
{
    const size_t n = clusters.reordered_indices.size();
    dataset::ResidualPQCache cache(cache_dir, dataset_name);
    if (cache.get_cache_dir().empty()) {
        throw std::runtime_error("IVF_PQ_CACHE_DIR is not configured");
    }
    const auto load = [&]() {
        if constexpr (Mapped) return cache.try_map(spec);
        else return cache.try_load(spec);
    };
    auto cached = load();
    if (cached.hit) return cached;
    require_build_policy("plain_pq:" + dataset_name);

    std::unique_ptr<transform::OPQTransform> opq;
    if (rotation) {
        if (rotation->size() !=
            static_cast<size_t>(source.spec().dim) * source.spec().dim) {
            throw std::invalid_argument("invalid OPQ rotation shape");
        }
        opq = std::make_unique<transform::OPQTransform>(source.spec().dim);
        opq->set_matrix(*rotation);
        opq->prepare();
    }
    std::vector<float> transformed;
    auto transform_vectors = [&](const float* vectors, size_t count) -> const float* {
        if (!opq) return vectors;
        transformed.resize(count * static_cast<size_t>(source.spec().dim));
        opq->apply_host_to_host(vectors, transformed.data(), count, nullptr);
        return transformed.data();
    };
    std::vector<float> converted;

    std::vector<float> codebook;
    if constexpr (Mapped) {
        auto partial = cache.try_load(spec);
        if (partial.codebook_hit) codebook = std::move(partial.codebook);
    } else {
        if (cached.codebook_hit) codebook = std::move(cached.codebook);
    }
    if (codebook.empty()) {
        const int train_n = std::min<int>(n, std::max(params.k, params.train_n));
        dataset::VectorBuffer train_buffer;
        if (source_ids_are_prefix) {
            train_buffer = source.load_base(static_cast<size_t>(train_n));
        } else {
            std::vector<uint32_t> train_ids(static_cast<size_t>(train_n));
            for (int row = 0; row < train_n; ++row) {
                train_ids[static_cast<size_t>(row)] = source_id_at(static_cast<size_t>(row));
            }
            train_buffer = source.load_base_rows(train_ids);
        }
        ivf::PlainPQTrainConfig train_cfg{};
        train_cfg.n = train_n;
        train_cfg.dim = source.spec().dim;
        train_cfg.n_subspace = params.m;
        train_cfg.n_codes = params.k;
        train_cfg.iters = params.iters;
        train_cfg.seed = seed;
        const auto train_view = train_buffer.view();
        if (!opq && train_view.dtype == dataset::VectorDType::U8) {
            ivf::train_plain_pq_codebooks_gpu(train_view.u8(), train_cfg, codebook);
        } else {
            const float* train_vectors = as_fp32(train_view, converted);
            train_vectors = transform_vectors(train_vectors, train_view.n);
            ivf::train_plain_pq_codebooks_gpu(train_vectors, train_cfg, codebook);
        }
    }

    std::vector<uint8_t> codes(n * static_cast<size_t>(params.m));
    if (dense_source_span) {
        if (source_end <= source_begin || source_end - source_begin > static_cast<size_t>(INT_MAX)) {
            throw std::invalid_argument("plain PQ source span is invalid");
        }
        const size_t source_span = source_end - source_begin;
        std::vector<int32_t> reordered_position(source_span, -1);
        for (size_t reordered = 0; reordered < n; ++reordered) {
            const idx_t local = clusters.reordered_indices[reordered];
            if (local < 0 || static_cast<size_t>(local) >= n) {
                throw std::runtime_error("cluster cache contains an invalid reordered index");
            }
            const size_t source_id = source_id_at(static_cast<size_t>(local));
            if (source_id < source_begin || source_id >= source_end) {
                throw std::runtime_error("plain PQ source ID is outside the declared span");
            }
            reordered_position[source_id - source_begin] = static_cast<int32_t>(reordered);
        }
        std::vector<uint8_t> native_u8;
        std::vector<float> native_f32;
        std::vector<uint8_t> chunk_codes;
        for (size_t begin = source_begin; begin < source_end; begin += params.encode_chunk) {
            const int count = static_cast<int>(
                std::min<size_t>(params.encode_chunk, source_end - begin));
            dataset::VectorView view{};
            view.dtype = source.spec().dtype;
            view.n = static_cast<size_t>(count);
            view.dim = source.spec().dim;
            if (view.dtype == dataset::VectorDType::U8) {
                native_u8.resize(static_cast<size_t>(count) * view.dim);
                if (source.read_base_range(begin, count, native_u8.data()) !=
                    static_cast<size_t>(count)) {
                    throw std::runtime_error("short read while encoding plain PQ cache");
                }
                view.data = native_u8.data();
            } else {
                native_f32.resize(static_cast<size_t>(count) * view.dim);
                if (source.read_base_range(begin, count, native_f32.data()) !=
                    static_cast<size_t>(count)) {
                    throw std::runtime_error("short read while encoding plain PQ cache");
                }
                view.data = native_f32.data();
            }
            const float* vectors = as_fp32(view, converted);
            vectors = transform_vectors(vectors, static_cast<size_t>(count));
            chunk_codes.resize(static_cast<size_t>(count) * params.m);
            transform::opq_encode_plain_pq_gpu(
                vectors, count, source.spec().dim, params.m, params.k, codebook.data(),
                chunk_codes.data(), count);
            for (int row = 0; row < count; ++row) {
                const int32_t dst = reordered_position[begin + static_cast<size_t>(row) - source_begin];
                if (dst >= 0) {
                    std::memcpy(codes.data() + static_cast<size_t>(dst) * params.m,
                                chunk_codes.data() + static_cast<size_t>(row) * params.m,
                                static_cast<size_t>(params.m));
                }
            }
        }
    } else {
        for (size_t begin = 0; begin < n; begin += params.encode_chunk) {
            const int count = static_cast<int>(std::min<size_t>(params.encode_chunk, n - begin));
            std::vector<uint32_t> ids(static_cast<size_t>(count));
            for (int row = 0; row < count; ++row) {
                const idx_t local = clusters.reordered_indices[begin + static_cast<size_t>(row)];
                if (local < 0 || static_cast<size_t>(local) >= n) {
                    throw std::runtime_error("cluster cache contains an invalid reordered index");
                }
                ids[static_cast<size_t>(row)] = source_id_at(static_cast<size_t>(local));
            }
            auto buffer = source.load_base_rows(ids);
            const float* vectors = as_fp32(buffer.view(), converted);
            vectors = transform_vectors(vectors, static_cast<size_t>(count));
            transform::opq_encode_plain_pq_gpu(
                vectors, count, source.spec().dim, params.m, params.k, codebook.data(),
                codes.data() + begin * static_cast<size_t>(params.m), count);
        }
    }
    cache.save(spec, codebook.data(), codes.data());
    std::vector<uint8_t>().swap(codes);
    auto result = load();
    if (!result.hit) throw std::runtime_error("plain PQ cache generation failed");
    return result;
}

}  // namespace

PQCaches ensure_pq(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const PlainPQParams& pq) {
    if (n_vectors == 0 || n_vectors > static_cast<size_t>(INT_MAX) ||
        pq.m <= 0 || pq.k <= 0 || pq.iters <= 0 || pq.train_n <= 0 ||
        pq.encode_chunk <= 0 || source.spec().dim % pq.m != 0) {
        throw std::invalid_argument("invalid static plain PQ cache request");
    }
    PQCaches out;
    out.clusters = ensure_kmeans(source, n_vectors, nlist, kmeans);
    out.pq = ensure_pq(source, out.clusters, kmeans, pq);
    return out;
}

dataset::PQCacheResult ensure_pq(
    const dataset::DatasetSource& source,
    const dataset::ClusterCache::CacheResult& clusters,
    const KMeansSpec& kmeans,
    const PlainPQParams& pq) {
    const size_t n_vectors = clusters.reordered_indices.size();
    const int nlist = static_cast<int>(clusters.cluster_counts.size());
    if (!valid_cluster_cache(clusters, n_vectors, source.spec().dim, nlist) ||
        n_vectors > static_cast<size_t>(INT_MAX) ||
        pq.m <= 0 || pq.k <= 0 || pq.iters <= 0 || pq.train_n <= 0 ||
        pq.encode_chunk <= 0 || source.spec().dim % pq.m != 0) {
        throw std::invalid_argument("invalid static plain PQ cache metadata");
    }
    const std::string name = dataset::dataset_key_name(source.spec());
    auto pq_spec = dataset::make_plain_pq_cache_request(
        static_cast<int>(n_vectors), source.spec().dim, nlist,
        pq.m, pq.k, pq.iters, pq.train_n);
    pq_spec.cluster_layout_id = clusters.layout_id;
    return ensure_plain_pq_cache(
        source, name, clusters, pq_spec, pq, clusters.entry_dir + "/pq",
        0, n_vectors, true, true, nullptr, 12345,
        [](size_t local) { return static_cast<uint32_t>(local); });
}

MappedPQCaches ensure_pq_mapped(
    const dataset::DatasetSource& source, size_t n_vectors, int nlist,
    const KMeansSpec& kmeans, const PlainPQParams& pq) {
    if (n_vectors == 0 || n_vectors > static_cast<size_t>(INT_MAX) ||
        pq.m <= 0 || pq.k <= 0 || pq.k > 256 || pq.iters <= 0 || pq.train_n <= 0 ||
        pq.encode_chunk <= 0 || source.spec().dim % pq.m != 0)
        throw std::invalid_argument("invalid mapped plain PQ cache request");
    MappedPQCaches out;
    out.clusters = ensure_kmeans_mapped(source, n_vectors, nlist, kmeans);
    auto spec = dataset::make_plain_pq_cache_request(
        static_cast<int>(n_vectors), source.spec().dim, nlist,
        pq.m, pq.k, pq.iters, pq.train_n);
    spec.cluster_layout_id = out.clusters.layout_id;
    out.pq = ensure_plain_pq_cache<true>(
        source, dataset::dataset_key_name(source.spec()), out.clusters, spec, pq,
        out.clusters.entry_dir + "/pq", 0, n_vectors, true, true, nullptr, 12345,
        [](size_t local) { return static_cast<uint32_t>(local); });
    return out;
}

PQCaches ensure_opq(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const PlainPQParams& pq,
    const std::vector<float>& rotation,
    int opq_iters,
    int opq_train_n,
    unsigned seed) {
    if (n_vectors == 0 || n_vectors > static_cast<size_t>(INT_MAX) ||
        opq_iters <= 0 || opq_train_n <= 0 ||
        rotation.size() != static_cast<size_t>(source.spec().dim) *
                source.spec().dim) {
        throw std::invalid_argument("invalid OPQ cache request");
    }
    PQCaches out;
    out.clusters = ensure_kmeans(source, n_vectors, nlist, kmeans);
    const std::string name = dataset::dataset_key_name(source.spec());
    dataset::PQCacheRequest spec = dataset::make_plain_pq_cache_request(
        static_cast<int>(n_vectors), source.spec().dim, nlist,
        pq.m, pq.k, pq.iters, pq.train_n);
    spec.cluster_layout_id = out.clusters.layout_id;
    spec.trainer = "opq-gpu-v1";
    spec.opq_iters = opq_iters;
    spec.opq_train_rows = opq_train_n;
    spec.opq_seed = seed;
    out.pq = ensure_plain_pq_cache(
        source, name, out.clusters, spec, pq, out.clusters.entry_dir + "/pq",
        0, n_vectors, true, true, &rotation, seed,
        [](size_t local) { return static_cast<uint32_t>(local); });
    return out;
}


}  // namespace paraselect::ivf_cache
