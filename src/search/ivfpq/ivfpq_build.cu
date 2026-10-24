#include "search/ivfpq/ivfpq_build.h"

#include "transform/opq_cache.cuh"
#include "transform/opq_trainer.cuh"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

namespace paraselect::ivfpq {
using namespace paraselect::search;
namespace {

std::vector<float> load_fp32_prefix(
    const paraselect::dataset::DatasetSource& source, size_t count) {
    auto buffer = source.load_base(count);
    const auto view = buffer.view();
    std::vector<float> output(view.n * static_cast<size_t>(view.dim));
    if (view.dtype == paraselect::dataset::VectorDType::U8) {
        std::transform(
            view.u8(), view.u8() + output.size(), output.begin(),
            [](uint8_t value) { return static_cast<float>(value); });
    } else {
        std::copy_n(view.fp32(), output.size(), output.data());
    }
    return output;
}

std::vector<float> load_or_train_rotation(const IVFPQBuildConfig& config) {
    const int dim = config.source.spec().dim;
    const int train_n = config.opq_train_n > 0
        ? config.opq_train_n
        : static_cast<int>(config.n_vectors);
    paraselect::transform::OPQTrainSpec spec;
    spec.n = train_n;
    spec.dim = dim;
    spec.pq_m = config.pq_m;
    spec.pq_k = config.pq_k;
    spec.opq_iters = config.opq_iters;
    spec.pq_iters = config.pq_iters;
    spec.seed = config.seed;
    const std::string cache_dir = paraselect::transform::default_opq_cache_dir();
    const std::string cache_key = paraselect::transform::make_opq_cache_key(
        paraselect::dataset::dataset_key_name(config.source.spec()),
        static_cast<int>(config.n_vectors), dim, spec);
    std::vector<float> rotation;
    if (paraselect::transform::load_opq_rotation_cache(
            cache_dir, cache_key, dim, rotation)) {
        return rotation;
    }
    std::vector<float> training = load_fp32_prefix(config.source, train_n);
    auto trained = paraselect::transform::train_opq_transform(training.data(), spec);
    rotation = std::move(trained.rotation_rowmajor_out_in);
    paraselect::transform::save_opq_rotation_cache(
        cache_dir, cache_key, spec, rotation);
    return rotation;
}

template <typename T>
std::vector<T> materialize_reordered(
    const paraselect::dataset::DatasetSource& source,
    const std::vector<idx_t>& reordered_ids,
    size_t chunk_rows) {
    const size_t n = reordered_ids.size();
    const int dim = source.spec().dim;
    std::vector<int32_t> source_to_local(n, -1);
    for (size_t local = 0; local < n; ++local) {
        const idx_t source_id = reordered_ids[local];
        if (source_id < 0 || static_cast<size_t>(source_id) >= n ||
            source_to_local[static_cast<size_t>(source_id)] != -1) {
            throw std::runtime_error("invalid IVF reordered IDs");
        }
        source_to_local[static_cast<size_t>(source_id)] =
            static_cast<int32_t>(local);
    }
    std::vector<T> output(n * static_cast<size_t>(dim));
    std::vector<T> chunk(std::min(chunk_rows, n) * static_cast<size_t>(dim));
    for (size_t begin = 0; begin < n; begin += chunk_rows) {
        const size_t count = std::min(chunk_rows, n - begin);
        if (source.read_base_range(begin, count, chunk.data()) != count) {
            throw std::runtime_error("short base read while preparing IVFPQ rerank");
        }
        for (size_t row = 0; row < count; ++row) {
            const size_t local = static_cast<size_t>(source_to_local[begin + row]);
            std::copy_n(
                chunk.data() + row * dim,
                dim,
                output.data() + local * dim);
        }
    }
    return output;
}

void validate_build_config(const IVFPQBuildConfig& config) {
    const auto& spec = config.source.spec();
    if ((config.transform != IVFPQTransformKind::Plain &&
         config.transform != IVFPQTransformKind::OPQ) ||
        (config.fine_storage != IVFPQFineStorage::Host &&
         config.fine_storage != IVFPQFineStorage::Resident) ||
        (config.exact_rerank != IVFPQExactRerank::Disabled &&
         config.exact_rerank != IVFPQExactRerank::Flat)) {
        throw std::invalid_argument("unknown IVFPQ build strategy");
    }
    if (config.n_vectors == 0 ||
        config.n_vectors > config.source.base_info().n ||
        config.n_vectors > static_cast<size_t>(INT_MAX) ||
        config.nlist <= 0 ||
        config.n_vectors < static_cast<size_t>(config.nlist) ||
        config.pq_m <= 0 || config.pq_k <= 0 || config.pq_k > 256 ||
        config.pq_iters <= 0 || config.pq_train_n <= 0 ||
        spec.dim % config.pq_m != 0 ||
        config.prepare_chunk_rows == 0 ||
        config.prepare_chunk_rows > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("invalid IVFPQ build config");
    }
    if (config.transform == IVFPQTransformKind::Plain &&
        (config.opq_iters != 0 || config.opq_train_n != 0)) {
        throw std::invalid_argument("plain IVFPQ cannot specify OPQ training");
    }
    if (config.transform == IVFPQTransformKind::OPQ &&
        (config.opq_iters <= 0 ||
         (config.opq_train_n > 0
              ? config.opq_train_n < config.pq_k ||
                    static_cast<size_t>(config.opq_train_n) > config.n_vectors
              : config.n_vectors < static_cast<size_t>(config.pq_k)))) {
        throw std::invalid_argument("invalid IVFPQ OPQ config");
    }
    if (config.fine_storage == IVFPQFineStorage::Resident &&
        config.resident_combo.empty()) {
        throw std::invalid_argument("resident IVFPQ requires a combo");
    }
    if (config.adc_strategy ==
            paraselect::search::PQAdcSearchStrategy::Threshold &&
        (config.fine_storage != IVFPQFineStorage::Resident ||
         config.exact_rerank != IVFPQExactRerank::Flat ||
         config.n_rerank <= 0 || config.threshold.seed_probes <= 0 ||
         config.threshold.candidate_capacity < config.n_rerank ||
         config.threshold.candidate_capacity > 8192)) {
        throw std::invalid_argument("invalid IVFPQ threshold build config");
    }
    if (config.exact_rerank == IVFPQExactRerank::Flat &&
        config.n_rerank <= 0) {
        throw std::invalid_argument("IVFPQ Flat rerank requires n_rerank");
    }
}

}  // namespace

BuildArtifacts build_artifacts(const IVFPQBuildConfig& config) {
    validate_build_config(config);
    BuildArtifacts out;
    paraselect::ivf_cache::PlainPQParams pq;
    pq.m = config.pq_m;
    pq.k = config.pq_k;
    pq.iters = config.pq_iters;
    pq.train_n = config.pq_train_n;
    pq.encode_chunk = static_cast<int>(config.prepare_chunk_rows);
    if (config.transform == IVFPQTransformKind::OPQ) {
        const std::vector<float> rotation = load_or_train_rotation(config);
        out.caches = paraselect::ivf_cache::ensure_opq(
            config.source,
            config.n_vectors,
            config.nlist,
            config.kmeans,
            pq,
            rotation,
            config.opq_iters,
            config.opq_train_n > 0
                ? config.opq_train_n
                : static_cast<int>(config.n_vectors),
            config.seed);
        out.transform = std::make_unique<paraselect::transform::OPQTransform>(
            config.source.spec().dim);
        out.transform->set_matrix(rotation);
        out.transform->prepare();
        out.transformed_centroids.resize(out.caches.clusters.centroids.size());
        out.transform->apply_host_to_host(
            out.caches.clusters.centroids.data(),
            out.transformed_centroids.data(),
            static_cast<size_t>(config.nlist),
            nullptr);
    } else {
        out.caches = paraselect::ivf_cache::ensure_pq(
            config.source,
            config.n_vectors,
            config.nlist,
            config.kmeans,
            pq);
    }

    if (config.exact_rerank == IVFPQExactRerank::Flat) {
        if (config.source.spec().dtype == paraselect::dataset::VectorDType::U8) {
            out.rerank_u8 = materialize_reordered<uint8_t>(
                config.source,
                out.caches.clusters.reordered_indices,
                config.prepare_chunk_rows);
        } else if (
            config.source.spec().dtype == paraselect::dataset::VectorDType::F32) {
            out.rerank_fp32 = materialize_reordered<float>(
                config.source,
                out.caches.clusters.reordered_indices,
                config.prepare_chunk_rows);
        } else {
            throw std::invalid_argument("unsupported IVFPQ rerank dtype");
        }
    }
    return out;
}

}  // namespace paraselect::ivfpq
