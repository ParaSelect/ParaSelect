#include "index/ivf_cache.h"
#include "index/ivf_cache/cache_policy.h"

#include "index/ivf_cache/cache_file.h"
#include "index/ivf_cache/source_vectors.h"

#include "transform/pca_transform.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace paraselect::ivf_cache {

using detail::TemporaryMapping;
using detail::load_fp32_prefix;

dataset::PCACacheMappedResult ensure_pca_mapped(
    const dataset::DatasetSource& source,
    const dataset::ClusterCache::MappedResult& clusters,
    const PCAParams& params) {
    const size_t n_vectors = clusters.reordered_indices.size();
    const int source_dim = source.spec().dim;
    if (!clusters.hit || n_vectors == 0 || n_vectors > INT_MAX ||
        clusters.layout_id.empty() || params.dim <= 0 || params.dim > source_dim ||
        params.train_n <= 0 || params.encode_chunk <= 0) {
        throw std::invalid_argument("invalid PCA cache request");
    }

    dataset::PCACacheSpec spec;
    spec.n_vectors = static_cast<int>(n_vectors);
    spec.source_dim = source_dim;
    spec.dim = params.dim;
    spec.train_n = static_cast<int>(
        std::min<size_t>(n_vectors, static_cast<size_t>(params.train_n)));
    spec.cluster_layout_id = clusters.layout_id;
    const std::string name = dataset::dataset_key_name(source.spec());
    dataset::PCACache cache(clusters.entry_dir + "/pca", name);
    auto mapped = cache.try_map(spec);
    if (mapped.hit) return mapped;
    require_build_policy(clusters.entry_dir + "/pca");

    std::cerr << "[ivf_cache:pca] train prefix n=" << spec.train_n
              << " dim=" << source_dim << "->" << spec.dim
              << " full_encode_n=" << n_vectors << std::endl;
    std::unique_ptr<float, decltype(&std::free)> train(
        load_fp32_prefix(source, static_cast<size_t>(spec.train_n)), &std::free);
    transform::PCATransform model = transform::fit_pca_transform(
        train.get(), static_cast<size_t>(spec.train_n), source_dim, spec.dim);
    transform::PCAProjector projector(model);
    std::vector<float> scales(static_cast<size_t>(spec.dim));
    projector.calibrate_scales(
        train.get(), static_cast<size_t>(spec.train_n), scales.data());
    train.reset();

    TemporaryMapping<uint32_t> source_to_local_workspace(
        clusters.entry_dir, "pca-source-layout", n_vectors);
    uint32_t* source_to_local = source_to_local_workspace.data();
    std::fill(source_to_local, source_to_local + n_vectors, UINT32_MAX);
    for (uint32_t local = 0; local < static_cast<uint32_t>(n_vectors); ++local) {
        const idx_t source_id = clusters.reordered_indices[local];
        if (source_id < 0 || static_cast<size_t>(source_id) >= n_vectors) {
            throw std::runtime_error("cluster cache contains an invalid PCA source ID");
        }
        uint32_t& destination = source_to_local[static_cast<size_t>(source_id)];
        if (destination != UINT32_MAX) {
            throw std::runtime_error("cluster cache PCA source IDs are not a permutation");
        }
        destination = local;
    }

    dataset::PCACacheWriter writer = cache.create(spec);
    int8_t* output = writer.vectors();
    const size_t chunk_rows = static_cast<size_t>(params.encode_chunk);
    std::vector<uint8_t> native_u8;
    std::vector<float> native_fp32;
    std::vector<float> input;
    std::vector<int8_t> quantized;
    uint64_t clipped = 0;
    for (size_t begin = 0; begin < n_vectors; begin += chunk_rows) {
        const size_t rows = std::min(chunk_rows, n_vectors - begin);
        const size_t input_values = rows * static_cast<size_t>(source_dim);
        input.resize(input_values);
        if (source.spec().dtype == dataset::VectorDType::U8) {
            native_u8.resize(input_values);
            if (source.read_base_range(begin, rows, native_u8.data()) != rows) {
                throw std::runtime_error("short read during PCA storage encoding");
            }
            std::transform(
                native_u8.begin(), native_u8.end(), input.begin(),
                [](uint8_t value) { return static_cast<float>(value); });
        } else {
            native_fp32.resize(input_values);
            if (source.read_base_range(begin, rows, native_fp32.data()) != rows) {
                throw std::runtime_error("short read during PCA storage encoding");
            }
            std::copy(native_fp32.begin(), native_fp32.end(), input.begin());
        }
        quantized.resize(rows * static_cast<size_t>(spec.dim));
        clipped += projector.apply_quantized(
            input.data(), rows, scales.data(), quantized.data());
        for (size_t row = 0; row < rows; ++row) {
            const uint32_t local = source_to_local[begin + row];
            if (local == UINT32_MAX) {
                throw std::runtime_error("cluster cache PCA layout is incomplete");
            }
            int8_t* destination = output + static_cast<size_t>(local) * spec.dim;
            std::memcpy(
                destination,
                quantized.data() + row * static_cast<size_t>(spec.dim),
                static_cast<size_t>(spec.dim));
        }
        if (((begin / chunk_rows) & 15U) == 15U || begin + rows == n_vectors) {
            std::cerr << "[ivf_cache:pca] full encoding "
                      << (begin + rows) << "/" << n_vectors << std::endl;
        }
    }
    const double clip_rate = static_cast<double>(clipped) /
        (static_cast<double>(n_vectors) * spec.dim);
    std::cerr << "[ivf_cache:pca] quantized_clip_rate=" << clip_rate << std::endl;
    writer.commit(model.projection, model.bias, scales);

    mapped = cache.try_map(spec);
    if (!mapped.hit) throw std::runtime_error("manifest cache generation failed for PCA asset");
    if (!mapped.hit) throw std::runtime_error("PCA cache generation failed");
    return mapped;
}

}  // namespace paraselect::ivf_cache
