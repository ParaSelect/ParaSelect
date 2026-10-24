#include "index/ivf_cache.h"
#include "index/ivf_cache/cache_policy.h"

#include "index/ivf_cache/cache_file.h"

#include "dataset/cache/cache_fs.h"
#include "dataset/cache/cache_manifest.h"
#include "quantization/turboquant/turboquant_1bit_encode.cuh"
#include "utils/check_cuda_error.cuh"

#include <unistd.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace paraselect::ivf_cache {
namespace {

using detail::WritableBinaryMapping;

}  // namespace

MappedTurboQuantCaches ensure_turboquant_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const TurboQuantParams& params)
{
    const int dim = source.spec().dim;
    if (n_vectors == 0 || n_vectors > static_cast<size_t>(INT_MAX) ||
        nlist <= 0 || params.rotation_seed < 0 || params.encode_chunk == 0 ||
        dim != 128 || source.spec().dtype != dataset::VectorDType::U8) {
        throw std::invalid_argument(
            "invalid mapped TurboQuant cache request (requires SIFT128 U8)");
    }

    MappedTurboQuantCaches out;
    out.clusters = ensure_kmeans_mapped(source, n_vectors, nlist, kmeans);
    const std::string name = dataset::dataset_key_name(source.spec());
    const std::string tq_dir = out.clusters.entry_dir + "/tq";
    dataset::TurboQuantCacheRequest spec;
    spec.n_vectors = static_cast<int>(n_vectors);
    spec.dim = dim;
    spec.n_lists = nlist;
    const int words_per_code = spec.words_per_code();
    spec.rotation_seed = params.rotation_seed;
    spec.cluster_layout_id = out.clusters.layout_id;
    dataset::TurboQuantCache cache(tq_dir, name);
    out.tq = cache.try_map(spec);
    if (out.tq.hit) return out;
    require_build_policy(tq_dir);

    // Same source -> (cluster, reordered row) mapping as the residual PQ cache
    // builder, so both caches describe the identical cluster-reordered layout.
    detail::TemporaryMapping<uint64_t> source_layout_workspace(
        out.clusters.entry_dir, "tq-source-layout", n_vectors);
    uint64_t* source_layout = source_layout_workspace.data();
    std::fill(source_layout, source_layout + n_vectors, UINT64_MAX);
    for (int cluster = 0; cluster < nlist; ++cluster) {
        const idx_t begin = out.clusters.cluster_offsets[static_cast<size_t>(cluster)];
        const idx_t end = out.clusters.cluster_offsets[static_cast<size_t>(cluster) + 1u];
        if (begin < 0 || end < begin || static_cast<size_t>(end) > n_vectors) {
            throw std::runtime_error("cluster cache contains invalid offsets");
        }
        for (idx_t position = begin; position < end; ++position) {
            const idx_t source_id =
                out.clusters.reordered_indices[static_cast<size_t>(position)];
            if (source_id < 0 || static_cast<size_t>(source_id) >= n_vectors) {
                throw std::runtime_error("cluster cache contains an out-of-range source ID");
            }
            uint64_t& destination = source_layout[static_cast<size_t>(source_id)];
            if (destination != UINT64_MAX) {
                throw std::runtime_error("cluster cache reordered IDs are not a permutation");
            }
            destination = (static_cast<uint64_t>(static_cast<uint32_t>(cluster)) << 32) |
                          static_cast<uint32_t>(position);
        }
    }

    const std::string entry_dir = tq_dir + "/.building-" + std::to_string(::getpid());
    if (!dataset::cache_fs::ensure_dir_exists(entry_dir)) {
        throw std::runtime_error("failed to create TurboQuant cache directory: " + entry_dir);
    }

    WritableBinaryMapping<uint32_t> codes_file(
        entry_dir + "/codes.u32bin", static_cast<uint32_t>(n_vectors),
        static_cast<uint32_t>(words_per_code));
    WritableBinaryMapping<float> scales_file(
        entry_dir + "/scales.f32bin", static_cast<uint32_t>(n_vectors), 1u);
    // Encoding consumes the source in sequential chunks while the cache must be
    // stored in cluster-reordered order. Accumulate the encoded rows in a
    // destination-ordered host buffer first and flush it sequentially: writing
    // the mapped files directly from the scatter produced ~8x write
    // amplification on the 20 GB cache.
    std::vector<uint32_t> host_codes(
        static_cast<size_t>(n_vectors) * static_cast<size_t>(words_per_code));
    std::vector<float> host_scales(static_cast<size_t>(n_vectors));

    const size_t encode_chunk = params.encode_chunk;
    float* d_residual = nullptr;
    uint32_t* d_codes = nullptr;
    float* d_scales = nullptr;
    CUDA_CHECK(cudaMalloc(&d_residual, encode_chunk * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_codes, encode_chunk * words_per_code * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, encode_chunk * sizeof(float)));

    std::vector<uint8_t> native_u8;
    std::vector<float> residual;
    std::vector<uint32_t> chunk_codes;
    std::vector<float> chunk_scales;
    try {
        for (size_t begin = 0; begin < n_vectors; begin += encode_chunk) {
            const size_t count = std::min(encode_chunk, n_vectors - begin);
            const size_t values = count * static_cast<size_t>(dim);
            native_u8.resize(values);
            if (source.read_base_range(begin, count, native_u8.data()) != count) {
                throw std::runtime_error("short read during TurboQuant encoding");
            }
            residual.resize(values);
            for (size_t row = 0; row < count; ++row) {
                const uint32_t cluster = static_cast<uint32_t>(
                    source_layout[begin + row] >> 32);
                if (cluster >= static_cast<uint32_t>(nlist)) {
                    throw std::runtime_error("cluster cache contains invalid assignment");
                }
                const float* centroid = out.clusters.centroids.data() +
                    static_cast<size_t>(cluster) * dim;
                const uint8_t* vector = native_u8.data() + row * dim;
                float* destination = residual.data() + row * dim;
                for (int d = 0; d < dim; ++d) {
                    destination[d] = static_cast<float>(vector[d]) - centroid[d];
                }
            }
            CUDA_CHECK(cudaMemcpy(
                d_residual, residual.data(), values * sizeof(float),
                cudaMemcpyHostToDevice));
            paraselect::search::turboquant::turboquant_1bit_encode_residual_l1_gpu(
                {d_residual, static_cast<int>(count), dim, dim,
                 static_cast<std::uint32_t>(params.rotation_seed)},
                {nullptr, d_codes, d_scales, nullptr}, nullptr);
            chunk_codes.resize(count * static_cast<size_t>(words_per_code));
            chunk_scales.resize(count);
            CUDA_CHECK(cudaMemcpy(
                chunk_codes.data(), d_codes,
                chunk_codes.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                chunk_scales.data(), d_scales,
                count * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t row = 0; row < count; ++row) {
                const size_t destination =
                    static_cast<uint32_t>(source_layout[begin + row]);
                std::memcpy(
                    host_codes.data() + destination * words_per_code,
                    chunk_codes.data() + row * words_per_code,
                    static_cast<size_t>(words_per_code) * sizeof(uint32_t));
                host_scales[destination] = chunk_scales[row];
            }
            if (((begin / encode_chunk) & 15u) == 15u || begin + count == n_vectors) {
                std::cerr << "[ivf_cache:tq] full encoding "
                          << (begin + count) << "/" << n_vectors << std::endl;
            }
        }
        CUDA_CHECK(cudaFree(d_residual));
        CUDA_CHECK(cudaFree(d_codes));
        CUDA_CHECK(cudaFree(d_scales));
    } catch (...) {
        if (d_residual) cudaFree(d_residual);
        if (d_codes) cudaFree(d_codes);
        if (d_scales) cudaFree(d_scales);
        throw;
    }
    // Sequential flush: full-page writes, no scatter write amplification.
    std::memcpy(
        codes_file.data(), host_codes.data(),
        host_codes.size() * sizeof(uint32_t));
    std::memcpy(
        scales_file.data(), host_scales.data(),
        host_scales.size() * sizeof(float));
    codes_file.commit();
    scales_file.commit();
    cache.publish_staged(entry_dir, spec);

    out.tq = cache.try_map(spec);
    if (!out.tq.hit) {
        throw std::runtime_error("mapped TurboQuant cache generation failed");
    }
    return out;
}

}  // namespace paraselect::ivf_cache
