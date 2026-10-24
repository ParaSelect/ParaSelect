#include "index/ivf_cache.h"
#include "index/ivf_cache/cache_policy.h"

#include "index/ivf_cache/cache_file.h"
#include "index/ivf_cache/source_vectors.h"

#include "dataset/assignment_materializer.cuh"
#include "dataset/cache/cache_fs.h"
#include "dataset/cache/cache_manifest.h"
#include "dataset/dataset.cuh"
#include "index/row_correction/row_correction.h"
#include "quantization/pq/residual_pq_trainer.cuh"
#include "quantization/pq/pq_encoder.cuh"
#include "transform/opq_trainer.cuh"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <unistd.h>

namespace paraselect::ivf_cache {
namespace {


using detail::TemporaryMapping;
using detail::WritableBinaryMapping;
using detail::load_fp32_prefix;
using detail::write_binary_atomic;

row_correction::Manifest make_row_correction_manifest(
    const std::string& dataset_name,
    const std::string& pq_entry,
    const dataset::PQCacheRequest& spec) {
    return {
        dataset_name,
        dataset::cache_manifest::Json::parse(
            std::ifstream(pq_entry + "/manifest.json")).dump(),
        spec.cluster_layout_id,
        "residual_pq",
        spec.codes_layout,
        static_cast<std::uint64_t>(spec.n_vectors),
        spec.dim,
        spec.n_lists,
        spec.pq_m,
        spec.pq_k,
        spec.dsub()};
}

row_correction::BuildResult ensure_row_correction(
    const std::string& entry_dir,
    const row_correction::Manifest& manifest,
    const dataset::ClusterCache::MappedResult& clusters,
    const dataset::PQCacheMappedResult& pq) {
    if (row_correction::manifest_matches(entry_dir, manifest)) {
        (void)row_correction::Asset::open(entry_dir, manifest);
        row_correction::BuildResult hit;
        hit.cache_hit = true;
        return hit;
    }
    require_build_policy(entry_dir);
    return row_correction::build_pq(
        entry_dir,
        manifest,
        {clusters.centroids.data(),
         reinterpret_cast<const std::int64_t*>(clusters.cluster_offsets.data()),
         pq.codes.data(),
         pq.codebook.data()});
}

}  // namespace

dataset::PQCacheRequest make_pq_cache_request(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const std::string& cluster_layout_id,
    const ResidualPQParams& params,
    const std::string& trainer,
    int training_rows) {
    dataset::PQCacheRequest spec;
    spec.n_vectors = static_cast<int>(n_vectors);
    spec.dim = source.spec().dim;
    spec.n_lists = nlist;
    spec.cluster_layout_id = cluster_layout_id;
    spec.pq_m = params.m;
    spec.pq_k = params.k;
    spec.pq_iters = 25;
    spec.trainer = trainer;
    spec.training_sampling = "prefix";
    spec.training_rows = training_rows;
    return spec;
}

PQCaches ensure_respq(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const ResidualPQParams& pq) {
    auto mapped = ensure_respq_mapped(source, n_vectors, nlist, kmeans, pq);
    PQCaches out;
    auto& clusters = out.clusters;
    clusters.hit = mapped.clusters.hit;
    clusters.request = mapped.clusters.request;
    clusters.entry_dir = mapped.clusters.entry_dir;
    clusters.layout_id = mapped.clusters.layout_id;
    clusters.centroids.assign(mapped.clusters.centroids.data(),
        mapped.clusters.centroids.data() + mapped.clusters.centroids.size());
    clusters.reordered_indices.assign(mapped.clusters.reordered_indices.data(),
        mapped.clusters.reordered_indices.data() + mapped.clusters.reordered_indices.size());
    clusters.cluster_counts.assign(mapped.clusters.cluster_counts.data(),
        mapped.clusters.cluster_counts.data() + mapped.clusters.cluster_counts.size());
    clusters.cluster_offsets.assign(mapped.clusters.cluster_offsets.data(),
        mapped.clusters.cluster_offsets.data() + mapped.clusters.cluster_offsets.size());
    dataset::ResidualPQCache cache(clusters.entry_dir + "/pq",
        dataset::dataset_key_name(source.spec()));
    out.pq = cache.try_load(mapped.pq.request);
    if (!out.pq.hit) throw std::runtime_error("residual PQ cache disappeared after preparation");
    return out;
}

MappedPQCaches ensure_respq_mapped(
    const dataset::DatasetSource& source,
    size_t n_vectors,
    int nlist,
    const KMeansSpec& kmeans,
    const ResidualPQParams& pq,
    ResidualPQBuildTiming* timing) {
    using BuildClock=std::chrono::steady_clock;
    if(timing)*timing={};
    const auto started=timing?BuildClock::now():BuildClock::time_point{};
    auto checkpoint=started;
    ResidualPQBuildTiming unused;
    auto& measured=timing?*timing:unused;
    auto record=[&](double& field) {
        if(timing) {
            const auto now=BuildClock::now();
            field+=std::chrono::duration<double>(now-checkpoint).count();checkpoint=now;
        }
    };
    if (n_vectors == 0 || n_vectors > static_cast<size_t>(INT_MAX) ||
        pq.m <= 0 || pq.k <= 0 || pq.max_gpus <= 0 || pq.train_n <= 0 ||
        pq.encode_chunk <= 0 || source.spec().dim % pq.m != 0) {
        throw std::invalid_argument("invalid mapped residual PQ cache request");
    }

    MappedPQCaches out;
    out.clusters = ensure_kmeans_mapped(source, n_vectors, nlist, kmeans);
    const std::string name = dataset::dataset_key_name(source.spec());
    const std::string pq_dir = out.clusters.entry_dir + "/pq";
    const int train_n = static_cast<int>(std::min<size_t>(n_vectors, pq.train_n));
    const auto spec = make_pq_cache_request(
        source, n_vectors, nlist, out.clusters.layout_id, pq,
        "gpu-stream-v1", train_n);
    dataset::ResidualPQCache cache(pq_dir, name);
    out.pq = cache.try_map(spec);
    record(measured.lookup_s);
    if (out.pq.hit) {
        if (pq.ensure_row_correction) {
            const auto correction = ensure_row_correction(
                out.pq.entry_dir,
                make_row_correction_manifest(name, out.pq.entry_dir, spec),
                out.clusters,
                out.pq);
            measured.row_correction_build_s += correction.compute_s;
            measured.row_correction_quantize_s += correction.quantize_s;
            measured.row_correction_commit_s += correction.commit_s;
        }
        measured.cache_hit = true;
        measured.total_s = measured.lookup_s + measured.row_correction_build_s +
            measured.row_correction_quantize_s + measured.row_correction_commit_s;
        return out;
    }
    require_build_policy(pq_dir);

    TemporaryMapping<uint64_t> source_layout_workspace(
        out.clusters.entry_dir, "respq-source-layout", n_vectors);
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

    record(measured.layout_s);
    std::cerr << "[ivf_cache:respq] train prefix n=" << train_n
              << " full_encode_n=" << n_vectors << std::endl;
    std::unique_ptr<float, decltype(&std::free)> train(
        load_fp32_prefix(source, static_cast<size_t>(train_n)), &std::free);
    std::vector<idx_t> train_assignments(static_cast<size_t>(train_n));
    std::vector<int> train_counts(static_cast<size_t>(nlist), 0);
    for (int row = 0; row < train_n; ++row) {
        const uint32_t cluster = static_cast<uint32_t>(
            source_layout[static_cast<size_t>(row)] >> 32);
        if (cluster >= static_cast<uint32_t>(nlist)) {
            throw std::runtime_error("cluster cache contains invalid assignment");
        }
        train_assignments[static_cast<size_t>(row)] = static_cast<idx_t>(cluster);
        ++train_counts[cluster];
    }
    std::vector<idx_t> train_offsets(static_cast<size_t>(nlist) + 1u, 0);
    for (int cluster = 0; cluster < nlist; ++cluster) {
        train_offsets[static_cast<size_t>(cluster) + 1u] =
            train_offsets[static_cast<size_t>(cluster)] + train_counts[static_cast<size_t>(cluster)];
    }
    cpu_cluster_reorder_vectors_inplace(
        train.get(), train_assignments.data(), train_n, source.spec().dim, nlist);
    train_assignments.clear();
    train_assignments.shrink_to_fit();

    quantization::pq::ResidualPQCase train_case;
    train_case.n = train_n;
    train_case.dim = source.spec().dim;
    train_case.nlist = nlist;
    train_case.m = pq.m;
    train_case.k = pq.k;
    train_case.iters = spec.pq_iters;
    train_case.device_id = 0;
    train_case.max_gpus = pq.max_gpus;
    quantization::pq::ResidualPQInput input;
    input.data = train.get();
    input.coarse_centroids = out.clusters.centroids.data();
    input.cluster_offsets = train_offsets.data();
    input.cluster_counts = train_counts.data();
    std::vector<float> codebook(
        static_cast<size_t>(pq.m) * pq.k * spec.dsub());
    record(measured.train_prepare_s);
    quantization::pq::ResidualPQTrainerGPU(train_case).fit(input, codebook.data());
    record(measured.train_s);
    train.reset();
    train_offsets.clear();
    train_counts.clear();

    const std::string entry_dir = cache.get_cache_dir() + "/.building-" +
        std::to_string(static_cast<long long>(::getpid()));
    if (!dataset::cache_fs::ensure_dir_exists(entry_dir)) {
        throw std::runtime_error("failed to create residual PQ cache directory: " + entry_dir);
    }
    // Original-row encoding scatters into list order. Keep that random write
    // workload in RAM: MAP_SHARED output repeatedly dirties the whole codes
    // file across chunks and causes large background-writeback amplification.
    // Capacity is fixed; publish once, sequentially, after every row is encoded.
    std::vector<uint8_t> reordered_codes(n_vectors * static_cast<size_t>(pq.m));
    uint8_t* output_codes = reordered_codes.data();
    std::vector<uint8_t> native_u8;
    std::vector<float> native_fp32;
    std::vector<float> residual;
    std::vector<uint8_t> chunk_codes;
    const size_t encode_chunk = static_cast<size_t>(pq.encode_chunk);
    quantization::pq::PQEncoderGPU encoder(codebook.data(), source.spec().dim,
        pq.m, pq.k, std::min(encode_chunk, n_vectors));
    record(measured.encode_prepare_s);
    for (size_t begin = 0; begin < n_vectors; begin += encode_chunk) {
        const size_t count = std::min(encode_chunk, n_vectors - begin);
        const size_t values = count * static_cast<size_t>(source.spec().dim);
        residual.resize(values);
        if (source.spec().dtype == dataset::VectorDType::U8) {
            native_u8.resize(values);
            if (source.read_base_range(begin, count, native_u8.data()) != count) {
                throw std::runtime_error("short read during residual PQ encoding");
            }
            std::transform(
                native_u8.begin(), native_u8.end(), residual.begin(),
                [](uint8_t value) { return static_cast<float>(value); });
        } else {
            native_fp32.resize(values);
            if (source.read_base_range(begin, count, native_fp32.data()) != count) {
                throw std::runtime_error("short read during residual PQ encoding");
            }
            std::copy(native_fp32.begin(), native_fp32.end(), residual.begin());
        }
        for (size_t row = 0; row < count; ++row) {
            const uint32_t cluster = static_cast<uint32_t>(
                source_layout[begin + row] >> 32);
            const float* centroid = out.clusters.centroids.data() +
                static_cast<size_t>(cluster) * source.spec().dim;
            float* vector = residual.data() + row * source.spec().dim;
            for (int d = 0; d < source.spec().dim; ++d) vector[d] -= centroid[d];
        }
        chunk_codes.resize(count * static_cast<size_t>(pq.m));
        record(measured.residual_io_s);
        encoder.encode(residual.data(), static_cast<int>(count), chunk_codes.data());
        record(measured.encode_s);
        for (size_t row = 0; row < count; ++row) {
            const size_t destination = static_cast<uint32_t>(source_layout[begin + row]);
            std::memcpy(
                output_codes + destination * pq.m,
                chunk_codes.data() + row * pq.m,
                static_cast<size_t>(pq.m));
        }
        record(measured.code_reorder_s);
        if (((begin / encode_chunk) & 15u) == 15u || begin + count == n_vectors) {
            std::cerr << "[ivf_cache:respq] full encoding "
                      << (begin + count) << "/" << n_vectors << std::endl;
        }
    }
    WritableBinaryMapping<uint8_t> codes_file(
        entry_dir + "/codes.u8bin", static_cast<uint32_t>(n_vectors),
        static_cast<uint32_t>(pq.m));
    std::memcpy(codes_file.data(), reordered_codes.data(), reordered_codes.size());
    codes_file.commit();
    write_binary_atomic(
        entry_dir + "/codebook.fbin",
        static_cast<uint32_t>(pq.m * pq.k),
        static_cast<uint32_t>(spec.dsub()), codebook.data());
    cache.publish_staged(entry_dir, spec);

    out.pq = cache.try_map(spec);
    if (!out.pq.hit) {
        throw std::runtime_error("mapped residual PQ cache generation failed");
    }
    if (pq.ensure_row_correction) {
        const auto correction = ensure_row_correction(
            out.pq.entry_dir,
            make_row_correction_manifest(name, out.pq.entry_dir, spec),
            out.clusters,
            out.pq);
        measured.row_correction_build_s += correction.compute_s;
        measured.row_correction_quantize_s += correction.quantize_s;
        measured.row_correction_commit_s += correction.commit_s;
    }
    record(measured.commit_s);
    if(timing)measured.total_s=std::chrono::duration<double>(BuildClock::now()-started).count();
    return out;
}


}  // namespace paraselect::ivf_cache
