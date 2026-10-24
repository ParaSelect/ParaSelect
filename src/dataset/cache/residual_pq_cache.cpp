#include "residual_pq_cache.h"

#include "cache_fs.h"
#include "cache_manifest.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

namespace paraselect {
namespace dataset {

namespace {

cache_manifest::Json manifest_identity(const std::string& dataset, const PQCacheRequest& spec) {
    const int train_rows = spec.training_rows > 0 ? spec.training_rows : spec.n_vectors;
    if ((spec.training_sampling != "full" && spec.training_sampling != "prefix" &&
         spec.training_sampling != "sample") || train_rows <= 0 || train_rows > spec.n_vectors ||
        (spec.training_sampling == "full" && train_rows != spec.n_vectors) ||
        (spec.training_sampling != "full" && spec.training_rows <= 0))
        throw std::invalid_argument("invalid PQ training recipe");
    auto result = cache_manifest::identity(spec.pq_type == "residual" ? "residual_pq" : "pq", dataset);
    result.update({{"n_vectors", spec.n_vectors}, {"nlist", spec.n_lists}, {"dim", spec.dim},
                   {"m", spec.pq_m}, {"k", spec.pq_k}, {"dsub", spec.dsub()},
                   {"pq_type", spec.pq_type}, {"pq_iters", spec.pq_iters},
                   {"training", {{"sampling", spec.training_sampling}, {"rows", train_rows}}},
                   {"trainer", spec.trainer},
                   {"cluster_layout_id", spec.cluster_layout_id}, {"codes_layout", spec.codes_layout}});
    if (spec.opq_iters > 0) {
        if (spec.trainer != "opq-gpu-v1" || spec.opq_train_rows <= 0)
            throw std::invalid_argument("invalid OPQ build recipe");
        result["transform"] = {{"kind", "opq"}, {"iters", spec.opq_iters},
            {"training_rows", spec.opq_train_rows}, {"seed", spec.opq_seed}};
    } else if (spec.trainer == "opq-gpu-v1") {
        throw std::invalid_argument("OPQ build recipe is missing");
    }
    return result;
}

bool file_size_is(const std::string& path, size_t expected) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return (size_t)st.st_size == expected;
}

bool atomic_rename(const std::string& tmp, const std::string& dst) {
    return std::rename(tmp.c_str(), dst.c_str()) == 0;
}

}  // namespace

ResidualPQCache::ResidualPQCache(std::string cache_dir, std::string dataset_name)
    : cache_dir_(std::move(cache_dir)), dataset_name_(std::move(dataset_name)) {
    if (cache_dir_.empty() || dataset_name_.empty())
        throw std::invalid_argument("ResidualPQCache requires cache root and dataset");
}

bool ResidualPQCache::load_codebook(
    const std::string& path,
    const PQCacheRequest& spec,
    std::vector<float>& out) const
{
    constexpr size_t header_bytes = sizeof(uint32_t) * 2;
    const uint32_t expected_n = (uint32_t)(spec.pq_m * spec.pq_k);
    const uint32_t expected_dim = (uint32_t)spec.dsub();
    const size_t expected_values = (size_t)expected_n * expected_dim;
    const size_t expected_bytes = header_bytes + expected_values * sizeof(float);
    if (!file_size_is(path, expected_bytes)) return false;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0;
    uint32_t dim = 0;
    bool ok = std::fread(&n, sizeof(uint32_t), 1, f) == 1 &&
              std::fread(&dim, sizeof(uint32_t), 1, f) == 1 &&
              n == expected_n &&
              dim == expected_dim;
    if (ok) {
        out.resize(expected_values);
        ok = std::fread(out.data(), sizeof(float), expected_values, f) == expected_values;
    }
    std::fclose(f);
    if (!ok) out.clear();
    return ok;
}

bool ResidualPQCache::load_codes(
    const std::string& path,
    const PQCacheRequest& spec,
    std::vector<uint8_t>& out) const
{
    constexpr size_t header_bytes = sizeof(uint32_t) * 2;
    const uint32_t expected_n = (uint32_t)spec.n_vectors;
    const uint32_t expected_dim = (uint32_t)spec.pq_m;
    const size_t expected_values = (size_t)expected_n * expected_dim;
    const size_t expected_bytes = header_bytes + expected_values * sizeof(uint8_t);
    if (!file_size_is(path, expected_bytes)) return false;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0;
    uint32_t dim = 0;
    bool ok = std::fread(&n, sizeof(uint32_t), 1, f) == 1 &&
              std::fread(&dim, sizeof(uint32_t), 1, f) == 1 &&
              n == expected_n &&
              dim == expected_dim;
    if (ok) {
        out.resize(expected_values);
        ok = std::fread(out.data(), sizeof(uint8_t), expected_values, f) == expected_values;
    }
    std::fclose(f);
    if (!ok) out.clear();
    return ok;
}

PQCacheResult ResidualPQCache::try_load(
    const PQCacheRequest& spec) const
{
    PQCacheResult result;
    result.request = spec;
    if (cache_dir_.empty()) return result;
    const std::string entry_dir = cache_manifest::find(
        cache_dir_, manifest_identity(dataset_name_, spec));
    if (entry_dir.empty()) return result;
    result.entry_dir = entry_dir;
    if (!load_codebook(entry_dir + "/codebook.fbin", spec, result.codebook)) return result;
    result.codebook_hit = true;

    if (load_codes(entry_dir + "/codes.u8bin", spec, result.codes)) {
        result.hit = true;
        std::cout << "[PQCache] : " << entry_dir << std::endl;
    } else {
        std::cout << "[PQCache] codebook hit only: " << entry_dir << std::endl;
    }
    return result;
}

PQCacheMappedResult ResidualPQCache::try_map(
    const PQCacheRequest& spec) const {
    PQCacheMappedResult result;
    result.request = spec;
    result.entry_dir = cache_manifest::find(cache_dir_, manifest_identity(dataset_name_, spec));
    if (result.entry_dir.empty()) return result;
    try {
        result.codebook = MappedArray<float>::open_binary(
            result.entry_dir + "/codebook.fbin", spec.pq_m * spec.pq_k, spec.dsub());
        result.codes = MappedArray<uint8_t>::open_binary(
            result.entry_dir + "/codes.u8bin", spec.n_vectors, spec.pq_m);
    } catch (const std::exception& error) {
        std::cerr << "[PQCache] manifest payload mismatch: " << error.what() << "\n";
        result.codebook = {};
        result.codes = {};
        return result;
    }
    result.hit = true;
    std::cout << "[PQCache] manifest hit: " << result.entry_dir << "\n";
    return result;
}

void ResidualPQCache::publish_staged(const std::string& entry,
    const PQCacheRequest& spec) const {
    const auto manifest = manifest_identity(dataset_name_, spec);
    const auto codebook = MappedArray<float>::open_binary(
        entry + "/codebook.fbin", spec.pq_m * spec.pq_k, spec.dsub());
    const auto codes = MappedArray<uint8_t>::open_binary(
        entry + "/codes.u8bin", spec.n_vectors, spec.pq_m);
    (void)codebook;
    (void)codes;
    const std::string canonical = cache_manifest::canonical_entry(cache_dir_, manifest);
    if (entry != canonical) {
        std::error_code ec;
        if (std::filesystem::exists(canonical)) {
            if (std::filesystem::exists(canonical + "/manifest.json")) {
                std::filesystem::remove_all(entry, ec);
                if (ec) throw std::runtime_error("cannot discard duplicate PQ build: " + ec.message());
                return;
            }
            std::filesystem::remove_all(canonical, ec);
            if (ec) throw std::runtime_error("cannot remove incomplete PQ cache: " + ec.message());
        }
        std::filesystem::rename(entry, canonical, ec);
        if (ec) throw std::runtime_error("cannot publish PQ cache: " + ec.message());
    }
    write_meta(canonical + "/meta.json", dataset_name_, spec);
    cache_manifest::publish(canonical, manifest);
}

void ResidualPQCache::write_meta(
    const std::string& path,
    const std::string& dataset_name,
    const PQCacheRequest& spec) const
{
    const auto manifest = manifest_identity(dataset_name, spec);
    cache_manifest::Json metadata = {
        {"format", "paraselect-pq-cache"}, {"dataset_name", dataset_name},
        {"n_vectors", spec.n_vectors}, {"n_dim", spec.dim}, {"n_lists", spec.n_lists},
        {"cluster_layout_id", spec.cluster_layout_id}, {"pq_type", spec.pq_type},
        {"pq_m", spec.pq_m}, {"pq_k", spec.pq_k}, {"pq_d_sub", spec.dsub()},
        {"pq_iters", spec.pq_iters}, {"trainer", spec.trainer},
        {"training", manifest.at("training")}, {"codebook_layout", "[M,K,dsub]"},
        {"codes_layout", spec.codes_layout}};
    if (manifest.contains("transform")) metadata["transform"] = manifest.at("transform");
    std::ofstream meta(path);
    meta << metadata.dump(2) << '\n';
    meta.close();
    if (!meta) throw std::runtime_error("cannot write PQ build metadata: " + path);
}

void ResidualPQCache::save(
    const PQCacheRequest& spec,
    const float* codebook,
    const uint8_t* codes) const
{
    if (cache_dir_.empty() || !codebook) return;
    const auto manifest = manifest_identity(dataset_name_, spec);
    const std::string entry_dir = cache_manifest::canonical_entry(cache_dir_, manifest);
    if (!cache_fs::ensure_dir_exists(entry_dir)) {
        std::cerr << "[PQCache] failed to create cache dir: " << entry_dir << std::endl;
        return;
    }

    const std::string meta_path = entry_dir + "/meta.json";
    const std::string codebook_path = entry_dir + "/codebook.fbin";
    const std::string codes_path = entry_dir + "/codes.u8bin";
    const std::string meta_tmp = meta_path + ".tmp";
    const std::string codebook_tmp = codebook_path + ".tmp";
    const std::string codes_tmp = codes_path + ".tmp";

    {
        FILE* f = std::fopen(codebook_tmp.c_str(), "wb");
        if (!f) return;
        const uint32_t n = (uint32_t)(spec.pq_m * spec.pq_k);
        const uint32_t dim = (uint32_t)spec.dsub();
        const size_t values = (size_t)n * dim;
        std::fwrite(&n, sizeof(uint32_t), 1, f);
        std::fwrite(&dim, sizeof(uint32_t), 1, f);
        std::fwrite(codebook, sizeof(float), values, f);
        std::fclose(f);
    }
    if (codes) {
        FILE* f = std::fopen(codes_tmp.c_str(), "wb");
        if (!f) return;
        const uint32_t n = (uint32_t)spec.n_vectors;
        const uint32_t dim = (uint32_t)spec.pq_m;
        const size_t values = (size_t)n * dim;
        std::fwrite(&n, sizeof(uint32_t), 1, f);
        std::fwrite(&dim, sizeof(uint32_t), 1, f);
        std::fwrite(codes, sizeof(uint8_t), values, f);
        std::fclose(f);
    }
    write_meta(meta_tmp, dataset_name_, spec);
    // Unpublish before replacing any payload. Incomplete/codebook-only entries are not hits.
    std::filesystem::remove(entry_dir + "/manifest.json");
    bool ok = atomic_rename(codebook_tmp, codebook_path) &&
              atomic_rename(meta_tmp, meta_path);
    if (ok && codes) {
        ok = atomic_rename(codes_tmp, codes_path);
    } else if (ok) {
        std::remove(codes_path.c_str());
    }
    if (!ok) {
        std::cerr << "[PQCache] : " << entry_dir << std::endl;
        return;
    }
    if (codes) cache_manifest::publish(entry_dir, manifest);
    std::cout << "[PQCache] : " << entry_dir << std::endl;
}

}  // namespace dataset
}  // namespace paraselect
