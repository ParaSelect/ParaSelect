#include "turboquant_cache.h"
#include "cache_manifest.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace paraselect {
namespace dataset {

namespace {

cache_manifest::Json manifest_identity(const std::string& dataset, const TurboQuantCacheRequest& spec) {
    auto identity = cache_manifest::identity("turboquant", dataset);
    identity.update({{"n_vectors", spec.n_vectors}, {"dim", spec.dim},
        {"nlist", spec.n_lists}, {"words_per_code", spec.words_per_code()},
        {"rotation_seed", spec.rotation_seed}, {"builder", spec.builder},
        {"cluster_layout_id", spec.cluster_layout_id}, {"codes_layout", spec.codes_layout}});
    return identity;
}
}  // namespace

TurboQuantCache::TurboQuantCache(std::string cache_dir, std::string dataset_name)
    : cache_dir_(std::move(cache_dir)), dataset_name_(std::move(dataset_name)) {
    if (cache_dir_.empty() || dataset_name_.empty())
        throw std::invalid_argument("TurboQuantCache requires root and dataset");
}

TurboQuantCacheMappedResult TurboQuantCache::try_map(
    const TurboQuantCacheRequest& spec) const {
    TurboQuantCacheMappedResult result;
    result.request = spec;
    const auto identity = manifest_identity(dataset_name_, spec);
    result.entry_dir = cache_manifest::find(cache_dir_, identity);
    if (result.entry_dir.empty()) return result;
    try {
        result.codes = MappedArray<uint32_t>::open_binary(
            result.entry_dir + "/codes.u32bin", spec.n_vectors, spec.words_per_code());
        result.scales = MappedArray<float>::open_binary(
            result.entry_dir + "/scales.f32bin", spec.n_vectors, 1u);
    } catch (const std::exception& error) {
        std::cerr << "[TQCache] manifest payload mismatch: " << error.what() << "\n";
        result.codes = {};
        result.scales = {};
        return result;
    }
    result.hit = true;
    std::cout << "[TQCache] manifest hit: " << result.entry_dir << "\n";
    return result;
}

void TurboQuantCache::publish_staged(const std::string& entry_dir,
    const TurboQuantCacheRequest& spec) const {
    const auto manifest = manifest_identity(dataset_name_, spec);
    const auto codes = MappedArray<uint32_t>::open_binary(
        entry_dir + "/codes.u32bin", spec.n_vectors, spec.words_per_code());
    const auto scales = MappedArray<float>::open_binary(
        entry_dir + "/scales.f32bin", spec.n_vectors, 1u);
    const std::string canonical = cache_manifest::canonical_entry(cache_dir_, manifest);
    if (entry_dir != canonical) {
        std::error_code ec;
        if (std::filesystem::exists(canonical)) {
            if (std::filesystem::exists(canonical + "/manifest.json")) {
                std::filesystem::remove_all(entry_dir, ec);
                if (ec) throw std::runtime_error("cannot discard duplicate TurboQuant build: " + ec.message());
                return;
            }
            std::filesystem::remove_all(canonical, ec);
            if (ec) throw std::runtime_error("cannot remove incomplete TurboQuant cache: " + ec.message());
        }
        std::filesystem::rename(entry_dir, canonical, ec);
        if (ec) throw std::runtime_error("cannot publish TurboQuant cache: " + ec.message());
    }
    cache_manifest::publish(canonical, manifest);
}

}  // namespace dataset
}  // namespace paraselect
