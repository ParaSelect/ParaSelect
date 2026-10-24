#include "cluster_cache.h"
#include "cache_manifest.h"

#include "cache_fs.h"

#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace paraselect {
namespace dataset {

namespace {

bool validate_layout(const ClusterCache::MappedResult& result) {
    const size_t nlist = result.cluster_counts.size();
    if (nlist == 0 || result.request.n_vectors < 0 ||
        result.cluster_offsets.size() != nlist + 1u ||
        result.cluster_offsets[0] != 0 ||
        result.cluster_offsets[nlist] != static_cast<idx_t>(result.request.n_vectors)) {
        return false;
    }
    for (size_t i = 0; i < nlist; ++i) {
        const idx_t begin = result.cluster_offsets[i];
        const idx_t end = result.cluster_offsets[i + 1u];
        if (end < begin || end - begin != static_cast<idx_t>(result.cluster_counts[i])) {
            return false;
        }
    }
    return true;
}


}  // namespace

ClusterCache::ClusterCache(const std::string& cache_dir) {
    if (!cache_dir.empty()) {
        cache_dir_ = cache_dir;
        return;
    }
    const char* env_dir = std::getenv("IVF_CLUSTER_CACHE_DIR");
    if (env_dir && *env_dir) {
        cache_dir_ = env_dir;
    }
}

std::string ClusterCache::compute_layout_id(const std::string& entry_dir) const {
    static constexpr const char* kFiles[] = {
        "centroids.fbin",
        "reordered_indices.ibin",
        "cluster_counts.ibin",
        "cluster_offsets.ibin",
    };
    uint64_t hash = 1469598103934665603ULL;
    auto update = [&](const void* data, size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < bytes; ++i) {
            hash ^= p[i];
            hash *= 1099511628211ULL;
        }
    };
    constexpr size_t kSampleBytes = 4096;
    unsigned char sample[kSampleBytes];
    for (const char* name : kFiles) {
        const std::string path = entry_dir + "/" + name;
        struct stat st {};
        if (::stat(path.c_str(), &st) != 0 || st.st_size <= 0) return {};
        update(name, std::strlen(name));
        update(&st.st_size, sizeof(st.st_size));
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return {};
        const size_t first_bytes = std::min(kSampleBytes, static_cast<size_t>(st.st_size));
        const ssize_t first_read = ::pread(fd, sample, first_bytes, 0);
        if (first_read != static_cast<ssize_t>(first_bytes)) {
            ::close(fd);
            return {};
        }
        update(sample, first_bytes);
        if (static_cast<size_t>(st.st_size) > first_bytes) {
            const off_t offset = st.st_size - static_cast<off_t>(first_bytes);
            const ssize_t last_read = ::pread(fd, sample, first_bytes, offset);
            if (last_read != static_cast<ssize_t>(first_bytes)) {
                ::close(fd);
                return {};
            }
            update(sample, first_bytes);
        }
        ::close(fd);
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

std::string ClusterCache::read_manifest_layout_id(const std::string& entry_dir) const {
    std::ifstream input(entry_dir + "/manifest.json");
    if (!input) return {};
    try {
        const auto manifest = cache_manifest::Json::parse(input);
        const auto field = manifest.find("cluster_layout_id");
        if (field == manifest.end() || !field->is_string()) return {};
        return field->get<std::string>();
    } catch (const cache_manifest::Json::exception&) { return {}; }
}

ClusterCache::CacheResult ClusterCache::try_load(
    const std::string& dataset_name,
    const ClusterCacheSpec& spec) {
    const auto mapped = try_map(dataset_name, spec);
    CacheResult result;
    result.request = mapped.request;
    if (!mapped.hit) return result;
    result.entry_dir = mapped.entry_dir;
    result.layout_id = mapped.layout_id;
    result.centroids.assign(mapped.centroids.data(),
        mapped.centroids.data() + mapped.centroids.size());
    result.reordered_indices.assign(mapped.reordered_indices.data(),
        mapped.reordered_indices.data() + mapped.reordered_indices.size());
    result.cluster_counts.assign(mapped.cluster_counts.data(),
        mapped.cluster_counts.data() + mapped.cluster_counts.size());
    result.cluster_offsets.assign(mapped.cluster_offsets.data(),
        mapped.cluster_offsets.data() + mapped.cluster_offsets.size());
    result.hit = true;
    return result;
}

ClusterCache::MappedResult ClusterCache::try_map(
    const std::string& dataset_name,
    const ClusterCacheSpec& spec) {
    MappedResult result;
    result.request = spec;
    if (cache_dir_.empty()) return result;
    auto identity = cache_manifest::identity("cluster", dataset_name);
    identity.update({{"n_vectors", spec.n_vectors}, {"nlist", spec.n_lists},
                     {"dim", spec.dim}, {"build", cache_manifest::Json::parse(spec.build_recipe_json)}});
    if (!spec.layout_id.empty()) identity["cluster_layout_id"] = spec.layout_id;
    result.entry_dir = cache_manifest::find(cache_dir_, identity);
    if (result.entry_dir.empty()) return result;
    const auto manifest = cache_manifest::Json::parse(std::ifstream(result.entry_dir + "/manifest.json"));
    if (manifest.at("build") != cache_manifest::Json::parse(spec.build_recipe_json))
        return MappedResult{spec};
    try {
        result.centroids = MappedArray<float>::open_binary(
            result.entry_dir + "/centroids.fbin", spec.n_lists, spec.dim);
        result.reordered_indices = MappedArray<idx_t>::open_binary(
            result.entry_dir + "/reordered_indices.ibin", spec.n_vectors, 1);
        result.cluster_counts = MappedArray<int>::open_binary(
            result.entry_dir + "/cluster_counts.ibin", spec.n_lists, 1);
        result.cluster_offsets = MappedArray<idx_t>::open_binary(
            result.entry_dir + "/cluster_offsets.ibin", spec.n_lists + 1, 1);
    } catch (const std::exception& error) {
        std::cerr << "[ClusterCache] manifest payload mismatch: " << error.what() << "\n";
        return MappedResult{spec};
    }
    if (!validate_layout(result)) return MappedResult{spec};
    result.layout_id = read_manifest_layout_id(result.entry_dir);
    if (result.layout_id.empty()) return MappedResult{spec};
    result.hit = true;
    std::cout << "[ClusterCache] manifest hit: " << result.entry_dir << "\n";
    return result;
}


std::string ClusterCache::publish_staged(
    const std::string& entry_dir,
    const std::string& dataset_name,
    const ClusterCacheSpec& spec) {
    if (spec.n_vectors <= 0 || spec.n_lists <= 0 || spec.dim <= 0)
        throw std::invalid_argument("invalid staged cluster shape");
    MappedResult staged;
    staged.request = spec;
    staged.centroids = MappedArray<float>::open_binary(
        entry_dir + "/centroids.fbin", spec.n_lists, spec.dim);
    staged.reordered_indices = MappedArray<idx_t>::open_binary(
        entry_dir + "/reordered_indices.ibin", spec.n_vectors, 1);
    staged.cluster_counts = MappedArray<int>::open_binary(
        entry_dir + "/cluster_counts.ibin", spec.n_lists, 1);
    staged.cluster_offsets = MappedArray<idx_t>::open_binary(
        entry_dir + "/cluster_offsets.ibin", spec.n_lists + 1, 1);
    if (!validate_layout(staged))
        throw std::runtime_error("invalid staged cluster offsets/counts");
    const auto layout_id = compute_layout_id(entry_dir);
    if (layout_id.empty()) throw std::runtime_error("cannot identify staged cluster layout");
    auto manifest = cache_manifest::identity("cluster", dataset_name);
    manifest.update({{"n_vectors", spec.n_vectors}, {"nlist", spec.n_lists},
        {"dim", spec.dim}, {"n_clusters_actual", spec.n_lists},
        {"build", cache_manifest::Json::parse(spec.build_recipe_json)}, {"cluster_layout_id", layout_id}});
    const auto canonical = cache_manifest::canonical_entry(cache_dir_, manifest);
    if (canonical != entry_dir) {
        if (std::filesystem::exists(canonical))
            throw std::runtime_error("cluster publication destination already exists: " + canonical);
        std::filesystem::rename(entry_dir, canonical);
    }
    cache_manifest::publish(canonical, manifest);
    return canonical;
}

void ClusterCache::save(
    const std::string& dataset_name,
    const ClusterCacheSpec& spec,
    const float* centroids,
    const idx_t* reordered_indices,
    const int* cluster_counts,
    const idx_t* cluster_offsets,
    int n_clusters) {
    if (cache_dir_.empty()) return;
    const std::string entry_dir = cache_dir_ + "/.building-" +
        std::to_string(static_cast<long long>(::getpid()));
    if (!cache_fs::ensure_dir_exists(entry_dir)) {
        std::cerr << "[ClusterCache] failed to create cache dir: " << entry_dir << std::endl;
        return;
    }

    const cache_manifest::Json metadata{{"dataset_name",dataset_name},
        {"n_lists",spec.n_lists},{"build",cache_manifest::Json::parse(spec.build_recipe_json)},
        {"n_vectors",spec.n_vectors},{"n_dim",spec.dim},{"n_clusters_actual",n_clusters}};
    std::ofstream meta_file(entry_dir + "/meta.json");
    meta_file << metadata.dump(2) << std::endl;
    meta_file.close();
    if (!meta_file) throw std::runtime_error("cannot write cluster metadata");

    FILE* f = fopen((entry_dir + "/centroids.fbin").c_str(), "wb");
    if (f) {
        uint32_t n = static_cast<uint32_t>(n_clusters);
        uint32_t dim = static_cast<uint32_t>(spec.dim);
        fwrite(&n, sizeof(uint32_t), 1, f);
        fwrite(&dim, sizeof(uint32_t), 1, f);
        fwrite(centroids, sizeof(float), static_cast<size_t>(n_clusters) * static_cast<size_t>(spec.dim), f);
        fclose(f);
    }

    f = fopen((entry_dir + "/reordered_indices.ibin").c_str(), "wb");
    if (f) {
        uint32_t n = static_cast<uint32_t>(spec.n_vectors);
        uint32_t dim = 1;
        fwrite(&n, sizeof(uint32_t), 1, f);
        fwrite(&dim, sizeof(uint32_t), 1, f);
        fwrite(reordered_indices, sizeof(idx_t), static_cast<size_t>(spec.n_vectors), f);
        fclose(f);
    }

    f = fopen((entry_dir + "/cluster_counts.ibin").c_str(), "wb");
    if (f) {
        uint32_t n = static_cast<uint32_t>(n_clusters);
        uint32_t dim = 1;
        fwrite(&n, sizeof(uint32_t), 1, f);
        fwrite(&dim, sizeof(uint32_t), 1, f);
        fwrite(cluster_counts, sizeof(int), static_cast<size_t>(n_clusters), f);
        fclose(f);
    }

    f = fopen((entry_dir + "/cluster_offsets.ibin").c_str(), "wb");
    if (f) {
        uint32_t n = static_cast<uint32_t>(n_clusters + 1);
        uint32_t dim = 1;
        fwrite(&n, sizeof(uint32_t), 1, f);
        fwrite(&dim, sizeof(uint32_t), 1, f);
        fwrite(cluster_offsets, sizeof(idx_t), static_cast<size_t>(n_clusters) + 1u, f);
        fclose(f);
    }

    const auto canonical_dir = publish_staged(entry_dir, dataset_name, spec);
    std::cout << "[ClusterCache] : " << canonical_dir << std::endl;
}

}  // namespace dataset
}  // namespace paraselect
