#include "transform/opq_cache.cuh"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace paraselect::transform {
namespace {

constexpr uint32_t kMagic = 0x3151504f;  // OPQ1
constexpr uint32_t kVersion = 1;

struct CacheHeader {
    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    int32_t n = 0;
    int32_t dim = 0;
    int32_t pq_m = 0;
    int32_t pq_k = 0;
    int32_t opq_iters = 0;
    int32_t pq_iters = 0;
    uint32_t seed = 0;
};

std::string sanitize_key(std::string key) {
    for (char& c : key) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) c = '_';
    }
    return key;
}

std::string cache_path(const std::string& cache_dir, const std::string& cache_key) {
    return cache_dir + "/" + sanitize_key(cache_key) + ".opq";
}

}  // namespace

std::string default_opq_cache_dir() {
    const char* env = std::getenv("PARASELECT_OPQ_CACHE_DIR");
    if (env && env[0] != '\0') return std::string(env);
    const char* cluster_env = std::getenv("IVF_CLUSTER_CACHE_DIR");
    if (cluster_env && cluster_env[0] != '\0') return std::string(cluster_env) + "/opq";
    return "results/opq_cache";
}

std::string make_opq_cache_key(
    const std::string& dataset_name,
    int n,
    int dim,
    const OPQTrainSpec& spec)
{
    return dataset_name +
           "__n" + std::to_string(n) +
           "__train" + std::to_string(spec.n) +
           "__d" + std::to_string(dim) +
           "__M" + std::to_string(spec.pq_m) +
           "__K" + std::to_string(spec.pq_k) +
           "__opq" + std::to_string(spec.opq_iters) +
           "__pq" + std::to_string(spec.pq_iters) +
           "__seed" + std::to_string(spec.seed);
}

bool load_opq_rotation_cache(
    const std::string& cache_dir,
    const std::string& cache_key,
    int dim,
    std::vector<float>& rotation_out)
{
    rotation_out.clear();
    const std::string path = cache_path(cache_dir, cache_key);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    CacheHeader h;
    const bool header_ok = std::fread(&h, sizeof(h), 1, f) == 1;
    if (!header_ok || h.magic != kMagic || h.version != kVersion || h.dim != dim) {
        std::fclose(f);
        return false;
    }
    std::vector<float> rotation((size_t)dim * dim);
    const bool data_ok = std::fread(rotation.data(), sizeof(float), rotation.size(), f) == rotation.size();
    std::fclose(f);
    if (!data_ok) return false;
    rotation_out = std::move(rotation);
    return true;
}

void save_opq_rotation_cache(
    const std::string& cache_dir,
    const std::string& cache_key,
    const OPQTrainSpec& spec,
    const std::vector<float>& rotation)
{
    if (spec.dim <= 0 || rotation.size() != (size_t)spec.dim * spec.dim) {
        throw std::invalid_argument("save_opq_rotation_cache: invalid rotation shape");
    }
    std::filesystem::create_directories(cache_dir);
    const std::string path = cache_path(cache_dir, cache_key);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("save_opq_rotation_cache: cannot open " + path);

    CacheHeader h;
    h.n = spec.n;
    h.dim = spec.dim;
    h.pq_m = spec.pq_m;
    h.pq_k = spec.pq_k;
    h.opq_iters = spec.opq_iters;
    h.pq_iters = spec.pq_iters;
    h.seed = spec.seed;
    const bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1 &&
                    std::fwrite(rotation.data(), sizeof(float), rotation.size(), f) == rotation.size();
    std::fclose(f);
    if (!ok) throw std::runtime_error("save_opq_rotation_cache: short write " + path);
}

}  // namespace paraselect::transform
