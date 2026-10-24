#include "clustering/kmeans/kmeans_cache_recipe.h"
#include "index/ivf_cache.h"
#include "index/ivf_cache/kmeans.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace paraselect;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class F>
void expect_required_miss(F call) {
    try { call(); }
    catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("required cache is missing or incompatible:") != 0)
            throw std::runtime_error(std::string("cache miss bypassed REQUIRE; actual error: ") + e.what());
        return;
    }
    throw std::runtime_error("REQUIRE accepted missing cache");
}

int main() {
    char pattern[] = "/tmp/paraselect-kmeans-policy-XXXXXX";
    const char* directory = ::mkdtemp(pattern);
    if (!directory) return 1;
    const std::filesystem::path root(directory);
    try {
        const auto base = root / "base.u8bin";
        // Header only: REQUIRE must reject the miss without loading the payload.
        { std::ofstream out(base, std::ios::binary);
          const uint32_t shape[] = {2, 128};
          out.write(reinterpret_cast<const char*>(shape), sizeof(shape)); }
        setenv("PARASELECT_SIFT_BASE", base.c_str(), 1);
        setenv("PARASELECT_SIFT_QUERY", base.c_str(), 1);
        setenv("PARASELECT_SIFT_GT_1M", base.c_str(), 1);
        const auto cache_root = root / "cache";
        setenv("IVF_CLUSTER_CACHE_DIR", cache_root.c_str(), 1);
        auto spec = dataset::describe_dataset("sift", "1M");
        spec.n_vectors = 2;
        dataset::DatasetSource source(spec);
        KMeansSpec kmeans;
        for (const char* policy : {"require", "default"}) {
            if (std::string(policy) == "default") unsetenv("PARASELECT_CACHE_POLICY");
            else setenv("PARASELECT_CACHE_POLICY", policy, 1);
            expect_required_miss([&] { ivf_cache::ensure_kmeans(source, 2, 1, kmeans); });
            expect_required_miss([&] { ivf_cache::ensure_kmeans_mapped(source, 2, 1, kmeans); });
            require(!std::filesystem::exists(cache_root), "REQUIRE wrote cache files");
        }
        dataset::ClusterCache cache(cache_root.string());
        const std::vector<float> centroids(128, 0.0f);
        const idx_t reordered[] = {0, 1}, offsets[] = {0, 2};
        const int counts[] = {2};
        cache.save(dataset::dataset_key_name(spec),
                   {2, 1, 128, make_kmeans_cache_recipe(kmeans)},
                   centroids.data(), reordered, counts, offsets, 1);
        for (const char* policy : {"require", "build"}) {
            setenv("PARASELECT_CACHE_POLICY", policy, 1);
            require(ivf_cache::ensure_kmeans(source, 2, 1, kmeans).hit, "owning hit rejected");
            require(ivf_cache::ensure_kmeans_mapped(source, 2, 1, kmeans).hit, "mapped hit rejected");
        }
        std::filesystem::remove_all(root);
        std::cout << "[test_kmeans_cache_policy] passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(root);
        std::cerr << "[test_kmeans_cache_policy] " << e.what() << '\n';
        return 1;
    }
}
