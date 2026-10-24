#include "dataset/cache/pca_cache.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_cache_round_trip() {
    std::string pattern = "/tmp/paraselect-pca-cache-XXXXXX";
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    char* directory = ::mkdtemp(path.data());
    if (!directory) throw std::runtime_error("failed to create PCA cache test directory");
    const std::filesystem::path root(directory);
    try {
        paraselect::dataset::PCACacheSpec spec;
        spec.n_vectors = 4;
        spec.source_dim = 3;
        spec.dim = 2;
        spec.train_n = 4;
        spec.cluster_layout_id = "layout-a";

        paraselect::dataset::PCACache missing((root / "missing").string(), "synthetic");
        require(!missing.try_map(spec).hit,
                "missing cache root must return a miss so BUILD can proceed");
        paraselect::dataset::PCACache cache(root.string(), "synthetic");
        auto writer = cache.create(spec);
        const int8_t vectors[] = {1, 2, 3, 4, 5, 6, 7, 8};
        std::copy(std::begin(vectors), std::end(vectors), writer.vectors());
        writer.commit(
            {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
            {-1.0f, -2.0f},
            {0.25f, 0.5f});

        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_directory()) {
                std::filesystem::rename(entry.path(), root / "renamed-asset");
                break;
            }
        }
        const auto mapped = cache.try_map(spec);
        require(mapped.hit, "new PCA cache should map");
        require(std::equal(std::begin(vectors), std::end(vectors), mapped.vectors.data()),
                "mapped PCA vectors mismatch");
        require(mapped.projection.size() == 6 && mapped.bias.size() == 2 &&
                    mapped.scales.size() == 2,
                "mapped PCA model shape mismatch");

        auto wrong_layout = spec;
        wrong_layout.cluster_layout_id = "layout-b";
        require(!cache.try_map(wrong_layout).hit,
                "PCA cache must be bound to the IVF layout");
        auto wrong_quantization = spec;
        wrong_quantization.quantization = "different-encoding";
        require(!cache.try_map(wrong_quantization).hit,
                "PCA cache must match quantization");
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_cache_round_trip();
    } catch (const std::exception& error) {
        std::cerr << "[test_pca_cache] failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "[test_pca_cache] passed" << std::endl;
    return 0;
}
