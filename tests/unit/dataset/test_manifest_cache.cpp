#include "dataset/cache/residual_pq_cache.h"
#include "dataset/cache/cache_manifest.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace paraselect::dataset;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    char pattern[] = "/tmp/paraselect-manifest-XXXXXX";
    const std::filesystem::path root(mkdtemp(pattern));
    try {
        PQCacheRequest spec;
        spec.n_vectors = 2; spec.dim = 2; spec.n_lists = 1;
        spec.pq_m = 1; spec.pq_k = 2; spec.pq_iters = 25;
        spec.cluster_layout_id = "parent-a";
        spec.training_sampling = "prefix";
        spec.training_rows = 2;
        ResidualPQCache cache(root.string(), "synthetic");
        const float book[] = {1,2,3,4};
        const uint8_t codes[] = {0,1};
        cache.save(spec, book, codes);
        const auto saved = std::filesystem::path(
            cache_manifest::find(root.string(),
                cache_manifest::identity("residual_pq", "synthetic")));
        check(!saved.empty(), "manifest lookup must find compactly named asset");
        const auto published = cache_manifest::Json::parse(std::ifstream(saved / "manifest.json"));
        check(!published.contains("train_tag"), "training identity must be structured");
        check(published.at("training").at("rows") == 2 &&
              published.at("training").at("sampling") == "prefix", "typed training recipe missing");
        auto different_training = spec; different_training.training_rows = 1;
        check(!cache.try_map(different_training).hit, "different training corpus must not match");
        const auto entry = root / "renamed" / "nested";
        std::filesystem::create_directories(entry.parent_path());
        std::filesystem::rename(saved, entry);
        check(cache.try_map(spec).hit, "renamed PQ must map");
        check(cache.try_load(spec).hit,
              "owned PQ load must ignore caller diagnostic tag");
        auto meta = cache_manifest::Json::parse(std::ifstream(entry / "meta.json"));
        meta["algo_tag"] = "renamed-diagnostic-tag";
        std::ofstream(entry / "meta.json") << meta.dump();
        check(cache.try_load(spec).hit,
              "metadata JSON whitespace must not reject a manifest hit");
        std::filesystem::remove(entry / "meta.json");
        const auto owned = cache.try_load(spec);
        check(owned.hit && owned.codebook == std::vector<float>(book, book + 4) &&
                  owned.codes == std::vector<uint8_t>(codes, codes + 2),
              "manifest and payload must suffice for owned PQ loading");
        check(cache.try_map(spec).hit,
              "owned and mapped loaders must agree without diagnostic metadata");
        std::ofstream(entry / "meta.json") << meta.dump(2);
        auto wrong = spec; wrong.cluster_layout_id = "parent-b";
        check(!cache.try_map(wrong).hit, "PQ must reject different parent layout");
        wrong = spec; wrong.pq_iters = 2;
        check(!cache.try_map(wrong).hit, "PQ must compare numbers exactly");
        wrong = spec; wrong.pq_type = "plain";
        check(!cache.try_map(wrong).hit, "plain and residual PQ must not collide");
        wrong = spec; wrong.codes_layout = "source-row";
        check(!cache.try_map(wrong).hit, "PQ must reject different code ordering");
        auto manifest = cache_manifest::Json::parse(std::ifstream(entry / "manifest.json"));
        std::ofstream(entry / "manifest.json") << manifest.dump();
        check(cache.try_map(spec).hit, "JSON whitespace must not affect matching");
        manifest["pq_iters"] = "25";
        std::ofstream(entry / "manifest.json") << manifest.dump();
        check(!cache.try_map(spec).hit, "string must not match integer");
        manifest["pq_iters"] = 25;
        std::ofstream(entry / "manifest.json") << manifest.dump();
        const auto fixture = root / "fixture";
        std::filesystem::copy(entry, fixture, std::filesystem::copy_options::recursive);
        const auto building = root / ".building-test";
        std::filesystem::copy(entry, building, std::filesystem::copy_options::recursive);
        std::filesystem::remove(building / "manifest.json");
        std::filesystem::remove_all(entry);
        cache.publish_staged(building.string(), spec);
        const auto canonical = cache_manifest::canonical_entry(
            root.string(), cache_manifest::Json::parse(std::ifstream(fixture / "manifest.json")));
        check(std::filesystem::exists(canonical) &&
                  std::filesystem::path(canonical).filename().string().rfind("asset-", 0) == 0,
              "temporary PQ build must publish to canonical asset");
        check(!std::filesystem::exists(building), "temporary PQ directory must be removed after publish");
        std::filesystem::remove_all(canonical);
        std::filesystem::copy(fixture, root / "duplicate", std::filesystem::copy_options::recursive);
        bool ambiguous = false;
        try { cache.try_map(spec); }
        catch (const std::runtime_error&) { ambiguous = true; }
        check(ambiguous, "ambiguity must not be treated as a miss and rebuilt");
        std::filesystem::remove_all(root / "duplicate");
        std::filesystem::resize_file(fixture / "codes.u8bin", 8);
        check(!cache.try_map(spec).hit, "truncated payload must not hit");
        check(!ResidualPQCache((root / "missing").string(), "synthetic").try_map(spec).hit,
              "absent root must be a miss");
        std::filesystem::remove_all(root);
        std::cout << "[test_manifest_cache] passed\n";
    } catch (const std::exception& e) {
        std::filesystem::remove_all(root);
        std::cerr << e.what() << '\n'; return 1;
    }
}
