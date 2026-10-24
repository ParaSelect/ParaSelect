#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "common/test_context.cuh"
#include "src/dataset/cache/cluster_cache.h"
#include "src/dataset/cache/cache_manifest.h"
#include "src/dataset/common/dataset_spec.h"
#include "src/dataset/common/dataset_source.h"
#include "src/dataset/common/mapped_array.h"
#include "src/dataset/cache/residual_pq_cache.h"
#include "src/dataset/common/vector_buffer.h"

namespace {

template <typename T>
void write_header_bin(
    const std::filesystem::path& path,
    uint32_t n,
    uint32_t dim,
    const std::vector<T>& values);

void test_dataset_names(paraselect::test::TestContext& ctx) {
    using namespace paraselect::dataset;

    const DatasetSpec sift = describe_dataset("sift", "100M");
    ctx.check_eq(display_dataset_name(sift), std::string("SIFT100M"), "SIFT display name");
    ctx.check_eq(dataset_key_name(sift), std::string("SIFT100M-128-euclidean"), "SIFT dataset key name");
    ctx.check_eq(groundtruth_dataset_name(sift), std::string("sift100M"), "SIFT groundtruth name");
    ctx.check_eq(benchmark_dataset_name(sift), std::string("sift100M"), "SIFT benchmark name");
    ctx.check(sift.dtype == VectorDType::U8, "SIFT native dtype is u8");

    const DatasetSpec deep = describe_dataset("deep", "1000M");
    ctx.check_eq(display_dataset_name(deep), std::string("DEEP1B"), "DEEP 1000M display normalizes to 1B");
    ctx.check_eq(dataset_key_name(deep), std::string("DEEP1B-96-euclidean"), "DEEP dataset key name");
    ctx.check(deep.dtype == VectorDType::F32, "DEEP native dtype is fp32");
    ctx.check_eq(deep.n_vectors, size_t{1000000000}, "DEEP1B physical row count");

    const DatasetSpec spacev = describe_dataset("spacev", "1B");
    ctx.check_eq(display_dataset_name(spacev), std::string("SPACEV1B"), "SPACEV display name");
    ctx.check_eq(
        spacev.n_vectors, size_t{1402020720},
        "SPACEV1B uses the complete public release");

    const DatasetSpec text = describe_dataset("text", "100M");
    ctx.check_eq(display_dataset_name(text), std::string("TEXT100M"), "TEXT display name");
    ctx.check_eq(dataset_key_name(text), std::string("TEXT100M-200-euclidean"), "TEXT dataset key name");
    ctx.check_eq(groundtruth_dataset_name(text), std::string("text100M"), "TEXT groundtruth name");
}

void test_vector_buffer(paraselect::test::TestContext& ctx) {
    using namespace paraselect::dataset;
    VectorBuffer buffer(VectorDType::U8, new uint8_t[8]{}, 2, 4, [](void* p) { delete[] static_cast<uint8_t*>(p); });
    ctx.check_eq(buffer.view().bytes(), size_t{8}, "u8 buffer bytes");
    ctx.check(buffer.view().u8() != nullptr && buffer.view().fp32() == nullptr, "typed u8 view");
    VectorBuffer moved(std::move(buffer));
    ctx.check(buffer.empty(), "moved-from buffer is empty");
    ctx.check_eq(moved.view().n, size_t{2}, "move preserves shape");
}

void test_mapped_array(paraselect::test::TestContext& ctx) {
    using paraselect::dataset::MappedArray;
    const std::filesystem::path root = "/tmp/paraselect_mapped_array_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto path = root / "vectors.u8bin";
    write_header_bin(path, 4, 3, std::vector<uint8_t>{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    const auto exact = MappedArray<uint8_t>::open_binary(path, 4, 3);
    ctx.check_eq(exact.size(), size_t{12}, "strict mapping size");
    ctx.check_eq(static_cast<int>(exact[11]), 11, "strict mapping payload");

    const auto prefix = MappedArray<uint8_t>::open_binary_prefix(path, 2, 3);
    ctx.check_eq(prefix.size(), size_t{6}, "prefix mapping logical size");
    ctx.check_eq(static_cast<int>(prefix[5]), 5, "prefix mapping payload");

    bool strict_rejected_prefix = false;
    try {
        (void)MappedArray<uint8_t>::open_binary(path, 2, 3);
    } catch (const std::runtime_error&) {
        strict_rejected_prefix = true;
    }
    ctx.check(strict_rejected_prefix, "strict cache mapping rejects a larger file");

    bool oversized_prefix_rejected = false;
    try {
        (void)MappedArray<uint8_t>::open_binary_prefix(path, 5, 3);
    } catch (const std::runtime_error&) {
        oversized_prefix_rejected = true;
    }
    ctx.check(oversized_prefix_rejected, "dataset prefix cannot exceed physical rows");
    std::filesystem::remove_all(root);
}

void test_cluster_cache_query_view(paraselect::test::TestContext& ctx) {
    using paraselect::dataset::ClusterCache;
    const std::filesystem::path root = "/tmp/paraselect_cluster_cache_test";
    std::filesystem::remove_all(root);

    ClusterCache cache(root.string());
    const std::vector<float> centroids{0.0f, 1.0f, 10.0f, 11.0f};
    const std::vector<idx_t> reordered_ids{2, 0, 3, 1};
    const std::vector<int> counts{2, 2};
    const std::vector<idx_t> offsets{0, 2, 4};
    const paraselect::dataset::ClusterCacheSpec cluster_spec{4, 2, 2, R"({"family":"synthetic","revision":1})"};
    cache.save(
        "unit", cluster_spec, centroids.data(), reordered_ids.data(),
        counts.data(), offsets.data(), 2);

    const auto entry = std::filesystem::path(
        paraselect::dataset::cache_manifest::find(
            root.string(), paraselect::dataset::cache_manifest::Json{
                {"schema", "paraselect.cache"}, {"schema_version", 2},
                {"asset_type", "cluster"}, {"dataset", "unit"},
                {"nlist", 2}, {"build", {{"family","synthetic"},{"revision",1}}}}));
    ctx.check(
        !std::filesystem::exists(entry / "assignments.u32bin"),
        "cluster query cache does not contain assignments");
    const auto mapped = cache.try_map("unit", cluster_spec);
    ctx.check(mapped.hit, "cluster cache maps without assignments");
    ctx.check(!mapped.layout_id.empty(), "cluster cache exposes a layout identity");
    ctx.check_eq(mapped.reordered_indices.size(), size_t{4}, "mapped reordered ID count");
    ctx.check_eq(mapped.cluster_offsets[2], idx_t{4}, "mapped terminal offset");

    const auto centroids_path = std::filesystem::path(entry) / "centroids.fbin";
    const auto timestamp = std::filesystem::last_write_time(centroids_path);
    std::filesystem::last_write_time(centroids_path, timestamp + std::chrono::seconds(1));
    const auto remapped_after_touch = cache.try_map("unit", cluster_spec);
    ctx.check_eq(
        remapped_after_touch.layout_id, mapped.layout_id,
        "cluster layout identity does not depend on file timestamps");
    std::filesystem::remove(std::filesystem::path(entry) / "meta.json");
    const auto mapped_without_diagnostic_meta = cache.try_map("unit", cluster_spec);
    ctx.check(mapped_without_diagnostic_meta.hit,
              "cluster manifest and payload must suffice without meta.json");

    {
        std::ofstream manifest(std::filesystem::path(entry) / "manifest.json");
        manifest << "{\n  \"schema\": \"paraselect.cache\",\n"
                 << "  \"schema_version\": 2,\n"
                 << "  \"asset_type\": \"cluster\",\n"
                 << "  \"dataset\": \"unit\",\n"
                 << "  \"n_vectors\": 4,\n"
                 << "  \"nlist\": 2,\n"
                 << "  \"dim\": 2,\n"
                 << "  \"build\": {\"family\":\"synthetic\",\"revision\":1},\n"
                 << "  \"cluster_layout_id\": \"manifest-layout\"\n}\n";
    }
    const auto remapped_with_manifest = cache.try_map("unit", cluster_spec);
    ctx.check_eq(
        remapped_with_manifest.layout_id, std::string("manifest-layout"),
        "parent manifest owns cluster layout identity");

    paraselect::dataset::PQCacheRequest pq_spec;
    pq_spec.n_vectors = 4;
    pq_spec.dim = 2;
    pq_spec.n_lists = 2;
    pq_spec.cluster_layout_id = remapped_with_manifest.layout_id;
    pq_spec.pq_m = 1;
    pq_spec.pq_k = 2;
    pq_spec.pq_iters = 1;
    pq_spec.trainer = "unit";
    pq_spec.training_sampling = "full";
    paraselect::dataset::ResidualPQCache pq_cache((entry / "pq").string(), "unit");
    const std::vector<float> codebook{0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<uint8_t> codes{0, 1, 0, 1};
    pq_cache.save(pq_spec, codebook.data(), codes.data());
    ctx.check(
        pq_cache.try_map(pq_spec).hit,
        "PQ cache maps for its exact cluster layout");
    pq_spec.cluster_layout_id += "-different";
    ctx.check(
        !pq_cache.try_map(pq_spec).hit,
        "PQ cache rejects a different cluster layout");

    const auto check_rejected = [&](const char* reason) {
        const auto owned = cache.try_load("unit", cluster_spec);
        const auto invalid = cache.try_map("unit", cluster_spec);
        ctx.check(!owned.hit && !invalid.hit, reason);
        ctx.check_eq(owned.request.n_vectors, cluster_spec.n_vectors,
                     "owning miss preserves request");
        ctx.check_eq(invalid.request.n_vectors, cluster_spec.n_vectors,
                     "mapped miss preserves request");
        ctx.check(owned.centroids.empty() && invalid.centroids.size() == 0,
                  "miss exposes no partial payload");
    };
    write_header_bin(entry / "centroids.fbin", 1, 4, centroids);
    check_rejected("same-size wrong centroid shape must not hit");
    write_header_bin(entry / "centroids.fbin", 2, 2, centroids);
    write_header_bin(entry / "cluster_offsets.ibin", 3, 1,
                     std::vector<idx_t>{0, 3, 4});
    check_rejected("counts must agree with every offset interval");
    write_header_bin(entry / "cluster_offsets.ibin", 3, 1, offsets);
    std::filesystem::resize_file(entry / "reordered_indices.ibin", 8);
    check_rejected("truncated reordered IDs must not hit");

    std::filesystem::remove_all(root);
}

template <typename T>
void write_header_bin(
    const std::filesystem::path& path, uint32_t n, uint32_t dim, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}

void write_groundtruth_bin(
    const std::filesystem::path& path, uint32_t n, uint32_t k,
    const std::vector<int>& ids, const std::vector<float>& distances) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&k), sizeof(k));
    out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(int));
    out.write(reinterpret_cast<const char*>(distances.data()), distances.size() * sizeof(float));
}

template <typename T>
void write_vecs(
    const std::filesystem::path& path, uint32_t n, uint32_t dim, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    for (uint32_t row = 0; row < n; ++row) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(values.data() + static_cast<size_t>(row) * dim),
                  static_cast<size_t>(dim) * sizeof(T));
    }
}

template <typename T>
std::vector<T> make_vectors(uint32_t n, uint32_t dim) {
    std::vector<T> values(static_cast<size_t>(n) * dim);
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t column = 0; column < dim; ++column) {
            values[static_cast<size_t>(row) * dim + column] =
                static_cast<T>(row * 10 + column % 10);
        }
    }
    return values;
}

void set_dataset_paths(
    const std::string& dataset_name,
    const std::filesystem::path& root,
    const std::filesystem::path& base,
    const std::filesystem::path& query,
    const std::filesystem::path& ids,
    const std::filesystem::path& distances) {
    const std::string prefix = "PARASELECT_" +
        paraselect::dataset::to_upper_ascii(dataset_name);
    setenv((prefix + "_ROOT").c_str(), root.c_str(), 1);
    setenv((prefix + "_BASE").c_str(), base.c_str(), 1);
    setenv((prefix + "_QUERY").c_str(), query.c_str(), 1);
    setenv((prefix + "_GT_1M").c_str(), ids.c_str(), 1);
    setenv((prefix + "_GT_DIST_1M").c_str(), distances.c_str(), 1);
}

void verify_source(
    paraselect::test::TestContext& ctx,
    const std::string& dataset_name,
    paraselect::dataset::VectorDType dtype,
    int dim,
    int expected_first = 20,
    int expected_second = 0) {
    using namespace paraselect::dataset;
    DatasetSpec spec = describe_dataset(dataset_name, "1M");
    spec.n_vectors = 3;
    const DatasetSource source(std::move(spec));
    ctx.check(source.spec().dtype == dtype, "source uses registered dtype");
    ctx.check_eq(source.base_info().n, size_t{3}, "base info row count");
    ctx.check_eq(source.query_info().n, size_t{2}, "query info row count");

    const VectorBuffer base = source.load_base();
    ctx.check_eq(base.view().n, size_t{3}, "base materialize row count");
    ctx.check_eq(base.view().dim, dim, "base materialize dim");
    const VectorBuffer query = source.load_queries(1);
    ctx.check_eq(query.view().n, size_t{1}, "query limit");

    const VectorBuffer rows = source.load_base_rows({2, 0});
    if (dtype == VectorDType::U8) {
        ctx.check_eq(static_cast<int>(rows.view().u8()[0]), expected_first, "u8 selected row order");
        ctx.check_eq(static_cast<int>(rows.view().u8()[dim]), expected_second, "u8 selected second row");
    } else {
        ctx.check_eq(rows.view().fp32()[0], 20.0f, "fp32 selected row order");
        ctx.check_eq(rows.view().fp32()[dim], 0.0f, "fp32 selected second row");
    }

    const GroundTruth gt = source.load_groundtruth(1);
    ctx.check_eq(gt.n, size_t{1}, "groundtruth query limit");
    ctx.check_eq(gt.k, 2, "groundtruth width");
    ctx.check_eq(gt.ids[0], 2, "groundtruth ids");
    ctx.check_eq(gt.distances[1], 0.2f, "groundtruth distances");
}

void test_dataset_source(paraselect::test::TestContext& ctx) {
    using namespace paraselect::dataset;
    const std::filesystem::path root = "/tmp/paraselect_dataset_source_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const std::vector<int> gt_ids{2, 1, 0, 2};
    const std::vector<float> gt_distances{0.1f, 0.2f, 0.3f, 0.4f};

    const auto sift_base = root / "sift_base.bvecs";
    const auto sift_query = root / "sift_query.bvecs";
    const auto sift_ids = root / "sift.ivecs";
    const auto sift_distances = root / "sift.fvecs";
    write_vecs(sift_base, 3, 128, make_vectors<uint8_t>(3, 128));
    write_vecs(sift_query, 2, 128, make_vectors<uint8_t>(2, 128));
    write_vecs(sift_ids, 2, 2, gt_ids);
    write_vecs(sift_distances, 2, 2, gt_distances);
    set_dataset_paths("sift", root, sift_base, sift_query, sift_ids, sift_distances);
    verify_source(ctx, "sift", VectorDType::U8, 128);

    DatasetSpec oversized = describe_dataset("sift", "1M");
    oversized.n_vectors = 4;
    bool short_base_rejected = false;
    try {
        (void)DatasetSource(std::move(oversized)).base_info();
    } catch (const std::runtime_error&) {
        short_base_rejected = true;
    }
    ctx.check(short_base_rejected, "base shorter than logical dataset is rejected");

    const auto sift_u8bin_base = root / "sift_base.u8bin";
    write_header_bin(sift_u8bin_base, 3, 128, make_vectors<uint8_t>(3, 128));
    set_dataset_paths("sift", root, sift_u8bin_base, sift_query, sift_ids, sift_distances);
    verify_source(ctx, "sift", VectorDType::U8, 128);

    const auto unknown_base = root / "sift_base.unknown";
    write_header_bin(unknown_base, 3, 128, make_vectors<uint8_t>(3, 128));
    set_dataset_paths("sift", root, unknown_base, sift_query, sift_ids, sift_distances);
    bool unknown_rejected = false;
    try {
        const DatasetSource unknown_source(describe_dataset("sift", "1M"));
        (void)unknown_source.base_info();
    } catch (const std::runtime_error&) {
        unknown_rejected = true;
    }
    ctx.check(unknown_rejected, "unknown vector extension is rejected");

    const auto deep_base = root / "deep.fbin";
    const auto deep_query = root / "deep_query.fbin";
    const auto deep_ids = root / "groundtruth_deep.bin";
    const auto deep_distances = root / "unused.fbin";
    write_header_bin(deep_base, 3, 96, make_vectors<float>(3, 96));
    write_header_bin(deep_query, 2, 96, make_vectors<float>(2, 96));
    write_groundtruth_bin(deep_ids, 2, 2, gt_ids, gt_distances);
    set_dataset_paths("deep", root, deep_base, deep_query, deep_ids, deep_distances);
    verify_source(ctx, "deep", VectorDType::F32, 96);

    const auto spacev_base = root / "spacev.i8bin";
    const auto spacev_query = root / "spacev_query.i8bin";
    const auto spacev_ids = root / "spacev.ibin";
    const auto spacev_distances = root / "spacev_gt.fbin";
    auto spacev_base_values = make_vectors<int8_t>(3, 100);
    auto spacev_query_values = make_vectors<int8_t>(2, 100);
    spacev_base_values[0] = -128;
    spacev_base_values[1] = -1;
    spacev_base_values[2] = 0;
    spacev_base_values[3] = 127;
    write_header_bin(spacev_base, 3, 100, spacev_base_values);
    write_header_bin(spacev_query, 2, 100, spacev_query_values);
    write_header_bin(spacev_ids, 2, 2, gt_ids);
    write_header_bin(spacev_distances, 2, 2, gt_distances);
    set_dataset_paths("spacev", root, spacev_base, spacev_query, spacev_ids, spacev_distances);
    verify_source(ctx, "spacev", VectorDType::U8, 100, 148, 0);
    DatasetSpec spacev_spec = describe_dataset("spacev", "1M");
    spacev_spec.n_vectors = 3;
    const DatasetSource spacev_source(std::move(spacev_spec));
    ctx.check(spacev_source.base_has_signed_byte_encoding(),
              "spacev base exposes signed byte encoding");
    const VectorBuffer normalized = spacev_source.load_base();
    ctx.check_eq(static_cast<int>(normalized.view().u8()[0]), 0, "i8 minimum maps to u8 minimum");
    ctx.check_eq(static_cast<int>(normalized.view().u8()[1]), 127, "i8 negative maps below midpoint");
    ctx.check_eq(static_cast<int>(normalized.view().u8()[2]), 128, "i8 zero maps to u8 midpoint");
    ctx.check_eq(static_cast<int>(normalized.view().u8()[3]), 255, "i8 maximum maps to u8 maximum");
    std::filesystem::remove_all(root);
}

} // namespace

int main() {
    paraselect::test::TestContext ctx("dataset_spec", paraselect::test::TestKind::Unit);
    ctx.run_case("dataset_names", [&] { test_dataset_names(ctx); });
    ctx.run_case("vector_buffer", [&] { test_vector_buffer(ctx); });
    ctx.run_case("mapped_array", [&] { test_mapped_array(ctx); });
    ctx.run_case("cluster_cache_query_view", [&] { test_cluster_cache_query_view(ctx); });
    ctx.run_case("dataset_source", [&] { test_dataset_source(ctx); });
    return ctx.exit_code();
}
