#include "dataset/common/dataset_source.h"
#include "dataset/temporary_exact_store.h"
#include "cpu/distance/distance_dispatch.h"
#include "search/ivf_common/ivf_exact_topk.h"
#include "search/ivf_common/file/ivf_file_rerank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace {

using paraselect::search::IVFExactRerankDType;
using paraselect::search::IVFFileReadConfig;
using paraselect::search::IVFFileStorageView;

void write_header_u8(
    const std::filesystem::path& path,
    uint32_t count,
    uint32_t dim,
    const std::vector<uint8_t>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    output.write(reinterpret_cast<const char*>(values.data()), values.size());
}

void write_header_fp32(
    const std::filesystem::path& path,
    uint32_t count,
    uint32_t dim,
    const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
}

bool empty_directory(const std::filesystem::path& path) {
    return std::filesystem::directory_iterator(path) == std::filesystem::directory_iterator();
}

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

template <typename Value>
void exact_rerank_reference(
    const Value* storage,
    const Value* queries,
    int count,
    int dim,
    const int* candidates,
    int batch,
    int candidate_count,
    int topk,
    int* output_ids,
    float* output_distances) {
    const auto distance = [] {
        if constexpr (std::is_same_v<Value, float>) {
            return paraselect::cpu_distance::select_fp32_l2_fn();
        } else {
            return paraselect::cpu_distance::select_u8_l2_fn();
        }
    }();
    std::vector<int> unique_candidates;
    std::vector<int> heap_ids(topk);
    std::vector<float> heap_distances(topk);
    for (int query = 0; query < batch; ++query) {
        unique_candidates.clear();
        for (int slot = 0; slot < candidate_count; ++slot) {
            const int id = candidates[static_cast<size_t>(query) * candidate_count + slot];
            if (id >= 0 && id < count) unique_candidates.push_back(id);
        }
        std::sort(unique_candidates.begin(), unique_candidates.end());
        unique_candidates.erase(
            std::unique(unique_candidates.begin(), unique_candidates.end()),
            unique_candidates.end());
        int heap_size = 0;
        for (const int id : unique_candidates) {
            const auto raw_distance = distance(
                queries + static_cast<size_t>(query) * dim,
                storage + static_cast<size_t>(id) * dim,
                dim);
            paraselect::search::ivf_insert_exact_topk(
                static_cast<float>(raw_distance), id, topk,
                heap_distances.data(), heap_ids.data(), heap_size);
        }
        paraselect::search::ivf_finalize_exact_topk(
            heap_distances.data(), heap_ids.data(), heap_size, topk);
        std::copy_n(heap_ids.data(), topk, output_ids + static_cast<size_t>(query) * topk);
        std::copy_n(
            heap_distances.data(), topk,
            output_distances + static_cast<size_t>(query) * topk);
    }
}

int test_fp32_file_rerank(
    const std::filesystem::path& root,
    const std::filesystem::path& store_dir) {
    constexpr uint32_t count = 2048;
    constexpr uint32_t dim = 96;
    std::vector<float> source(static_cast<size_t>(count) * dim);
    for (uint32_t row = 0; row < count; ++row) {
        for (uint32_t column = 0; column < dim; ++column) {
            source[static_cast<size_t>(row) * dim + column] =
                static_cast<float>(row) * 0.03125f +
                static_cast<float>(column) * 0.0078125f;
        }
    }

    const auto base_path = root / "base.fbin";
    const auto query_path = root / "query.fbin";
    const auto gt_path = root / "gt.bin";
    write_header_fp32(base_path, count, dim, source);
    write_header_fp32(query_path, 1, dim, std::vector<float>(dim));
    std::ofstream(gt_path, std::ios::binary).put('\0');
    ::setenv("PARASELECT_DEEP_BASE", base_path.c_str(), 1);
    ::setenv("PARASELECT_DEEP_QUERY", query_path.c_str(), 1);
    ::setenv("PARASELECT_DEEP_GT_1M", gt_path.c_str(), 1);

    paraselect::dataset::DatasetSpec spec =
        paraselect::dataset::describe_dataset("deep", "1M");
    spec.n_vectors = count;
    paraselect::dataset::DatasetSource dataset(std::move(spec));

    std::vector<idx_t> reordered(count);
    std::vector<float> host(static_cast<size_t>(count) * dim);
    for (uint32_t row = 0; row < count; ++row) {
        const uint32_t source_id = (row * 17) % count;
        reordered[row] = source_id;
        std::copy_n(
            source.data() + static_cast<size_t>(source_id) * dim,
            dim,
            host.data() + static_cast<size_t>(row) * dim);
    }

    paraselect::dataset::TemporaryExactStoreOptions options;
    options.directory = store_dir.string();
    options.source_chunk_rows = 13;
    auto store = paraselect::dataset::TemporaryExactStore::build(
        dataset, reordered.data(), count, dim, options);
    const IVFFileStorageView file = store.view();
    if (file.dtype != IVFExactRerankDType::FP32 || file.row_stride != 512) {
        return fail("fp32 exact store did not use the expected dtype and stride");
    }
    if (!empty_directory(store_dir)) {
        return fail("fp32 temporary file remained visible after open");
    }

    void* block = nullptr;
    if (::posix_memalign(&block, file.block_size, file.block_size) != 0 || !block) {
        return fail("could not allocate fp32 direct-read block");
    }
    const ssize_t received = ::pread(
        file.fd, block, file.block_size, static_cast<off_t>(file.payload_offset));
    if (received != static_cast<ssize_t>(file.block_size)) {
        std::free(block);
        return fail("fp32 direct block read failed");
    }
    for (size_t row = 0; row < file.block_size / file.row_stride; ++row) {
        const auto* stored = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(block) + row * file.row_stride);
        if (!std::equal(stored, stored + dim, host.data() + row * dim)) {
            std::free(block);
            return fail("fp32 reordered payload differs from expected permutation");
        }
        const uint8_t* padding = reinterpret_cast<const uint8_t*>(stored + dim);
        const size_t padding_bytes = file.row_stride - dim * sizeof(float);
        for (size_t i = 0; i < padding_bytes; ++i) {
            if (padding[i] != 0) {
                std::free(block);
                return fail("fp32 row padding is not zero initialized");
            }
        }
    }
    std::free(block);

    constexpr int batch = 3;
    constexpr int candidate_count = 12;
    constexpr int topk = 5;
    std::vector<float> queries(static_cast<size_t>(batch) * dim);
    std::copy_n(source.data() + 7 * dim, dim, queries.data());
    std::copy_n(source.data() + 31 * dim, dim, queries.data() + dim);
    std::copy_n(source.data() + 52 * dim, dim, queries.data() + 2 * dim);
    const std::vector<int> candidates = {
        0, 7, 16, 31, 127, 255, 511, 1023, 1535, 2047, 0, -1,
        1, 8, 17, 32, 128, 256, 512, 1024, 1536, 2000, 8, -1,
        2, 9, 18, 33, 129, 257, 513, 1025, 1537, 1900, 2, -1,
    };
    std::vector<int> host_ids(batch * topk), file_ids(batch * topk);
    std::vector<float> host_distances(batch * topk), file_distances(batch * topk);
    exact_rerank_reference(
        host.data(), queries.data(), count, dim,
        candidates.data(), batch, candidate_count, topk, host_ids.data(),
        host_distances.data());

    IVFFileReadConfig read_config;
    read_config.read_block_bytes = file.block_size;
    read_config.ring_count = 2;
    read_config.queue_depth_per_ring = 8;
    read_config.cache_capacity_bytes = 32 * file.block_size;
    auto reader = paraselect::search::make_ivf_file_reader(file, read_config);
    paraselect::search::IVFRerankStats stats;
    paraselect::search::ivf_file_rerank(
        *reader, file, nullptr, queries.data(), candidates.data(), batch,
        candidate_count, topk, file_ids.data(), file_distances.data(), {}, &stats);
    if (host_ids != file_ids) return fail("fp32 file rerank IDs differ from host rerank");
    for (size_t i = 0; i < host_distances.size(); ++i) {
        if (host_distances[i] != file_distances[i]) {
            return fail("fp32 file rerank distance differs from host rerank");
        }
    }
    if (stats.distance_evaluations <= 0 || stats.cache_misses <= 0) {
        return fail("fp32 file rerank did not execute file distances");
    }

    IVFFileReadConfig no_cache_config = read_config;
    no_cache_config.cache_capacity_bytes = 0;
    auto no_cache_reader = paraselect::search::make_ivf_file_reader(
        file, no_cache_config);
    if (no_cache_reader->capacity_blocks() != 0 ||
        no_cache_reader->contains(0) ||
        no_cache_reader->stats().prefill_bytes != 0) {
        return fail("fp32 no-cache reader exposed cache state");
    }
    paraselect::search::IVFRerankStats no_cache_stats;
    paraselect::search::ivf_file_rerank(
        *no_cache_reader, file, nullptr, queries.data(), candidates.data(), batch,
        candidate_count, topk, file_ids.data(), file_distances.data(), {},
        &no_cache_stats);
    if (host_ids != file_ids) return fail("fp32 no-cache IDs differ from host rerank");
    for (size_t i = 0; i < host_distances.size(); ++i) {
        if (host_distances[i] != file_distances[i]) {
            return fail("fp32 no-cache distance differs from host rerank");
        }
    }
    const auto no_cache_reader_stats = no_cache_reader->stats();
    if (no_cache_reader_stats.capacity_bytes != 0 ||
        no_cache_reader_stats.prefill_bytes != 0 ||
        no_cache_reader_stats.hits != 0 ||
        no_cache_reader_stats.misses == 0) {
        return fail("fp32 no-cache reader reported invalid statistics");
    }
    paraselect::search::ivf_file_rerank(
        *no_cache_reader, file, nullptr, queries.data(), candidates.data(), batch,
        candidate_count, topk, file_ids.data(), file_distances.data(), {}, nullptr);
    const auto repeated_no_cache_stats = no_cache_reader->stats();
    if (repeated_no_cache_stats.hits != 0 ||
        repeated_no_cache_stats.misses != 2 * no_cache_reader_stats.misses) {
        return fail("fp32 no-cache reader reused blocks across calls");
    }
    return 0;
}

}  // namespace

int main() {
    constexpr uint32_t count = 10240;
    constexpr uint32_t dim = 128;
    const std::filesystem::path root =
        std::filesystem::path(".tmp") / ("ivf-file-rerank-unit-" + std::to_string(::getpid()));
    const std::filesystem::path store_dir = root / "store";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(store_dir);

    std::vector<uint8_t> source(static_cast<size_t>(count) * dim);
    for (uint32_t row = 0; row < count; ++row) {
        for (uint32_t column = 0; column < dim; ++column) {
            uint8_t value = static_cast<uint8_t>((row * 13 + column * 7) & 255);
            if (column == 0) value = static_cast<uint8_t>(row & 255);
            if (column == 1) value = static_cast<uint8_t>((row >> 8) & 255);
            source[static_cast<size_t>(row) * dim + column] = value;
        }
    }
    const auto base_path = root / "base.u8bin";
    const auto query_path = root / "query.u8bin";
    const auto gt_path = root / "gt.ivecs";
    write_header_u8(base_path, count, dim, source);
    write_header_u8(query_path, 1, dim, std::vector<uint8_t>(dim));
    write_header_u8(gt_path, 1, 1, std::vector<uint8_t>(4));
    ::setenv("PARASELECT_SIFT_BASE", base_path.c_str(), 1);
    ::setenv("PARASELECT_SIFT_QUERY", query_path.c_str(), 1);
    ::setenv("PARASELECT_SIFT_GT_1M", gt_path.c_str(), 1);

    paraselect::dataset::DatasetSpec spec =
        paraselect::dataset::describe_dataset("sift", "1M");
    spec.n_vectors = count;
    paraselect::dataset::DatasetSource dataset(std::move(spec));

    std::vector<idx_t> reordered(count);
    std::vector<uint8_t> host(static_cast<size_t>(count) * dim);
    for (uint32_t row = 0; row < count; ++row) {
        const uint32_t source_id = (row * 17) % count;
        reordered[row] = source_id;
        std::copy_n(
            source.data() + static_cast<size_t>(source_id) * dim,
            dim,
            host.data() + static_cast<size_t>(row) * dim);
    }

    {
        paraselect::dataset::TemporaryExactStoreOptions options;
        options.directory = store_dir.string();
        options.source_chunk_rows = 11;
        auto store = paraselect::dataset::TemporaryExactStore::build(
            dataset, reordered.data(), count, dim, options);
        const IVFFileStorageView file = store.view();
        if (!empty_directory(store_dir)) return fail("temporary file remained visible after open");

        void* block = nullptr;
        if (::posix_memalign(&block, file.block_size, file.block_size) != 0 || !block) {
            return fail("could not allocate aligned direct-read block");
        }
        const ssize_t received = ::pread(
            file.fd, block, file.block_size, static_cast<off_t>(file.payload_offset));
        if (received != static_cast<ssize_t>(file.block_size)) {
            std::free(block);
            return fail("direct block read failed");
        }
        if (!std::equal(host.begin(), host.begin() + file.block_size, static_cast<uint8_t*>(block))) {
            std::free(block);
            return fail("reordered file payload differs from expected permutation");
        }
        std::free(block);

        constexpr int batch = 3;
        constexpr int candidate_count = 12;
        constexpr int topk = 5;
        std::vector<uint8_t> queries(static_cast<size_t>(batch) * dim);
        std::copy_n(source.data() + 7 * dim, dim, queries.data());
        std::copy_n(source.data() + 31 * dim, dim, queries.data() + dim);
        std::copy_n(source.data() + 52 * dim, dim, queries.data() + 2 * dim);
        std::vector<int> candidates = {
            0, 1024, 2048, 8192, 9216, 7, 2055, 8500, 9500, 0, -1, 4,
            1, 1100, 2200, 8300, 9300, 3, 2250, 8600, 9600, 9300, 9, 1,
            2, 1200, 2300, 8400, 9400, 6, 2350, 8700, 9700, 2, -1, 9400,
        };
        std::vector<int> host_ids(batch * topk), file_ids(batch * topk);
        std::vector<float> host_distances(batch * topk), file_distances(batch * topk);
        paraselect::search::IVFRerankStats file_stats;
        IVFFileReadConfig read_config;
        read_config.read_block_bytes = 128 * 1024;
        read_config.ring_count = 1;
        read_config.queue_depth_per_ring = 1;
        read_config.cache_capacity_bytes = 8 * read_config.read_block_bytes;
        auto reader = paraselect::search::make_ivf_file_reader(
            file, read_config);
        const auto initial_cache = reader->contains_many({0, 1, 2, 8, 9});
        if (initial_cache != std::vector<uint8_t>({1, 1, 1, 0, 0})) {
            return fail("file rerank cache probe did not distinguish hit and miss blocks");
        }
        exact_rerank_reference(
            host.data(), queries.data(), count, dim,
            candidates.data(), batch, candidate_count, topk, host_ids.data(),
            host_distances.data());
        paraselect::search::ivf_file_rerank(
            *reader, file, queries.data(), nullptr, candidates.data(), batch, candidate_count, topk,
            file_ids.data(), file_distances.data(), {}, &file_stats);
        if (host_ids != file_ids) return fail("file rerank IDs differ from host rerank");
        for (size_t i = 0; i < host_distances.size(); ++i) {
            if (std::abs(host_distances[i] - file_distances[i]) > 0.0f) {
                return fail("file rerank distance differs from host rerank");
            }
        }
        std::set<std::pair<int, int>> unique_candidates;
        for (int query = 0; query < batch; ++query) {
            for (int slot = 0; slot < candidate_count; ++slot) {
                const int id = candidates[static_cast<size_t>(query) * candidate_count + slot];
                if (id >= 0) unique_candidates.emplace(query, id);
            }
        }
        const long long unique_valid_candidates =
            static_cast<long long>(unique_candidates.size());
        if (file_stats.unique_blocks != 5 || file_stats.threads <= 0 ||
            file_stats.distance_evaluations != unique_valid_candidates) {
            return fail("file rerank did not report I/O statistics");
        }
        if (file_stats.cache_hits != 3 || file_stats.cache_misses != 2 ||
            file_stats.bytes_read <= 0 || file_stats.bytes_reused <= 0) {
            return fail("file rerank did not stream mixed cache hits and misses");
        }
        if (file_stats.read_block_bytes !=
            static_cast<long long>(read_config.read_block_bytes)) {
            return fail("file rerank reported the wrong read block size");
        }
        paraselect::search::IVFRerankStats reused_stats;
        paraselect::search::ivf_file_rerank(
            *reader, file, queries.data(), nullptr, candidates.data(), batch, candidate_count, topk,
            file_ids.data(), file_distances.data(), {}, &reused_stats);
        if (reused_stats.cache_hits != 5 || reused_stats.cache_misses != 0 ||
            reused_stats.bytes_read != 0 || reused_stats.bytes_reused <= 0) {
            return fail("file rerank did not reuse cached blocks");
        }

        IVFFileReadConfig multi_ring_config;
        multi_ring_config.read_block_bytes = file.block_size;
        multi_ring_config.ring_count = 4;
        multi_ring_config.queue_depth_per_ring = 8;
        multi_ring_config.cache_capacity_bytes = 24 * file.block_size;
        auto multi_ring_reader = paraselect::search::make_ivf_file_reader(
            file, multi_ring_config);
        std::vector<paraselect::search::IVFFileReader::Request> requests;
        for (uint64_t block_id = 24; block_id < 40; ++block_id) {
            requests.push_back({block_id, file.block_size});
        }
        const auto loaded = multi_ring_reader->read(requests);
        if (loaded.size() != requests.size()) {
            return fail("multi-ring reader returned the wrong block count");
        }
        for (size_t i = 0; i < loaded.size(); ++i) {
            const size_t offset = static_cast<size_t>(requests[i].block) * file.block_size;
            if (!std::equal(
                    host.begin() + offset,
                    host.begin() + offset + file.block_size,
                    loaded[i].data())) {
                return fail("multi-ring reader data differs from reordered storage");
            }
        }
        const auto multi_ring_stats = multi_ring_reader->stats();
        if (multi_ring_stats.misses != requests.size() ||
            multi_ring_stats.bytes_read != requests.size() * file.block_size) {
            return fail("multi-ring reader reported the wrong I/O counters");
        }
    }

    if (!empty_directory(store_dir)) return fail("temporary file remained after owner destruction");
    if (const int result = test_fp32_file_rerank(root, store_dir); result != 0) {
        return result;
    }
    if (!empty_directory(store_dir)) {
        return fail("fp32 temporary file remained after owner destruction");
    }
    std::filesystem::remove_all(root);
    return 0;
}
