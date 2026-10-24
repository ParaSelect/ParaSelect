#include "search/ivf_common/file/ivf_file_rerank.h"

#include "search/ivf_common/ivf_exact_topk.h"
#include "cpu/distance/distance_dispatch.h"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/persistent_thread_pool.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace paraselect::search {

namespace {

struct BlockCandidateRef {
    uint64_t block = 0;
    int query = 0;
    int id = 0;
    int pq_rank = 0;
};

struct BlockRange {
    uint64_t block = 0;
    size_t candidate_begin = 0;
    size_t candidate_end = 0;
    size_t bytes = 0;
    int best_pq_rank = 0;
};

struct LoadedBlockWindow {
    size_t block_begin = 0;
    size_t block_end = 0;
    std::vector<paraselect::search::IVFFileReader::Block> loaded;
};

struct FileThreadState {
    std::vector<float> heap_distances;
    std::vector<int> heap_ids;
    std::vector<int> heap_sizes;
    double distance_ms = 0.0;
    long long cache_hits = 0;
    long long cache_misses = 0;
    long long bytes_read = 0;
    long long bytes_reused = 0;
    long long distance_evaluations = 0;
};

struct FilePoolState { std::vector<FileThreadState> threads; };

FilePoolState& state_for(int threads) {
    static FilePoolState state;
    if (static_cast<int>(state.threads.size()) != threads) state.threads.resize(threads);
    return state;
}

std::mutex& run_mutex() {
    static std::mutex value;
    return value;
}

}  // namespace

template <bool UseFloat>
void rerank_file_impl(
    paraselect::search::IVFFileReader& reader,
    const IVFFileStorageView& storage,
    const uint8_t* queries_u8,
    const float* queries_fp32,
    const int* candidates,
    int batch_n,
    int candidate_count,
    int topk,
    int* output_ids,
    float* output_distances,
    const IVFFileBlockSelection& block_selection,
    IVFRerankStats* stats) {
    std::lock_guard<std::mutex> guard(run_mutex());
    const size_t read_block_size = reader.block_size();
    const auto plan_begin = std::chrono::steady_clock::now();
    std::vector<BlockCandidateRef> refs;
    refs.reserve(static_cast<size_t>(batch_n) * candidate_count);
    for (int query = 0; query < batch_n; ++query) {
        for (int slot = 0; slot < candidate_count; ++slot) {
            const int id = candidates[static_cast<size_t>(query) * candidate_count + slot];
            if (id < 0 || id >= storage.count) continue;
            const uint64_t byte = static_cast<uint64_t>(id) * storage.row_stride;
            refs.push_back({byte / read_block_size, query, id, slot});
        }
    }
    std::sort(refs.begin(), refs.end(), [](const BlockCandidateRef& left, const BlockCandidateRef& right) {
        if (left.block != right.block) return left.block < right.block;
        if (left.query != right.query) return left.query < right.query;
        if (left.id != right.id) return left.id < right.id;
        return left.pq_rank < right.pq_rank;
    });
    refs.erase(
        std::unique(refs.begin(), refs.end(), [](const BlockCandidateRef& left, const BlockCandidateRef& right) {
            return left.block == right.block && left.query == right.query && left.id == right.id;
        }),
        refs.end());

    std::vector<BlockRange> considered_block_ranges;
    std::vector<uint64_t> considered_block_ids;
    considered_block_ranges.reserve(refs.size());
    considered_block_ids.reserve(refs.size());
    long long blocks_considered = 0;
    for (size_t begin = 0; begin < refs.size();) {
        size_t end = begin + 1;
        while (end < refs.size() && refs[end].block == refs[begin].block) ++end;
        ++blocks_considered;
        int best_pq_rank = refs[begin].pq_rank;
        for (size_t ref = begin + 1; ref < end; ++ref) {
            best_pq_rank = std::min(best_pq_rank, refs[ref].pq_rank);
        }
        const size_t candidate_refs = end - begin;
        if (stats) {
            if (stats->block_candidate_ref_histogram.size() <= candidate_refs) {
                stats->block_candidate_ref_histogram.resize(candidate_refs + 1, 0);
            }
            ++stats->block_candidate_ref_histogram[candidate_refs];
        }
        const size_t block_offset = static_cast<size_t>(refs[begin].block) * read_block_size;
        const size_t payload_bytes = static_cast<size_t>(storage.count) * storage.row_stride;
        const size_t useful_bytes = std::min(read_block_size, payload_bytes - block_offset);
        const size_t direct_bytes =
            (useful_bytes + storage.block_size - 1) / storage.block_size * storage.block_size;
        considered_block_ranges.push_back({
            refs[begin].block, begin, end, direct_bytes, best_pq_rank});
        considered_block_ids.push_back(refs[begin].block);
        begin = end;
    }
    const std::vector<uint8_t> initially_cached =
        reader.contains_many(considered_block_ids);

    std::vector<BlockRange> blocks;
    std::vector<BlockCandidateRef> selected_refs;
    selected_refs.reserve(refs.size());
    long long blocks_skipped = 0;
    for (size_t block_index = 0;
         block_index < considered_block_ranges.size();
         ++block_index) {
        const BlockRange& candidate = considered_block_ranges[block_index];
        const size_t candidate_refs = candidate.candidate_end - candidate.candidate_begin;
        const bool cached = initially_cached[block_index] != 0;
        const bool keep = ivf_file_block_selected(
            block_selection,
            cached,
            candidate_refs,
            candidate.best_pq_rank);
        if (!keep) {
            ++blocks_skipped;
            continue;
        }
        const size_t selected_begin = selected_refs.size();
        selected_refs.insert(
            selected_refs.end(),
            refs.begin() + candidate.candidate_begin,
            refs.begin() + candidate.candidate_end);
        blocks.push_back({
            candidate.block,
            selected_begin,
            selected_refs.size(),
            candidate.bytes,
            candidate.best_pq_rank});
    }
    refs.swap(selected_refs);
    const auto plan_end = std::chrono::steady_clock::now();

    const int thread_count =
        std::max(1, std::min<int>(
            paraselect::resolve_cpu_fine_thread_count(),
            std::max<size_t>(1, blocks.size())));
    const auto l2 = [] {
        if constexpr (UseFloat) {
            return paraselect::cpu_distance::select_fp32_l2_fn();
        } else {
            return paraselect::cpu_distance::select_u8_l2_fn();
        }
    }();
    FilePoolState& state = state_for(thread_count);
    paraselect::PersistentThreadPool& pool =
        paraselect::persistent_thread_pool_for_threads(thread_count);
    for (int tid = 0; tid < thread_count; ++tid) {
        FileThreadState& local = state.threads[tid];
        local.heap_distances.resize(static_cast<size_t>(batch_n) * topk);
        local.heap_ids.resize(static_cast<size_t>(batch_n) * topk);
        local.heap_sizes.assign(batch_n, 0);
        local.distance_ms = 0.0;
        local.cache_hits = local.cache_misses = 0;
        local.bytes_read = local.bytes_reused = 0;
        local.distance_evaluations = 0;
    }

    const size_t max_window_blocks = reader.capacity_blocks() == 0
        ? std::max<size_t>(
              1,
              static_cast<size_t>(reader.ring_count()) *
                  static_cast<size_t>(reader.queue_depth_per_ring()))
        : std::max<size_t>(1, reader.capacity_blocks() / 2);
    std::vector<std::pair<size_t, size_t>> windows;
    for (size_t begin = 0; begin < blocks.size();) {
        const size_t end = std::min(blocks.size(), begin + max_window_blocks);
        windows.push_back({begin, end});
        begin = end;
    }

    auto load_window = [&](size_t block_begin, size_t block_end) {
        std::vector<paraselect::search::IVFFileReader::Request> requests;
        requests.reserve(block_end - block_begin);
        for (size_t i = block_begin; i < block_end; ++i) {
            requests.push_back({blocks[i].block, blocks[i].bytes});
        }
        LoadedBlockWindow window;
        window.block_begin = block_begin;
        window.block_end = block_end;
        window.loaded = reader.read(requests);
        return window;
    };

    double read_ms = 0.0;
    std::future<LoadedBlockWindow> pending;
    if (!windows.empty()) {
        const auto [begin, end] = windows.front();
        pending = std::async(std::launch::async, load_window, begin, end);
    }
    for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
        const auto wait_begin = std::chrono::steady_clock::now();
        LoadedBlockWindow current = pending.get();
        read_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_begin).count();
        if (window_index + 1 < windows.size()) {
            const auto [begin, end] = windows[window_index + 1];
            pending = std::async(std::launch::async, load_window, begin, end);
        }

        pool.run([&](int tid) {
            FileThreadState& local = state.threads[tid];
            const size_t window_blocks = current.block_end - current.block_begin;
            const size_t begin = window_blocks * tid / thread_count;
            const size_t end = window_blocks * (tid + 1) / thread_count;
            for (size_t local_block = begin; local_block < end; ++local_block) {
                const size_t block_index = current.block_begin + local_block;
                const BlockRange& range = blocks[block_index];
                const auto& block = current.loaded[local_block];
                if (block.reused()) {
                    ++local.cache_hits;
                    local.bytes_reused += static_cast<long long>(range.bytes);
                } else {
                    ++local.cache_misses;
                    local.bytes_read += static_cast<long long>(range.bytes);
                }

                const auto distance_begin = std::chrono::steady_clock::now();
                for (size_t ref_index = range.candidate_begin;
                     ref_index < range.candidate_end;
                     ++ref_index) {
                    const BlockCandidateRef& ref = refs[ref_index];
                    const int query_id = ref.query;
                    float* heap_distances =
                        local.heap_distances.data() + static_cast<size_t>(query_id) * topk;
                    int* heap_ids = local.heap_ids.data() + static_cast<size_t>(query_id) * topk;
                    int& heap_size = local.heap_sizes[query_id];
                    const size_t byte_offset =
                        static_cast<size_t>(ref.id) * storage.row_stride -
                        static_cast<size_t>(range.block) * read_block_size;
                    float distance;
                    if constexpr (UseFloat) {
                        const float* query =
                            queries_fp32 + static_cast<size_t>(query_id) * storage.dim;
                        const float* vector = reinterpret_cast<const float*>(
                            block.data() + byte_offset);
                        distance = l2(query, vector, storage.dim);
                    } else {
                        const uint8_t* query =
                            queries_u8 + static_cast<size_t>(query_id) * storage.dim;
                        distance = static_cast<float>(
                            l2(query, block.data() + byte_offset, storage.dim));
                    }
                    ivf_insert_exact_topk(
                        distance,
                        ref.id,
                        topk,
                        heap_distances,
                        heap_ids,
                        heap_size);
                    ++local.distance_evaluations;
                }
                local.distance_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - distance_begin).count();
            }
        });
    }

    const auto topk_begin = std::chrono::steady_clock::now();
    const int topk_threads = std::max(1, std::min(thread_count, batch_n));
    paraselect::persistent_thread_pool_for_threads(topk_threads).run([&](int tid) {
        std::vector<float> heap_distances(topk);
        std::vector<int> heap_ids(topk);
        const int query_begin = batch_n * tid / topk_threads;
        const int query_end = batch_n * (tid + 1) / topk_threads;
        for (int query = query_begin; query < query_end; ++query) {
            int heap_size = 0;
            for (const FileThreadState& local : state.threads) {
                const int local_size = local.heap_sizes[query];
                const float* local_distances =
                    local.heap_distances.data() + static_cast<size_t>(query) * topk;
                const int* local_ids = local.heap_ids.data() + static_cast<size_t>(query) * topk;
                for (int i = 0; i < local_size; ++i) {
                    ivf_insert_exact_topk(
                        local_distances[i], local_ids[i], topk,
                        heap_distances.data(), heap_ids.data(), heap_size);
                }
            }
            ivf_finalize_exact_topk(
                heap_distances.data(), heap_ids.data(), heap_size, topk);
            std::copy_n(
                heap_ids.data(), topk,
                output_ids + static_cast<size_t>(query) * topk);
            std::copy_n(
                heap_distances.data(), topk,
                output_distances + static_cast<size_t>(query) * topk);
        }
    });
    const auto topk_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->plan_ms = std::chrono::duration<double, std::milli>(
            plan_end - plan_begin).count();
        stats->topk_ms = std::chrono::duration<double, std::milli>(
            topk_end - topk_begin).count();
        stats->unique_blocks = static_cast<long long>(blocks.size());
        stats->blocks_considered = blocks_considered;
        stats->blocks_skipped = blocks_skipped;
        stats->read_block_bytes = static_cast<long long>(read_block_size);
        stats->threads = thread_count;
        stats->io_rings = reader.ring_count();
        stats->io_depth_per_ring = reader.queue_depth_per_ring();
        stats->read_ms = read_ms;
        for (const FileThreadState& local : state.threads) {
            stats->distance_ms = std::max(stats->distance_ms, local.distance_ms);
            stats->cache_hits += local.cache_hits;
            stats->cache_misses += local.cache_misses;
            stats->bytes_read += local.bytes_read;
            stats->bytes_reused += local.bytes_reused;
            stats->distance_evaluations += local.distance_evaluations;
        }
        stats->avg_valid_candidates =
            static_cast<double>(stats->distance_evaluations) / batch_n;
    }
}

void ivf_file_rerank(
    paraselect::search::IVFFileReader& reader,
    const IVFFileStorageView& storage,
    const uint8_t* queries_u8,
    const float* queries_fp32,
    const int* candidates,
    int batch_n,
    int candidate_count,
    int topk,
    int* output_ids,
    float* output_distances,
    const IVFFileBlockSelection& block_selection,
    IVFRerankStats* stats) {
    if (storage.fd < 0 || !candidates || !output_ids || !output_distances ||
        batch_n <= 0 || candidate_count <= 0 || topk <= 0) {
        throw std::invalid_argument("invalid file IVF rerank request");
    }
    switch (storage.dtype) {
        case IVFExactRerankDType::U8:
            if (!queries_u8) {
                throw std::invalid_argument("u8 file rerank requires u8 queries");
            }
            rerank_file_impl<false>(
                reader, storage, queries_u8, nullptr, candidates, batch_n,
                candidate_count, topk, output_ids, output_distances,
                block_selection, stats);
            return;
        case IVFExactRerankDType::FP32:
            if (!queries_fp32) {
                throw std::invalid_argument("fp32 file rerank requires fp32 queries");
            }
            rerank_file_impl<true>(
                reader, storage, nullptr, queries_fp32, candidates, batch_n,
                candidate_count, topk, output_ids, output_distances,
                block_selection, stats);
            return;
        case IVFExactRerankDType::Disabled:
            throw std::invalid_argument("file rerank dtype is disabled");
    }
    throw std::invalid_argument("unknown file rerank dtype");
}

void ivf_file_rerank_prewarm(const IVFFileStorageView& storage) {
    std::lock_guard<std::mutex> guard(run_mutex());
    const int threads = std::max(1, paraselect::resolve_cpu_fine_thread_count());
    (void)state_for(threads);
    paraselect::persistent_thread_pool_for_threads(threads).run([](int) {});
}

}  // namespace paraselect::search
