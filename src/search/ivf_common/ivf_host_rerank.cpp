#include "search/ivf_common/ivf_host_rerank.h"

#include "search/ivf_common/ivf_rerank_policy.h"
#include "search/ivf_common/ivf_exact_topk.h"
#include "cpu/distance/distance_dispatch.h"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/persistent_thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace {

using paraselect::search::IVFHostStorageView;

struct ThreadState {
    std::vector<int> candidates;
    std::vector<int> dedup_table;
    std::vector<int> dedup_slots;
    std::vector<float> heap_distances;
    std::vector<int> heap_ids;
    bool pinned = false;
    long long valid = 0;
    long long gap_count = 0;
    long long gap_sum = 0;
    long long max_gap = 0;
    double plan_ms = 0.0;
    double distance_ms = 0.0;
    double topk_ms = 0.0;
    double other_ms = 0.0;
};

struct PoolState { std::vector<ThreadState> threads; };

PoolState& state_for(int threads) {
    static PoolState state;
    if (static_cast<int>(state.threads.size()) != threads) state.threads.resize(threads);
    return state;
}

std::mutex& run_mutex() {
    static std::mutex value;
    return value;
}

}  // namespace

void ivf_rerank_host(
    const IVFHostStorageView& storage,
    const uint8_t* queries_u8,
    const float* queries_fp32,
    bool use_float,
    const int* candidates,
    int batch_n,
    int candidate_count,
    int candidate_stride,
    int topk,
    int* output_ids,
    float* output_distances,
    int max_threads,
    bool pin_threads,
    int prefetch_ahead,
    paraselect::search::IVFRerankStats* stats,
    const int* candidate_counts,
    const int* candidate_order,
    idx_t candidate_order_count) {
    if (!candidates || !output_ids || !output_distances || batch_n <= 0 ||
        candidate_count <= 0 || candidate_stride < candidate_count || topk <= 0 ||
        storage.count <= 0 || storage.dim <= 0) {
        throw std::invalid_argument("invalid host IVF rerank request");
    }
    if ((use_float && (!storage.fp32 || !queries_fp32)) ||
        (!use_float && (!storage.u8 || !queries_u8))) {
        throw std::invalid_argument("host IVF rerank dtype does not match storage");
    }
    if (candidate_counts) {
        for (int q = 0; q < batch_n; ++q) {
            if (candidate_counts[q] < 0 || candidate_counts[q] > candidate_count)
                throw std::invalid_argument("host rerank candidate count exceeds row width");
        }
    }
    if (candidate_order_count < 0 || (candidate_order_count && !candidate_order))
        throw std::invalid_argument("invalid candidate row mapping size");
    const idx_t scan_rows = candidate_order_count ? candidate_order_count : storage.count;
    if (candidate_order && storage.source_rows)
        throw std::invalid_argument("cannot restore candidates before reordered host storage");
    if (stats) *stats = paraselect::search::IVFRerankStats{};

    const int available = std::max(1, max_threads > 0 ? max_threads : omp_get_max_threads());
    const int thread_count = paraselect::search::ivf_rerank_thread_count(available, batch_n);
    if (prefetch_ahead < 0) {
        throw std::invalid_argument("host rerank prefetch distance must be nonnegative");
    }
    const bool pin = pin_threads;
    const int prefetch = prefetch_ahead;
    // Pipeline rerank passes a null stats pointer. Keep all per-candidate
    // clocks out of that hot path; diagnostic callers opt in by requesting
    // stats explicitly.
    const bool collect_breakdown = stats != nullptr;
    const auto fp32_l2 = paraselect::cpu_distance::select_fp32_l2_fn();
    const auto fp32_pf = paraselect::cpu_distance::select_fp32_prefetch_fn();
    const auto u8_l2 = paraselect::cpu_distance::select_u8_l2_fn();
    const auto u8_pf = paraselect::cpu_distance::select_u8_prefetch_fn();

    std::lock_guard<std::mutex> guard(run_mutex());
    PoolState& state = state_for(thread_count);
    for (ThreadState& local : state.threads) {
        local.valid = local.gap_count = local.gap_sum = local.max_gap = 0;
        local.plan_ms = local.distance_ms = local.topk_ms = local.other_ms = 0.0;
    }
    paraselect::PersistentThreadPool& pool = paraselect::persistent_thread_pool_for_threads(thread_count);
    pool.run([&](int tid) {
        ThreadState& local = state.threads[tid];
        if (pin && !local.pinned) {
            paraselect::pin_thread_to_physical_core(tid);
            local.pinned = true;
        }
        local.candidates.reserve(candidate_count);
        // Order-preserving open-addressing dedup: preserves the GPU candidate
        // order (identical top-k to the previous sort+unique path) without the
        // O(n log n) sort of the random-access candidate list.
        size_t dedup_size = 1;
        while (dedup_size < 2 * static_cast<size_t>(candidate_count)) dedup_size <<= 1;
        if (local.dedup_table.size() != dedup_size) {
            local.dedup_table.assign(dedup_size, -1);
        }
        local.dedup_slots.reserve(candidate_count);
        local.heap_distances.resize(topk);
        local.heap_ids.resize(topk);
        const int q_begin = batch_n * tid / thread_count;
        const int q_end = batch_n * (tid + 1) / thread_count;
        for (int q = q_begin; q < q_end; ++q) {
            local.candidates.clear();
            const int* query_candidates = candidates + static_cast<size_t>(q) * candidate_stride;
            const int count = candidate_counts ? candidate_counts[q] : candidate_count;
            if (candidate_order) {
                for (int i = 0; i < count; ++i) {
                    const int scan_id = query_candidates[i];
                    if (scan_id < 0) continue;
                    if (scan_id >= scan_rows)
                        throw std::runtime_error("scan candidate exceeds row order");
                    const int id = candidate_order[scan_id];
                    if (id >= 0 && id < storage.count) local.candidates.push_back(id);
                }
            } else for (int i = 0; i < count; ++i) {
                const int id = query_candidates[i];
                if (id >= 0 && id < storage.count) local.candidates.push_back(id);
            }
            const auto plan_begin = collect_breakdown
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            {
                const size_t mask = local.dedup_table.size() - 1;
                size_t out = 0;
                for (size_t idx = 0; idx < local.candidates.size(); ++idx) {
                    const int id = local.candidates[idx];
                    size_t slot =
                        (static_cast<size_t>(static_cast<uint32_t>(id)) * 2654435761u) & mask;
                    while (local.dedup_table[slot] != -1 && local.dedup_table[slot] != id) {
                        slot = (slot + 1) & mask;
                    }
                    if (local.dedup_table[slot] == id) continue;
                    local.dedup_table[slot] = id;
                    local.dedup_slots.push_back(static_cast<int>(slot));
                    local.candidates[out++] = id;
                }
                local.candidates.resize(out);
                for (int slot : local.dedup_slots) local.dedup_table[slot] = -1;
                local.dedup_slots.clear();
            }
            if (collect_breakdown) {
                local.plan_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - plan_begin).count();
            }
            local.valid += static_cast<long long>(local.candidates.size());
            for (size_t i = 1; i < local.candidates.size(); ++i) {
                const long long gap = local.candidates[i] - local.candidates[i - 1];
                local.gap_count++;
                local.gap_sum += gap;
                local.max_gap = std::max(local.max_gap, gap);
            }

            int heap_size = 0;
            auto insert = [&](float distance, int row) {
                // Most candidates cannot enter the heap: avoid their ID gather.
                if (heap_size == topk && distance > local.heap_distances[0]) return;
                const int id = storage.source_rows ? storage.source_rows[row] : row;
                paraselect::search::ivf_insert_exact_topk(
                    distance, id, topk, local.heap_distances.data(),
                    local.heap_ids.data(), heap_size);
            };
            const uint8_t* query_u8 = queries_u8 ? queries_u8 + static_cast<size_t>(q) * storage.dim : nullptr;
            const float* query_fp32 = queries_fp32 ? queries_fp32 + static_cast<size_t>(q) * storage.dim : nullptr;
            const size_t candidate_total = local.candidates.size();
            size_t i = 0;
            // Batched distance computation: issue four independent random row
            // reads before any heap update so the core overlaps four misses per
            // thread. The rerank is latency bound, so this is the main MLP knob.
            for (; i + 4 <= candidate_total; i += 4) {
                if (prefetch > 0) {
                    const size_t prefetch_begin = i + 4 + static_cast<size_t>(prefetch);
                    for (size_t j = 0; j < 4 && prefetch_begin + j < candidate_total; ++j) {
                        const int id = local.candidates[prefetch_begin + j];
                        if (use_float) fp32_pf(storage.fp32 + static_cast<size_t>(id) * storage.dim, storage.dim);
                        else u8_pf(storage.u8 + static_cast<size_t>(id) * storage.dim, storage.dim);
                    }
                }
                const int id0 = local.candidates[i + 0];
                const int id1 = local.candidates[i + 1];
                const int id2 = local.candidates[i + 2];
                const int id3 = local.candidates[i + 3];
                const auto distance_begin = collect_breakdown
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                const float d0 = use_float
                    ? fp32_l2(query_fp32, storage.fp32 + static_cast<size_t>(id0) * storage.dim, storage.dim)
                    : static_cast<float>(u8_l2(query_u8, storage.u8 + static_cast<size_t>(id0) * storage.dim, storage.dim));
                const float d1 = use_float
                    ? fp32_l2(query_fp32, storage.fp32 + static_cast<size_t>(id1) * storage.dim, storage.dim)
                    : static_cast<float>(u8_l2(query_u8, storage.u8 + static_cast<size_t>(id1) * storage.dim, storage.dim));
                const float d2 = use_float
                    ? fp32_l2(query_fp32, storage.fp32 + static_cast<size_t>(id2) * storage.dim, storage.dim)
                    : static_cast<float>(u8_l2(query_u8, storage.u8 + static_cast<size_t>(id2) * storage.dim, storage.dim));
                const float d3 = use_float
                    ? fp32_l2(query_fp32, storage.fp32 + static_cast<size_t>(id3) * storage.dim, storage.dim)
                    : static_cast<float>(u8_l2(query_u8, storage.u8 + static_cast<size_t>(id3) * storage.dim, storage.dim));
                if (collect_breakdown) {
                    local.distance_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - distance_begin).count();
                }
                const auto topk_begin = collect_breakdown
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                insert(d0, id0);
                insert(d1, id1);
                insert(d2, id2);
                insert(d3, id3);
                if (collect_breakdown) {
                    local.topk_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - topk_begin).count();
                }
            }
            for (; i < candidate_total; ++i) {
                if (prefetch > 0 &&
                    i + static_cast<size_t>(prefetch) < candidate_total) {
                    const int id = local.candidates[i + static_cast<size_t>(prefetch)];
                    if (use_float) fp32_pf(storage.fp32 + static_cast<size_t>(id) * storage.dim, storage.dim);
                    else u8_pf(storage.u8 + static_cast<size_t>(id) * storage.dim, storage.dim);
                }
                const int id = local.candidates[i];
                const auto distance_begin = collect_breakdown
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                const float distance = use_float
                    ? fp32_l2(query_fp32, storage.fp32 + static_cast<size_t>(id) * storage.dim, storage.dim)
                    : static_cast<float>(u8_l2(query_u8, storage.u8 + static_cast<size_t>(id) * storage.dim, storage.dim));
                if (collect_breakdown) {
                    local.distance_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - distance_begin).count();
                }
                const auto topk_begin = collect_breakdown
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                insert(distance, id);
                if (collect_breakdown) {
                    local.topk_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - topk_begin).count();
                }
            }
            const auto topk_begin = collect_breakdown
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            paraselect::search::ivf_finalize_exact_topk(
                local.heap_distances.data(), local.heap_ids.data(), heap_size, topk);
            if (collect_breakdown) {
                local.topk_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - topk_begin).count();
            }
            const auto output_begin = collect_breakdown
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            std::copy_n(local.heap_ids.data(), topk, output_ids + static_cast<size_t>(q) * topk);
            std::copy_n(local.heap_distances.data(), topk, output_distances + static_cast<size_t>(q) * topk);
            if (collect_breakdown) {
                local.other_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - output_begin).count();
            }
        }
    });

    if (stats) {
        long long valid = 0, gap_count = 0, gap_sum = 0, max_gap = 0;
        for (const ThreadState& local : state.threads) {
            valid += local.valid;
            gap_count += local.gap_count;
            gap_sum += local.gap_sum;
            max_gap = std::max(max_gap, local.max_gap);
            stats->plan_ms = std::max(stats->plan_ms, local.plan_ms);
            stats->distance_ms = std::max(stats->distance_ms, local.distance_ms);
            stats->topk_ms = std::max(stats->topk_ms, local.topk_ms);
            stats->other_ms = std::max(stats->other_ms, local.other_ms);
        }
        stats->avg_valid_candidates = static_cast<double>(valid) / batch_n;
        stats->avg_adjacent_gap = gap_count ? static_cast<double>(gap_sum) / gap_count : 0.0;
        stats->max_adjacent_gap = static_cast<double>(max_gap);
        stats->threads = thread_count;
        stats->pinned = pin ? 1 : 0;
    }
}

void ivf_prewarm_host_rerank(
    int max_threads, int batch_n, bool pin_threads) {
    const int available = std::max(1, max_threads > 0 ? max_threads : omp_get_max_threads());
    const int thread_count = paraselect::search::ivf_rerank_thread_count(available, batch_n);
    std::lock_guard<std::mutex> guard(run_mutex());
    PoolState& state = state_for(thread_count);
    const bool pin = pin_threads;
    paraselect::persistent_thread_pool_for_threads(thread_count).run([&](int tid) {
        if (pin && !state.threads[tid].pinned) {
            paraselect::pin_thread_to_physical_core(tid);
            state.threads[tid].pinned = true;
        }
    });
}
