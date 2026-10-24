#include "quantization/pca/pca_storage.h"
#include "search/ivf_common/ivf_pca_refine.h"

#include "cpu/utils/persistent_thread_pool.h"
#include "search/ivf_common/ivf_rerank_policy.h"
#include "search/ivf_common/ivf_exact_topk.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace paraselect::search {
namespace {

struct ThreadState {
    std::vector<int> candidates;
    std::vector<float> heap_distances;
    std::vector<int> heap_ids;
};

struct PoolState {
    std::vector<ThreadState> threads;
    std::vector<float> projected_queries;
    std::vector<long long> valid_counts;
};

PoolState& state_for(int threads) {
    static PoolState state;
    if (static_cast<int>(state.threads.size()) != threads) state.threads.resize(threads);
    return state;
}

std::mutex& run_mutex() {
    static std::mutex value;
    return value;
}

int thread_count_for(int max_threads, int batch_n) {
    const int available = std::max(
        1, max_threads > 0 ? max_threads : omp_get_max_threads());
    return ivf_rerank_thread_count(available, batch_n);
}

void validate_cpu_storage(const paraselect::quantization::PCAStorageView& storage) {
    if (!storage.vectors || !storage.projection || !storage.bias ||
        !storage.scales || storage.count <= 0 || storage.source_dim <= 0 ||
        storage.dim <= 0 || storage.dim > storage.source_dim) {
        throw std::invalid_argument("invalid CPU IVF PCA storage");
    }
    for (int d = 0; d < storage.dim; ++d) {
        if (!(storage.scales[d] > 0.0f) || !std::isfinite(storage.scales[d]) ||
            !std::isfinite(storage.bias[d])) {
            throw std::invalid_argument("IVF PCA storage has invalid quantization data");
        }
    }
}

}  // namespace

void ivf_pca_refine_cpu(
    const paraselect::quantization::PCAStorageView& storage,
    const float* queries,
    const int* candidates,
    int batch_n,
    int candidate_stride,
    int output_count,
    int max_threads,
    int* output_candidates,
    IVFPCARefineStats* stats) {
    validate_cpu_storage(storage);
    if (!queries || !candidates || !output_candidates || batch_n <= 0 ||
        candidate_stride <= 0 || output_count <= 0 || output_count > candidate_stride) {
        throw std::invalid_argument("invalid CPU IVF PCA refinement request");
    }

    const int threads = thread_count_for(max_threads, batch_n);
    std::lock_guard<std::mutex> guard(run_mutex());
    PoolState& state = state_for(threads);
    state.projected_queries.resize(static_cast<size_t>(batch_n) * storage.dim);
    state.valid_counts.assign(static_cast<size_t>(threads), 0);
    PersistentThreadPool& pool = persistent_thread_pool_for_threads(threads);

    const auto transform_begin = std::chrono::steady_clock::now();
    pool.run([&](int tid) {
        const int query_begin = batch_n * tid / threads;
        const int query_end = batch_n * (tid + 1) / threads;
        for (int query = query_begin; query < query_end; ++query) {
            const float* input = queries + static_cast<size_t>(query) * storage.source_dim;
            float* output = state.projected_queries.data() +
                static_cast<size_t>(query) * storage.dim;
            for (int out_dim = 0; out_dim < storage.dim; ++out_dim) {
                const float* projection = storage.projection +
                    static_cast<size_t>(out_dim) * storage.source_dim;
                float value = storage.bias[out_dim];
                for (int source_dim = 0; source_dim < storage.source_dim; ++source_dim) {
                    value += projection[source_dim] * input[source_dim];
                }
                output[out_dim] = value;
            }
        }
    });
    const auto transform_end = std::chrono::steady_clock::now();

    const auto select_begin = std::chrono::steady_clock::now();
    pool.run([&](int tid) {
        ThreadState& local = state.threads[tid];
        local.candidates.reserve(candidate_stride);
        local.heap_distances.resize(output_count);
        local.heap_ids.resize(output_count);
        long long local_valid = 0;
        const int query_begin = batch_n * tid / threads;
        const int query_end = batch_n * (tid + 1) / threads;
        for (int query = query_begin; query < query_end; ++query) {
            local.candidates.clear();
            const int* input_candidates =
                candidates + static_cast<size_t>(query) * candidate_stride;
            for (int slot = 0; slot < candidate_stride; ++slot) {
                const int id = input_candidates[slot];
                if (id == -1) continue;
                if (id < 0 || id >= storage.count) {
                    throw std::out_of_range("CPU IVF PCA candidate is outside storage");
                }
                local.candidates.push_back(id);
            }
            std::sort(local.candidates.begin(), local.candidates.end());
            if (std::adjacent_find(local.candidates.begin(), local.candidates.end()) !=
                local.candidates.end()) {
                throw std::invalid_argument("CPU IVF PCA candidates must be unique");
            }
            local_valid += static_cast<long long>(local.candidates.size());

            const float* projected_query = state.projected_queries.data() +
                static_cast<size_t>(query) * storage.dim;
            int heap_size = 0;
            for (int id : local.candidates) {
                const int8_t* vector = storage.vectors +
                    static_cast<size_t>(id) * storage.dim;
                float distance = 0.0f;
                for (int d = 0; d < storage.dim; ++d) {
                    const float value = static_cast<float>(vector[d]) * storage.scales[d];
                    const float delta = projected_query[d] - value;
                    distance += delta * delta;
                }
                ivf_insert_exact_topk(
                    distance, id, output_count,
                    local.heap_distances.data(), local.heap_ids.data(), heap_size);
            }
            ivf_finalize_exact_topk(
                local.heap_distances.data(), local.heap_ids.data(), heap_size, output_count);
            std::copy_n(
                local.heap_ids.data(), output_count,
                output_candidates + static_cast<size_t>(query) * output_count);
        }
        state.valid_counts[static_cast<size_t>(tid)] = local_valid;
    });
    const auto select_end = std::chrono::steady_clock::now();

    if (stats) {
        long long valid_total = 0;
        for (long long count : state.valid_counts) valid_total += count;
        stats->query_transform_ms = std::chrono::duration<double, std::milli>(
            transform_end - transform_begin).count();
        stats->select_ms = std::chrono::duration<double, std::milli>(
            select_end - select_begin).count();
        stats->avg_input_candidates = static_cast<double>(valid_total) / batch_n;
        stats->threads = threads;
    }
}

void ivf_pca_refine_cpu_prewarm(int max_threads, int batch_n) {
    const int threads = thread_count_for(max_threads, batch_n);
    std::lock_guard<std::mutex> guard(run_mutex());
    (void)state_for(threads);
    persistent_thread_pool_for_threads(threads).run([](int) {});
}

}  // namespace paraselect::search
