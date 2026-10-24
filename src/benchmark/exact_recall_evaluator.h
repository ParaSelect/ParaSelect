#pragma once

#include "utils/pch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include "cpu/utils/cpu_affinity.h"

namespace paraselect {

struct ExactRecallResult {
    float recall_at_1 = -1.0f;
    float recall_at_5 = -1.0f;
    float recall_at_10 = -1.0f;
};

inline int resolve_exact_recall_threads(int n_query) {
    const char* env = std::getenv("PARASELECT_RECALL_THREADS");
    if (env && *env) {
        int v = std::atoi(env);
        if (v > 0) return std::max(1, std::min(v, std::max(1, n_query)));
    }
    const int threads = paraselect::resolve_cpu_fine_thread_count();
    return std::max(1, std::min(threads, std::max(1, n_query)));
}

inline float exact_l2_distance_sq(const float* query, const float* base, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float diff = query[d] - base[d];
        sum += diff * diff;
    }
    return sum;
}

inline float exact_cosine_distance(const float* query, const float* base, int dim) {
    float dot = 0.0f;
    float q_norm2 = 0.0f;
    float b_norm2 = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float q = query[d];
        const float b = base[d];
        dot += q * b;
        q_norm2 += q * q;
        b_norm2 += b * b;
    }
    const float denom = std::sqrt(std::max(q_norm2, 0.0f)) *
                        std::sqrt(std::max(b_norm2, 0.0f));
    if (denom <= 0.0f) return 1.0f;
    return 1.0f - dot / denom;
}

inline float exact_distance_for_recall(
    const float* query,
    const float* base,
    int dim,
    int distance_mode) {
    return distance_mode == L2_DISTANCE ?
        exact_l2_distance_sq(query, base, dim) :
        exact_cosine_distance(query, base, dim);
}

inline ExactRecallResult compute_exact_threshold_recall(
    const float* queries,
    const float* base,
    int n_base,
    int dim,
    const idx_t* result_indices,
    int n_query,
    int k_result,
    const float* gt_distances,
    int gt_k,
    int distance_mode,
    int num_threads = 0) {
    ExactRecallResult out;
    if (!queries || !base || !result_indices || !gt_distances ||
        n_base <= 0 || dim <= 0 || n_query <= 0 || k_result <= 0 || gt_k <= 0) {
        return out;
    }

    const int recall_ks[3] = {1, 5, 10};
    long long hit_counts[3] = {0, 0, 0};
    const float eps = distance_mode == L2_DISTANCE ? 1e-3f : 1e-5f;
    const int threads = num_threads > 0 ?
        std::max(1, std::min(num_threads, n_query)) :
        resolve_exact_recall_threads(n_query);

    std::vector<std::thread> workers;
    std::vector<std::array<long long, 3>> local_hits((size_t)threads);
    const int chunk = (n_query + threads - 1) / threads;

    for (int tid = 0; tid < threads; ++tid) {
        const int q_begin = tid * chunk;
        const int q_end = std::min(n_query, q_begin + chunk);
        workers.emplace_back([&, tid, q_begin, q_end]() {
            auto& hits = local_hits[(size_t)tid];
            hits = {0, 0, 0};
            for (int q = q_begin; q < q_end; ++q) {
                const float* qv = queries + (size_t)q * dim;
                for (int rk_i = 0; rk_i < 3; ++rk_i) {
                    const int rk = recall_ks[rk_i];
                    if (rk > k_result || rk > gt_k) continue;
                    const float threshold = gt_distances[(size_t)q * gt_k + (rk - 1)] + eps;
                    for (int j = 0; j < rk; ++j) {
                        const idx_t id = result_indices[(size_t)q * k_result + j];
                        if (id < 0 || id >= n_base) continue;
                        const float* bv = base + (size_t)id * dim;
                        const float exact = exact_distance_for_recall(qv, bv, dim, distance_mode);
                        if (exact <= threshold) {
                            hits[(size_t)rk_i]++;
                        }
                    }
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (const auto& hits : local_hits) {
        for (int i = 0; i < 3; ++i) {
            hit_counts[i] += hits[(size_t)i];
        }
    }

    out.recall_at_1 = (gt_k >= 1 && k_result >= 1) ?
        static_cast<float>(hit_counts[0]) / static_cast<float>(n_query) : -1.0f;
    out.recall_at_5 = (gt_k >= 5 && k_result >= 5) ?
        static_cast<float>(hit_counts[1]) / static_cast<float>(n_query * 5) : -1.0f;
    out.recall_at_10 = (gt_k >= 10 && k_result >= 10) ?
        static_cast<float>(hit_counts[2]) / static_cast<float>(n_query * 10) : -1.0f;
    return out;
}

}  // namespace paraselect
