#ifndef CPU_BRUTE_FORCE_SCAN_H
#define CPU_BRUTE_FORCE_SCAN_H

#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <queue>
#include "cpu_distance.h"
#include "src/utils/pch.h"

/**
 * CPU (,)
 *
 *  query  data , topk
 *  topk , (dist, index)
 *
 * @param h_query  [n_query, n_dim]
 * @param h_data  [n_data, n_dim]
 * @param n_query
 * @param n_data
 * @param n_dim
 * @param k  topk
 * @param distance_mode  (L2_DISTANCE  COSINE_DISTANCE)
 * @param h_topk_dist :topk  [n_query, k]
 * @param h_topk_index :topk  [n_query, k]
 */
inline void cpu_brute_force_scan(
    const float* h_query,
    const float* h_data,
    int n_query,
    int n_data,
    int n_dim,
    int k,
    int distance_mode,
    float* h_topk_dist,
    idx_t* h_topk_index)
{
    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());

    // 1.  L2 ,
    std::vector<float> query_norms(n_query);
    std::vector<float> data_norms(n_data);

    for (int q = 0; q < n_query; ++q) {
        float norm = 0.0f;
        for (int i = 0; i < n_dim; ++i) {
            float v = h_query[q * n_dim + i];
            norm += v * v;
        }
        query_norms[q] = norm;
    }

    for (int d = 0; d < n_data; ++d) {
        float norm = 0.0f;
        for (int i = 0; i < n_dim; ++i) {
            float v = h_data[d * n_dim + i];
            norm += v * v;
        }
        data_norms[d] = norm;
    }

    // 2.  query
    auto worker = [&](int start, int end) {
        for (int q = start; q < end; ++q) {
            // () k
            std::priority_queue<std::pair<float, idx_t>> heap;

            const float* q_vec = h_query + q * n_dim;
            const float q_norm = query_norms[q];

            for (int d = 0; d < n_data; ++d) {
                const float* d_vec = h_data + d * n_dim;
                const float d_norm = data_norms[d];

                //  dot product
                float dot = 0.0f;
                for (int i = 0; i < n_dim; ++i) {
                    dot += q_vec[i] * d_vec[i];
                }

                float dist;
                if (distance_mode == L2_DISTANCE) {
                    dist = q_norm + d_norm - 2.0f * dot;
                } else {
                    float cos_sim = dot / (std::sqrt(q_norm) * std::sqrt(d_norm));
                    dist = 1.0f - cos_sim;
                }

                //  topk
                if (static_cast<int>(heap.size()) < k) {
                    heap.emplace(dist, static_cast<idx_t>(d));
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.emplace(dist, static_cast<idx_t>(d));
                }
            }

            // (,)
            for (int i = k - 1; i >= 0 && !heap.empty(); --i) {
                auto& item = heap.top();
                h_topk_dist[q * k + i] = item.first;
                h_topk_index[q * k + i] = item.second;
                heap.pop();
            }
        }
    };

    std::vector<std::thread> threads;
    int chunk = (n_query + num_threads - 1) / num_threads;
    for (unsigned t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(n_query, start + chunk);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto& t : threads) t.join();
}

#endif // CPU_BRUTE_FORCE_SCAN_H
