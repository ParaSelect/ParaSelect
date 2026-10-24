#ifndef CPU_SEARCH_H
#define CPU_SEARCH_H

#include <vector>
#include <thread>
#include <algorithm>
#include <limits>
#include "src/clustering/kmeans/kmeans.cuh"
#include "cpu/utils/cpu_distance.h"
#include "cpu/utils/cpu_affinity.h"

// CPUk()-
void cpu_cos_distance_topk(const float** query_vectors, const float** data_vectors,
    idx_t** topk_index, float** topk_dist,
    int n_query, int n_batch, int n_dim, int k) {

    // L2(,)
    float* query_norms = (float*)malloc(n_query * sizeof(float));
    float* data_norms = (float*)malloc(n_batch * sizeof(float));

    // query
    compute_squared_sums_batch(query_vectors, query_norms, n_query, n_dim);

    // data
    compute_squared_sums_batch(data_vectors, data_norms, n_batch, n_dim);

    // ,CPU
    paraselect::parallel_chunk_on_physical_cores(n_query,
        [&](int start_q, int end_q, int core_id) {
            (void)core_id;
            // querycos_pairs
            std::vector<std::pair<float, idx_t>> cos_pairs(n_batch);

            for (int i = start_q; i < end_q; i++) {
                // querydata
                for (int j = 0; j < n_batch; j++) {
                    //  dot_product
                    float dot = dot_product(query_vectors[i], data_vectors[j], n_dim);
                    float cos_sim;
                    //
                    if (query_norms[i] < 1e-6f || data_norms[j] < 1e-6f)
                    cos_sim = 0.0f;  // ,0
                    else
                    cos_sim = 1.0f - (dot / sqrt(query_norms[i] * data_norms[j]));

                    // ( j)
                    cos_pairs[j] = std::make_pair(cos_sim, j);
                }

                // partial_sortk,
                int topk_count = std::min(k, n_batch);
                std::partial_sort(cos_pairs.begin(), cos_pairs.begin() + topk_count, cos_pairs.end(),
                [](const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b)
                {
                    if(a.first != b.first){
                        return a.first < b.first;  // ()
                    }
                    else{
                        return a.second > b.second;
                    }
                });

                // k
                for (int j = 0; j < topk_count; ++j) {
                    topk_index[i][j] = cos_pairs[j].second;
                    topk_dist[i][j] = cos_pairs[j].first;
                }
            }
        });

    free(query_norms);
    free(data_norms);
}

/**
 * CPU
 * GPU
 */

/**
 * CPU Coarse-Fine()
 *
 * topk:
 * 1. Coarse:querycluster,n_probescluster
 * 2. Fine:cluster,,topk
 *
 * @param n_query
 * @param dim
 * @param k cluster
 * @param n_probes cluster
 * @param topk topk
 * @param query_batch  [n_query, dim]
 * @param reordered_data  [n, dim](cluster)
 * @param centroids  [k, dim]
 * @param cluster_info cluster(offsetscounts)
 * @param distance_mode (L2_DISTANCE  COSINE_DISTANCE)
 * @param out_index  [n_query, topk]()
 * @param out_dist  [n_query, topk]
 * @param coarse_index  [n_query, n_probes],nullptr
 * @param coarse_dist  [n_query, n_probes],nullptr
 */
inline void cpu_coarse_fine_search(
    int n_query,
    int dim,
    int k,
    int n_probes,
    int topk,
    const float* query_batch,      // [n_query, dim]
    const float* reordered_data,   // [n, dim] (cluster)
    const float* centroids,        // [k, dim]
    const ClusterInfo& cluster_info, // cluster
    DistanceType distance_mode,    // L2_DISTANCE  COSINE_DISTANCE
    int** out_index,               // [n_query, topk] ()
    float** out_dist,              // [n_query, topk]
    int** coarse_index = nullptr, // [n_query, n_probes] ()
    float** coarse_dist = nullptr  // [n_query, n_probes] ()
) {
    //
    int n_total = 0;
    if (cluster_info.k > 0) {
        n_total = (int)cluster_info.offsets[cluster_info.k];
    }
    struct Pair { float dist; int idx; };

    // ,CPU
    paraselect::parallel_chunk_on_physical_cores(n_query,
        [&](int start, int end, int core_id) {
            (void)core_id;
            std::vector<Pair> tmp(k);
            std::vector<Pair> fine_buffer;

            for (int qi = start; qi < end; ++qi) {
                const float* query = query_batch + (size_t)qi * dim;

                // Coarse: querycluster
                for (int cid = 0; cid < k; ++cid) {
                    const float* center = centroids + (size_t)cid * dim;
                    float dist = (distance_mode == L2_DISTANCE)
                        ? l2_distance_squared(query, center, dim)
                        : cosine_distance(query, center, dim);
                    tmp[cid] = {dist, cid};
                }

                // n_probescluster
                std::partial_sort(tmp.begin(), tmp.begin() + n_probes, tmp.end(),
                                [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

                // ()
                if (coarse_index != nullptr && coarse_dist != nullptr) {
                    for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                        coarse_index[qi][probe_idx] = tmp[probe_idx].idx;
                        coarse_dist[qi][probe_idx] = tmp[probe_idx].dist;
                    }
                }

                // Fine: cluster
                fine_buffer.clear();
                for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                    int cid = tmp[probe_idx].idx;
                    long long cluster_offset = cluster_info.offsets[cid];
                    int cluster_size = cluster_info.counts[cid];

                    for (int vid = 0; vid < cluster_size; ++vid) {
                        int global_idx = (int)(cluster_offset + vid);
                        const float* vec = reordered_data + (size_t)global_idx * dim;
                        float dist = (distance_mode == L2_DISTANCE)
                            ? l2_distance_squared(query, vec, dim)
                            : cosine_distance(query, vec, dim);
                        fine_buffer.push_back({dist, global_idx});
                    }
                }

                // topk
                if (fine_buffer.size() > (size_t)topk) {
                    std::nth_element(fine_buffer.begin(), fine_buffer.begin() + topk, fine_buffer.end(),
                                   [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                    fine_buffer.resize(topk);
                    std::sort(fine_buffer.begin(), fine_buffer.end(),
                             [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                } else {
                    std::sort(fine_buffer.begin(), fine_buffer.end(),
                             [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }

            // ()
            for (int i = 0; i < topk && i < (int)fine_buffer.size(); ++i) {
                int reordered_pos = fine_buffer[i].idx;
                if (reordered_pos >= 0 && reordered_pos < n_total) {
                    out_index[qi][i] = reordered_pos;  //
                } else {
                    out_index[qi][i] = -1;
                }
                out_dist[qi][i] = fine_buffer[i].dist;
            }
            // (topk)
            for (int i = (int)fine_buffer.size(); i < topk; ++i) {
                out_index[qi][i] = -1;
                out_dist[qi][i] = std::numeric_limits<float>::max();
            }
        }
    });
}

/**
 * CPU Coarse-Fine()
 *
 * topk:
 * 1. Coarse:querycluster,n_probescluster
 * 2. Fine:cluster,,topk
 *
 * @param n_query
 * @param dim
 * @param k cluster
 * @param n_probes cluster
 * @param topk topk
 * @param query_batch  [n_query, dim]
 * @param reordered_data  [n, dim](cluster)
 * @param centroids  [k, dim]
 * @param cluster_info cluster(offsetscounts)
 * @param distance_mode (L2_DISTANCE  COSINE_DISTANCE)
 * @param out_index  [n_query, topk]()
 * @param out_dist  [n_query, topk]
 * @param coarse_index  [n_query, n_probes],nullptr
 * @param coarse_dist  [n_query, n_probes],nullptr
 */
 inline void cpu_ivf_search(
    int n_query,
    int dim,
    int k,
    int n_probes,
    int topk,
    const float* query_batch,      // [n_query, dim]
    const float* reordered_data,   // [n, dim] (cluster)
    const float* centroids,        // [k, dim]
    const ClusterInfo& cluster_info, // cluster
    DistanceType distance_mode,    // L2_DISTANCE  COSINE_DISTANCE
    idx_t** out_index,               // [n_query, topk] ()
    float** out_dist,              // [n_query, topk]
    idx_t** coarse_index = nullptr, // [n_query, n_probes] ()
    float** coarse_dist = nullptr,  // [n_query, n_probes] ()
    const idx_t* reordered_indices = nullptr  // [n] ()
) {
    //
    int n_total = 0;
    if (cluster_info.k > 0) {
        n_total = (int)cluster_info.offsets[cluster_info.k];
    }
    struct Pair { float dist; idx_t idx; };

    // ,CPU
    paraselect::parallel_chunk_on_physical_cores(n_query,
        [&](int start, int end, int core_id) {
            (void)core_id;
            std::vector<Pair> tmp(k);
            std::vector<Pair> fine_buffer;

            for (int qi = start; qi < end; ++qi) {
                const float* query = query_batch + (size_t)qi * dim;

                // Coarse: querycluster
                for (int cid = 0; cid < k; ++cid) {
                    const float* center = centroids + (size_t)cid * dim;
                    float dist = (distance_mode == L2_DISTANCE)
                        ? l2_distance_squared(query, center, dim)
                        : cosine_distance(query, center, dim);
                    tmp[cid] = {dist, (idx_t)cid};
                }

                // n_probescluster
                std::partial_sort(tmp.begin(), tmp.begin() + n_probes, tmp.end(),
                                [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

                // ()
                if (coarse_index != nullptr && coarse_dist != nullptr) {
                    for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                        coarse_index[qi][probe_idx] = tmp[probe_idx].idx;
                        coarse_dist[qi][probe_idx] = tmp[probe_idx].dist;
                    }
                }

                // Fine: cluster
                fine_buffer.clear();
                for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                    idx_t cid = tmp[probe_idx].idx;
                    long long cluster_offset = cluster_info.offsets[cid];
                    int cluster_size = cluster_info.counts[cid];

                    for (int vid = 0; vid < cluster_size; ++vid) {
                        idx_t global_idx = (idx_t)(cluster_offset + vid);
                        const float* vec = reordered_data + (size_t)global_idx * dim;
                        float dist = (distance_mode == L2_DISTANCE)
                            ? l2_distance_squared(query, vec, dim)
                            : cosine_distance(query, vec, dim);
                        fine_buffer.push_back({dist, global_idx});
                    }
                }

                // topk
                if (fine_buffer.size() > (size_t)topk) {
                    std::nth_element(fine_buffer.begin(), fine_buffer.begin() + topk, fine_buffer.end(),
                                   [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                    fine_buffer.resize(topk);
                    std::sort(fine_buffer.begin(), fine_buffer.end(),
                             [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                } else {
                    std::sort(fine_buffer.begin(), fine_buffer.end(),
                             [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                }

                // (,)
                for (int i = 0; i < topk && i < (int)fine_buffer.size(); ++i) {
                    idx_t reordered_pos = fine_buffer[i].idx;
                    if (reordered_pos >= 0 && reordered_pos < (idx_t)n_total) {
                        // ,
                        if (reordered_indices != nullptr) {
                            out_index[qi][i] = reordered_indices[reordered_pos];  //
                        } else {
                            out_index[qi][i] = reordered_pos;  // ,
                        }
                    } else {
                        out_index[qi][i] = -1;
                    }
                    out_dist[qi][i] = fine_buffer[i].dist;
                }
                // (topk)
                for (int i = (int)fine_buffer.size(); i < topk; ++i) {
                    out_index[qi][i] = -1;
                    out_dist[qi][i] = std::numeric_limits<float>::max();
                }
            }
        });
}

#endif // CPU_SEARCH_H

