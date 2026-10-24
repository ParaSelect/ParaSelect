#ifndef CPU_STAGE3_LAZY_REF_H
#define CPU_STAGE3_LAZY_REF_H

/**
 * IVF Stage3 lazy  CPU , `ivf_search_stage3_lazy` + chunk  ****:
 *  `pack_rowmajor_to_interleaved`, `compute_l2_norm_cpu_interleaved`,
 *  `interleaved_offset`( `gemv_interleaved_*_topk_chunk_kernel` ).
 *  row-major  + `compute_l2_norm` , GPU interleaved , compare_set_2D .
 *
 * : (q,p)  k  (dist, ), query  n_probes*k  (dist, idx)  k ( select_k ).
 *  ID  `d_topk_index`(Stage4 ) `compare_set_2D_with_mismatches*`; `compare_set_2D`.
 *
 *  lambda  `std::vector` , `[=]`  vector.
 */

#include "cpu/utils/cpu_array_utils.h"
#include "src/layout/ivf_interleaved_layout.cuh"
#include "cpu/utils/l2norm_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

namespace cpu_stage3_lazy_ref_detail {

inline int resolve_jobs_threads(int pack_threads, int n_jobs) {
    if (n_jobs <= 0)
        return 1;
    int nt = cpu_array_resolve_num_threads(pack_threads);
    if (nt > n_jobs)
        nt = n_jobs;
    return nt > 0 ? nt : 1;
}

/**
 *  GPU chunk , (dist, );, select_k/warpsort .
 * L2 = qn^2 + vn^2 - 2*dot;Cos = 1 - dot/(qn*vn)
 */
inline void probe_local_topk_pairs_chunk_aligned(
    bool use_cosine,
    const float* q_row,
    float qn,
    const float* cluster_vecs_rowmajor,
    int n_vec_in_cluster,
    int n_dim,
    int k,
    float* cand_row_k_dist,
    int* cand_row_k_compact_idx,
    int compact_base,
    float* scratch_ivf,
    float* scratch_norms) {
    if (n_vec_in_cluster <= 0 || n_dim <= 0 || k <= 0)
        return;

    paraselect::pack_rowmajor_to_interleaved(
        cluster_vecs_rowmajor, scratch_ivf, n_vec_in_cluster, n_dim);
    paraselect::compute_l2_norm_cpu_interleaved(scratch_ivf, scratch_norms, n_vec_in_cluster, n_dim);

    if (k >= n_vec_in_cluster) {
        for (int j = 0; j < n_vec_in_cluster; ++j) {
            float dot = 0.f;
            for (int d = 0; d < n_dim; ++d)
                dot += q_row[d] * scratch_ivf[paraselect::interleaved_offset(j, d, n_dim)];
            const float vn = scratch_norms[j];
            float dist = use_cosine ? (1.0f - dot / (qn * vn)) : (qn * qn + vn * vn - 2.f * dot);
            cand_row_k_dist[j] = dist;
            if (cand_row_k_compact_idx)
                cand_row_k_compact_idx[j] = compact_base + j;
        }
        for (int j = n_vec_in_cluster; j < k; ++j) {
            cand_row_k_dist[j] = std::numeric_limits<float>::infinity();
            if (cand_row_k_compact_idx)
                cand_row_k_compact_idx[j] = -1;
        }
        return;
    }

    std::vector<std::pair<float, int>> tmp(static_cast<std::size_t>(n_vec_in_cluster));
    for (int j = 0; j < n_vec_in_cluster; ++j) {
        float dot = 0.f;
        for (int d = 0; d < n_dim; ++d)
            dot += q_row[d] * scratch_ivf[paraselect::interleaved_offset(j, d, n_dim)];
        const float vn = scratch_norms[j];
        float dist = use_cosine ? (1.0f - dot / (qn * vn)) : (qn * qn + vn * vn - 2.f * dot);
        tmp[static_cast<std::size_t>(j)] = {dist, j};
    }
    std::partial_sort(
        tmp.begin(),
        tmp.begin() + k,
        tmp.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            if (a.first != b.first)
                return a.first < b.first;
            return a.second < b.second;
        });
    for (int j = 0; j < k; ++j) {
        cand_row_k_dist[j] = tmp[static_cast<std::size_t>(j)].first;
        if (cand_row_k_compact_idx)
            cand_row_k_compact_idx[j] = compact_base + tmp[static_cast<std::size_t>(j)].second;
    }
}

/** : probe_local_topk_pairs_chunk_aligned */
inline void probe_local_topk_distances_chunk_aligned(
    bool use_cosine,
    const float* q_row,
    float qn,
    const float* cluster_vecs_rowmajor,
    int n_vec_in_cluster,
    int n_dim,
    int k,
    float* cand_row_k,
    float* scratch_ivf,
    float* scratch_norms) {
    std::vector<int> idx_pad(static_cast<std::size_t>(k));
    probe_local_topk_pairs_chunk_aligned(
        use_cosine,
        q_row,
        qn,
        cluster_vecs_rowmajor,
        n_vec_in_cluster,
        n_dim,
        k,
        cand_row_k,
        idx_pad.data(),
        0,
        scratch_ivf,
        scratch_norms);
}

}  // namespace cpu_stage3_lazy_ref_detail

/**
 * @param h_qp_cluster  n_query*n_probes
 * @param h_compact_off  n_clusters+1()
 * @param h_out_topk_dist  [n_query, k]
 * @param h_out_topk_compact_idx ; GPU `map_candidate` ,`stage4` ( d_topk_index )
 * @param dbg_cand_l1_sum  (q,p)  top-k  L1 ( d_topk_dist_candidate ,)
 */
inline void cpu_ivf_lazy_stage3_topk_dist_ref_mt(
    bool use_cosine,
    const float* h_query_rm,
    const float* h_query_norm,
    const float* h_cluster_vecs_rm,
    const int* h_cluster_sizes,
    const int* h_compact_off,
    const int* h_qp_cluster,
    int n_query,
    int n_clusters,
    int n_dim,
    int n_probes,
    int k,
    float* h_out_topk_dist,
    int pack_threads,
    double* dbg_cand_l1_sum = nullptr,
    int* h_out_topk_compact_idx = nullptr) {
    if (!h_query_rm || !h_query_norm || !h_cluster_vecs_rm || !h_cluster_sizes || !h_compact_off ||
        !h_qp_cluster || !h_out_topk_dist || n_query <= 0 || n_dim <= 0 || n_probes <= 0 || k <= 0)
        return;

    int max_vecs = 0;
    for (int c = 0; c < n_clusters; ++c) {
        const int sz = h_cluster_sizes[c];
        if (sz > max_vecs)
            max_vecs = sz;
    }
    if (max_vecs <= 0)
        return;

    const size_t max_ivf_floats = paraselect::interleaved_storage_floats(max_vecs, n_dim);

    const int n_qp = n_query * n_probes;
    std::vector<float> cand(static_cast<std::size_t>(n_qp) * k);
    std::vector<int> cand_idx(static_cast<std::size_t>(n_qp) * k);

    {
        const int nt = cpu_stage3_lazy_ref_detail::resolve_jobs_threads(pack_threads, n_qp);
        if (nt <= 1) {
            std::vector<float> scratch_ivf(max_ivf_floats);
            std::vector<float> scratch_norms(static_cast<std::size_t>(max_vecs));
            for (int q = 0; q < n_query; ++q) {
                const float* qrow = h_query_rm + static_cast<std::size_t>(q) * n_dim;
                const float qn = h_query_norm[q];
                for (int p = 0; p < n_probes; ++p) {
                    const int c = h_qp_cluster[q * n_probes + p];
                    const int nvec = h_cluster_sizes[c];
                    const int row0 = h_compact_off[c];
                    const float* cbase = h_cluster_vecs_rm + static_cast<std::size_t>(row0) * n_dim;
                    const size_t row_off = static_cast<std::size_t>(q * n_probes + p) * static_cast<std::size_t>(k);
                    cpu_stage3_lazy_ref_detail::probe_local_topk_pairs_chunk_aligned(
                        use_cosine,
                        qrow,
                        qn,
                        cbase,
                        nvec,
                        n_dim,
                        k,
                        cand.data() + row_off,
                        cand_idx.data() + row_off,
                        row0,
                        scratch_ivf.data(),
                        scratch_norms.data());
                }
            }
        } else {
            const int chunk = (n_qp + nt - 1) / nt;
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(nt));
            std::vector<std::vector<float>> tls_ivf(static_cast<std::size_t>(nt));
            std::vector<std::vector<float>> tls_norms(static_cast<std::size_t>(nt));
            for (int t = 0; t < nt; ++t) {
                tls_ivf[static_cast<std::size_t>(t)].resize(max_ivf_floats);
                tls_norms[static_cast<std::size_t>(t)].resize(static_cast<std::size_t>(max_vecs));
            }
            for (int t = 0; t < nt; ++t) {
                const int i0 = t * chunk;
                const int i1 = std::min(i0 + chunk, n_qp);
                if (i0 >= i1)
                    break;
                workers.emplace_back([&, t, i0, i1]() {
                    float* sivf = tls_ivf[static_cast<std::size_t>(t)].data();
                    float* snorm = tls_norms[static_cast<std::size_t>(t)].data();
                    for (int i = i0; i < i1; ++i) {
                        const int q = i / n_probes;
                        const int p = i % n_probes;
                        const float* qrow = h_query_rm + static_cast<std::size_t>(q) * n_dim;
                        const float qn = h_query_norm[q];
                        const int c = h_qp_cluster[q * n_probes + p];
                        const int nvec = h_cluster_sizes[c];
                        const int row0 = h_compact_off[c];
                        const float* cbase = h_cluster_vecs_rm + static_cast<std::size_t>(row0) * n_dim;
                        const size_t row_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
                        cpu_stage3_lazy_ref_detail::probe_local_topk_pairs_chunk_aligned(
                            use_cosine,
                            qrow,
                            qn,
                            cbase,
                            nvec,
                            n_dim,
                            k,
                            cand.data() + row_off,
                            cand_idx.data() + row_off,
                            row0,
                            sivf,
                            snorm);
                    }
                });
            }
            for (auto& w : workers) {
                if (w.joinable())
                    w.join();
            }
        }
    }

    if (dbg_cand_l1_sum) {
        double acc = 0.0;
        for (float x : cand)
            acc += static_cast<double>(x);
        *dbg_cand_l1_sum = acc;
    }

    auto reduce_query_topk = [&](int q) {
        std::vector<std::pair<float, int>> pool(static_cast<std::size_t>(n_probes) * static_cast<std::size_t>(k));
        for (int p = 0; p < n_probes; ++p) {
            const size_t base = static_cast<std::size_t>(q * n_probes + p) * static_cast<std::size_t>(k);
            for (int j = 0; j < k; ++j) {
                pool[static_cast<std::size_t>(p) * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] = {
                    cand[base + static_cast<std::size_t>(j)], cand_idx[base + static_cast<std::size_t>(j)]};
            }
        }
        std::partial_sort(
            pool.begin(),
            pool.begin() + k,
            pool.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                if (a.first != b.first)
                    return a.first < b.first;
                return a.second < b.second;
            });
        for (int i = 0; i < k; ++i) {
            const size_t o = static_cast<std::size_t>(q) * static_cast<std::size_t>(k) + static_cast<std::size_t>(i);
            h_out_topk_dist[o] = pool[static_cast<std::size_t>(i)].first;
            if (h_out_topk_compact_idx)
                h_out_topk_compact_idx[o] = pool[static_cast<std::size_t>(i)].second;
        }
    };

    {
        const int nt = cpu_stage3_lazy_ref_detail::resolve_jobs_threads(pack_threads, n_query);
        if (nt <= 1) {
            for (int q = 0; q < n_query; ++q)
                reduce_query_topk(q);
        } else {
            const int chunk = (n_query + nt - 1) / nt;
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(nt));
            for (int t = 0; t < nt; ++t) {
                const int q0 = t * chunk;
                const int q1 = std::min(q0 + chunk, n_query);
                if (q0 >= q1)
                    break;
                workers.emplace_back([&, q0, q1]() {
                    for (int q = q0; q < q1; ++q)
                        reduce_query_topk(q);
                });
            }
            for (auto& w : workers) {
                if (w.joinable())
                    w.join();
            }
        }
    }
}

#endif  // CPU_STAGE3_LAZY_REF_H
