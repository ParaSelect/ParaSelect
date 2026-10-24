#ifndef CPU_IVF_HOST_PREPARE_H
#define CPU_IVF_HOST_PREPARE_H

/**
 * IVF / interleaved  host :query , query  interleaved  vec L2 norm.
 *  `test_utils.cuh::compute_norms`  `test_gemv_interleaved_pipeline.cu`  query , query().
 */

#include "cpu/utils/cpu_array_utils.h"
#include "cpu/utils/l2norm_cpu.h"
#include "cpu_distance.h"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

/** pack_threads<=0  hardware_concurrency, n_query */
inline int cpu_ivf_resolve_query_threads(int pack_threads, int n_query) {
    if (n_query <= 0)
        return 1;
    int nt = cpu_array_resolve_num_threads(pack_threads);
    return std::min(nt, n_query);
}

/**
 * row-major [n_vec, n_dim]: L2  norms( `compute_norms` ,)
 */
inline void compute_norms_multithreaded(
    const float* data,
    float* norms,
    int n_vec,
    int n_dim,
    int pack_threads) {
    if (!data || !norms || n_vec <= 0 || n_dim <= 0)
        return;

    const int nt = cpu_ivf_resolve_query_threads(pack_threads, n_vec);
    if (nt <= 1) {
        for (int i = 0; i < n_vec; ++i) {
            norms[i] = compute_l2_norm(
                data + static_cast<std::size_t>(i) * static_cast<std::size_t>(n_dim),
                n_dim);
        }
        return;
    }

    const int chunk = (n_vec + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int i0 = t * chunk;
        const int i1 = std::min(i0 + chunk, n_vec);
        if (i0 >= i1)
            break;
        workers.emplace_back([=]() {
            for (int i = i0; i < i1; ++i) {
                norms[i] = compute_l2_norm(
                    data + static_cast<std::size_t>(i) * static_cast<std::size_t>(n_dim),
                    n_dim);
            }
        });
    }
    for (auto& w : workers) {
        if (w.joinable())
            w.join();
    }
}

/**
 *  query q:`init_array_interleaved_multithreaded`  h_ivf  + `compute_l2_norm_cpu_interleaved`  h_vec_norm .
 *  q ; pack_threads  interleaved ( q ).
 */
inline void init_interleaved_and_vec_norms_per_query_multithreaded(
    float* h_ivf_base,
    std::size_t interleaved_floats,
    float* h_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    unsigned seed_base,
    float min_val,
    float max_val,
    int pack_threads) {
    if (!h_ivf_base || !h_vec_norm || n_query <= 0 || n_vec <= 0 || n_dim <= 0)
        return;

    const int nt = cpu_ivf_resolve_query_threads(pack_threads, n_query);
    if (nt <= 1) {
        for (int q = 0; q < n_query; ++q) {
            float* ivf_blk = h_ivf_base + q * interleaved_floats;
            float* vn_blk = h_vec_norm + static_cast<std::size_t>(q) * n_vec;
            init_array_interleaved_multithreaded(
                ivf_blk,
                n_vec,
                n_dim,
                seed_base + static_cast<unsigned>(q),
                min_val,
                max_val,
                pack_threads);
            paraselect::compute_l2_norm_cpu_interleaved(ivf_blk, vn_blk, n_vec, n_dim);
        }
        return;
    }

    const int chunk = (n_query + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int q0 = t * chunk;
        const int q1 = std::min(q0 + chunk, n_query);
        if (q0 >= q1)
            break;
        workers.emplace_back([=]() {
            for (int q = q0; q < q1; ++q) {
                float* ivf_blk = h_ivf_base + q * interleaved_floats;
                float* vn_blk = h_vec_norm + static_cast<std::size_t>(q) * n_vec;
                init_array_interleaved_multithreaded(
                    ivf_blk,
                    n_vec,
                    n_dim,
                    seed_base + static_cast<unsigned>(q),
                    min_val,
                    max_val,
                    pack_threads);
                paraselect::compute_l2_norm_cpu_interleaved(ivf_blk, vn_blk, n_vec, n_dim);
            }
        });
    }
    for (auto& w : workers) {
        if (w.joinable())
            w.join();
    }
}

#endif  // CPU_IVF_HOST_PREPARE_H
