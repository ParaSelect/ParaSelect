#ifndef CPU_FINE_H
#define CPU_FINE_H

/**
 * (fine) CPU :query row-major ×  interleaved( query  interleaved, interleaved_floats).
 * : query ( cpu_search  query ).
 * : `h_ref`  nth_element  query  k (`fill_cpu_topk_dist_from_ref_multithreaded`), chunk top-k  GPU  multiset .
 */

#include "src/layout/ivf_interleaved_layout.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

namespace cpu_fine_detail {

inline int resolve_num_threads(int num_threads, int n_query) {
    if (n_query <= 0)
        return 1;
    unsigned int nt = 0;
    if (num_threads > 0)
        nt = static_cast<unsigned int>(num_threads);
    else {
        nt = std::thread::hardware_concurrency();
        if (nt == 0)
            nt = 4u;
    }
    nt = std::min(nt, static_cast<unsigned int>(n_query));
    return static_cast<int>(nt);
}

/** [q_begin, q_end)  query  h_out[q*n_vec + v] */
inline void compute_ref_l2_interleaved_range(
    const float* h_query,
    const float* h_qn,
    const float* h_ivf_base,
    std::size_t interleaved_floats,
    const float* h_vec_norm,
    int n_vec,
    int n_dim,
    float* h_out,
    int q_begin,
    int q_end) {
    for (int q = q_begin; q < q_end; ++q) {
        const float* qq = h_query + static_cast<std::size_t>(q) * n_dim;
        const float qn = h_qn[q];
        const float* ivf = h_ivf_base + q * interleaved_floats;
        for (int v = 0; v < n_vec; ++v) {
            float dot = 0.f;
            for (int d = 0; d < n_dim; ++d)
                dot += qq[d] * ivf[paraselect::interleaved_offset(v, d, n_dim)];
            const float vn = h_vec_norm[static_cast<std::size_t>(q) * n_vec + v];
            h_out[static_cast<std::size_t>(q) * n_vec + v] =
                qn * qn + vn * vn - 2.f * dot;
        }
    }
}

inline void compute_ref_cos_interleaved_range(
    const float* h_query,
    const float* h_qn,
    const float* h_ivf_base,
    std::size_t interleaved_floats,
    const float* h_vec_norm,
    int n_vec,
    int n_dim,
    float* h_out,
    int q_begin,
    int q_end) {
    for (int q = q_begin; q < q_end; ++q) {
        const float* qq = h_query + static_cast<std::size_t>(q) * n_dim;
        const float qn = h_qn[q];
        const float* ivf = h_ivf_base + q * interleaved_floats;
        for (int v = 0; v < n_vec; ++v) {
            float dot = 0.f;
            for (int d = 0; d < n_dim; ++d)
                dot += qq[d] * ivf[paraselect::interleaved_offset(v, d, n_dim)];
            const float vn = h_vec_norm[static_cast<std::size_t>(q) * n_vec + v];
            float outv;
            if (qn >= 1e-6f && vn >= 1e-6f) {
                float cos_sim = dot / (qn * vn);
                outv = 1.0f - cos_sim;
            } else {
                outv = 1.0f;
            }
            h_out[static_cast<std::size_t>(q) * n_vec + v] = outv;
        }
    }
}

/**
 * [q_begin, q_end): h_ref  k  h_cpu_topk_out; compare .
 * k >= n_vec  +inf( chunk ).
 */
inline void fill_cpu_topk_from_ref_rows_range(
    const float* h_ref,
    int n_vec,
    int k,
    float* h_cpu_topk_out,
    float** cpu_row_ptrs_out,
    float* h_gpu_topk_dist,
    float** gpu_row_ptrs_out,
    int q_begin,
    int q_end) {
    for (int q = q_begin; q < q_end; ++q) {
        const float* ref_row = h_ref + static_cast<std::size_t>(q) * n_vec;
        float* dst = h_cpu_topk_out + static_cast<std::size_t>(q) * k;
        if (k >= n_vec) {
            for (int j = 0; j < n_vec; ++j)
                dst[j] = ref_row[j];
            for (int j = n_vec; j < k; ++j)
                dst[j] = std::numeric_limits<float>::infinity();
        } else {
            std::vector<float> tmp(ref_row, ref_row + n_vec);
            std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
            std::copy(tmp.begin(), tmp.begin() + k, dst);
        }
        cpu_row_ptrs_out[static_cast<std::size_t>(q)] = dst;
        gpu_row_ptrs_out[static_cast<std::size_t>(q)] =
            h_gpu_topk_dist + static_cast<std::size_t>(q) * k;
    }
}

}  // namespace cpu_fine_detail

/**
 * CPU :L2  d[q][v] = ||q||^2 + ||v||^2 - 2 q·v( interleaved)
 * @param num_threads ;<=0  hardware_concurrency( 1, n_query)
 */
inline void compute_ref_l2(
    const float* h_query,
    const float* h_qn,
    const float* h_ivf_base,
    std::size_t interleaved_floats,
    const float* h_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* h_out,
    int num_threads = 0) {
    if (!h_query || !h_qn || !h_ivf_base || !h_vec_norm || !h_out || n_query <= 0 ||
        n_vec <= 0 || n_dim <= 0)
        return;

    const int nt = cpu_fine_detail::resolve_num_threads(num_threads, n_query);
    const int chunk = (n_query + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int q0 = t * chunk;
        const int q1 = std::min(q0 + chunk, n_query);
        if (q0 >= q1)
            break;
        workers.emplace_back(
            [=]() {
                cpu_fine_detail::compute_ref_l2_interleaved_range(
                    h_query,
                    h_qn,
                    h_ivf_base,
                    interleaved_floats,
                    h_vec_norm,
                    n_vec,
                    n_dim,
                    h_out,
                    q0,
                    q1);
            });
    }
    for (auto& w : workers) {
        if (w.joinable())
            w.join();
    }
}

/** CPU : 1 - cos_sim( launch_gemm_interleaved_cos ) */
inline void compute_ref_cos(
    const float* h_query,
    const float* h_qn,
    const float* h_ivf_base,
    std::size_t interleaved_floats,
    const float* h_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    float* h_out,
    int num_threads = 0) {
    if (!h_query || !h_qn || !h_ivf_base || !h_vec_norm || !h_out || n_query <= 0 ||
        n_vec <= 0 || n_dim <= 0)
        return;

    const int nt = cpu_fine_detail::resolve_num_threads(num_threads, n_query);
    const int chunk = (n_query + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int q0 = t * chunk;
        const int q1 = std::min(q0 + chunk, n_query);
        if (q0 >= q1)
            break;
        workers.emplace_back(
            [=]() {
                cpu_fine_detail::compute_ref_cos_interleaved_range(
                    h_query,
                    h_qn,
                    h_ivf_base,
                    interleaved_floats,
                    h_vec_norm,
                    n_vec,
                    n_dim,
                    h_out,
                    q0,
                    q1);
            });
    }
    for (auto& w : workers) {
        if (w.joinable())
            w.join();
    }
}

/**
 *  h_ref(, n_vec) query  k (nth_element), query .
 * @param h_gpu_topk_dist  gpu_row_ptrs_out[q] (,)
 */
inline void fill_cpu_topk_dist_from_ref_multithreaded(
    const float* h_ref,
    int n_query,
    int n_vec,
    int k,
    float* h_cpu_topk_out,
    float** cpu_row_ptrs_out,
    float* h_gpu_topk_dist,
    float** gpu_row_ptrs_out,
    int num_threads = 0) {
    if (!h_ref || !h_cpu_topk_out || !cpu_row_ptrs_out || !h_gpu_topk_dist || !gpu_row_ptrs_out ||
        n_query <= 0 || n_vec <= 0 || k <= 0)
        return;

    const int nt = cpu_fine_detail::resolve_num_threads(num_threads, n_query);
    if (nt <= 1 || n_query <= 1) {
        cpu_fine_detail::fill_cpu_topk_from_ref_rows_range(
            h_ref,
            n_vec,
            k,
            h_cpu_topk_out,
            cpu_row_ptrs_out,
            h_gpu_topk_dist,
            gpu_row_ptrs_out,
            0,
            n_query);
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
        workers.emplace_back(
            [=]() {
                cpu_fine_detail::fill_cpu_topk_from_ref_rows_range(
                    h_ref,
                    n_vec,
                    k,
                    h_cpu_topk_out,
                    cpu_row_ptrs_out,
                    h_gpu_topk_dist,
                    gpu_row_ptrs_out,
                    q0,
                    q1);
            });
    }
    for (auto& w : workers) {
        if (w.joinable())
            w.join();
    }
}

/**
 * `test_gemv_interleaved_pipeline_chunk` : host interleaved  CPU .
 * 1) `compute_ref_{l2,cos}`  `h_ref_distances_out`( q×n_vec);
 * 2) `fill_cpu_topk_dist_from_ref_multithreaded`  query  k  compare .
 * `h_gpu_topk_buffer`  device ; query .
 */
inline void cpu_chunk_topk_reference_from_host_interleaved_mt(
    bool use_cosine,
    const float* h_query,
    const float* h_query_norm,
    const float* h_ivf_base,
    std::size_t interleaved_floats,
    const float* h_vec_norm,
    int n_query,
    int n_vec,
    int n_dim,
    int k,
    float* h_ref_distances_out,
    float* h_cpu_topk_out,
    float** cpu_row_ptrs_out,
    float* h_gpu_topk_buffer,
    float** gpu_row_ptrs_out,
    int num_threads) {
    if (use_cosine) {
        compute_ref_cos(
            h_query,
            h_query_norm,
            h_ivf_base,
            interleaved_floats,
            h_vec_norm,
            n_query,
            n_vec,
            n_dim,
            h_ref_distances_out,
            num_threads);
    } else {
        compute_ref_l2(
            h_query,
            h_query_norm,
            h_ivf_base,
            interleaved_floats,
            h_vec_norm,
            n_query,
            n_vec,
            n_dim,
            h_ref_distances_out,
            num_threads);
    }
    fill_cpu_topk_dist_from_ref_multithreaded(
        h_ref_distances_out,
        n_query,
        n_vec,
        k,
        h_cpu_topk_out,
        cpu_row_ptrs_out,
        h_gpu_topk_buffer,
        gpu_row_ptrs_out,
        num_threads);
}

#endif  // CPU_FINE_H
