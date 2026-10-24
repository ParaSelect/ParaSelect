/**
 * @file test_gemv_interleaved_pipeline_chunk.cu
 *
 * @brief `launch_gemv_interleaved_{l2,cos}_topk_chunk_kernel` ( `src/search/kernels/fine/gpu_fine/chunk/`).
 *
 *  `test_gemv_interleaved_pipeline.cu` :query row-major; query  interleaved ;
 *  H2D interleaved( per-chunk norm),query / query_norm / vec_norm .
 *
 * chunk  **top-k**;`launch_gemv_interleaved_*_topk_chunk`  `k`  WarpSort ( Stage3 lazy / `select_k` ),
 *  `k`( `main`  10);`n_vec`  `k`, launch  `chunk_num_blocks = ceil(n_vec/32)`.
 * CPU : `cpu_utils/cpu_fine.h`  `cpu_chunk_topk_reference_from_host_interleaved_mt`( +  top-k); `cpu_ivf_host_prepare.h`  `compute_norms_multithreaded`.
 *
 * round / warmup /  stream  pipeline .
 */

#include "src/utils/pch.h"
#include "src/utils/nsight_nvtx.h"
#include "src/search/kernels/fine/gpu_fine/gemv_interleaved.cuh"
#include "cpu/utils/cpu_array_utils.h"
#include "common/test_utils.cuh"
#include "common/params_macros.cuh"
#include "cpu/search/cpu_fine.h"
#include "cpu/search/cpu_ivf_host_prepare.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define EPSILON 1e-3f

static constexpr int kSlots = 4;
/**  WarpSort (`k` ) */
static constexpr int kWarpSortCap = 256;

static void launch_chunk_fine(
    bool use_cosine,
    float* d_query,
    float* d_chunk_vector,
    float* d_chunk_norm,
    int* d_probe_vector_offset,
    int* d_probe_vector_offset_compact,
    int* d_probe_vector_count,
    int* d_entry_cluster_id,
    int* d_entry_query_start,
    int* d_entry_query_count,
    int* d_entry_queries,
    int* d_entry_probe_indices,
    float* d_query_norm,
    int n_vec,
    int n_dim,
    int k,
    float* d_topk_dist_base,
    idx_t* d_topk_index_base,
    cudaStream_t st) {
    const int chunk_num_blocks = (n_vec + 31) / 32;
    /*  query_global_id  row , topk  */
    constexpr int kQB = 8;
    const dim3 block_dim(32 * kQB, 1, 1);
    if (use_cosine) {
        launch_gemv_interleaved_cos_topk_chunk(
            block_dim,
            n_dim,
            d_query,
            d_chunk_vector,
            d_chunk_norm,
            d_probe_vector_offset,
            d_probe_vector_offset_compact,
            d_probe_vector_count,
            d_entry_cluster_id,
            d_entry_query_start,
            d_entry_query_count,
            d_entry_queries,
            d_entry_probe_indices,
            d_query_norm,
            1,
            1,
            k,
            0,
            chunk_num_blocks,
            d_topk_dist_base,
            d_topk_index_base,
            st);
    } else {
        launch_gemv_interleaved_l2_topk_chunk(
            block_dim,
            n_dim,
            d_query,
            d_chunk_vector,
            d_chunk_norm,
            d_probe_vector_offset,
            d_probe_vector_offset_compact,
            d_probe_vector_count,
            d_entry_cluster_id,
            d_entry_query_start,
            d_entry_query_count,
            d_entry_queries,
            d_entry_probe_indices,
            d_query_norm,
            1,
            1,
            k,
            0,
            chunk_num_blocks,
            d_topk_dist_base,
            d_topk_index_base,
            st);
    }
}

/**
 * : q  query  interleaved + chunk norm; query  topk ; launch chunk .
 */
static void pipeline_submit_round_per_query(
    int i,
    int slot,
    int q,
    cudaStream_t st_copy,
    cudaStream_t st_compute,
    cudaEvent_t ev_copy_done[kSlots],
    cudaEvent_t ev_gem_done[kSlots],
    const float* h_ivf_base,
    size_t interleaved_bytes,
    size_t interleaved_floats,
    int n_vec,
    int chunk_norm_floats,
    const float* h_vec_norm,
    const float* h_topk_init_dist,
    const int* h_topk_init_idx,
    int k,
    float* d_query,
    float* d_query_norm,
    float* d_chunk_vector[kSlots],
    float* d_chunk_norm[kSlots],
    int* d_probe_vector_offset,
    int* d_probe_vector_offset_compact,
    int* d_probe_vector_count,
    int* d_entry_cluster_id,
    int* d_entry_query_start,
    int* d_entry_query_count,
    int* d_entry_queries,
    int* d_entry_probe_indices,
    float* d_topk_dist,
    idx_t* d_topk_index,
    int n_dim,
    bool use_cosine,
    int* h_entry_ring) {
    if (i >= kSlots)
        cudaStreamWaitEvent(st_copy, ev_gem_done[slot], 0);

    const float* h_src_ivf = h_ivf_base + q * interleaved_floats;
    const float* vn_row = h_vec_norm + (size_t)q * n_vec;

    cudaMemcpyAsync(
        d_chunk_vector[slot],
        h_src_ivf,
        interleaved_bytes,
        cudaMemcpyHostToDevice,
        st_copy);

    float* h_cn = (float*)std::malloc(static_cast<size_t>(chunk_norm_floats) * sizeof(float));
    const int n_blocks = (n_vec + 31) / 32;
    for (int b = 0; b < n_blocks; ++b) {
        for (int lane = 0; lane < 32; ++lane) {
            int vi = b * 32 + lane;
            h_cn[b * 32 + lane] =
                (vi < n_vec) ? vn_row[vi] : 0.f;
        }
    }
    cudaMemcpyAsync(
        d_chunk_norm[slot],
        h_cn,
        static_cast<size_t>(chunk_norm_floats) * sizeof(float),
        cudaMemcpyHostToDevice,
        st_copy);
    std::free(h_cn);

    cudaEventRecord(ev_copy_done[slot], st_copy);

    cudaStreamWaitEvent(st_compute, ev_copy_done[slot], 0);

    cudaMemcpyAsync(
        d_topk_dist + (size_t)q * k,
        h_topk_init_dist,
        static_cast<size_t>(k) * sizeof(float),
        cudaMemcpyHostToDevice,
        st_compute);
    cudaMemcpyAsync(
        d_topk_index + (size_t)q * k,
        h_topk_init_idx,
        static_cast<size_t>(k) * sizeof(idx_t),
        cudaMemcpyHostToDevice,
        st_compute);

    /*  submit  pinned : Async H2D  *h=q (CUDA ) */
    h_entry_ring[i] = q;
    cudaMemcpyAsync(
        d_entry_queries,
        h_entry_ring + i,
        sizeof(int),
        cudaMemcpyHostToDevice,
        st_compute);

    launch_chunk_fine(
        use_cosine,
        d_query,
        d_chunk_vector[slot],
        d_chunk_norm[slot],
        d_probe_vector_offset,
        d_probe_vector_offset_compact,
        d_probe_vector_count,
        d_entry_cluster_id,
        d_entry_query_start,
        d_entry_query_count,
        d_entry_queries,
        d_entry_probe_indices,
        d_query_norm,
        n_vec,
        n_dim,
        k,
        d_topk_dist,
        d_topk_index,
        st_compute);
    cudaEventRecord(ev_gem_done[slot], st_compute);
}

static std::vector<double> run_case(
    int n_query,
    int n_vec,
    int n_dim,
    bool use_cosine,
    int n_rounds,
    int n_warmup,
    int pack_threads,
    int k) {
    if (k > kWarpSortCap || k < 1) {
        COUT_ENDL("run_case: invalid k (must be 1..kWarpSortCap)");
        return {0.0, 0.0, 0.0, 0.0, 0.0};
    }

    float* d_query = nullptr;
    float* d_query_norm = nullptr;
    float* d_chunk_vector[kSlots];
    float* d_chunk_norm[kSlots];
    float* d_topk_dist = nullptr;
    idx_t* d_topk_index = nullptr;
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_offset_compact = nullptr;
    int* d_probe_vector_count = nullptr;
    int* d_entry_cluster_id = nullptr;
    int* d_entry_query_start = nullptr;
    int* d_entry_query_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probe_indices = nullptr;

    const size_t ivf_bytes = paraselect::interleaved_storage_bytes(n_vec, n_dim);
    const size_t interleaved_floats = paraselect::interleaved_storage_floats(n_vec, n_dim);
    const int chunk_norm_floats = ((n_vec + 31) / 32) * 32;
    const size_t vec_norm_bytes = (size_t)n_vec * sizeof(float);
    const size_t total_ivf_bytes = ivf_bytes * (size_t)n_query;
    const size_t total_vec_norm_bytes = vec_norm_bytes * (size_t)n_query;

    float* h_query = nullptr;
    float* h_query_norm = nullptr;
    float* h_ivf = nullptr;
    float* h_vec_norm = nullptr;

    cudaHostAlloc(&h_query, (size_t)n_query * n_dim * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_query_norm, (size_t)n_query * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_ivf, total_ivf_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_vec_norm, total_vec_norm_bytes, cudaHostAllocDefault);

    for (int s = 0; s < kSlots; ++s) {
        cudaMalloc(&d_chunk_vector[s], ivf_bytes);
        cudaMalloc(&d_chunk_norm[s], static_cast<size_t>(chunk_norm_floats) * sizeof(float));
    }

    cudaMalloc(&d_query, (size_t)n_query * n_dim * sizeof(float));
    cudaMalloc(&d_query_norm, (size_t)n_query * sizeof(float));
    cudaMalloc(&d_topk_dist, (size_t)n_query * k * sizeof(float));
    cudaMalloc(&d_topk_index, (size_t)n_query * k * sizeof(idx_t));

    cudaMalloc(&d_probe_vector_offset, sizeof(int));
    cudaMalloc(&d_probe_vector_offset_compact, sizeof(int));
    cudaMalloc(&d_probe_vector_count, sizeof(int));
    cudaMalloc(&d_entry_cluster_id, sizeof(int));
    cudaMalloc(&d_entry_query_start, sizeof(int));
    cudaMalloc(&d_entry_query_count, sizeof(int));
    cudaMalloc(&d_entry_queries, sizeof(int));
    cudaMalloc(&d_entry_probe_indices, sizeof(int));

    int h_off = 0, h_compact = 0;
    idx_t h_cnt = n_vec;
    int h_ec = 0, h_qs = 0, h_qc = 1, h_eq = 0, h_pi = 0;
    cudaMemcpy(d_probe_vector_offset, &h_off, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_offset_compact, &h_compact, sizeof(int), cudaMemcpyHostToDevice);
    const int h_cnt_i = (int)h_cnt;
    cudaMemcpy(d_probe_vector_count, &h_cnt_i, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_entry_cluster_id, &h_ec, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_entry_query_start, &h_qs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_entry_query_count, &h_qc, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_entry_probe_indices, &h_pi, sizeof(int), cudaMemcpyHostToDevice);

    std::vector<float> h_topk_init_dist(static_cast<size_t>(k), std::numeric_limits<float>::infinity());
    std::vector<int> h_topk_init_idx(static_cast<size_t>(k), -1);
    /* : cudaMemcpy, int;overlap : submit  */
    int* h_single_query_id = nullptr;
    cudaHostAlloc(&h_single_query_id, sizeof(int), cudaHostAllocDefault);
    const int max_submits = n_warmup * n_query + n_rounds * n_query + 128;
    int* h_entry_ring = nullptr;
    cudaHostAlloc(&h_entry_ring, static_cast<size_t>(max_submits) * sizeof(int), cudaHostAllocDefault);

    init_array_multithreaded<float>(
        h_query,
        (size_t)n_query * (size_t)n_dim,
        12345u,
        -1.0f,
        1.0f,
        pack_threads);
    compute_norms_multithreaded(h_query, h_query_norm, n_query, n_dim, pack_threads);

    init_interleaved_and_vec_norms_per_query_multithreaded(
        h_ivf,
        interleaved_floats,
        h_vec_norm,
        n_query,
        n_vec,
        n_dim,
        900000u,
        -1.0f,
        1.0f,
        pack_threads);

    cudaMemcpy(d_query, h_query, (size_t)n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_norm, h_query_norm, (size_t)n_query * sizeof(float), cudaMemcpyHostToDevice);

    float* h_ref = (float*)std::malloc((size_t)n_query * n_vec * sizeof(float));
    float* h_gpu_topk_dist = (float*)std::malloc((size_t)n_query * k * sizeof(float));
    float* h_cpu_topk_dist = (float*)std::malloc((size_t)n_query * k * sizeof(float));
    std::vector<float*> cpu_row_ptrs(static_cast<size_t>(n_query));
    std::vector<float*> gpu_row_ptrs(static_cast<size_t>(n_query));

    double cpu_ref_ms = 0.0;
    {
        auto __t_ref0 = std::chrono::steady_clock::now();
        cpu_chunk_topk_reference_from_host_interleaved_mt(
            use_cosine,
            h_query,
            h_query_norm,
            h_ivf,
            interleaved_floats,
            h_vec_norm,
            n_query,
            n_vec,
            n_dim,
            k,
            h_ref,
            h_cpu_topk_dist,
            cpu_row_ptrs.data(),
            h_gpu_topk_dist,
            gpu_row_ptrs.data(),
            pack_threads);
        auto __t_ref1 = std::chrono::steady_clock::now();
        cpu_ref_ms = std::chrono::duration<double, std::milli>(__t_ref1 - __t_ref0).count();
    }

    cudaStream_t st_copy = nullptr, st_compute = nullptr;
    cudaEvent_t ev_copy[kSlots], ev_gem[kSlots];
    for (int s = 0; s < kSlots; ++s) {
        cudaEventCreate(&ev_copy[s]);
        cudaEventCreate(&ev_gem[s]);
    }
    cudaStreamCreate(&st_copy);
    cudaStreamCreate(&st_compute);
#if defined(USE_NVTX) && USE_NVTX
    NVTX_NAME_CUDA_STREAM(st_copy, "gemv_pipeline_chunk/st_copy_H2D");
    NVTX_NAME_CUDA_STREAM(st_compute, "gemv_pipeline_chunk/st_compute_chunk");
    static const char* const k_ev_copy_names[kSlots] = {
        "gemv_pipeline_chunk/ev_copy_done_slot0",
        "gemv_pipeline_chunk/ev_copy_done_slot1",
        "gemv_pipeline_chunk/ev_copy_done_slot2",
        "gemv_pipeline_chunk/ev_copy_done_slot3",
    };
    static const char* const k_ev_gem_names[kSlots] = {
        "gemv_pipeline_chunk/ev_gem_done_slot0",
        "gemv_pipeline_chunk/ev_gem_done_slot1",
        "gemv_pipeline_chunk/ev_gem_done_slot2",
        "gemv_pipeline_chunk/ev_gem_done_slot3",
    };
    for (int s = 0; s < kSlots; ++s) {
        NVTX_NAME_CUDA_EVENT(ev_copy[s], k_ev_copy_names[s]);
        NVTX_NAME_CUDA_EVENT(ev_gem[s], k_ev_gem_names[s]);
    }
#endif
    auto run_serial = [&]() {
        cudaDeviceSynchronize();
        NVTX_PUSH("gemv_pipeline_chunk/serial/warmup");
        for (int w = 0; w < n_warmup; ++w) {
            for (int q = 0; q < n_query; ++q) {
                const int slot = q % kSlots;
                cudaMemcpy(d_chunk_vector[slot], h_ivf + q * interleaved_floats, ivf_bytes, cudaMemcpyHostToDevice);
                float* tmp_cn = (float*)std::malloc(static_cast<size_t>(chunk_norm_floats) * sizeof(float));
                const float* vn_row = h_vec_norm + (size_t)q * n_vec;
                const int n_blocks = (n_vec + 31) / 32;
                for (int b = 0; b < n_blocks; ++b)
                    for (int lane = 0; lane < 32; ++lane) {
                        int vi = b * 32 + lane;
                        tmp_cn[b * 32 + lane] = (vi < n_vec) ? vn_row[vi] : 0.f;
                    }
                cudaMemcpy(d_chunk_norm[slot], tmp_cn, static_cast<size_t>(chunk_norm_floats) * sizeof(float), cudaMemcpyHostToDevice);
                std::free(tmp_cn);

                cudaMemcpy(
                    d_topk_dist + (size_t)q * k,
                    h_topk_init_dist.data(),
                    static_cast<size_t>(k) * sizeof(float),
                    cudaMemcpyHostToDevice);
                cudaMemcpy(
                    d_topk_index + (size_t)q * k,
                    h_topk_init_idx.data(),
                    static_cast<size_t>(k) * sizeof(idx_t),
                    cudaMemcpyHostToDevice);
                *h_single_query_id = q;
                cudaMemcpy(d_entry_queries, h_single_query_id, sizeof(int), cudaMemcpyHostToDevice);

                launch_chunk_fine(
                    use_cosine,
                    d_query,
                    d_chunk_vector[slot],
                    d_chunk_norm[slot],
                    d_probe_vector_offset,
                    d_probe_vector_offset_compact,
                    d_probe_vector_count,
                    d_entry_cluster_id,
                    d_entry_query_start,
                    d_entry_query_count,
                    d_entry_queries,
                    d_entry_probe_indices,
                    d_query_norm,
                    n_vec,
                    n_dim,
                    k,
                    d_topk_dist,
                    d_topk_index,
                    0);
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline_chunk/serial/timed_start");
        NVTX_PUSH("gemv_pipeline_chunk/serial/timed");
        auto __t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < n_rounds; ++r) {
            for (int q = 0; q < n_query; ++q) {
                const int slot = q % kSlots;
                cudaMemcpy(d_chunk_vector[slot], h_ivf + q * interleaved_floats, ivf_bytes, cudaMemcpyHostToDevice);
                float* tmp_cn = (float*)std::malloc(static_cast<size_t>(chunk_norm_floats) * sizeof(float));
                const float* vn_row = h_vec_norm + (size_t)q * n_vec;
                const int n_blocks = (n_vec + 31) / 32;
                for (int b = 0; b < n_blocks; ++b)
                    for (int lane = 0; lane < 32; ++lane) {
                        int vi = b * 32 + lane;
                        tmp_cn[b * 32 + lane] = (vi < n_vec) ? vn_row[vi] : 0.f;
                    }
                cudaMemcpy(d_chunk_norm[slot], tmp_cn, static_cast<size_t>(chunk_norm_floats) * sizeof(float), cudaMemcpyHostToDevice);
                std::free(tmp_cn);

                cudaMemcpy(
                    d_topk_dist + (size_t)q * k,
                    h_topk_init_dist.data(),
                    static_cast<size_t>(k) * sizeof(float),
                    cudaMemcpyHostToDevice);
                cudaMemcpy(
                    d_topk_index + (size_t)q * k,
                    h_topk_init_idx.data(),
                    static_cast<size_t>(k) * sizeof(idx_t),
                    cudaMemcpyHostToDevice);
                *h_single_query_id = q;
                cudaMemcpy(d_entry_queries, h_single_query_id, sizeof(int), cudaMemcpyHostToDevice);

                launch_chunk_fine(
                    use_cosine,
                    d_query,
                    d_chunk_vector[slot],
                    d_chunk_norm[slot],
                    d_probe_vector_offset,
                    d_probe_vector_offset_compact,
                    d_probe_vector_count,
                    d_entry_cluster_id,
                    d_entry_query_start,
                    d_entry_query_count,
                    d_entry_queries,
                    d_entry_probe_indices,
                    d_query_norm,
                    n_vec,
                    n_dim,
                    k,
                    d_topk_dist,
                    d_topk_index,
                    0);
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        auto __t1 = std::chrono::steady_clock::now();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline_chunk/serial/timed_end");
        return std::chrono::duration<double, std::milli>(__t1 - __t0).count();
    };

    auto run_overlap = [&]() {
        cudaDeviceSynchronize();
        int submit_idx = 0;
        NVTX_PUSH("gemv_pipeline_chunk/overlap/warmup");
        for (int w = 0; w < n_warmup; ++w) {
            for (int q = 0; q < n_query; ++q) {
                const int slot = q % kSlots;
                pipeline_submit_round_per_query(
                    submit_idx++,
                    slot,
                    q,
                    st_copy,
                    st_compute,
                    ev_copy,
                    ev_gem,
                    h_ivf,
                    ivf_bytes,
                    interleaved_floats,
                    n_vec,
                    chunk_norm_floats,
                    h_vec_norm,
                    h_topk_init_dist.data(),
                    h_topk_init_idx.data(),
                    k,
                    d_query,
                    d_query_norm,
                    d_chunk_vector,
                    d_chunk_norm,
                    d_probe_vector_offset,
                    d_probe_vector_offset_compact,
                    d_probe_vector_count,
                    d_entry_cluster_id,
                    d_entry_query_start,
                    d_entry_query_count,
                    d_entry_queries,
                    d_entry_probe_indices,
                    d_topk_dist,
                    d_topk_index,
                    n_dim,
                    use_cosine,
                    h_entry_ring);
            }
        }
        cudaDeviceSynchronize();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline_chunk/overlap/timed_start");
        NVTX_PUSH("gemv_pipeline_chunk/overlap/timed");
        auto __t0o = std::chrono::steady_clock::now();
        for (int r = 0; r < n_rounds; ++r) {
            for (int q = 0; q < n_query; ++q) {
                const int slot = q % kSlots;
                pipeline_submit_round_per_query(
                    submit_idx++,
                    slot,
                    q,
                    st_copy,
                    st_compute,
                    ev_copy,
                    ev_gem,
                    h_ivf,
                    ivf_bytes,
                    interleaved_floats,
                    n_vec,
                    chunk_norm_floats,
                    h_vec_norm,
                    h_topk_init_dist.data(),
                    h_topk_init_idx.data(),
                    k,
                    d_query,
                    d_query_norm,
                    d_chunk_vector,
                    d_chunk_norm,
                    d_probe_vector_offset,
                    d_probe_vector_offset_compact,
                    d_probe_vector_count,
                    d_entry_cluster_id,
                    d_entry_query_start,
                    d_entry_query_count,
                    d_entry_queries,
                    d_entry_probe_indices,
                    d_topk_dist,
                    d_topk_index,
                    n_dim,
                    use_cosine,
                    h_entry_ring);
            }
        }
        cudaStreamSynchronize(st_copy);
        cudaStreamSynchronize(st_compute);
        auto __t1o = std::chrono::steady_clock::now();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline_chunk/overlap/timed_end");
        return std::chrono::duration<double, std::milli>(__t1o - __t0o).count();
    };

    COUT_ENDL("run serial");
    NVTX_PUSH("gemv_pipeline_chunk/run_serial");
    double serial_ms = run_serial();
    NVTX_POP();

    COUT_ENDL("run overlap");
    NVTX_PUSH("gemv_pipeline_chunk/run_overlap");
    double overlap_ms = run_overlap();
    NVTX_POP();
    double speedup = (overlap_ms > 1e-6) ? (serial_ms / overlap_ms) : 0.0;

    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu_topk_dist, d_topk_dist, (size_t)n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    COUT_ENDL("compare_set_2D");


    const bool pass_overlap_vs_cpu = compare_set_2D<float>(
        cpu_row_ptrs.data(),
        gpu_row_ptrs.data(),
        n_query,
        k,
        EPSILON);
    const double pass_rate = pass_overlap_vs_cpu ? 1.0 : 0.0;

    cudaStreamDestroy(st_copy);
    cudaStreamDestroy(st_compute);
    for (int s = 0; s < kSlots; ++s) {
        cudaEventDestroy(ev_copy[s]);
        cudaEventDestroy(ev_gem[s]);
    }

    cudaFree(d_query);
    cudaFree(d_query_norm);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    cudaFree(d_probe_vector_offset);
    cudaFree(d_probe_vector_offset_compact);
    cudaFree(d_probe_vector_count);
    cudaFree(d_entry_cluster_id);
    cudaFree(d_entry_query_start);
    cudaFree(d_entry_query_count);
    cudaFree(d_entry_queries);
    cudaFree(d_entry_probe_indices);
    for (int s = 0; s < kSlots; ++s) {
        cudaFree(d_chunk_vector[s]);
        cudaFree(d_chunk_norm[s]);
    }
    cudaFreeHost(h_query);
    cudaFreeHost(h_query_norm);
    cudaFreeHost(h_ivf);
    cudaFreeHost(h_vec_norm);
    cudaFreeHost(h_single_query_id);
    cudaFreeHost(h_entry_ring);

    std::free(h_ref);
    std::free(h_gpu_topk_dist);
    std::free(h_cpu_topk_dist);

    return {pass_rate, serial_ms, overlap_ms, speedup, cpu_ref_ms};
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    MetricsCollector metrics;
    metrics.set_columns(
        "pass_rate",
        "n_query",
        "n_vec",
        "n_dim",
        "k_topk",
        "mode",
        "pack_threads",
        "n_rounds",
        "cpu_ref_ms",
        "serial_ms",
        "overlap_ms",
        "speedup");
    metrics.set_num_repeats(1);

    cudaSetDevice(0);

    COUT_ENDL(
        " launch_gemv_interleaved_*_topk_chunk ( k ,compare_set_2D)...");

    const int n_rounds = 1;
    const int n_warmup = 1;

    PARAM_3D(
        // n_query, (2, 3),
        // n_dim, (128, 31, 17, 7, 768),
        // n_query, (2, 10, 1000),
        n_query, (10),
        // n_dim, (128, 200, 768),
        n_dim, (128),
        // n_vec, (12, 256, 441, 1000)){
        n_vec, (10000)){
            // n_vec, (12, 32, 33, 50, 128, 220,  225, 10000, 37291)){
        PARAM_2D(pack_threads, (4), use_cosine, (0)){
            const int k = 10;
           if (QUIET) {
                COUT_VAL(
                    "[run] pack_threads=", pack_threads,
                    " n_query=", n_query,
                    " n_vec=", n_vec,
                    " n_dim=", n_dim,
                    " k=", k,
                    " mode=", use_cosine);
            }
            auto r = run_case(
                n_query,
                n_vec,
                n_dim,
                use_cosine != 0,
                n_rounds,
                n_warmup,
                pack_threads,
                k);
            metrics.add_row(
                r[0],
                (double)n_query,
                (double)n_vec,
                (double)n_dim,
                (double)k,
                (double)use_cosine,
                (double)pack_threads,
                (double)n_rounds,
                r[4],
                r[1],
                r[2],
                r[3]);
        }
    }

    cudaDeviceSynchronize();

    COUT_ENDL();
    COUT_ENDL("==========  ==========");
    metrics.print_table();
    metrics.export_csv("gemv_interleaved_pipeline_chunk_metrics.csv");

    bool all_pass = true;
    for (size_t i = 0; i < metrics.size(); ++i) {
        auto row = metrics.get_row(i);
        if (row.size() > 0 && row[0] < 1.0)
            all_pass = false;
    }
    COUT_ENDL(all_pass ? "all_test_passed ✅" : "some tests failed ❌");
    return all_pass ? 0 : 1;
}
