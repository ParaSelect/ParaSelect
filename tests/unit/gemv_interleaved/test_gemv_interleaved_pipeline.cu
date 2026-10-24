/**
 * @file test_gemv_interleaved_pipeline.cu
 *
 * @brief `launch_gemm_interleaved_*` :query **row-major**;** query ** `n_vec` (host row-major
 *  query  `init_array_interleaved_multithreaded`  interleaved( row-major ).
 *
 * :`gemm_interleaved_*` **** interleaved ** query **; query,, **`n_query`
 * `n_query=1`  launch**  `d_out`( IVF" query  list"). H2D+launch  query , PCIe
 *  **L2 ** host  `compute_l2_norm_cpu_interleaved` ,** H2D**  `d_vec_norm_all`;
 *  ** interleaved**, vec norm("norm ,").
 *
 * `pass_rate`  **overlap **  `d_out`  CPU .
 *
 * **round **:** =  `n_query` ** H2D+launch();**`n_rounds`** = " query".
 * `n_warmup`  ****( query).
 *
 * **Nsight Systems**: `PARASELECT_USE_NVTX`( ON), **NVTX**  `nvtxMarkA` ;
 *  `run_test_gemv_interleaved_pipeline_nsys.sh`  `make_all.sh --test-nsys`  `nsys-rep`.
 */

#include "src/utils/pch.h"
#include "src/utils/nsight_nvtx.h"
#include "src/search/kernels/fine/gpu_fine/gemv_interleaved.cuh"
#include "cpu/utils/l2norm_cpu.h"
#include "cpu/utils/cpu_array_utils.h"
#include "common/test_utils.cuh"
#include "common/params_macros.cuh"
#include "cpu/search/cpu_fine.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define EPSILON 1e-3f

static constexpr int kSlots = 4;

static void launch_gemm_fine(
    bool use_cosine,
    const float* d_q,
    const float* d_ivf,
    const float* d_qn,
    const float* d_vn,
    int n_query_batch,
    int n_vec,
    int n_dim,
    float* d_out,
    cudaStream_t st) {
    if (use_cosine)
        launch_gemm_interleaved_cos(
            d_q, d_ivf, d_qn, d_vn, n_query_batch, n_vec, n_dim, d_out, st);
    else
        launch_gemm_interleaved_l2(
            d_q, d_ivf, d_qn, d_vn, n_query_batch, n_vec, n_dim, d_out, st);
}

/**
 *  q  query  interleaved;**vec L2 norm  GPU**(`d_vec_norm_all + q*n_vec`),.
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
    const float* d_query,
    const float* d_query_norm,
    const float* d_vec_norm_all,
    float* d_interleaved[kSlots],
    float* d_out,
    int n_dim,
    bool use_cosine) {
    if (i >= kSlots)
        cudaStreamWaitEvent(st_copy, ev_gem_done[slot], 0);

    const float* h_src_ivf = h_ivf_base + q * interleaved_floats;
    const float* d_vn_row = d_vec_norm_all + (size_t)q * n_vec;

    cudaMemcpyAsync(
        d_interleaved[slot],
        h_src_ivf,
        interleaved_bytes,
        cudaMemcpyHostToDevice,
        st_copy);
    cudaEventRecord(ev_copy_done[slot], st_copy);

    cudaStreamWaitEvent(st_compute, ev_copy_done[slot], 0);
    launch_gemm_fine(
        use_cosine,
        d_query + (size_t)q * n_dim,
        d_interleaved[slot],
        d_query_norm + q,
        d_vn_row,
        1,
        n_vec,
        n_dim,
        d_out + (size_t)q * n_vec,
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
    int pack_threads) {
    float* d_query = nullptr;
    float* d_query_norm = nullptr;
    float* d_interleaved[kSlots];
    float* d_vec_norm_all = nullptr;
    float* d_out = nullptr;

    const size_t ivf_bytes = paraselect::interleaved_storage_bytes(n_vec, n_dim);
    const size_t interleaved_floats = paraselect::interleaved_storage_floats(n_vec, n_dim);
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
        cudaMalloc(&d_interleaved[s], ivf_bytes);
    }

    cudaMalloc(&d_query, (size_t)n_query * n_dim * sizeof(float));
    cudaMalloc(&d_query_norm, (size_t)n_query * sizeof(float));
    cudaMalloc(&d_vec_norm_all, total_vec_norm_bytes);
    cudaMalloc(&d_out, (size_t)n_query * n_vec * sizeof(float));

    init_array_multithreaded<float>(
        h_query,
        (size_t)n_query * (size_t)n_dim,
        12345u,
        -1.0f,
        1.0f,
        pack_threads);
    compute_norms(h_query, h_query_norm, n_query, n_dim);

    for (int q = 0; q < n_query; ++q) {
        float* ivf_blk = h_ivf + q * interleaved_floats;
        float* vn_blk = h_vec_norm + (size_t)q * n_vec;
        init_array_interleaved_multithreaded(
            ivf_blk,
            n_vec,
            n_dim,
            900000u + (unsigned)q,
            -1.0f,
            1.0f,
            pack_threads);
        paraselect::compute_l2_norm_cpu_interleaved(ivf_blk, vn_blk, n_vec, n_dim);
    }

    cudaMemcpy(d_query, h_query, (size_t)n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_norm, h_query_norm, (size_t)n_query * sizeof(float), cudaMemcpyHostToDevice);
    /*  vec L2 norm  host,, norm */
    cudaMemcpy(d_vec_norm_all, h_vec_norm, total_vec_norm_bytes, cudaMemcpyHostToDevice);

    float* h_ref = (float*)std::malloc((size_t)n_query * n_vec * sizeof(float));
    float* h_gpu = (float*)std::malloc((size_t)n_query * n_vec * sizeof(float));

    /* CPU  */
    if (use_cosine)
        compute_ref_cos(
            h_query,
            h_query_norm,
            h_ivf,
            interleaved_floats,
            h_vec_norm,
            n_query,
            n_vec,
            n_dim,
            h_ref,
            pack_threads);
    else
        compute_ref_l2(
            h_query,
            h_query_norm,
            h_ivf,
            interleaved_floats,
            h_vec_norm,
            n_query,
            n_vec,
            n_dim,
            h_ref,
            pack_threads);

    cudaStream_t st_copy = nullptr, st_compute = nullptr;
    cudaEvent_t ev_copy[kSlots], ev_gem[kSlots];
    for (int s = 0; s < kSlots; ++s) {
        cudaEventCreate(&ev_copy[s]);
        cudaEventCreate(&ev_gem[s]);
    }
    cudaStreamCreate(&st_copy);
    cudaStreamCreate(&st_compute);
#if defined(USE_NVTX) && USE_NVTX
    /* nvtxNameCuda* ,/stream  →  */
    NVTX_NAME_CUDA_STREAM(st_copy, "gemv_pipeline/st_copy_H2D");
    NVTX_NAME_CUDA_STREAM(st_compute, "gemv_pipeline/st_compute_gemm");
    static const char* const k_ev_copy_names[kSlots] = {
        "gemv_pipeline/ev_copy_done_slot0",
        "gemv_pipeline/ev_copy_done_slot1",
        "gemv_pipeline/ev_copy_done_slot2",
        "gemv_pipeline/ev_copy_done_slot3",
    };
    static const char* const k_ev_gem_names[kSlots] = {
        "gemv_pipeline/ev_gem_done_slot0",
        "gemv_pipeline/ev_gem_done_slot1",
        "gemv_pipeline/ev_gem_done_slot2",
        "gemv_pipeline/ev_gem_done_slot3",
    };
    for (int s = 0; s < kSlots; ++s) {
        NVTX_NAME_CUDA_EVENT(ev_copy[s], k_ev_copy_names[s]);
        NVTX_NAME_CUDA_EVENT(ev_gem[s], k_ev_gem_names[s]);
    }
#endif

    auto run_serial = [&]() {
        cudaDeviceSynchronize();
        NVTX_PUSH("gemv_pipeline/serial/warmup");
        /* warmup:n_warmup , query */
        for (int w = 0; w < n_warmup; ++w) {
            for (int q = 0; q < n_query; ++q) {
                const int slot = q % kSlots;
                cudaMemcpy(d_interleaved[slot], h_ivf + q * interleaved_floats, ivf_bytes, cudaMemcpyHostToDevice);
                launch_gemm_fine(
                    use_cosine,
                    d_query + (size_t)q * n_dim,
                    d_interleaved[slot],
                    d_query_norm + q,
                    d_vec_norm_all + (size_t)q * n_vec,
                    1,
                    n_vec,
                    n_dim,
                    d_out + (size_t)q * n_vec,
                    0);
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline/serial/timed_start");
        NVTX_PUSH("gemv_pipeline/serial/timed");
        auto __t0 = std::chrono::steady_clock::now();
        /* :n_rounds , query */
        for (int r = 0; r < n_rounds; ++r) {
            for (int q = 0; q < n_query; ++q) {
                const int slot = q % kSlots;
                cudaMemcpy(d_interleaved[slot], h_ivf + q * interleaved_floats, ivf_bytes, cudaMemcpyHostToDevice);
                launch_gemm_fine(
                    use_cosine,
                    d_query + (size_t)q * n_dim,
                    d_interleaved[slot],
                    d_query_norm + q,
                    d_vec_norm_all + (size_t)q * n_vec,
                    1,
                    n_vec,
                    n_dim,
                    d_out + (size_t)q * n_vec,
                    0);
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        auto __t1 = std::chrono::steady_clock::now();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline/serial/timed_end");
        return std::chrono::duration<double, std::milli>(__t1 - __t0).count();
    };

    auto run_overlap = [&]() {
        cudaDeviceSynchronize();
        int submit_idx = 0;
        NVTX_PUSH("gemv_pipeline/overlap/warmup");
        /* warmup:n_warmup , q=0..n_query-1 */
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
                    d_query,
                    d_query_norm,
                    d_vec_norm_all,
                    d_interleaved,
                    d_out,
                    n_dim,
                    use_cosine);
            }
        }
        cudaDeviceSynchronize();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline/overlap/timed_start");
        NVTX_PUSH("gemv_pipeline/overlap/timed");
        auto __t0o = std::chrono::steady_clock::now();
        /* :n_rounds , q=0..n_query-1 */
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
                    d_query,
                    d_query_norm,
                    d_vec_norm_all,
                    d_interleaved,
                    d_out,
                    n_dim,
                    use_cosine);
            }
        }
        cudaStreamSynchronize(st_copy);
        cudaStreamSynchronize(st_compute);
        auto __t1o = std::chrono::steady_clock::now();
        NVTX_POP();
        NVTX_MARK("gemv_pipeline/overlap/timed_end");
        return std::chrono::duration<double, std::milli>(__t1o - __t0o).count();
    };

    NVTX_PUSH("gemv_pipeline/run_serial");
    double serial_ms = run_serial();
    NVTX_POP();
    NVTX_PUSH("gemv_pipeline/run_overlap");
    double overlap_ms = run_overlap();
    NVTX_POP();
    double speedup = (overlap_ms > 1e-6) ? (serial_ms / overlap_ms) : 0.0;

    /* pass_rate:overlap  d_out vs CPU( query) */
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu, d_out, (size_t)n_query * n_vec * sizeof(float), cudaMemcpyDeviceToHost);
    const bool pass_overlap_vs_cpu =
        compare_1D(h_gpu, h_ref, n_query * n_vec, EPSILON);
    double pass_rate = pass_overlap_vs_cpu ? 1.0 : 0.0;

    cudaStreamDestroy(st_copy);
    cudaStreamDestroy(st_compute);
    for (int s = 0; s < kSlots; ++s) {
        cudaEventDestroy(ev_copy[s]);
        cudaEventDestroy(ev_gem[s]);
    }

    cudaFree(d_query);
    cudaFree(d_query_norm);
    cudaFree(d_vec_norm_all);
    cudaFree(d_out);
    for (int s = 0; s < kSlots; ++s) {
        cudaFree(d_interleaved[s]);
    }
    cudaFreeHost(h_query);
    cudaFreeHost(h_query_norm);
    cudaFreeHost(h_ivf);
    cudaFreeHost(h_vec_norm);

    std::free(h_ref);
    std::free(h_gpu);

    return {pass_rate, serial_ms, overlap_ms, speedup};
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
        "mode",
        "pack_threads",
        "n_rounds",
        "serial_ms",
        "overlap_ms",
        "speedup");
    metrics.set_num_repeats(1);

    cudaSetDevice(0);

    COUT_ENDL(" launch_gemm_interleaved_* ( query  n_vec ,query row-major)...");

    // const int n_rounds = 64;
    const int n_rounds = 1;
    const int n_warmup = 0;

    PARAM_3D(
        n_query, (3, 10, 32, 64),
        n_dim, (128, 200, 3, 4, 8, 12, 16, 9, 7, 15, 17, 31, 33, 768),
        // n_dim, (128, 200, 768),
        // n_dim, (243),
        // n_vec, (12, 441, 1000, 2048)){
        // n_vec, (1)){
        n_vec, (12, 441, 1000, 2048)){
        // n_query, (64, 128),
        // n_vec, (2048, 4096)){

        PARAM_2D(
            // pack_threads, (1, 4),
            // n_dim, (128, 200, 768),
            // use_cosine, (0, 1)){
            pack_threads, (4),
            use_cosine, (0)){

            if (QUIET) {
                COUT_VAL(
                    "[run] pack_threads=",
                    pack_threads,
                    " n_query=",
                    n_query,
                    " n_vec=",
                    n_vec,
                    " n_dim=",
                    n_dim,
                    " mode=",
                    use_cosine);
            }
            auto r = run_case(
                n_query,
                n_vec,
                n_dim,
                use_cosine != 0,
                n_rounds,
                n_warmup,
                pack_threads);
            metrics.add_row(
                r[0],
                (double)n_query,
                (double)n_vec,
                (double)n_dim,
                (double)use_cosine,
                (double)pack_threads,
                (double)n_rounds,
                r[1],
                r[2],
                r[3]);
        }
    }

    cudaDeviceSynchronize();

    COUT_ENDL();
    COUT_ENDL("==========  ==========");
    metrics.print_table();
    metrics.export_csv("gemv_interleaved_pipeline_metrics.csv");

    bool all_pass = true;
    for (size_t i = 0; i < metrics.size(); ++i) {
        auto row = metrics.get_row(i);
        if (row.size() > 0 && row[0] < 1.0)
            all_pass = false;
    }
    COUT_ENDL(all_pass ? "all_test_passed ✅" : "some tests failed ❌");
    return all_pass ? 0 : 1;
}
