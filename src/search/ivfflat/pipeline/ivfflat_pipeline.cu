// ============================================================
// CPU (, batch  pool_slot=0)
// ============================================================
#include "utils/pch.h"
#include "search/ivfflat/pipeline/ivfflat_pipeline.cuh"
#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/ivfflat_search_ctx.h"
#include "utils/utils.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_executor.h"
#include "l2norm/l2norm.cuh"
#include "search/ivf_common/ivf_search_stats_collector.h"
#include "ivfflat_cpu_overlap.cuh"
#include "ivfflat_probe_metadata.cuh"

#include <cstring>
#include <chrono>

using paraselect::search::IVFSearchStats;
#include <stdexcept>
#include <tuple>
#include <vector>
#include <algorithm>

static void ivfflat_cpu_fine_job(void* owner, const paraselect::cpu_fine::CpuFineBatchView& view) {
    ivfflat_fine_cpu(static_cast<IVFSearchCtx*>(owner), view);
}

// Device-side fp32 -> fp16 query conversion for the FP16 coarse GEMM.
// Replaces the previous host-side scalar loop + synchronous cudaMemcpy, which
// showed up as ~0.8ms @bs2048 and ~2.0ms @bs8192 in query_preprocess_ms.
__global__ void ivfflat_fp32_to_fp16_query_kernel(const float* __restrict__ src,
                                                  __half* __restrict__ dst,
                                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half_rn(src[i]);
    }
}

static void ivfflat_pipeline_serial(
    IVFSearchCtx* ctx,
    IVFSearchStats* stats) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFFlatFineCtx* fine = &ctx->flat_fine;
    IVFSearchArg* arg = &ctx->arg;
    const int dim = arg->n_dim;
    const int k = arg->k;
    const int n_probes = arg->n_probes;
    const int n_query = arg->n_query;
    const int batch_size = arg->query_batch_size;
    const float* h_queries = coarse->h_queries;
    const uint8_t* h_queries_u8 = coarse->h_queries_u8;
    float* h_dist_out = coarse->h_topk_dist_out;
    idx_t* h_idx_out = coarse->h_topk_index_out;

    const int n_batches = (n_query + batch_size - 1) / batch_size;
    cudaStream_t stream = coarse->stream_0;
    const bool resident = arg->fine_kind == IVFFlatFineKind::Resident;

    ivfflat_prepare(ctx);
    const int coarse_sub_batch_size = (coarse->coarse_sub_batch_size > 0)
                                          ? std::min(coarse->coarse_sub_batch_size, batch_size)
                                          : batch_size;

    paraselect::search::IVFSearchStatsCollector collector_storage(stats);
    auto* collector = stats ? &collector_storage : nullptr;
    if (collector) {
        collector->startTotal();
    }
    auto time_stage = [&](const char* name, auto&& fn) {
        (void)name;
        if (!collector) {
            fn();
            return 0.0;
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    auto collect_cpu_fine_scan_stats = [&]() {
        long long candidate_vectors = 0;
        long long max_candidates_per_query = 0;
        int max_probe_list_size = 0;
        if (!fine->h_top_nprobe_batch || !fine->h_cluster_sizes) {
            return std::make_tuple(candidate_vectors, max_candidates_per_query, max_probe_list_size);
        }
        for (int q = 0; q < coarse->current_n_querys; ++q) {
            long long query_candidates = 0;
            for (int p = 0; p < n_probes; ++p) {
                const idx_t cluster = fine->h_top_nprobe_batch[(size_t)q * n_probes + p];
                if (cluster < 0 || cluster >= arg->n_total_clusters) continue;
                const int list_size = fine->h_cluster_sizes[(int)cluster];
                query_candidates += list_size;
                max_probe_list_size = std::max(max_probe_list_size, list_size);
            }
            candidate_vectors += query_candidates;
            max_candidates_per_query = std::max(max_candidates_per_query, query_candidates);
        }
        return std::make_tuple(candidate_vectors, max_candidates_per_query, max_probe_list_size);
    };

    long long total_candidate_vectors = 0;
    long long global_max_candidates_per_query = 0;
    int global_max_probe_list_size = 0;
    double resident_threshold_candidate_sum = 0.0;
    double resident_threshold_max_candidates = 0.0;
    paraselect::cpu_fine::CpuFineExecutor fine_executor(false);

    for (int bi = 0; bi < n_batches; ++bi) {
        const int batch_start = bi * batch_size;
        const int batch_curr = std::min(batch_size, n_query - batch_start);
        if (collector) {
            collector->startBatch(bi, batch_start, batch_curr);
        }

        const int sid = 0;
        const float* batch_queries = h_queries + (size_t)batch_start * dim;
        const uint8_t* batch_queries_u8 = h_queries_u8 ? (h_queries_u8 + (size_t)batch_start * dim) : nullptr;
        float* batch_dist_out = h_dist_out + (size_t)batch_start * k;
        idx_t* batch_idx_out = h_idx_out + (size_t)batch_start * k;
        coarse->h_queries = batch_queries;
        coarse->h_queries_u8 = batch_queries_u8;
        coarse->h_topk_dist_out = batch_dist_out;
        coarse->h_topk_index_out = batch_idx_out;
        double query_preprocess_ms = 0.0;
        double coarse_search_ms = 0.0;
        double coarse_stage1_ms = 0.0;
        double coarse_d2h_ms = 0.0;
        double coarse_gemm_ms = 0.0;
        double coarse_topk_ms = 0.0;
        double probe_metadata_ms = 0.0;
        double cpu_fine_ms = 0.0;

        IVFFlatCoarseCtx* coarse_buffers = coarse;
        if (!resident && !fine->h_top_nprobe_batch) {
            throw std::runtime_error("pipeline: missing h_top_nprobe_batch scratch; prepare was not completed");
        }
        if (resident && coarse_sub_batch_size < batch_curr) {
            throw std::runtime_error("resident IVFFlat does not support coarse sub-batch tiling yet");
        }

        for (int coarse_off = 0; coarse_off < batch_curr; coarse_off += coarse_sub_batch_size) {
            const int n_this = std::min(coarse_sub_batch_size, batch_curr - coarse_off);
            const float* sub_query = batch_queries + (size_t)coarse_off * dim;
            const uint8_t* sub_query_u8 = batch_queries_u8 ? (batch_queries_u8 + (size_t)coarse_off * dim) : nullptr;

            coarse->current_n_querys = n_this;

            query_preprocess_ms += time_stage("query_preprocess", [&]() {
                NVTX_PUSH("Query_Preprocess_H2D_Norm");
                cudaMemcpyAsync(coarse_buffers->d_pool_query_batch[sid],
                                sub_query,
                                (size_t)n_this * dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream);
                if (coarse_buffers->d_pool_query_batch_u8[sid]) {
                    if (!sub_query_u8) throw std::runtime_error("u8 resident query buffer missing");
                    cudaMemcpyAsync(coarse_buffers->d_pool_query_batch_u8[sid],
                                    sub_query_u8,
                                    (size_t)n_this * dim * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice, stream);
                }
                compute_l2_norm_gpu(
                    coarse_buffers->d_pool_query_batch[sid],
                    coarse_buffers->d_pool_query_norm[sid],
                    n_this,
                    arg->n_dim,
                    L2NORM_AUTO, stream);

                if (arg->fp16_coarse) {
                    const int n_elems = n_this * dim;
                    if (n_elems > 0 && coarse_buffers->d_pool_query_batch[sid] &&
                        coarse_buffers->d_pool_query_fp16[sid]) {
                        constexpr int kThreads = 256;
                        const int n_blocks = (n_elems + kThreads - 1) / kThreads;
                        ivfflat_fp32_to_fp16_query_kernel<<<n_blocks, kThreads, 0, stream>>>(
                            coarse_buffers->d_pool_query_batch[sid],
                            coarse_buffers->d_pool_query_fp16[sid],
                            n_elems);
                        CHECK_CUDA_ERRORS;
                    }
                }
                NVTX_POP();
            });

            coarse_search_ms += time_stage("coarse_search", [&]() {
                coarse->debug_coarse_gemm_ms = 0.0;
                coarse->debug_coarse_topk_ms = 0.0;
                coarse_stage1_ms += time_stage("coarse_stage1", [&]() {
                    ivfflat_coarse(ctx, sid);
                    CHECK_CUDA_ERRORS;
                });
                coarse_gemm_ms += coarse->debug_coarse_gemm_ms;
                coarse_topk_ms += coarse->debug_coarse_topk_ms;

                if (!resident) {
                    coarse_d2h_ms += time_stage("coarse_d2h", [&]() {
                        CUDA_CHECK(cudaMemcpyAsync(fine->h_top_nprobe_batch + (size_t)coarse_off * n_probes,
                                                   coarse->d_top_nprobe_index[sid],
                                                   (size_t)n_this * n_probes * sizeof(idx_t), cudaMemcpyDeviceToHost, stream));
                        if (coarse->h_coarse_dist != nullptr) {
                            CUDA_CHECK(cudaMemcpyAsync(coarse->h_coarse_dist[0] + (size_t)(batch_start + coarse_off) * n_probes,
                                                       coarse->d_pool_top_nprobe_dist[sid],
                                                       (size_t)n_this * n_probes * sizeof(float), cudaMemcpyDeviceToHost, stream));
                        }
                    });
                }
            });
        }

        coarse->current_n_querys = batch_curr;
        double probe_metadata_d2h_ms = 0.0;
        double resident_fine_ms = 0.0;
        paraselect::cpu_fine::thread_stats_detail::FineTimingSnapshot fine_timing{};
        if (resident) {
            fine->h_topk_dist_out = batch_dist_out;
            fine->h_topk_index_out = batch_idx_out;
            probe_metadata_ms += time_stage("probe_metadata", [&]() {
                ivfflat_metadata(ctx, sid);
            });
            resident_fine_ms += time_stage("gpu_resident_fine", [&]() {
                ivfflat_fine_gpu_resident(ctx, sid);
            });
            cpu_fine_ms = resident_fine_ms;
        } else {
            paraselect::cpu_fine::CpuProbeMetadataResult metadata{};
            probe_metadata_ms += time_stage("probe_metadata", [&]() {
                if (coarse_sub_batch_size >= batch_curr) {
                    metadata = paraselect::cpu_fine::transfer_cpu_probe_metadata(
                        ctx,
                        sid,
                        batch_curr,
                        stream,
                        {fine->h_cluster_query_offset_batch,
                         fine->h_cluster_query_data_batch,
                         fine->h_active_cluster_ids_batch},
                        "pipeline");
                    probe_metadata_d2h_ms += metadata.d2h_ms;
                }
            });
            if (fine->h_top_nprobe_batch && coarse->h_coarse_index) {
                std::memcpy(coarse->h_coarse_index[0] + (size_t)batch_start * n_probes,
                            fine->h_top_nprobe_batch,
                            (size_t)batch_curr * n_probes * sizeof(idx_t));
            }

            const auto [batch_candidate_vectors, batch_max_candidates_per_query, batch_max_probe_list_size] =
                collect_cpu_fine_scan_stats();
            total_candidate_vectors += batch_candidate_vectors;
            global_max_candidates_per_query =
                std::max(global_max_candidates_per_query, batch_max_candidates_per_query);
            global_max_probe_list_size =
                std::max(global_max_probe_list_size, batch_max_probe_list_size);

            paraselect::cpu_fine::CpuFineBatchView fine_view{};
            paraselect::search::ivfflat_stage::bind_cpu_fine_batch(
                &fine_view,
                paraselect::search::ivfflat_stage::CpuFineBatchBindConfig{
                    batch_curr,
                    k,
                    batch_queries,
                    batch_queries_u8,
                    batch_dist_out,
                    batch_idx_out,
                    fine->h_top_nprobe_batch,
                    fine->h_cluster_query_offset_batch,
                    fine->h_cluster_query_data_batch,
                    fine->h_active_cluster_ids_batch,
                    metadata.active_cluster_count,
                    fine->h_topk_local_idx_batch,
                    fine->h_query_u8_tmp,
                    fine->h_thread_dists,
                    fine->h_thread_idxs,
                    fine->h_thread_sizes,
                    fine->h_thread_touched,
                    fine->h_thread_topk_capacity,
                    fine->h_thread_query_capacity,
                });
            cpu_fine_ms += time_stage("cpu_fine", [&]() {
                fine_executor.submit({ctx, ivfflat_cpu_fine_job, fine_view});
                fine_executor.wait();
            });
            fine_timing = fine_executor.latest_timing();
        }
        if (collector) {
            collector->addStageTime("query_preprocess", query_preprocess_ms);
            collector->addStageTime("coarse_search", coarse_search_ms);
            collector->addStageTime("coarse_stage1", coarse_stage1_ms);
            collector->addStageTime("coarse_gemm", coarse_gemm_ms);
            collector->addStageTime("coarse_topk", coarse_topk_ms);
            collector->addStageTime("coarse_d2h", coarse_d2h_ms);
            collector->addStageTime("probe_metadata", probe_metadata_ms);
            collector->addStageTime("probe_metadata_d2h", probe_metadata_d2h_ms);
            collector->addStageTime("cpu_fine", cpu_fine_ms);
            if (resident) {
                collector->addStageTime("gpu_resident_fine", resident_fine_ms);
                collector->addStageTime("resident_entry_build", fine->debug_resident_entry_build_ms);
                collector->addStageTime("resident_probe_init", fine->debug_resident_probe_init_ms);
                collector->addStageTime("resident_scan", fine->debug_resident_scan_ms);
                collector->addStageTime("resident_threshold_seed_build", fine->debug_resident_threshold_seed_build_ms);
                collector->addStageTime("resident_threshold_sketch", fine->debug_resident_threshold_sketch_ms);
                collector->addStageTime("resident_threshold_scan", fine->debug_resident_threshold_scan_ms);
                collector->addStageTime("resident_threshold_reduce", fine->debug_resident_threshold_reduce_ms);
                collector->addField("resident_threshold_seed_probes",
                                    static_cast<double>(arg->threshold.seed_probes));
                collector->addField("resident_threshold_candidate_capacity",
                                    static_cast<double>(arg->threshold.candidate_capacity));
                resident_threshold_candidate_sum +=
                    fine->debug_resident_threshold_avg_candidates * batch_curr;
                resident_threshold_max_candidates = std::max(
                    resident_threshold_max_candidates,
                    fine->debug_resident_threshold_max_candidates);
                collector->addStageTime("resident_merge_map", fine->debug_resident_merge_map_ms);
                collector->addStageTime("resident_d2h", fine->debug_resident_d2h_ms);
            }
            collector->addStageTime("cpu_fine_query_prepare", fine_timing.query_prepare_ms);
            collector->addStageTime("cpu_fine_metadata_build", fine_timing.metadata_build_ms);
            collector->addStageTime("cpu_fine_scratch_clear", fine_timing.scratch_clear_ms);
            collector->addStageTime("cpu_fine_task_build", fine_timing.task_build_ms);
            collector->addStageTime("cpu_fine_scan", fine_timing.scan_ms);
            collector->addStageTime("cpu_fine_merge", fine_timing.merge_ms);
            collector->addStageTime("cpu_fine_runner", fine_timing.runner_ms);
            collector->addStageTime("cpu_fine_id_map", fine_timing.id_map_ms);
            collector->stopBatch();
        }
    }
    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERRORS;
    if (collector) {
        collector->stopTotal();
        collector->addField("coarse_stage1_ms", collector->avgStageTimeMs("coarse_stage1"));
        collector->addField("coarse_gemm_ms", collector->avgStageTimeMs("coarse_gemm"));
        collector->addField("coarse_topk_ms", collector->avgStageTimeMs("coarse_topk"));
        collector->addField("coarse_d2h_ms", collector->avgStageTimeMs("coarse_d2h"));
        collector->addField("probe_metadata_compact", (double)(fine->h_use_compact_metadata != 0));
        collector->addField("probe_metadata_d2h_ms", collector->avgStageTimeMs("probe_metadata_d2h"));
        collector->addField("cpu_fine_query_prepare_ms", collector->avgStageTimeMs("cpu_fine_query_prepare"));
        collector->addField("cpu_fine_metadata_build_ms", collector->avgStageTimeMs("cpu_fine_metadata_build"));
        collector->addField("cpu_fine_scratch_clear_ms", collector->avgStageTimeMs("cpu_fine_scratch_clear"));
        collector->addField("cpu_fine_task_build_ms", collector->avgStageTimeMs("cpu_fine_task_build"));
        collector->addField("cpu_fine_scan_ms", collector->avgStageTimeMs("cpu_fine_scan"));
        collector->addField("cpu_fine_merge_ms", collector->avgStageTimeMs("cpu_fine_merge"));
        collector->addField("cpu_fine_runner_ms", collector->avgStageTimeMs("cpu_fine_runner"));
        collector->addField("cpu_fine_id_map_ms", collector->avgStageTimeMs("cpu_fine_id_map"));
        collector->addField("gpu_resident_fine_ms", collector->avgStageTimeMs("gpu_resident_fine"));
        collector->addField("resident_entry_build_ms", collector->avgStageTimeMs("resident_entry_build"));
        collector->addField("resident_probe_init_ms", collector->avgStageTimeMs("resident_probe_init"));
        collector->addField("resident_scan_ms", collector->avgStageTimeMs("resident_scan"));
        collector->addField("resident_threshold_seed_build_ms", collector->avgStageTimeMs("resident_threshold_seed_build"));
        collector->addField("resident_threshold_sketch_ms", collector->avgStageTimeMs("resident_threshold_sketch"));
        collector->addField("resident_threshold_scan_ms", collector->avgStageTimeMs("resident_threshold_scan"));
        collector->addField("resident_threshold_reduce_ms", collector->avgStageTimeMs("resident_threshold_reduce"));
        collector->addField("resident_threshold_avg_candidates",
                            n_query > 0 ? resident_threshold_candidate_sum / n_query : 0.0);
        collector->addField("resident_threshold_max_candidates",
                            resident_threshold_max_candidates);
        collector->addField("resident_merge_map_ms", collector->avgStageTimeMs("resident_merge_map"));
        collector->addField("resident_d2h_ms", collector->avgStageTimeMs("resident_d2h"));
        collector->addField("cpu_fine_candidate_vectors", (double)total_candidate_vectors);
        collector->addField("cpu_fine_avg_candidates_per_query",
                            n_query > 0 ? (double)total_candidate_vectors / (double)n_query : 0.0);
        collector->addField("cpu_fine_max_candidates_per_query", (double)global_max_candidates_per_query);
        collector->addField("cpu_fine_max_probe_list_size", (double)global_max_probe_list_size);
        collector->finalizeCoreFields();
    }
}

// Experimental batch-level pipeline: run the query preprocess + coarse GEMM +
// probe metadata of batch n+1 on stream_1 while the resident fine scan of batch
// n runs on stream_0. Per-slot buffers and the per-batch host pointers are
// already double-buffered, so only events + slot selection are added. Stage
// timings are CUDA-event based (the serial collector path syncs per stage, which
// would serialize the pipeline). Enabled only when
// PARASELECT_IVFFLAT_RESIDENT_OVERLAP=1.
static void ivfflat_pipeline_resident_overlap(
    IVFSearchCtx* ctx,
    IVFSearchStats* stats) {
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFFlatFineCtx* fine = &ctx->flat_fine;
    IVFSearchArg* arg = &ctx->arg;
    const int dim = arg->n_dim;
    const int k = arg->k;
    const int n_query = arg->n_query;
    const int batch_size = arg->query_batch_size;
    const int n_batches = batch_size > 0 ? (n_query + batch_size - 1) / batch_size : 0;

    // The caller may have queued CUDA setup whose error was not consumed yet;
    // clear it so prepare's CHECK_CUDA_ERRORS does not report a stale handle.
    (void)cudaGetLastError();
    ivfflat_prepare(ctx);

    const float* h_queries = coarse->h_queries;
    const uint8_t* h_queries_u8 = coarse->h_queries_u8;
    float* h_dist_out = coarse->h_topk_dist_out;
    idx_t* h_idx_out = coarse->h_topk_index_out;

    cudaStream_t s_coarse =
        coarse->stream_1 ? coarse->stream_1 : (coarse->stream_0 ? coarse->stream_0 : 0);
    cudaStream_t s_fine = coarse->stream_0 ? coarse->stream_0 : 0;

    paraselect::search::IVFSearchStatsCollector collector_storage(stats);
    auto* collector = stats ? &collector_storage : nullptr;
    if (collector) collector->startTotal();

    std::vector<cudaEvent_t> ev((size_t)std::max(n_batches, 1) * 5, nullptr);
    auto EV = [&](int bi, int j) { return ev[(size_t)bi * 5 + j]; };
    // These events feed cudaEventElapsedTime, so timing must stay enabled.
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    cudaEvent_t ev_coarse[2] = {nullptr, nullptr};
    cudaEvent_t ev_fine[2] = {nullptr, nullptr};
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_coarse[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_fine[s], cudaEventDisableTiming));
    }

    // Cross-stream batch timing: coarse runs on s_coarse, fine on s_fine, so the
    // host wall clock around stopBatch() would collapse to ~0. Span the whole
    // pipeline with two timing events instead (same idea as ivfrespq's chunk
    // overlap pipeline) and report pipeline_span / n_batches.
    cudaEvent_t ev_pipe_start = nullptr;
    cudaEvent_t ev_pipe_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_pipe_start));
    CUDA_CHECK(cudaEventCreate(&ev_pipe_stop));

    // Resident fine sub-stage debug fields only hold the most recent batch;
    // accumulate across batches inside run_fine.
    double sum_entry_build = 0.0, sum_probe_init = 0.0, sum_resident_scan = 0.0;
    double sum_seed_build = 0.0, sum_sketch = 0.0, sum_threshold_scan = 0.0;
    double sum_threshold_reduce = 0.0, sum_merge_map = 0.0, sum_d2h = 0.0;
    double sum_coarse_stage1 = 0.0, sum_coarse_gemm = 0.0, sum_coarse_topk = 0.0;
    long double sum_candidates = 0.0;
    double max_candidates = 0.0;

    auto run_coarse = [&](int bi, int sid) {
        const int batch_start = bi * batch_size;
        const int batch_curr = std::min(batch_size, n_query - batch_start);
        const float* bq = h_queries + (size_t)batch_start * dim;
        const uint8_t* bq_u8 = h_queries_u8 ? h_queries_u8 + (size_t)batch_start * dim : nullptr;
        coarse->h_queries = bq;
        coarse->h_queries_u8 = bq_u8;
        coarse->h_topk_dist_out = h_dist_out + (size_t)batch_start * k;
        coarse->h_topk_index_out = h_idx_out + (size_t)batch_start * k;
        coarse->current_n_querys = batch_curr;

        CUDA_CHECK(cudaStreamWaitEvent(s_coarse, ev_fine[sid], 0));
        CUDA_CHECK(cudaEventRecord(EV(bi, 0), s_coarse));
        CUDA_CHECK(cudaMemcpyAsync(coarse->d_pool_query_batch[sid], bq,
                                   (size_t)batch_curr * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, s_coarse));
        if (coarse->d_pool_query_batch_u8[sid]) {
            CUDA_CHECK(cudaMemcpyAsync(coarse->d_pool_query_batch_u8[sid], bq_u8,
                                       (size_t)batch_curr * dim * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, s_coarse));
        }
        compute_l2_norm_gpu(coarse->d_pool_query_batch[sid], coarse->d_pool_query_norm[sid],
                            batch_curr, dim, L2NORM_AUTO, s_coarse);
        if (arg->fp16_coarse) {
            const int n_elems = batch_curr * dim;
            if (n_elems > 0 && coarse->d_pool_query_batch[sid] &&
                coarse->d_pool_query_fp16[sid]) {
                constexpr int kThreads = 256;
                const int n_blocks = (n_elems + kThreads - 1) / kThreads;
                ivfflat_fp32_to_fp16_query_kernel<<<n_blocks, kThreads, 0, s_coarse>>>(
                    coarse->d_pool_query_batch[sid], coarse->d_pool_query_fp16[sid], n_elems);
            }
        }
        CUDA_CHECK(cudaEventRecord(EV(bi, 1), s_coarse));
        coarse->debug_coarse_gemm_ms = 0.0;
        coarse->debug_coarse_topk_ms = 0.0;
        ivfflat_coarse(ctx, sid, s_coarse, /*measure=*/false);
        ivfflat_metadata(ctx, sid, s_coarse, /*sync=*/false);
        sum_coarse_gemm += coarse->debug_coarse_gemm_ms;
        sum_coarse_topk += coarse->debug_coarse_topk_ms;
        sum_coarse_stage1 += coarse->debug_coarse_gemm_ms + coarse->debug_coarse_topk_ms;
        CUDA_CHECK(cudaEventRecord(EV(bi, 2), s_coarse));
        CUDA_CHECK(cudaEventRecord(ev_coarse[sid], s_coarse));
    };

    auto run_fine = [&](int bi, int sid) {
        const int batch_start = bi * batch_size;
        const int batch_curr = std::min(batch_size, n_query - batch_start);
        coarse->current_n_querys = batch_curr;
        fine->h_topk_dist_out = h_dist_out + (size_t)batch_start * k;
        fine->h_topk_index_out = h_idx_out + (size_t)batch_start * k;
        CUDA_CHECK(cudaStreamWaitEvent(s_fine, ev_coarse[sid], 0));
        CUDA_CHECK(cudaEventRecord(EV(bi, 3), s_fine));
        ivfflat_fine_gpu_resident(ctx, sid);
        sum_entry_build += fine->debug_resident_entry_build_ms;
        sum_probe_init += fine->debug_resident_probe_init_ms;
        sum_resident_scan += fine->debug_resident_scan_ms;
        sum_seed_build += fine->debug_resident_threshold_seed_build_ms;
        sum_sketch += fine->debug_resident_threshold_sketch_ms;
        sum_threshold_scan += fine->debug_resident_threshold_scan_ms;
        sum_threshold_reduce += fine->debug_resident_threshold_reduce_ms;
        sum_merge_map += fine->debug_resident_merge_map_ms;
        sum_d2h += fine->debug_resident_d2h_ms;
        sum_candidates +=
            (long double)fine->debug_resident_threshold_avg_candidates * (long double)batch_curr;
        max_candidates = std::max(max_candidates, fine->debug_resident_threshold_max_candidates);
        CUDA_CHECK(cudaEventRecord(EV(bi, 4), s_fine));
        CUDA_CHECK(cudaEventRecord(ev_fine[sid], s_fine));
    };

    if (n_batches > 0) {
        CUDA_CHECK(cudaEventRecord(ev_pipe_start, s_coarse));
        run_coarse(0, 0);
        for (int bi = 0; bi < n_batches; ++bi) {
            const int sid = bi & 1;
            if (bi + 1 < n_batches) run_coarse(bi + 1, (bi + 1) & 1);
            run_fine(bi, sid);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev_pipe_stop, s_fine));
    CUDA_CHECK(cudaStreamSynchronize(s_fine));
    CUDA_CHECK(cudaStreamSynchronize(s_coarse));
    CHECK_CUDA_ERRORS;

    float pipeline_span_ms = 0.0f;
    if (n_batches > 0) {
        CUDA_CHECK(cudaEventElapsedTime(&pipeline_span_ms, ev_pipe_start, ev_pipe_stop));
    }

    if (collector) {
        const double per_batch_ms =
            n_batches > 0 ? (double)pipeline_span_ms / (double)n_batches : 0.0;
        const double inv_batches = n_batches > 0 ? 1.0 / (double)n_batches : 0.0;
        for (int bi = 0; bi < n_batches; ++bi) {
            const int batch_start = bi * batch_size;
            const int batch_curr = std::min(batch_size, n_query - batch_start);
            float t_pre = 0.f, t_coarse = 0.f, t_fine = 0.f;
            cudaEventElapsedTime(&t_pre, EV(bi, 0), EV(bi, 1));
            cudaEventElapsedTime(&t_coarse, EV(bi, 1), EV(bi, 2));
            cudaEventElapsedTime(&t_fine, EV(bi, 3), EV(bi, 4));
            collector->addStageTime("query_preprocess", t_pre);
            collector->addStageTime("coarse_search", t_coarse);
            collector->addStageTime("probe_metadata", 0.0);
            collector->addStageTime("cpu_fine", t_fine);
            collector->addStageTime("gpu_resident_fine", t_fine);
            collector->addBatchTime(bi, batch_start, batch_curr, per_batch_ms);
        }
        collector->stopTotal();
        collector->finalizeCoreFields();
        collector->addField("pipeline_overlap_span_ms", (double)pipeline_span_ms);
        collector->addField("coarse_stage1_ms", sum_coarse_stage1 * inv_batches);
        collector->addField("coarse_gemm_ms", sum_coarse_gemm * inv_batches);
        collector->addField("coarse_topk_ms", sum_coarse_topk * inv_batches);
        collector->addField("resident_entry_build_ms", sum_entry_build * inv_batches);
        collector->addField("resident_probe_init_ms", sum_probe_init * inv_batches);
        collector->addField("resident_scan_ms", sum_resident_scan * inv_batches);
        collector->addField("resident_threshold_seed_build_ms", sum_seed_build * inv_batches);
        collector->addField("resident_threshold_sketch_ms", sum_sketch * inv_batches);
        collector->addField("resident_threshold_scan_ms", sum_threshold_scan * inv_batches);
        collector->addField("resident_threshold_reduce_ms", sum_threshold_reduce * inv_batches);
        collector->addField("resident_merge_map_ms", sum_merge_map * inv_batches);
        collector->addField("resident_d2h_ms", sum_d2h * inv_batches);
        collector->addField("resident_threshold_seed_probes",
                            static_cast<double>(arg->threshold.seed_probes));
        collector->addField("resident_threshold_candidate_capacity",
                            static_cast<double>(arg->threshold.candidate_capacity));
        collector->addField("resident_threshold_avg_candidates",
                            n_query > 0 ? (double)(sum_candidates / (long double)n_query) : 0.0);
        collector->addField("resident_threshold_max_candidates", (double)max_candidates);
        collector->addField("gpu_resident_fine_ms", collector->avgStageTimeMs("gpu_resident_fine"));
    }

    cudaEventDestroy(ev_pipe_start);
    cudaEventDestroy(ev_pipe_stop);
    for (auto& e : ev) cudaEventDestroy(e);
    for (int s = 0; s < 2; ++s) {
        cudaEventDestroy(ev_coarse[s]);
        cudaEventDestroy(ev_fine[s]);
    }
}

void ivfflat_pipeline(
    IVFSearchCtx* ctx,
    IVFSearchStats* stats) {
    if (ctx->arg.cpu_batch_pipeline_overlap && ctx->arg.fine_kind == IVFFlatFineKind::Cpu) {
        ivfflat_cpu_overlap(ctx, stats);
        return;
    }
    if (ctx->arg.fine_kind == IVFFlatFineKind::Resident) {
        const char* overlap = std::getenv("PARASELECT_IVFFLAT_RESIDENT_OVERLAP");
        if (overlap && overlap[0] == '1') {
            ivfflat_pipeline_resident_overlap(ctx, stats);
            return;
        }
    }
    ivfflat_pipeline_serial(ctx, stats);
}
