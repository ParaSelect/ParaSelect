#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/ivfrespq_timing.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "search/ivfrespq/pipeline/ivfrespq_pipeline.cuh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

void ivfrespq_pipeline_cpu(IVFRESPQSearchCtx* ctx) {
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    const int nprobe = arg.n_probes;
    const int query_batch_size = coarse.query_batch_size;
    const int query_tile = coarse.query_tile;
    int batch_id = 0;
    double total_coarse_ms = 0.0;
    double total_pq_fine_ms = 0.0;
    double total_map_ids_ms = 0.0;
    double total_batch_ms = 0.0;

    for (int batch_start = 0; batch_start < arg.n_query; batch_start += query_batch_size) {
        const int batch_n = std::min(query_batch_size, arg.n_query - batch_start);
        const float* batch_query = arg.h_queries_fp32 + (size_t)batch_start * dim;
        const auto batch_t0 = std::chrono::high_resolution_clock::now();
        if (ctx->collector) {
            ctx->collector->startBatch(batch_id, batch_start, batch_n);
        }

        const auto coarse_t0 = std::chrono::high_resolution_clock::now();
        fine.runtime.d_query_batch_valid = 0;
        fine.runtime.d_top_nprobe_batch_valid = 0;
        fine.runtime.d_top_nprobe_dist_batch_valid = 0;
        fine.runtime.h_top_nprobe_batch_valid = 0;
        const bool consume_probe_on_gpu = fine.runtime.consume_probe_on_gpu != 0;
        const bool reuse_query_on_gpu = fine.runtime.reuse_query_on_gpu != 0;
        if (reuse_query_on_gpu) {
            ivfrespq_coarse_preload_query_batch(ctx, batch_query, batch_n);
            ivfrespq_coarse_preloaded_to_device(
                ctx, batch_n, fine.device.d_top_nprobe_batch, fine.device.d_top_nprobe_dist_batch);
            fine.runtime.d_query_batch_valid = 1;
            fine.runtime.d_top_nprobe_batch_valid = 1;
            fine.runtime.d_top_nprobe_dist_batch_valid = fine.device.d_top_nprobe_dist_batch ? 1 : 0;
        }
        for (int coarse_off = 0; !reuse_query_on_gpu && coarse_off < batch_n; coarse_off += query_tile) {
            const int n_this = std::min(query_tile, batch_n - coarse_off);
            const float* sub_query = batch_query + (size_t)coarse_off * dim;
            if (consume_probe_on_gpu) {
                ivfrespq_coarse_to_device(
                    ctx,
                    sub_query,
                    n_this,
                    fine.device.d_top_nprobe_batch + (size_t)coarse_off * nprobe,
                    fine.device.d_top_nprobe_dist_batch
                        ? fine.device.d_top_nprobe_dist_batch + (size_t)coarse_off * nprobe
                        : nullptr);
                fine.runtime.d_top_nprobe_batch_valid = 1;
                fine.runtime.d_top_nprobe_dist_batch_valid = fine.device.d_top_nprobe_dist_batch ? 1 : 0;
            } else {
                idx_t* sub_coarse_out = fine.host.h_top_nprobe_batch + (size_t)coarse_off * nprobe;
                ivfrespq_coarse(ctx, sub_query, n_this, sub_coarse_out);
                fine.runtime.h_top_nprobe_batch_valid = 1;
                if (fine.runtime.use_residual_pq_gpu && fine.device.d_top_nprobe_batch) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        fine.device.d_top_nprobe_batch + (size_t)coarse_off * nprobe,
                        coarse.d_top_nprobe_index,
                        (size_t)n_this * nprobe * sizeof(idx_t),
                        cudaMemcpyDeviceToDevice,
                        coarse.stream));
                    fine.runtime.d_top_nprobe_batch_valid = 1;
                    if (fine.device.d_top_nprobe_dist_batch) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            fine.device.d_top_nprobe_dist_batch +
                                (size_t)coarse_off * nprobe,
                            coarse.d_top_nprobe_dist,
                            (size_t)n_this * nprobe * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            coarse.stream));
                        fine.runtime.d_top_nprobe_dist_batch_valid = 1;
                    }
                }
                if (arg.h_coarse_cluster_ids_out) {
                    for (int i = 0; i < n_this * nprobe; ++i) {
                        arg.h_coarse_cluster_ids_out[(size_t)(batch_start + coarse_off) * nprobe + i] =
                            (int)sub_coarse_out[i];
                    }
                }
            }
        }
        if (consume_probe_on_gpu) {
            CUDA_SYNC_CHECK(coarse.stream, "ivfrespq stage1 coarse batch for gpu");
        }
        const auto coarse_t1 = std::chrono::high_resolution_clock::now();
        const double coarse_ms = std::chrono::duration<double, std::milli>(coarse_t1 - coarse_t0).count();
        total_coarse_ms += coarse_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime(
                "coarse_search",
                coarse_ms);
        }

        const auto fine_t0 = std::chrono::high_resolution_clock::now();
        ivfrespq_fine(ctx, batch_start, batch_n);
        const auto fine_t1 = std::chrono::high_resolution_clock::now();
        const double pq_fine_ms = std::chrono::duration<double, std::milli>(fine_t1 - fine_t0).count();
        total_pq_fine_ms += pq_fine_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime(
                "cpu_fine",
                pq_fine_ms);
            ctx->collector->addStageTime("pq_fine_total", pq_fine_ms);
        }

        const auto map_t0 = std::chrono::high_resolution_clock::now();
        if (!arg.candidate_only) {
            ivfrespq_map_ids(ctx, batch_start, batch_n);
        }
        const auto map_t1 = std::chrono::high_resolution_clock::now();
        const double map_ids_ms = std::chrono::duration<double, std::milli>(map_t1 - map_t0).count();
        total_map_ids_ms += map_ids_ms;
        const auto batch_t1 = std::chrono::high_resolution_clock::now();
        total_batch_ms += std::chrono::duration<double, std::milli>(batch_t1 - batch_t0).count();
        if (ctx->collector) {
            ctx->collector->addStageTime(
                "probe_metadata",
                map_ids_ms);
            ctx->collector->addStageTime("pq_map_ids", map_ids_ms);
            ctx->collector->stopBatch();
        }
        ++batch_id;
    }

    if (ctx->collector) {
        const auto per_batch = [&](const char* name) {
            return ctx->collector->avgStageTimePerBatchMs(name);
        };
        const double avg_adc_ms = per_batch("pq_adc_lut_scan");
        const double avg_residual_ms = per_batch("pq_residual");
        const double avg_lut_build_ms = per_batch("pq_lut_build");
        const double avg_lut_d2h_ms = per_batch("pq_lut_d2h");
        const double avg_adc_lookup_ms = per_batch("pq_adc_lookup");
        const double avg_heap_push_ms = per_batch("pq_heap_push");
        const double avg_rerank_ms = per_batch("pq_rerank");
        const double avg_rpq_gpu_search_ms = per_batch("rpq_gpu_search");
        const double avg_rpq_query_lut_ms = per_batch("rpq_query_lut");
        const double avg_rpq_probe_schedule_ms =
            per_batch("rpq_probe_schedule");
        const double avg_rpq_tile_schedule_ms =
            per_batch("rpq_tile_schedule");
        const double avg_rpq_fine_gpu_wall_ms =
            per_batch("rpq_fine_gpu_wall");
        const double avg_rpq_probe_topk_ms = per_batch("rpq_probe_topk");
        const double avg_rpq_lut_build_ms = per_batch("rpq_lut_build");
        const double avg_rpq_adc_scan_ms = per_batch("rpq_adc_scan");
        const double avg_rpq_global_reduce_ms = per_batch("rpq_global_reduce");
        const double avg_rpq_desc_ms = per_batch("rpq_desc");
        const double avg_rpq_group_merge_ms = per_batch("rpq_group_merge");
        const double avg_rpq_final_merge_ms = per_batch("rpq_final_merge");
        const double avg_rpq_candidate_desc_ms = per_batch("rpq_candidate_desc");
        const double avg_rpq_pq_scan_ms = per_batch("rpq_pq_scan");
        const double avg_rpq_candidate_reduce_ms = per_batch("rpq_candidate_reduce");
        const double avg_rpq_candidate_cap_select_ms =
            per_batch("rpq_candidate_cap_select");
        const double avg_rpq_final_copy_ms = per_batch("rpq_final_copy");
        const double avg_rpq_seed_compact_ms = per_batch("rpq_seed_compact");
        const double avg_rpq_seed_adc_ms = per_batch("rpq_seed_adc");
        const double avg_rpq_threshold_policy_ms = per_batch("rpq_threshold_policy");
        const double avg_rpq_scan_setup_ms = per_batch("rpq_scan_setup");
        const double avg_rpq_scan_kernel_ms = per_batch("rpq_scan_kernel");
        const double avg_rpq_d2h_ids_ms = per_batch("rpq_d2h_ids");
        const double avg_rpq_cpu_rerank_ms = per_batch("rpq_cpu_rerank");
        const double avg_batch_ms = batch_id > 0 ? total_batch_ms / (double)batch_id : 0.0;
        const double avg_coarse_ms = batch_id > 0 ? total_coarse_ms / (double)batch_id : 0.0;
        const double avg_pq_fine_ms = batch_id > 0 ? total_pq_fine_ms / (double)batch_id : 0.0;
        const double avg_map_ids_ms = batch_id > 0 ? total_map_ids_ms / (double)batch_id : 0.0;
        const paraselect::search::IVFResPQSerialTiming timing{
            avg_batch_ms,
            avg_coarse_ms,
            avg_pq_fine_ms,
            avg_map_ids_ms,
            avg_rpq_fine_gpu_wall_ms,
            avg_rpq_cpu_rerank_ms,
        };
        ctx->collector->addField("ivfrespq_pipeline", "serial");
        ctx->collector->addField("ivfrespq_batches", (double)batch_id);
        ctx->collector->addField("ivfrespq_coarse_total_ms", avg_coarse_ms);
        ctx->collector->addField("ivfrespq_pq_fine_total_ms", avg_pq_fine_ms);
        ctx->collector->addField("ivfrespq_adc_lut_scan_total_ms", avg_adc_ms);
        ctx->collector->addField("ivfrespq_residual_total_ms", avg_residual_ms);
        ctx->collector->addField("ivfrespq_lut_build_total_ms", avg_lut_build_ms);
        ctx->collector->addField("ivfrespq_lut_d2h_total_ms", avg_lut_d2h_ms);
        ctx->collector->addField("ivfrespq_adc_lookup_total_ms", avg_adc_lookup_ms);
        ctx->collector->addField("ivfrespq_heap_push_total_ms", avg_heap_push_ms);
        ctx->collector->addField("ivfrespq_rerank_total_ms", avg_rerank_ms);
        ctx->collector->addField("ivfrespq_pq_fine_overhead_ms", timing.fine_overhead_ms());
        ctx->collector->addField("ivfrespq_stage_sum_ms", timing.stage_sum_ms());
        ctx->collector->addField("ivfrespq_timing_closure_error_ms", timing.closure_error_ms());
        ctx->collector->addField("ivfrespq_map_ids_total_ms", avg_map_ids_ms);
        ctx->collector->addField("ivfrespq_batch_total_ms", avg_batch_ms);
        ctx->collector->addField("ivfrespq_avg_batch_ms", avg_batch_ms);
        for (const char* stage : {"rpq_query_prepare", "rpq_seed_count", "rpq_candidate_readback"}) {
            ctx->collector->addField(std::string(stage) + "_ms",
                ctx->collector->avgStageTimePerBatchMs(stage));
        }
        ctx->collector->addField("rpq_gpu_search_ms", avg_rpq_gpu_search_ms);
        ctx->collector->addField("rpq_query_lut_ms", avg_rpq_query_lut_ms);
        ctx->collector->addField("rpq_probe_schedule_ms", avg_rpq_probe_schedule_ms);
        ctx->collector->addField("rpq_tile_schedule_ms", avg_rpq_tile_schedule_ms);
        ctx->collector->addField("rpq_fine_gpu_wall_ms", avg_rpq_fine_gpu_wall_ms);
        ctx->collector->addField("rpq_probe_topk_ms", avg_rpq_probe_topk_ms);
        ctx->collector->addField("rpq_lut_build_ms", avg_rpq_lut_build_ms);
        ctx->collector->addField("rpq_adc_scan_ms", avg_rpq_adc_scan_ms);
        ctx->collector->addField("rpq_global_reduce_ms", avg_rpq_global_reduce_ms);
        ctx->collector->addField("rpq_desc_ms", avg_rpq_desc_ms);
        ctx->collector->addField("rpq_group_merge_ms", avg_rpq_group_merge_ms);
        ctx->collector->addField("rpq_final_merge_ms", avg_rpq_final_merge_ms);
        ctx->collector->addField("rpq_candidate_desc_ms", avg_rpq_candidate_desc_ms);
        ctx->collector->addField("rpq_pq_scan_ms", avg_rpq_pq_scan_ms);
        ctx->collector->addField("rpq_candidate_reduce_ms", avg_rpq_candidate_reduce_ms);
        ctx->collector->addField("rpq_candidate_cap_select_ms", avg_rpq_candidate_cap_select_ms);
        ctx->collector->addField("rpq_final_copy_ms", avg_rpq_final_copy_ms);
        ctx->collector->addField("rpq_seed_compact_ms", avg_rpq_seed_compact_ms);
        ctx->collector->addField("rpq_seed_adc_ms", avg_rpq_seed_adc_ms);
        ctx->collector->addField("rpq_threshold_policy_ms", avg_rpq_threshold_policy_ms);
        ctx->collector->addField("rpq_scan_setup_ms", avg_rpq_scan_setup_ms);
        ctx->collector->addField("rpq_scan_kernel_ms", avg_rpq_scan_kernel_ms);
        ctx->collector->addField("rpq_d2h_ids_ms", avg_rpq_d2h_ids_ms);
        ctx->collector->addField("rpq_cpu_rerank_ms", avg_rpq_cpu_rerank_ms);
        ctx->collector->addField("pq_scanned_codes", (double)ctx->pq_scanned_codes_total);
        ctx->collector->addField(
            "pq_physical_code_rows", (double)ctx->pq_physical_code_rows_total);
        ctx->collector->addField("pq_adc_lookups", (double)ctx->pq_adc_lookups_total);
        ctx->collector->addField(
            "pq_adc_codes_per_ms",
            total_pq_fine_ms > 0.0 && avg_adc_ms > 0.0
                ? (double)ctx->pq_scanned_codes_total / (avg_adc_ms * (double)batch_id)
                : 0.0);
    }
}

namespace {

struct IVFRESPQChunkSlot {
    IVFRESPQBatchView view{};
    idx_t* h_top_nprobe_batch = nullptr;
    int* h_rerank_candidate_idx = nullptr;
    int* h_topk_local_idx_batch = nullptr;
    bool owns_storage = true;

    void borrow(const IVFRESPQSearchCtx& ctx) {
        h_top_nprobe_batch = ctx.fine.host.h_top_nprobe_batch;
        h_rerank_candidate_idx = ctx.fine.host.h_rerank_candidate_idx;
        h_topk_local_idx_batch = ctx.fine.host.h_topk_local_idx_batch;
        owns_storage = false;
    }

    void allocate(const IVFRESPQSearchCtx& ctx) {
        const IVFResSearchState& arg = ctx.arg;
        const IVFRESPQFineCtx& fine = ctx.fine;
        const int batch_size = ctx.coarse.query_batch_size;
        if (!fine.runtime.consume_probe_on_gpu) {
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&h_top_nprobe_batch),
                (size_t)batch_size * arg.n_probes * sizeof(idx_t)));
        }
        if (fine.runtime.use_residual_pq_gpu) {
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&h_rerank_candidate_idx),
                (size_t)batch_size * fine.runtime.candidate_width * sizeof(int)));
        }
        h_topk_local_idx_batch = static_cast<int*>(
            std::malloc((size_t)batch_size * arg.k * sizeof(int)));
        if (!h_topk_local_idx_batch || (fine.runtime.use_residual_pq_gpu && !h_rerank_candidate_idx)) {
            throw std::bad_alloc();
        }
    }

    void release() {
        if (!owns_storage) return;
        if (h_top_nprobe_batch) cudaFreeHost(h_top_nprobe_batch);
        if (h_rerank_candidate_idx) cudaFreeHost(h_rerank_candidate_idx);
        std::free(h_topk_local_idx_batch);
        h_top_nprobe_batch = nullptr;
        h_rerank_candidate_idx = nullptr;
        h_topk_local_idx_batch = nullptr;
    }

    ~IVFRESPQChunkSlot() { release(); }
};

struct IVFRESPQCpuChunkTiming {
    double rerank_ms = 0.0;
    double map_ms = 0.0;
};

static void bind_ivfrespq_slot(IVFRESPQSearchCtx* ctx, const IVFRESPQChunkSlot& slot) {
    IVFRESPQFineCtx& fine = ctx->fine;
    fine.host.h_top_nprobe_batch = slot.view.h_top_nprobe_batch;
    fine.host.h_rerank_candidate_idx = slot.view.h_rerank_candidate_idx;
    fine.host.h_topk_local_idx_batch = slot.view.h_topk_local_idx_batch;
}

static void fill_ivfrespq_slot_view(
    IVFRESPQSearchCtx* ctx,
    IVFRESPQChunkSlot& slot,
    int batch_id,
    int batch_start,
    int batch_n) {
    const IVFResSearchState& arg = ctx->arg;
    const IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    slot.view = {};
    slot.view.batch_id = batch_id;
    slot.view.batch_start = batch_start;
    slot.view.batch_n = batch_n;
    slot.view.h_queries_fp32 = arg.h_queries_fp32 + (size_t)batch_start * dim;
    slot.view.h_queries_u8 = fine.assets.h_queries_u8 ? fine.assets.h_queries_u8 + (size_t)batch_start * dim : nullptr;
    slot.view.h_topk_dist_out = arg.h_topk_dist_out + (size_t)batch_start * arg.k;
    slot.view.h_topk_index_out = arg.h_topk_index_out + (size_t)batch_start * arg.k;
    slot.view.h_top_nprobe_batch = slot.h_top_nprobe_batch;
    slot.view.h_rerank_candidate_idx = slot.h_rerank_candidate_idx;
    slot.view.h_topk_local_idx_batch = slot.h_topk_local_idx_batch;
}

static double ivfrespq_chunk_overlap_gpu_work(IVFRESPQSearchCtx* ctx, const IVFRESPQBatchView& view) {
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    const int nprobe = arg.n_probes;
    const int query_tile = coarse.query_tile;
    const auto t0 = std::chrono::high_resolution_clock::now();

    fine.runtime.d_query_batch_valid = 0;
    fine.runtime.d_top_nprobe_batch_valid = 0;
    fine.runtime.d_top_nprobe_dist_batch_valid = 0;
    fine.runtime.h_top_nprobe_batch_valid = 0;
    const bool consume_probe_on_gpu = fine.runtime.consume_probe_on_gpu != 0;
    const bool reuse_query_on_gpu = fine.runtime.reuse_query_on_gpu != 0;
    if (reuse_query_on_gpu) {
        ivfrespq_coarse_preload_query_batch(ctx, view.h_queries_fp32, view.batch_n);
        ivfrespq_coarse_preloaded_to_device(
            ctx, view.batch_n, fine.device.d_top_nprobe_batch, fine.device.d_top_nprobe_dist_batch);
        fine.runtime.d_query_batch_valid = 1;
        fine.runtime.d_top_nprobe_batch_valid = 1;
        fine.runtime.d_top_nprobe_dist_batch_valid = fine.device.d_top_nprobe_dist_batch ? 1 : 0;
    }
    for (int coarse_off = 0; !reuse_query_on_gpu && coarse_off < view.batch_n; coarse_off += query_tile) {
        const int n_this = std::min(query_tile, view.batch_n - coarse_off);
        const float* sub_query = view.h_queries_fp32 + (size_t)coarse_off * dim;
        if (consume_probe_on_gpu) {
            ivfrespq_coarse_to_device(
                ctx,
                sub_query,
                n_this,
                fine.device.d_top_nprobe_batch + (size_t)coarse_off * nprobe,
                fine.device.d_top_nprobe_dist_batch
                    ? fine.device.d_top_nprobe_dist_batch + (size_t)coarse_off * nprobe
                    : nullptr);
            fine.runtime.d_top_nprobe_batch_valid = 1;
            fine.runtime.d_top_nprobe_dist_batch_valid = fine.device.d_top_nprobe_dist_batch ? 1 : 0;
        } else {
            idx_t* sub_coarse_out = view.h_top_nprobe_batch + (size_t)coarse_off * nprobe;
            ivfrespq_coarse(ctx, sub_query, n_this, sub_coarse_out);
            fine.runtime.h_top_nprobe_batch_valid = 1;
            if (fine.runtime.use_residual_pq_gpu && fine.device.d_top_nprobe_batch) {
                CUDA_CHECK(cudaMemcpyAsync(
                    fine.device.d_top_nprobe_batch + (size_t)coarse_off * nprobe,
                    coarse.d_top_nprobe_index,
                    (size_t)n_this * nprobe * sizeof(idx_t),
                    cudaMemcpyDeviceToDevice,
                    coarse.stream));
                fine.runtime.d_top_nprobe_batch_valid = 1;
                if (fine.device.d_top_nprobe_dist_batch) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        fine.device.d_top_nprobe_dist_batch +
                            (size_t)coarse_off * nprobe,
                        coarse.d_top_nprobe_dist,
                        (size_t)n_this * nprobe * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        coarse.stream));
                    fine.runtime.d_top_nprobe_dist_batch_valid = 1;
                }
            }
            if (arg.h_coarse_cluster_ids_out) {
                for (int i = 0; i < n_this * nprobe; ++i) {
                    arg.h_coarse_cluster_ids_out[(size_t)(view.batch_start + coarse_off) * nprobe + i] =
                        (int)sub_coarse_out[i];
                }
            }
        }
    }
    if (consume_probe_on_gpu) {
        CUDA_SYNC_CHECK(coarse.stream, "ivfrespq chunk_overlap coarse batch for gpu");
    }
    ivfrespq_fine_residual_pq_gpu_candidates(ctx, view);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static IVFRESPQCpuChunkTiming ivfrespq_chunk_overlap_cpu_work(
    IVFRESPQSearchCtx* ctx,
    IVFRESPQBatchView view) {
    IVFRESPQCpuChunkTiming timing;
    const auto rerank_t0 = std::chrono::high_resolution_clock::now();
    ivfrespq_rerank_residual_pq_candidates_cpu(ctx, view);
    const auto rerank_t1 = std::chrono::high_resolution_clock::now();
    timing.rerank_ms = std::chrono::duration<double, std::milli>(rerank_t1 - rerank_t0).count();

    const auto map_t0 = std::chrono::high_resolution_clock::now();
    ivfrespq_map_ids_view(ctx, view);
    const auto map_t1 = std::chrono::high_resolution_clock::now();
    timing.map_ms = std::chrono::duration<double, std::milli>(map_t1 - map_t0).count();
    return timing;
}

struct IVFRESPQCpuChunkJob {
    IVFRESPQSearchCtx* owner = nullptr;
    IVFRESPQBatchView view{};
    bool prewarm = false;
    int prewarm_batch_n = 0;
};

} // namespace

class IVFRESPQCpuChunkExecutor {
public:
    IVFRESPQCpuChunkExecutor() {
        // Start only after every mutex, condition variable and job flag exists.
        worker_ = std::thread([this]() { worker_loop(); });
    }

    IVFRESPQCpuChunkExecutor(const IVFRESPQCpuChunkExecutor&) = delete;
    IVFRESPQCpuChunkExecutor& operator=(const IVFRESPQCpuChunkExecutor&) = delete;

    ~IVFRESPQCpuChunkExecutor() noexcept {
        stop_worker_noexcept();
    }

    void submit(const IVFRESPQCpuChunkJob& job) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return job_done_ && !has_job_; });
        pending_job_ = job;
        latest_timing_ = {};
        error_ = nullptr;
        job_done_ = false;
        has_job_ = true;
        lock.unlock();
        cv_.notify_one();
    }

    IVFRESPQCpuChunkTiming wait() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return job_done_; });
        auto error = error_;
        error_ = nullptr;
        IVFRESPQCpuChunkTiming timing = latest_timing_;
        lock.unlock();
        if (error) std::rethrow_exception(error);
        return timing;
    }

private:
    void worker_loop() {
        for (;;) {
            IVFRESPQCpuChunkJob job{};
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this]() { return has_job_ || stop_; });
                if (stop_ && !has_job_) return;
                job = pending_job_;
                has_job_ = false;
            }

            IVFRESPQCpuChunkTiming timing{};
            std::exception_ptr error;
            try {
                if (job.prewarm) {
                    ivfrespq_prewarm_residual_pq_rerank_cpu(job.owner, job.prewarm_batch_n);
                } else {
                    timing = ivfrespq_chunk_overlap_cpu_work(job.owner, job.view);
                }
            } catch (...) {
                error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                latest_timing_ = timing;
                error_ = error;
                job_done_ = true;
            }
            cv_.notify_all();
        }
    }

    void stop_worker_noexcept() noexcept {
        if (!worker_.joinable()) return;
        try {
            wait();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_one();
        worker_.join();
    }

    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    IVFRESPQCpuChunkJob pending_job_{};
    IVFRESPQCpuChunkTiming latest_timing_{};
    bool has_job_ = false;
    bool job_done_ = true;
    bool stop_ = false;
    std::exception_ptr error_;
};

void ivfrespq_prepare_cpu_pipeline(IVFRESPQSearchCtx* ctx) {
    if (!ctx->arg.cpu_batch_pipeline_overlap || ctx->arg.candidate_only ||
        !ctx->fine.runtime.use_residual_pq_gpu || ctx->fine.device.use_resident_rerank)
        return;
    auto executor = std::make_shared<IVFRESPQCpuChunkExecutor>();
    IVFRESPQCpuChunkJob prewarm{};
    prewarm.owner = ctx;
    prewarm.prewarm = true;
    prewarm.prewarm_batch_n = ctx->coarse.query_batch_size;
    executor->submit(prewarm);
    (void)executor->wait();
    ctx->cpu_executor = std::move(executor);
}

void ivfrespq_pipeline_chunk_overlap(IVFRESPQSearchCtx* ctx) {
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    IVFRESPQFineCtx& fine = ctx->fine;
    if (!fine.runtime.use_residual_pq_gpu) {
        ivfrespq_pipeline_cpu(ctx);
        return;
    }

    const int query_batch_size = coarse.query_batch_size;
    int batch_id = 0;
    double total_gpu_ms = 0.0;
    double total_cpu_rerank_ms = 0.0;
    double total_map_ms = 0.0;

    // The scheduler borrows only host output bindings. Never restore a copy of
    // the entire fine context: GPU workspace ownership and audit state belong
    // to the GPU stage and may have changed while processing batches.
    const auto owner_probes = fine.host.h_top_nprobe_batch;
    const auto owner_candidates = fine.host.h_rerank_candidate_idx;
    const auto owner_topk = fine.host.h_topk_local_idx_batch;
    auto restore_host_bindings = [&]() {
        fine.host.h_top_nprobe_batch = owner_probes;
        fine.host.h_rerank_candidate_idx = owner_candidates;
        fine.host.h_topk_local_idx_batch = owner_topk;
    };
    IVFRESPQChunkSlot slots[2];
    // Prepare already owns one complete host output buffer. A second buffer
    // is needed only when another GPU batch can overlap CPU consumption.
    slots[0].borrow(*ctx);
    if (arg.n_query > query_batch_size) slots[1].allocate(*ctx);

    if (!ctx->cpu_executor) throw std::logic_error("CPU pipeline was not prepared");
    auto& cpu_executor = *ctx->cpu_executor;

    bool have_cpu_job = false;
    const auto pipeline_t0 = std::chrono::high_resolution_clock::now();

    auto wait_cpu = [&]() {
        if (!have_cpu_job) return;
        IVFRESPQCpuChunkTiming timing = cpu_executor.wait();
        have_cpu_job = false;
        total_cpu_rerank_ms += timing.rerank_ms;
        total_map_ms += timing.map_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime("pq_rerank", timing.rerank_ms);
            ctx->collector->addStageTime("rpq_cpu_rerank", timing.rerank_ms);
            ctx->collector->addStageTime("probe_metadata", timing.map_ms);
            ctx->collector->addStageTime("pq_map_ids", timing.map_ms);
        }
    };

    try {
        for (int batch_start = 0; batch_start < arg.n_query; batch_start += query_batch_size, ++batch_id) {
            const int batch_n = std::min(query_batch_size, arg.n_query - batch_start);
            IVFRESPQChunkSlot& slot = slots[batch_id & 1];
            fill_ivfrespq_slot_view(ctx, slot, batch_id, batch_start, batch_n);
            bind_ivfrespq_slot(ctx, slot);
            const double gpu_ms = ivfrespq_chunk_overlap_gpu_work(ctx, slot.view);
            total_gpu_ms += gpu_ms;
            if (ctx->collector) {
                ctx->collector->addStageTime("ivfrespq_chunk_gpu", gpu_ms);
            }

            wait_cpu();

            IVFRESPQCpuChunkJob cpu_job{};
            cpu_job.owner = ctx;
            cpu_job.view = slot.view;
            cpu_executor.submit(cpu_job);
            have_cpu_job = true;
        }
        wait_cpu();
        cudaStreamSynchronize(coarse.stream);
        CHECK_CUDA_ERRORS;
    } catch (...) {
        if (have_cpu_job) {
            try {
                cpu_executor.wait();
            } catch (...) {}
            have_cpu_job = false;
        }
        restore_host_bindings();
        throw;
    }

    const auto pipeline_t1 = std::chrono::high_resolution_clock::now();
    const double pipeline_wall_ms = std::chrono::duration<double, std::milli>(pipeline_t1 - pipeline_t0).count();
    restore_host_bindings();

    if (ctx->collector) {
        const double avg_gpu_ms = batch_id > 0 ? total_gpu_ms / (double)batch_id : 0.0;
        const double avg_rerank_ms = batch_id > 0 ? total_cpu_rerank_ms / (double)batch_id : 0.0;
        const double avg_map_ms = batch_id > 0 ? total_map_ms / (double)batch_id : 0.0;
        const double avg_batch_ms = batch_id > 0 ? pipeline_wall_ms / (double)batch_id : 0.0;
        const double avg_adc_ms = ctx->collector->avgStageTimeMs("pq_adc_lut_scan");
        const double avg_rpq_gpu_search_ms = ctx->collector->avgStageTimeMs("rpq_gpu_search");
        const double avg_rpq_probe_schedule_ms =
            ctx->collector->avgStageTimeMs("rpq_probe_schedule");
        const double avg_rpq_tile_schedule_ms =
            ctx->collector->avgStageTimeMs("rpq_tile_schedule");
        const double avg_rpq_fine_gpu_wall_ms =
            ctx->collector->avgStageTimeMs("rpq_fine_gpu_wall");
        const double avg_rpq_probe_topk_ms = ctx->collector->avgStageTimeMs("rpq_probe_topk");
        const double avg_rpq_global_reduce_ms = ctx->collector->avgStageTimeMs("rpq_global_reduce");
        const double avg_rpq_d2h_ids_ms = ctx->collector->avgStageTimeMs("rpq_d2h_ids");
        const double avg_rpq_desc_ms = ctx->collector->avgStageTimeMs("rpq_desc");
        const double avg_rpq_group_merge_ms = ctx->collector->avgStageTimeMs("rpq_group_merge");
        const double avg_rpq_final_merge_ms = ctx->collector->avgStageTimeMs("rpq_final_merge");
        const double avg_rpq_candidate_desc_ms = ctx->collector->avgStageTimeMs("rpq_candidate_desc");
        const double avg_rpq_pq_scan_ms = ctx->collector->avgStageTimeMs("rpq_pq_scan");
        const double avg_rpq_candidate_reduce_ms = ctx->collector->avgStageTimeMs("rpq_candidate_reduce");
        const double avg_rpq_candidate_cap_select_ms =
            ctx->collector->avgStageTimeMs("rpq_candidate_cap_select");
        const double avg_rpq_final_copy_ms = ctx->collector->avgStageTimeMs("rpq_final_copy");
        const double avg_rpq_seed_compact_ms = ctx->collector->avgStageTimeMs("rpq_seed_compact");
        const double avg_rpq_seed_adc_ms = ctx->collector->avgStageTimeMs("rpq_seed_adc");
        const double avg_rpq_threshold_policy_ms = ctx->collector->avgStageTimeMs("rpq_threshold_policy");
        const double avg_rpq_scan_setup_ms = ctx->collector->avgStageTimeMs("rpq_scan_setup");
        const double avg_rpq_scan_kernel_ms = ctx->collector->avgStageTimeMs("rpq_scan_kernel");
        const double avg_rpq_cpu_sort_ms = ctx->collector->avgStageTimeMs("rpq_cpu_rerank_sort");
        const double avg_rpq_cpu_dist_heap_ms = ctx->collector->avgStageTimeMs("rpq_cpu_rerank_dist_heap");
        const double avg_rpq_cpu_other_ms = ctx->collector->avgStageTimeMs("rpq_cpu_rerank_other");
        ctx->collector->addField("ivfrespq_pipeline", "chunk_overlap");
        ctx->collector->addField("ivfrespq_batches", (double)batch_id);
        ctx->collector->addField("ivfrespq_chunk_gpu_ms", avg_gpu_ms);
        ctx->collector->addField("ivfrespq_pq_fine_total_ms", avg_gpu_ms + avg_rerank_ms);
        ctx->collector->addField("ivfrespq_adc_lut_scan_total_ms", avg_adc_ms);
        ctx->collector->addField("ivfrespq_rerank_total_ms", avg_rerank_ms);
        ctx->collector->addField("ivfrespq_map_ids_total_ms", avg_map_ms);
        ctx->collector->addField("ivfrespq_batch_total_ms", avg_batch_ms);
        ctx->collector->addField("ivfrespq_avg_batch_ms", avg_batch_ms);
        for (const char* stage : {"rpq_query_prepare", "rpq_seed_count", "rpq_candidate_readback"}) {
            ctx->collector->addField(std::string(stage) + "_ms",
                ctx->collector->avgStageTimePerBatchMs(stage));
        }
        ctx->collector->addField("rpq_gpu_search_ms", avg_rpq_gpu_search_ms);
        ctx->collector->addField("rpq_probe_schedule_ms", avg_rpq_probe_schedule_ms);
        ctx->collector->addField("rpq_tile_schedule_ms", avg_rpq_tile_schedule_ms);
        ctx->collector->addField("rpq_fine_gpu_wall_ms", avg_rpq_fine_gpu_wall_ms);
        ctx->collector->addField("rpq_probe_topk_ms", avg_rpq_probe_topk_ms);
        ctx->collector->addField("rpq_global_reduce_ms", avg_rpq_global_reduce_ms);
        ctx->collector->addField("rpq_d2h_ids_ms", avg_rpq_d2h_ids_ms);
        ctx->collector->addField("rpq_desc_ms", avg_rpq_desc_ms);
        ctx->collector->addField("rpq_group_merge_ms", avg_rpq_group_merge_ms);
        ctx->collector->addField("rpq_final_merge_ms", avg_rpq_final_merge_ms);
        ctx->collector->addField("rpq_candidate_desc_ms", avg_rpq_candidate_desc_ms);
        ctx->collector->addField("rpq_pq_scan_ms", avg_rpq_pq_scan_ms);
        ctx->collector->addField("rpq_candidate_reduce_ms", avg_rpq_candidate_reduce_ms);
        ctx->collector->addField("rpq_candidate_cap_select_ms", avg_rpq_candidate_cap_select_ms);
        ctx->collector->addField("rpq_final_copy_ms", avg_rpq_final_copy_ms);
        ctx->collector->addField("rpq_seed_compact_ms", avg_rpq_seed_compact_ms);
        ctx->collector->addField("rpq_seed_adc_ms", avg_rpq_seed_adc_ms);
        ctx->collector->addField("rpq_threshold_policy_ms", avg_rpq_threshold_policy_ms);
        ctx->collector->addField("rpq_scan_setup_ms", avg_rpq_scan_setup_ms);
        ctx->collector->addField("rpq_scan_kernel_ms", avg_rpq_scan_kernel_ms);
        ctx->collector->addField("rpq_cpu_rerank_ms", avg_rerank_ms);
        ctx->collector->addField("rpq_cpu_rerank_sort_ms", avg_rpq_cpu_sort_ms);
        ctx->collector->addField("rpq_cpu_rerank_dist_heap_ms", avg_rpq_cpu_dist_heap_ms);
        ctx->collector->addField("rpq_cpu_rerank_other_ms", avg_rpq_cpu_other_ms);
        ctx->collector->addField("pq_scanned_codes", (double)ctx->pq_scanned_codes_total);
        ctx->collector->addField(
            "pq_physical_code_rows", (double)ctx->pq_physical_code_rows_total);
        ctx->collector->addField("pq_adc_lookups", (double)ctx->pq_adc_lookups_total);
    }
}
