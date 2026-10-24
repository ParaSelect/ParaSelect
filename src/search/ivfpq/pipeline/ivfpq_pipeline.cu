#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/pipeline/ivfpq_pipeline.cuh"
#include "search/ivfpq/stage/ivfpq_stages.cuh"

#include <algorithm>
#include <chrono>

void ivfpq_run_pipeline(IVFPQSearchContext* ctx) {
    const auto& queries = ctx->queries;
    const auto& params = ctx->params;
    const auto& output = ctx->output;
    const int query_batch_size = ctx->query_batch_size;
    int batch_id = 0;
    double total_batch_ms = 0.0;
    double total_coarse_ms = 0.0;
    double total_pq_ms = 0.0;
    double total_map_ms = 0.0;
    double total_rerank_ms = 0.0;

    for (int batch_start = 0; batch_start < queries.count; batch_start += query_batch_size) {
        const int batch_n = std::min(query_batch_size, queries.count - batch_start);
        const float* batch_query = queries.fp32 + (size_t)batch_start * queries.dim;
        if (ctx->collector) ctx->collector->startBatch(batch_id, batch_start, batch_n);
        const auto batch_t0 = std::chrono::high_resolution_clock::now();

        const auto coarse_t0 = std::chrono::high_resolution_clock::now();
        if (ctx->resident_codes.enabled && !output.coarse_ids) {
            ivfpq_coarse_to_device(ctx, batch_query, batch_n);
        } else {
            ivfpq_coarse(ctx, batch_query, batch_n, ctx->top_nprobe_batch.data());
            if (output.coarse_ids) {
                for (int i = 0; i < batch_n * params.nprobe; ++i) {
                    output.coarse_ids[(size_t)batch_start * params.nprobe + i] =
                        (int)ctx->top_nprobe_batch[(size_t)i];
                }
            }
        }
        const auto coarse_t1 = std::chrono::high_resolution_clock::now();
        const double coarse_ms = std::chrono::duration<double, std::milli>(coarse_t1 - coarse_t0).count();
        total_coarse_ms += coarse_ms;
        if (ctx->collector) ctx->collector->addStageTime("coarse_search", coarse_ms);

        const auto pq_t0 = std::chrono::high_resolution_clock::now();
        if (ctx->resident_codes.enabled) {
            ivfpq_fine_resident_codes_gpu(ctx, batch_start, batch_n);
        } else {
            ivfpq_fine_cpu_lut(ctx, batch_start, batch_n);
        }
        const auto pq_t1 = std::chrono::high_resolution_clock::now();
        const double pq_ms = std::chrono::duration<double, std::milli>(pq_t1 - pq_t0).count();
        total_pq_ms += pq_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime("cpu_fine", pq_ms);
            ctx->collector->addStageTime("pq_fine_total", pq_ms);
        }

        if (ctx->pq_rerank_enabled) {
            const auto rerank_t0 = std::chrono::high_resolution_clock::now();
            ivfpq_rerank_cpu(ctx, batch_start, batch_n);
            const auto rerank_t1 = std::chrono::high_resolution_clock::now();
            total_rerank_ms += std::chrono::duration<double, std::milli>(rerank_t1 - rerank_t0).count();
        }

        const auto map_t0 = std::chrono::high_resolution_clock::now();
        ivfpq_map_ids(ctx, batch_start, batch_n);
        const auto map_t1 = std::chrono::high_resolution_clock::now();
        const double map_ms = std::chrono::duration<double, std::milli>(map_t1 - map_t0).count();
        total_map_ms += map_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime("probe_metadata", map_ms);
            ctx->collector->addStageTime("pq_map_ids", map_ms);
        }

        const auto batch_t1 = std::chrono::high_resolution_clock::now();
        total_batch_ms += std::chrono::duration<double, std::milli>(batch_t1 - batch_t0).count();
        if (ctx->collector) ctx->collector->stopBatch();
        ++batch_id;
    }

    if (ctx->collector) {
        const double batches = batch_id > 0 ? (double)batch_id : 1.0;
        const double avg_batch_ms = total_batch_ms / batches;
        const double avg_pq_ms = total_pq_ms / batches;
        ctx->collector->addField(
            "ivfpq_pipeline",
            ctx->resident_codes.enabled ? "serial_gpu_resident_codes" : "serial_cpu_plain_pq");
        ctx->collector->addField("ivfpq_batches", (double)batch_id);
        ctx->collector->addField("ivfpq_coarse_total_ms", total_coarse_ms / batches);
        ctx->collector->addField("ivfpq_pq_fine_total_ms", avg_pq_ms);
        ctx->collector->addField("ivfpq_lut_build_total_ms", ctx->collector->avgStageTimeMs("pq_lut_build"));
        ctx->collector->addField("ivfpq_adc_scan_total_ms", ctx->collector->avgStageTimeMs("pq_adc_scan"));
        ctx->collector->addField("ivfpq_adc_query_h2d_ms", ctx->collector->avgStageTimeMs("ivfpq_adc_query_h2d"));
        ctx->collector->addField("ivfpq_adc_kernel_ms", ctx->collector->avgStageTimeMs("ivfpq_adc_kernel"));
        ctx->collector->addField("ivfpq_adc_prefix_ms", ctx->collector->avgStageTimeMs("ivfpq_adc_prefix"));
        ctx->collector->addField("ivfpq_adc_d2h_ms", ctx->collector->avgStageTimeMs("ivfpq_adc_d2h"));
        ctx->collector->addField("ivfpq_map_ids_total_ms", total_map_ms / batches);
        ctx->collector->addField("ivfpq_rerank_total_ms", total_rerank_ms / batches);
        ctx->collector->addField("rpq_candidate_k", (double)ctx->pq_stage1_k);
        ctx->collector->addField("ivfpq_batch_total_ms", avg_batch_ms);
        ctx->collector->addField("ivfpq_avg_batch_ms", avg_batch_ms);
        ctx->collector->addField("pq_scanned_codes", (double)ctx->pq_scanned_codes_total);
        ctx->collector->addField("pq_adc_lookups", (double)ctx->pq_adc_lookups_total);
        if (ctx->threshold_query_total > 0) {
            ctx->collector->addField(
                "rpq_threshold_seed_probes",
                (double)ctx->options.threshold.seed_probes);
            ctx->collector->addField(
                "rpq_threshold_avg_candidates",
                (double)ctx->threshold_candidate_total /
                    (double)ctx->threshold_query_total);
            ctx->collector->addField(
                "rpq_threshold_max_candidates",
                (double)ctx->threshold_candidate_max);
        }
    }
}
