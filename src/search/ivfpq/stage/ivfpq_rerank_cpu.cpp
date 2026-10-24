#include "search/ivfpq/ivfpq_search_context.h"
#include "cpu/distance/distance_dispatch.h"
#include "search/ivf_common/ivf_host_rerank.h"
#include "search/ivf_common/ivf_rerank_policy.h"
#include "search/ivfpq/stage/ivfpq_stages.cuh"

#include <algorithm>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <vector>
#include <omp.h>

void ivfpq_rerank_cpu(IVFPQSearchContext* ctx, int batch_start, int batch_n) {
    if (!ctx || !ctx->pq_rerank_enabled || batch_n <= 0) return;
    const auto& index = ctx->index;
    const auto& queries = ctx->queries;
    const auto& params = ctx->params;
    const auto& options = ctx->options;
    const int topk = params.topk;
    const int stage1_k = ctx->pq_stage1_k;
    const int dim = queries.dim;
    const auto fp32_isa = paraselect::cpu_distance::select_fp32_distance_isa();
    const auto u8_isa = paraselect::cpu_distance::select_u8_distance_isa();
    const int max_threads = options.cpu_rerank_threads > 0
        ? options.cpu_rerank_threads
        : std::min(std::max(1, omp_get_max_threads()), 16);
    const int threads = paraselect::search::ivf_rerank_thread_count(max_threads, batch_n);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const paraselect::search::IVFHostStorageView storage{
        index.storage.rerank_fp32, index.storage.rerank_u8,
        index.lists.n_vectors, dim};
    // The adapter owns the retained prefix; never expose a GPU count beyond
    // the host row width to the common exact-rerank contract.
    if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
        for (int q = 0; q < batch_n; ++q)
            ctx->rerank_candidate_count_batch[q] =
                std::clamp(ctx->rerank_candidate_count_batch[q], 0, stage1_k);
    }
    const int* counts = options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
        ? ctx->rerank_candidate_count_batch.data() : nullptr;
    ivf_rerank_host(storage,
        queries.u8 ? queries.u8 + size_t(batch_start) * dim : nullptr,
        queries.fp32 ? queries.fp32 + size_t(batch_start) * dim : nullptr,
        options.rerank_dtype == IVFExactRerankDType::FP32,
        ctx->rerank_candidate_idx_batch.data(), batch_n, stage1_k, stage1_k, topk,
        ctx->topk_local_idx_batch.data(),
        ctx->output.distances + size_t(batch_start) * topk,
        max_threads, options.pin_rerank_threads, options.rerank_prefetch, nullptr, counts);

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ctx->collector) {
        ctx->collector->addStageTime("pq_rerank", ms);
        ctx->collector->addStageTime("ivfpq_cpu_rerank", ms);
        ctx->collector->addField("rpq_candidate_k", (double)stage1_k);
        ctx->collector->addField(
            "ivfpq_rerank_dtype",
            options.rerank_dtype == IVFExactRerankDType::U8 ? "u8" : "fp32");
        ctx->collector->addField("ivfpq_rerank_kernel_isa",
            options.rerank_dtype == IVFExactRerankDType::U8
                ? paraselect::cpu_distance::u8_distance_isa_name(u8_isa)
                : paraselect::cpu_distance::fp32_distance_isa_name(fp32_isa));
        ctx->collector->addField("ivfpq_cpu_rerank_threads", (double)threads);
    }
}
