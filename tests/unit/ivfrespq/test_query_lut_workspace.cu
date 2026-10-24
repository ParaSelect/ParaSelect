#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_prepare_internal.cuh"
#include <iostream>
#include <limits>
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"

int main() {
    IVFRESPQSearchCtx ctx{};
    ctx.arg.fp16_coarse = true;
    ctx.arg.residual_codec = paraselect::search::ResidualCodecKind::PQ;
    ctx.coarse.query_batch_size = 10000;
    paraselect::search::ResidualPQGpuSearchSpec spec{};
    spec.search_strategy = paraselect::search::PQAdcSearchStrategy::Threshold;
    spec.adc_mode = paraselect::search::ResidualPQAdcMode::QueryLut;
    try {
        // Query-LUT owns its seed scratch. A dense-seed reservation must not
        // influence this path, regardless of batch/probe size or free memory.
        for (auto impl : {paraselect::search::ResidualPQGpuImpl::Dense,
                          paraselect::search::ResidualPQGpuImpl::GroupedVectorChunk}) {
            spec.impl = impl;
            prepare_fine_scan_workspace(&ctx, spec, 8000,
                                        std::numeric_limits<size_t>::max() / 2);
            const auto& w = ctx.fine.device.rpq_grouped_workspace;
            if (w.d_scanned_codes || w.h_scanned_codes || w.d_pair_counts ||
                w.d_pair_offsets || w.d_chunk_desc || w.d_merge_a_dist ||
                ctx.fine.device.d_probe_candidate_idx)
                throw std::runtime_error("query-LUT allocated dense seed workspace");
            paraselect::search::residual_pq_grouped_workspace_release(
                &ctx.fine.device.rpq_grouped_workspace);
        }
        ctx.arg.n_probes = 8;
        ctx.fine.runtime.candidate_width = 512;
        ctx.fine.assets.residual_pq_gpu_spec = spec;
        auto& threshold = ctx.fine.assets.residual_pq_gpu_spec.threshold;
        threshold.candidate_capacity = 8192;
        threshold.candidate_k = threshold.n_rerank = 512;
        threshold.seed_probes = 8;
        for (int seed_rank : {512, 1024}) {
            threshold.seed_rank = seed_rank;
            ivfrespq_prepare_candidate_output(&ctx);
            auto& output = ctx.fine.device.rpq_threshold_workspace;
            if (output.candidate_capacity != seed_rank || output.max_queries != 10000)
                throw std::runtime_error("threshold output retained the scan capacity");
            paraselect::search::pq_adc_threshold_workspace_release(&output);
            cudaFreeHost(ctx.fine.host.h_rerank_candidate_idx);
            ctx.fine.host.h_rerank_candidate_idx = nullptr;
        }
        std::cout << "query-LUT workspace isolation passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
