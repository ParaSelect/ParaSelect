#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>

#include "utils/pch.h"
#include "search/ivfrespq/ivfrespq_search.cuh"
#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/scheduler/ivfrespq_scheduler.cuh"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "search/ivfflat/ivfflat_search_ctx.h"
#include "search/ivf_common/ivf_search_stats_collector.h"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_threshold_policy.cuh"
#include "utils/check_cuda_error.cuh"

#include <cstdio>
#include <array>
#include <stdexcept>

using namespace paraselect::search;

namespace {

const char* ivfrespq_search_impl_name(const IVFResSearchState* arg) {
    (void)arg;
    return "ivfrespq_residual_pq_gpu";
}

const char* ivfrespq_exact_storage_name(IVFExactStorageKind kind) {
    switch (kind) {
        case IVFExactStorageKind::Host: return "host";
        case IVFExactStorageKind::Resident: return "resident";
        case IVFExactStorageKind::File: return "file";
        case IVFExactStorageKind::None: break;
    }
    return "none";
}

void add_file_block_candidate_metrics(
    paraselect::search::IVFSearchStatsCollector* collector,
    const std::vector<long long>& histogram) {
    if (!collector || histogram.size() <= 1) return;

    long long block_accesses = 0;
    long long candidate_refs = 0;
    size_t max_refs = 0;
    for (size_t refs = 1; refs < histogram.size(); ++refs) {
        const long long blocks = histogram[refs];
        block_accesses += blocks;
        candidate_refs += static_cast<long long>(refs) * blocks;
        if (blocks > 0) max_refs = refs;
    }
    if (block_accesses == 0) return;

    const auto percentile = [&](long long numerator, long long denominator) {
        const long long target =
            (block_accesses * numerator + denominator - 1) / denominator;
        long long cumulative = 0;
        for (size_t refs = 1; refs < histogram.size(); ++refs) {
            cumulative += histogram[refs];
            if (cumulative >= target) return refs;
        }
        return max_refs;
    };

    collector->addField("rpq_file_block_accesses", static_cast<double>(block_accesses));
    collector->addField("rpq_file_candidate_refs", static_cast<double>(candidate_refs));
    collector->addField(
        "rpq_file_refs_per_block_mean",
        static_cast<double>(candidate_refs) / static_cast<double>(block_accesses));
    collector->addField("rpq_file_refs_per_block_p50", static_cast<double>(percentile(50, 100)));
    collector->addField("rpq_file_refs_per_block_p90", static_cast<double>(percentile(90, 100)));
    collector->addField("rpq_file_refs_per_block_p99", static_cast<double>(percentile(99, 100)));
    collector->addField("rpq_file_refs_per_block_max", static_cast<double>(max_refs));

    constexpr std::array<size_t, 4> thresholds = {1, 2, 4, 8};
    for (size_t threshold : thresholds) {
        long long blocks_below = 0;
        long long refs_below = 0;
        const size_t upper = std::min(threshold, histogram.size() - 1);
        for (size_t refs = 1; refs <= upper; ++refs) {
            blocks_below += histogram[refs];
            refs_below += static_cast<long long>(refs) * histogram[refs];
        }
        const std::string suffix = std::to_string(threshold);
        collector->addField(
            "rpq_file_blocks_le" + suffix + "_rate",
            static_cast<double>(blocks_below) / static_cast<double>(block_accesses));
        collector->addField(
            "rpq_file_refs_in_blocks_le" + suffix + "_rate",
            candidate_refs > 0
                ? static_cast<double>(refs_below) / static_cast<double>(candidate_refs)
                : 0.0);
    }
}

}  // namespace

static void ivfrespq_search_prepared(IVFRESPQSearchCtx& ctx, IVFSearchStats* stats) {
    const IVFResSearchState* arg = &ctx.arg;
    ctx.pq_scanned_codes_total = ctx.pq_adc_lookups_total = ctx.pq_physical_code_rows_total = 0;
    ctx.threshold_candidate_total = ctx.threshold_query_total = 0;
    ctx.threshold_candidate_max = 0;
    ctx.file_block_candidate_ref_histogram.clear();
    ctx.file_blocks_considered_total = ctx.file_blocks_skipped_total = 0;
    paraselect::search::IVFSearchStatsCollector collector(stats);

    try {
        ctx.collector = stats ? &collector : nullptr;
        if (collector) {
            collector.addField("search_impl", ivfrespq_search_impl_name(arg));
            collector.addField("pq_M", (double)arg->pq_M);
            collector.addField("pq_K", (double)arg->pq_K);
            collector.addField("pq_quantizer", arg->pq_distance_mode == PQAdcDistanceMode::Plain ? "plain" : "residual");
            collector.addField("pq_rerank_n", (double)arg->pq_rerank_n);
            collector.addField("ivfrespq_fp16_coarse", arg->fp16_coarse ? 1.0 : 0.0);
            collector.addField("ivfrespq_coarse_backend",
                paraselect::coarse::current::path_name(ctx.coarse.coarse_workspace->plan().path));
            collector.addField(
                "ivfrespq_query_tile",
                static_cast<double>(ctx.coarse.query_tile));
            collector.addField("exact_storage", ivfrespq_exact_storage_name(arg->exact_storage.kind));
            collector.addField(
                "pca_backend", ivf_refine_backend_name(arg->pca_backend));
            collector.addField(
                "rpq_adc_strategy",
                arg->adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
                    ? "threshold"
                    : "dense");
            collector.addField(
                "rpq_adc_mode",
                arg->residual_adc_mode ==
                        paraselect::search::ResidualPQAdcMode::QueryLut
                    ? "pq_query_lut"
                    : "probe_residual_lut");
            collector.addField(
                "rpq_lut_quantization",
                arg->query_lut_quantization ==
                        paraselect::search::PQAdcLutQuantizationMode::U8Affine
                    ? "u8_affine"
                    : "fp32");
            collector.addField(
                "rpq_query_lut_bias_encoding",
                arg->query_lut_bias.empty() ? "none" :
                (arg->query_lut_bias.u16 ? "u16_affine" : "fp32"));
            if (arg->adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
                collector.addField(
                    "rpq_threshold_seed_probes",
                    static_cast<double>(arg->adc_threshold.seed_probes));
                collector.addField(
                    "rpq_threshold_policy",
                    paraselect::search::pq_threshold_policy_name(
                        arg->adc_threshold.policy));
                collector.addField(
                    "rpq_threshold_scan_layout",
                    arg->adc_threshold.scan_layout ==
                            paraselect::search::PQAdcThresholdScanLayout::ListCentric
                        ? "list"
                        : "pair");
                collector.addField(
                    "rpq_list_scan_work_mode",
                    arg->adc_threshold.list_work_mode ==
                            paraselect::search::PQAdcListScanWorkMode::FixedTile
                        ? "tile"
                        : "entry");
                collector.addField(
                    "rpq_list_scan_row_mode",
                    arg->adc_threshold.list_row_mode == paraselect::search::PQAdcListScanRowMode::Interleaved
                        ? "interleaved"
                        : arg->adc_threshold.list_row_mode == paraselect::search::PQAdcListScanRowMode::QueryRowTile4
                            ? "query_row_tile4" : "serial");
        collector.addField("rpq_code_tile_consumer",
                    arg->adc_threshold.code_tile_consumer == paraselect::search::PQAdcCodeTileConsumer::U8Hybrid ? "u8_hybrid" : arg->adc_threshold.code_tile_consumer == paraselect::search::PQAdcCodeTileConsumer::Fp16Hybrid ? "fp16_hybrid" : "lut");
                collector.addField(
                    "rpq_lut_read_mode",
                    arg->adc_threshold.lut_read_mode ==
                            paraselect::search::PQAdcLutReadMode::Global
                        ? "global"
                        : "shared");
                collector.addField(
                    "rpq_shared_lut_layout",
                    arg->adc_threshold.shared_lut_layout ==
                            paraselect::search::PQAdcSharedLutLayout::Compact
                        ? "compact"
                        : "padded");
            }
            collector.addField(
                "rpq_file_sparse_ref_threshold",
                static_cast<double>(arg->file_block_selection.sparse_ref_threshold));
            collector.addField(
                "rpq_file_protected_candidate_rank",
                static_cast<double>(arg->file_block_selection.protected_candidate_rank));
            collector.startTotal();
        }
        ivfrespq_select_pipeline(&ctx);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (collector) {
            if (ctx.threshold_query_total > 0) {
                collector.addField(
                    "rpq_threshold_avg_candidates",
                    static_cast<double>(ctx.threshold_candidate_total) /
                        static_cast<double>(ctx.threshold_query_total));
                collector.addField(
                    "rpq_threshold_max_candidates",
                    static_cast<double>(ctx.threshold_candidate_max));
            }
            if (ctx.fine.assets.file_reader) {
                const auto file_stats = ctx.fine.assets.file_reader->stats();
                const double total_accesses =
                    static_cast<double>(file_stats.hits + file_stats.misses);
                collector.addField("rpq_file_cache_hits", (double)file_stats.hits);
                collector.addField("rpq_file_cache_misses", (double)file_stats.misses);
                collector.addField(
                    "rpq_file_cache_hit_rate",
                    total_accesses > 0.0 ? (double)file_stats.hits / total_accesses : 0.0);
                collector.addField(
                    "rpq_file_cache_bytes_read", (double)file_stats.bytes_read);
                collector.addField(
                    "rpq_file_cache_bytes_reused", (double)file_stats.bytes_reused);
                collector.addField(
                    "rpq_file_cache_capacity_bytes", (double)file_stats.capacity_bytes);
                collector.addField(
                    "rpq_file_prefill_bytes", (double)file_stats.prefill_bytes);
                collector.addField("rpq_file_prefill_ms", file_stats.prefill_ms);
                add_file_block_candidate_metrics(
                    &collector, ctx.file_block_candidate_ref_histogram);
                collector.addField(
                    "rpq_file_blocks_considered",
                    static_cast<double>(ctx.file_blocks_considered_total));
                collector.addField(
                    "rpq_file_blocks_skipped",
                    static_cast<double>(ctx.file_blocks_skipped_total));
                collector.addField(
                    "rpq_file_block_skip_rate",
                    ctx.file_blocks_considered_total > 0
                        ? static_cast<double>(ctx.file_blocks_skipped_total) /
                              static_cast<double>(ctx.file_blocks_considered_total)
                        : 0.0);
            }
            collector.stopTotal();
            collector.finalizeCoreFields();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[ivfrespq_search][exception] what=%s n_query=%d dim=%d nlist=%d nprobe=%d topk=%d\n",
                     e.what(),
                     arg ? arg->n_query : -1,
                     arg ? arg->n_dim : -1,
                     arg ? arg->lists.nlist : -1,
                     arg ? arg->n_probes : -1,
                     arg ? arg->k : -1);
        ctx.collector = nullptr;
        throw;
    }

    ctx.collector = nullptr;
}

namespace {

IVFResSearchState make_request_state(
    const IVFCoarseIndexView& coarse,
    const IVFListLayoutView& lists,
    const IVFPQCodeStorageView& pq,
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params,
    const IVFResPQSearchOptions& options) {
    IVFResSearchState state;
    state.h_queries_fp32 = queries.fp32;
    state.h_queries_u8 = queries.u8;
    state.h_queries_fp16 = queries.fp16;
    state.lists = lists;
    state.h_cluster_centers = coarse.centroids;
    state.pq.codes = pq.codes;
    state.pq.code_storage = pq.code_storage;
    state.pq.codebook = pq.codebook;
    state.n_query = queries.count;
    state.n_dim = queries.dim;
    state.n_probes = params.nprobe;
    state.k = params.topk;
    state.query_batch_size = params.batch_size;
    if (options.query_tile < 0) throw std::invalid_argument("query_tile must be nonnegative");
    state.query_tile = options.query_tile;
    state.distance_mode = params.distance;
    state.fp16_coarse = params.fp16_coarse;
    state.pq_M = pq.subvectors;
    state.pq_K = pq.codebook_size;
    state.pq_rerank_n = options.n_rerank;
    state.shortlist_count = options.shortlist_count;
    state.pca_backend = options.pca_backend;
    state.pq_use_float_rerank = options.float_rerank ? 1 : 0;
    state.return_adc_distances = options.return_adc_distances;
    state.fine_backend = options.fine_backend;
    state.gpu = options.gpu;
    state.cpu = options.cpu;
    state.resident_combo_name = options.resident_combo;
    state.cpu_batch_pipeline_overlap = options.overlap_cpu_pipeline;
    state.adc_strategy = options.adc_strategy;
    state.residual_codec = options.residual_codec;
    state.pq_distance_mode = options.pq_distance_mode;
    state.query_lut_codebook_fp16 = options.query_lut_codebook_fp16;
    state.query_lut_quantization = options.query_lut_quantization;
    state.adc_threshold = options.adc_threshold;
    state.file_block_selection = options.file_block_selection;
    state.file_read = options.file_read;
    return state;
}

}  // namespace

static IVFResSearchState make_search_state(const IVFResPQSearchRequest& request) {
    validate_ivf_index_shape(request.index.coarse, request.index.lists);
    if (request.queries.dim != request.index.coarse.dim) {
        throw std::invalid_argument("IVFResPQ query and index dimensions differ");
    }
    const auto& storage = request.index.storage;
    IVFResSearchState state = make_request_state(
        request.index.coarse,
        request.index.lists,
        storage.pq,
        request.queries,
        request.params,
        request.options);
    state.host_rows = storage.host_rows;
    state.exact_storage = storage.assets.exact;
    state.query_lut_bias = storage.assets.query_lut_bias;
    state.residual_adc_mode = request.options.adc_mode;
    state.turboquant.codes = storage.assets.tq_codes;
    state.turboquant.scales = storage.assets.tq_scales;
    state.h_residual_rotation = storage.assets.residual_rotation;
    state.pca_storage = storage.pca;
    state.h_topk_index_out = request.output.indices;
    state.h_topk_dist_out = request.output.distances;
    state.h_coarse_cluster_ids_out = request.output.coarse_ids;
    return state;
}

static IVFResSearchState make_candidate_state(
    const IVFResPQCandidateRequest& request) {
    validate_ivf_index_shape(request.index.coarse, request.index.lists);
    if (request.queries.dim != request.index.coarse.dim) {
        throw std::invalid_argument("IVFResPQ query and index dimensions differ");
    }
    IVFResSearchState state = make_request_state(
        request.index.coarse,
        request.index.lists,
        request.index.pq,
        request.queries,
        request.params,
        request.options);
    state.candidate_only = true;
    state.exact_storage = request.index.assets.exact;
    state.query_lut_bias = request.index.assets.query_lut_bias;
    state.residual_adc_mode = request.options.adc_mode;
    state.turboquant.codes = request.index.assets.tq_codes;
    state.turboquant.scales = request.index.assets.tq_scales;
    state.h_residual_rotation = request.index.assets.residual_rotation;
    state.h_candidate_local_ids_out = request.output.local_ids;
    state.h_candidate_adc_distances_out = request.output.adc_distances;
    state.candidate_capacity = request.output.capacity;

    return state;
}

struct paraselect::search::IVFResPQSession::Impl {
    IVFRESPQSearchCtx ctx;
    ~Impl() { ivfrespq_search_ctx_release(&ctx); }
};

paraselect::search::IVFResPQSession::IVFResPQSession(const IVFResPQSearchRequest& request)
    : impl_(std::make_unique<Impl>()) {
    auto state = make_search_state(request);
    ivfrespq_prepare(&impl_->ctx, &state);
}

paraselect::search::IVFResPQSession::IVFResPQSession(const IVFResPQCandidateRequest& request)
    : impl_(std::make_unique<Impl>()) {
    auto state = make_candidate_state(request);
    ivfrespq_prepare(&impl_->ctx, &state);
}

int paraselect::search::IVFResPQSession::candidate_count() const {
    return impl_->ctx.fine.runtime.candidate_width;
}

paraselect::search::IVFResPQSession::~IVFResPQSession() = default;

void paraselect::search::IVFResPQSession::search(IVFSearchStats* stats) {
    ivfrespq_search_prepared(impl_->ctx, stats);
}

void paraselect::search::ivfrespq_search(const IVFResPQSearchRequest& request, IVFSearchStats* stats) {
    IVFResPQSession session(request);
    session.search(stats);
}

int paraselect::search::ivfrespq_collect_candidates(const IVFResPQCandidateRequest& request) {
    IVFResPQSession session(request);
    session.search();
    return session.candidate_count();
}
