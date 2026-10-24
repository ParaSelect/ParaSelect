#include "search/ivfpq/ivfpq_index.h"
#include "search/ivfrespq/ivfrespq_search.cuh"

#include "search/ivf_common/ivf_search_stats_collector.h"
#include "search/ivfpq/ivfpq_build.h"
#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/pipeline/ivfpq_pipeline.cuh"
#include "search/ivfpq/stage/ivfpq_stages.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

std::string ivfpq_search_impl_name(const IVFPQSearchContext& ctx) {
    if (ctx.resident_codes.enabled) {
        const char* strategy = nullptr;
        switch (ctx.options.adc_strategy) {
            case paraselect::search::PQAdcSearchStrategy::Dense:
                strategy = "dense";
                break;
            case paraselect::search::PQAdcSearchStrategy::Threshold:
                strategy = "threshold";
                break;
            default:
                throw std::invalid_argument("ivfpq search strategy is invalid");
        }
        return std::string("ivfpq_plain_pq_gpu_adc_") +
            ctx.resident_codes.combo_name + "_" + strategy;
    }
    return "ivfpq_plain_pq_cpu_lut";
}

bool prepared_shape_matches(
    const IVFPQSearchContext& ctx,
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params) {
    return ctx.query_batch_size == std::min(params.batch_size, queries.count) &&
           ctx.params.nprobe == params.nprobe &&
           ctx.params.topk == params.topk &&
           ctx.params.distance == params.distance &&
           ctx.params.fp16_coarse == params.fp16_coarse;
}

void validate_search_request(
    const IVFPQSearchContext& ctx,
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params,
    const IVFSearchOutputView& output) {
    if (!queries.fp32 || queries.count <= 0 ||
        queries.dim != ctx.index.coarse.dim) {
        throw std::invalid_argument("invalid IVFPQ query batch");
    }
    if (params.nprobe <= 0 || params.nprobe > ctx.index.lists.nlist ||
        params.topk <= 0 || params.batch_size <= 0) {
        throw std::invalid_argument("invalid IVFPQ search parameters");
    }
    if (!output.indices || !output.distances) {
        throw std::invalid_argument("invalid IVFPQ search output");
    }
    if (ctx.options.rerank_dtype == IVFExactRerankDType::U8 &&
        (!ctx.index.storage.rerank_u8 || !queries.u8)) {
        throw std::invalid_argument("IVFPQ U8 rerank requires U8 base and queries");
    }
    if (ctx.options.rerank_dtype == IVFExactRerankDType::FP32 &&
        !ctx.index.storage.rerank_fp32) {
        throw std::invalid_argument("IVFPQ FP32 rerank requires FP32 base storage");
    }
    const bool rerank_enabled =
        ctx.options.rerank_dtype != IVFExactRerankDType::Disabled &&
        ctx.options.n_rerank > params.topk;
    if (ctx.options.adc_strategy ==
            paraselect::search::PQAdcSearchStrategy::Threshold &&
        (!rerank_enabled || ctx.options.threshold.seed_probes <= 0 ||
         ctx.options.n_rerank <= 0 ||
         ctx.options.threshold.candidate_capacity < ctx.options.n_rerank ||
         ctx.options.threshold.candidate_capacity > 8192)) {
        throw std::invalid_argument("invalid IVFPQ threshold ADC parameters");
    }
    if (ctx.options.adc_strategy !=
            paraselect::search::PQAdcSearchStrategy::Dense &&
        ctx.options.adc_strategy !=
            paraselect::search::PQAdcSearchStrategy::Threshold) {
        throw std::invalid_argument("unknown IVFPQ ADC strategy");
    }
}

}  // namespace

namespace paraselect::search {

struct IVFPQIndex::Impl {
    explicit Impl(IVFPQBuildConfig requested_config)
        : config(std::move(requested_config)) {
        const auto build_start = Clock::now();
        artifacts = paraselect::ivfpq::build_artifacts(config);

        const float* centroids = artifacts.transformed_centroids.empty()
            ? artifacts.caches.clusters.centroids.data()
            : artifacts.transformed_centroids.data();
        ctx.index.coarse = {
            centroids, config.nlist, config.source.spec().dim};
        ctx.index.lists = {
            artifacts.caches.clusters.cluster_offsets.data(),
            artifacts.caches.clusters.reordered_indices.data(),
            config.nlist,
            static_cast<idx_t>(config.n_vectors)};
        ctx.index.storage.rerank_fp32 =
            artifacts.rerank_fp32.empty() ? nullptr : artifacts.rerank_fp32.data();
        ctx.index.storage.rerank_u8 =
            artifacts.rerank_u8.empty() ? nullptr : artifacts.rerank_u8.data();
        ctx.index.storage.pq.codes = artifacts.caches.pq.codes.data();
        ctx.index.storage.pq.codebook = artifacts.caches.pq.codebook.data();
        ctx.index.storage.pq.subvectors = config.pq_m;
        ctx.index.storage.pq.codebook_size = config.pq_k;
        ctx.options.n_rerank = config.n_rerank;
        ctx.options.cpu_rerank_threads = config.cpu_rerank_threads;
        ctx.options.pin_rerank_threads = config.pin_rerank_threads;
        ctx.options.rerank_prefetch = config.rerank_prefetch;
        ctx.options.order_candidate_ids = config.order_candidate_ids;
        ctx.options.rerank_dtype = artifacts.rerank_u8.empty()
            ? (artifacts.rerank_fp32.empty()
                   ? IVFExactRerankDType::Disabled
                   : IVFExactRerankDType::FP32)
            : IVFExactRerankDType::U8;
        ctx.options.resident_codes =
            config.fine_storage == IVFPQFineStorage::Resident;
        ctx.options.resident_combo = config.resident_combo;
        ctx.options.adc_strategy = config.adc_strategy;
        ctx.options.threshold = config.threshold;
        validate_ivf_index_shape(ctx.index.coarse, ctx.index.lists);
        if (!ctx.index.storage.pq.codes || !ctx.index.storage.pq.codebook ||
            ctx.index.storage.pq.subvectors <= 0 ||
            ctx.index.storage.pq.codebook_size <= 0 ||
            ctx.index.coarse.dim % ctx.index.storage.pq.subvectors != 0) {
            throw std::runtime_error("IVFPQ cache produced invalid storage");
        }
        try {
            if (!uses_shared_pipeline()) ivfpq_prepare_index(&ctx);
        } catch (...) {
            ivfpq_release(&ctx);
            throw;
        }
        if (ctx.resident_codes.enabled) {
            std::vector<uint8_t>().swap(artifacts.caches.pq.codes);
            ctx.index.storage.pq.codes = nullptr;
        }
        build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - build_start).count();
    }

    bool uses_shared_pipeline() const {
        return config.transform == IVFPQTransformKind::Plain &&
               config.fine_storage == IVFPQFineStorage::Resident &&
               config.adc_strategy == PQAdcSearchStrategy::Threshold;
    }

    void search_shared(const IVFQueryBatchView& queries,
                       const IVFSearchParams& params,
                       const IVFSearchOutputView& output,
                       IVFSearchStats* stats) {
        validate_search_request(ctx, queries, params, output);
        IVFResPQSearchRequest request;
        request.index.coarse = ctx.index.coarse;
        request.index.lists = ctx.index.lists;
        request.index.storage.pq = ctx.index.storage.pq;
        request.index.storage.assets.exact = IVFExactStorageView::from_host({
            ctx.index.storage.rerank_fp32, ctx.index.storage.rerank_u8,
            ctx.index.lists.n_vectors, ctx.index.coarse.dim});
        request.queries = queries;
        request.params = params;
        request.output = output;
        auto& options = request.options;
        options.pq_distance_mode = PQAdcDistanceMode::Plain;
        options.n_rerank = config.n_rerank;
        options.float_rerank = ctx.options.rerank_dtype == IVFExactRerankDType::FP32;
        options.cpu.rerank_threads = config.cpu_rerank_threads;
        options.cpu.pin_rerank_threads = config.pin_rerank_threads;
        options.cpu.rerank_prefetch = config.rerank_prefetch;
        options.gpu.combo = config.resident_combo.c_str();
        options.gpu.candidate_count = config.threshold.candidate_k > 0
            ? config.threshold.candidate_k : config.n_rerank;
        options.overlap_cpu_pipeline = config.overlap_cpu_pipeline;
        options.adc_strategy = config.adc_strategy;
        options.adc_mode = ResidualPQAdcMode::QueryLut;
        options.query_lut_quantization = config.query_lut_quantization;
        options.adc_threshold = config.threshold;
        ivfrespq_search(request, stats);
    }

    ~Impl() { ivfpq_release(&ctx); }

    void bind_request(
        const IVFQueryBatchView& queries,
        const IVFSearchParams& params,
        const IVFSearchOutputView& output) {
        validate_search_request(ctx, queries, params, output);
        const auto transform_start = Clock::now();
        transformed_queries.clear();
        ctx.options.transformed_queries = nullptr;
        if (artifacts.transform) {
            transformed_queries.resize(
                static_cast<size_t>(queries.count) * queries.dim);
            artifacts.transform->apply_host_to_host(
                queries.fp32,
                transformed_queries.data(),
                static_cast<size_t>(queries.count),
                nullptr);
            ctx.options.transformed_queries = transformed_queries.data();
        }
        query_transform_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - transform_start).count();
        const bool reuse_prepared =
            prepared && prepared_shape_matches(ctx, queries, params);
        if (prepared && !reuse_prepared) prepared = false;
        ctx.queries = queries;
        ctx.params = params;
        ctx.output = output;
        if (!prepared) {
            ivfpq_prepare(&ctx);
            prepared = true;
        }
        ctx.coarse.d_top_nprobe_batch_valid = 0;
        ctx.pq_scanned_codes_total = 0;
        ctx.pq_adc_lookups_total = 0;
        ctx.threshold_candidate_total = 0;
        ctx.threshold_query_total = 0;
        ctx.threshold_candidate_max = 0;
        std::fill(
            ctx.rerank_candidate_count_batch.begin(),
            ctx.rerank_candidate_count_batch.end(),
            0);
    }

    void run(paraselect::search::IVFSearchStatsCollector* collector) {
        ctx.collector = collector;
        if (collector) {
            collector->addField("search_impl", ivfpq_search_impl_name(ctx));
            collector->addField(
                "pq_M", static_cast<double>(ctx.index.storage.pq.subvectors));
            collector->addField(
                "pq_K", static_cast<double>(ctx.index.storage.pq.codebook_size));
        }
        try {
            ivfpq_run_pipeline(&ctx);
            CUDA_CHECK(cudaDeviceSynchronize());
        } catch (const std::exception& error) {
            ctx.collector = nullptr;
            std::fprintf(
                stderr,
                "[IVFPQIndex][exception] what=%s n_query=%d dim=%d "
                "nlist=%d nprobe=%d topk=%d\n",
                error.what(),
                ctx.queries.count,
                ctx.queries.dim,
                ctx.index.lists.nlist,
                ctx.params.nprobe,
                ctx.params.topk);
            throw;
        }
        ctx.collector = nullptr;
    }

    IVFPQBuildConfig config;
    paraselect::ivfpq::BuildArtifacts artifacts;
    std::vector<float> transformed_queries;
    IVFPQSearchContext ctx;
    double build_ms = 0.0;
    double query_transform_ms = 0.0;
    bool prepared = false;
};

IVFPQIndex IVFPQIndex::build(IVFPQBuildConfig config) {
    return IVFPQIndex(std::make_unique<Impl>(std::move(config)));
}

IVFPQIndex::IVFPQIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

IVFPQIndex::~IVFPQIndex() = default;
IVFPQIndex::IVFPQIndex(IVFPQIndex&&) noexcept = default;
IVFPQIndex& IVFPQIndex::operator=(IVFPQIndex&&) noexcept = default;

void IVFPQIndex::search(
    const IVFQueryBatchView& queries,
    const IVFSearchParams& params,
    const IVFSearchOutputView& output,
    IVFSearchStats* stats) {
    if (impl_->uses_shared_pipeline()) {
        impl_->search_shared(queries, params, output, stats);
        return;
    }
    paraselect::search::IVFSearchStatsCollector collector(stats);
    auto* active_collector = stats ? &collector : nullptr;
    if (active_collector) active_collector->startTotal();
    impl_->bind_request(queries, params, output);
    impl_->run(active_collector);
    if (active_collector) {
        const double wall_ms = active_collector->stopTotal();
        const int batches =
            (queries.count + params.batch_size - 1) / params.batch_size;
        const double preprocess_per_batch =
            impl_->query_transform_ms / std::max(1, batches);
        active_collector->setAverageBatchTime(wall_ms / std::max(1, batches));
        active_collector->addStageTime("query_preprocess", preprocess_per_batch);
        active_collector->addToField("ivfpq_avg_batch_ms", preprocess_per_batch);
        active_collector->addToField("ivfpq_batch_total_ms", preprocess_per_batch);
        active_collector->finalizeCoreFields();
    }
}

size_t IVFPQIndex::size() const { return impl_->config.n_vectors; }
int IVFPQIndex::dim() const { return impl_->config.source.spec().dim; }
int IVFPQIndex::nlist() const { return impl_->config.nlist; }
double IVFPQIndex::build_time_ms() const { return impl_->build_ms; }

}  // namespace paraselect::search
