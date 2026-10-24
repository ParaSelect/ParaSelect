#include "search/ivfjl/ivfjl_search.cuh"

#include "search/ivf_common/ivf_search_stats_collector.h"
#include "l2norm/l2norm.cuh"
#include "search/ivfflat/pipeline/ivfflat_probe_metadata.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/ivfjl/ivfjl_prepare.cuh"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_executor.h"
#include "transform/orthogonal_projection_transform.cuh"
#include "utils/check_cuda_error.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace paraselect::search;

struct IVFJLSearchState {
    const float* h_queries_fp32 = nullptr;
    IVFListLayoutView lists;
    const float* h_cluster_vectors = nullptr;
    const float* h_cluster_centers = nullptr;
    const float* h_cluster_norm_in = nullptr;
    int n_query = 0;
    int original_dim = 0;
    int projected_dim = 0;
    int n_probes = 0;
    int k = 0;
    int query_batch_size = 0;
    int distance_mode = 0;
    unsigned long long jl_seed = 12345ULL;
    bool fp16_coarse = false;
    bool cpu_batch_pipeline_overlap = false;
    idx_t* h_topk_index_out = nullptr;
    float* h_topk_dist_out = nullptr;
    int* h_coarse_cluster_ids_out = nullptr;
    const char* cpu_fine_combo_name = nullptr;
};

namespace {

void validate_ivfjl_arg(const IVFJLSearchState* arg) {
    if (!arg) throw std::invalid_argument("ivfjl_search: arg is null");
    if (!arg->h_queries_fp32) throw std::invalid_argument("ivfjl_search: h_queries_fp32 is null");
    validate_ivf_list_layout(arg->lists);
    if (arg->lists.n_vectors > INT_MAX) {
        throw std::invalid_argument("IVFJL currently requires n_vectors <= INT_MAX");
    }
    if (!arg->h_cluster_vectors) throw std::invalid_argument("ivfjl_search: h_cluster_vectors is null");
    if (!arg->h_cluster_centers) throw std::invalid_argument("ivfjl_search: h_cluster_centers is null");
    if (!arg->h_cluster_norm_in) throw std::invalid_argument("ivfjl_search: h_cluster_norm_in is null");
    if (!arg->h_topk_index_out || !arg->h_topk_dist_out || !arg->h_coarse_cluster_ids_out) {
        throw std::invalid_argument("ivfjl_search: output buffers are null");
    }
    if (arg->n_query <= 0 || arg->original_dim <= 0 || arg->projected_dim <= 0 ||
        arg->lists.nlist <= 0 || arg->lists.n_vectors <= 0 || arg->n_probes <= 0 ||
        arg->k <= 0 || arg->query_batch_size <= 0) {
        throw std::invalid_argument("ivfjl_search: invalid shape");
    }
    if (arg->n_probes > arg->lists.nlist) {
        throw std::invalid_argument("ivfjl_search: n_probes exceeds n_total_cluster");
    }
    if (arg->distance_mode != (int)L2_DISTANCE) {
        throw std::invalid_argument("ivfjl_search: only L2 distance is supported");
    }
    if (arg->cpu_batch_pipeline_overlap) {
        throw std::invalid_argument("ivfjl_search: overlap pipeline is not implemented yet");
    }
    if (!arg->cpu_fine_combo_name || !arg->cpu_fine_combo_name[0]) {
        throw std::invalid_argument("ivfjl_search: CPU fine combo is required");
    }
}

void ivfjl_cpu_fine_job(void* owner, const paraselect::cpu_fine::CpuFineBatchView& view) {
    ivfflat_fine_cpu(static_cast<IVFSearchCtx*>(owner), view);
}

struct ScopedFloatPtrSwap {
    float*& slot;
    float* saved;

    ScopedFloatPtrSwap(float*& slot_ref, float* next) : slot(slot_ref), saved(slot_ref) {
        slot = next;
    }

    ~ScopedFloatPtrSwap() {
        slot = saved;
    }

    ScopedFloatPtrSwap(const ScopedFloatPtrSwap&) = delete;
    ScopedFloatPtrSwap& operator=(const ScopedFloatPtrSwap&) = delete;
};

void fill_ivfjl_ctx_from_arg(
    IVFSearchCtx* ctx, const IVFJLSearchState& arg, int query_batch, const int* cluster_counts) {
    IVFSearchArg& a = ctx->arg;
    a.n_query = arg.n_query;
    a.query_batch_size = query_batch;
    a.n_dim = arg.projected_dim;
    a.n_total_clusters = arg.lists.nlist;
    a.n_total_vectors = static_cast<int>(arg.lists.n_vectors);
    a.n_probes = arg.n_probes;
    a.k = arg.k;
    a.distance_mode = arg.distance_mode;
    a.max_probe_slots = 0;
    a.fp16_coarse = arg.fp16_coarse;
    a.cpu_batch_pipeline_overlap = false;
    std::strncpy(a.cpu_fine_combo_name, arg.cpu_fine_combo_name,
                 CPU_FINE_COMBO_NAME_MAX_LEN - 1);
    a.cpu_fine_combo_name[CPU_FINE_COMBO_NAME_MAX_LEN - 1] = '\0';

    ctx->flat_coarse.h_cluster_sizes = cluster_counts;
    ctx->flat_coarse.h_cluster_centers = arg.h_cluster_centers;
    ctx->flat_coarse.h_queries = nullptr;
    ctx->flat_coarse.h_queries_u8 = nullptr;
    ctx->flat_coarse.h_topk_dist_out = arg.h_topk_dist_out;
    ctx->flat_coarse.h_topk_index_out = arg.h_topk_index_out;

    ctx->flat_fine.h_cluster_sizes = cluster_counts;
    ctx->flat_fine.h_queries = nullptr;
    ctx->flat_fine.h_queries_u8 = nullptr;
    ctx->flat_fine.h_topk_dist_out = arg.h_topk_dist_out;
    ctx->flat_fine.h_topk_index_out = arg.h_topk_index_out;
    ctx->flat_fine.h_cluster_vectors = arg.h_cluster_vectors;
    ctx->flat_fine.h_cluster_vectors_u8 = nullptr;
    ctx->flat_fine.h_cluster_vectors_aosoa = nullptr;
    ctx->flat_fine.h_reordered_indices = arg.lists.reordered_ids;
    ctx->flat_fine.h_cluster_centers = arg.h_cluster_centers;
    ctx->flat_fine.h_cluster_norm = const_cast<float*>(arg.h_cluster_norm_in);
}

}  // namespace

static void ivfjl_search_impl(const IVFJLSearchState* user_arg, IVFSearchStats* stats) {
    validate_ivfjl_arg(user_arg);
    const IVFJLSearchState& arg = *user_arg;
    const std::vector<int> cluster_counts = derive_ivf_list_counts(arg.lists);

    const int query_batch = std::max(1, std::min(arg.query_batch_size, arg.n_query));
    const int n_batches = (arg.n_query + query_batch - 1) / query_batch;

    IVFSearchCtx ctx = {};
    ivf_search_ctx_zero(&ctx);
    float* d_query_raw = nullptr;
    std::vector<float> h_query_projected((size_t)query_batch * (size_t)arg.projected_dim);
    double query_transform_total_ms = 0.0;
    double query_preprocess_total_ms = 0.0;
    double coarse_total_ms = 0.0;
    double coarse_gemm_total_ms = 0.0;
    double coarse_topk_total_ms = 0.0;
    double coarse_d2h_total_ms = 0.0;
    double metadata_total_ms = 0.0;
    double fine_total_ms = 0.0;
    long long total_candidate_vectors = 0;
    long long max_candidates_per_query = 0;
    int max_probe_list_size = 0;
    paraselect::search::IVFSearchStatsCollector collector_storage(stats);
    auto* collector = stats ? &collector_storage : nullptr;

    try {
        fill_ivfjl_ctx_from_arg(&ctx, arg, query_batch, cluster_counts.data());
        CUDA_CHECK(cudaStreamCreate(&ctx.flat_coarse.stream_0));
        CUDA_CHECK(cudaStreamCreate(&ctx.flat_coarse.stream_1));
        for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaEventCreate(&ctx.flat_coarse.ev_coarse_done[i]));
        CUDA_CHECK(cudaEventCreate(&ctx.flat_coarse.ev_fine_done));

        ivfjl_prepare(&ctx);
        const int coarse_sub_batch = ctx.flat_coarse.coarse_sub_batch_size > 0
            ? std::min(ctx.flat_coarse.coarse_sub_batch_size, query_batch)
            : query_batch;
        cudaStream_t stream = ctx.flat_coarse.stream_0;

        paraselect::transform::OrthogonalProjectionTransform transform({
            arg.original_dim,
            arg.projected_dim,
            static_cast<uint64_t>(arg.jl_seed),
        });
        transform.prepare();
        CUDA_CHECK(cudaMalloc(&d_query_raw, (size_t)query_batch * arg.original_dim * sizeof(float)));

        if (collector) {
            collector->startTotal();
        }
        auto time_cpu_or_gpu = [&](auto&& fn) -> double {
            if (!collector) {
                fn();
                return 0.0;
            }
            const auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CHECK_CUDA_ERRORS;
            const auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        paraselect::cpu_fine::CpuFineExecutor fine_executor(false);
        for (int bi = 0; bi < n_batches; ++bi) {
            const int batch_start = bi * query_batch;
            const int batch_curr = std::min(query_batch, arg.n_query - batch_start);
            if (collector) collector->startBatch(bi, batch_start, batch_curr);

            idx_t* batch_idx_out = arg.h_topk_index_out + (size_t)batch_start * arg.k;
            float* batch_dist_out = arg.h_topk_dist_out + (size_t)batch_start * arg.k;
            ctx.flat_coarse.h_topk_index_out = batch_idx_out;
            ctx.flat_coarse.h_topk_dist_out = batch_dist_out;
            ctx.flat_fine.h_topk_index_out = batch_idx_out;
            ctx.flat_fine.h_topk_dist_out = batch_dist_out;

            double batch_query_transform_ms = time_cpu_or_gpu([&]() {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_query_raw,
                    arg.h_queries_fp32 + (size_t)batch_start * arg.original_dim,
                    (size_t)batch_curr * arg.original_dim * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
                transform.apply_device(d_query_raw, ctx.flat_coarse.d_pool_query_batch[0], (size_t)batch_curr, stream);
                CUDA_CHECK(cudaMemcpyAsync(h_query_projected.data(),
                                           ctx.flat_coarse.d_pool_query_batch[0],
                                           (size_t)batch_curr * arg.projected_dim * sizeof(float),
                                           cudaMemcpyDeviceToHost,
                                           stream));
            });
            query_transform_total_ms += batch_query_transform_ms;

            double batch_query_preprocess_ms = 0.0;
            double batch_coarse_ms = 0.0;
            double batch_coarse_d2h_ms = 0.0;
            for (int off = 0; off < batch_curr; off += coarse_sub_batch) {
                const int n_this = std::min(coarse_sub_batch, batch_curr - off);
                ctx.flat_coarse.current_n_querys = n_this;
                float* d_projected_sub = ctx.flat_coarse.d_pool_query_batch[0] + (size_t)off * arg.projected_dim;
                ScopedFloatPtrSwap query_ptr_guard(ctx.flat_coarse.d_pool_query_batch[0], d_projected_sub);
                batch_query_preprocess_ms += time_cpu_or_gpu([&]() {
                    compute_l2_norm_gpu(ctx.flat_coarse.d_pool_query_batch[0],
                                        ctx.flat_coarse.d_pool_query_norm[0],
                                        n_this,
                                        arg.projected_dim,
                                        L2NORM_AUTO,
                                        stream);
                });
                batch_coarse_ms += time_cpu_or_gpu([&]() {
                    ctx.flat_coarse.debug_coarse_gemm_ms = 0.0;
                    ctx.flat_coarse.debug_coarse_topk_ms = 0.0;
                    ivfflat_coarse(&ctx, 0);
                    CHECK_CUDA_ERRORS;
                    coarse_gemm_total_ms += ctx.flat_coarse.debug_coarse_gemm_ms;
                    coarse_topk_total_ms += ctx.flat_coarse.debug_coarse_topk_ms;
                });
                batch_coarse_d2h_ms += time_cpu_or_gpu([&]() {
                    CUDA_CHECK(cudaMemcpyAsync(ctx.flat_fine.h_top_nprobe_batch + (size_t)off * arg.n_probes,
                                               ctx.flat_coarse.d_top_nprobe_index[0],
                                               (size_t)n_this * arg.n_probes * sizeof(idx_t),
                                               cudaMemcpyDeviceToHost,
                                               stream));
                });
                for (int i = 0; i < n_this * arg.n_probes; ++i) {
                    arg.h_coarse_cluster_ids_out[(size_t)(batch_start + off) * arg.n_probes + i] =
                        (int)ctx.flat_fine.h_top_nprobe_batch[(size_t)off * arg.n_probes + i];
                }
            }
            query_preprocess_total_ms += batch_query_preprocess_ms;
            coarse_total_ms += batch_coarse_ms;
            coarse_d2h_total_ms += batch_coarse_d2h_ms;

            ctx.flat_coarse.current_n_querys = batch_curr;
            paraselect::cpu_fine::CpuProbeMetadataResult metadata{};
            double batch_metadata_ms = time_cpu_or_gpu([&]() {
                metadata = paraselect::cpu_fine::transfer_cpu_probe_metadata(
                    &ctx,
                    0,
                    batch_curr,
                    stream,
                    {ctx.flat_fine.h_cluster_query_offset_batch,
                     ctx.flat_fine.h_cluster_query_data_batch,
                     ctx.flat_fine.h_active_cluster_ids_batch},
                    "ivfjl");
            });
            metadata_total_ms += batch_metadata_ms;

            for (int q = 0; q < batch_curr; ++q) {
                long long query_candidates = 0;
                for (int p = 0; p < arg.n_probes; ++p) {
                    const idx_t cid = ctx.flat_fine.h_top_nprobe_batch[(size_t)q * arg.n_probes + p];
                    if (cid < 0 || cid >= arg.lists.nlist) continue;
                    const int list_size = cluster_counts[(int)cid];
                    query_candidates += list_size;
                    max_probe_list_size = std::max(max_probe_list_size, list_size);
                }
                total_candidate_vectors += query_candidates;
                max_candidates_per_query = std::max(max_candidates_per_query, query_candidates);
            }

            paraselect::cpu_fine::CpuFineBatchView fine_view{};
            fine_view.n_query = batch_curr;
            fine_view.h_queries = h_query_projected.data();
            fine_view.h_queries_u8 = nullptr;
            fine_view.h_topk_dist_out = batch_dist_out;
            fine_view.h_topk_index_out = batch_idx_out;
            fine_view.h_top_nprobe_batch = ctx.flat_fine.h_top_nprobe_batch;
            fine_view.h_cluster_query_offset_batch = ctx.flat_fine.h_cluster_query_offset_batch;
            fine_view.h_cluster_query_data_batch = ctx.flat_fine.h_cluster_query_data_batch;
            fine_view.h_active_cluster_ids_batch = ctx.flat_fine.h_active_cluster_ids_batch;
            fine_view.active_cluster_count = metadata.active_cluster_count;
            fine_view.h_topk_local_idx_batch = ctx.flat_fine.h_topk_local_idx_batch;
            fine_view.h_thread_dists = ctx.flat_fine.h_thread_dists;
            fine_view.h_thread_idxs = ctx.flat_fine.h_thread_idxs;
            fine_view.h_thread_sizes = ctx.flat_fine.h_thread_sizes;
            fine_view.h_thread_touched = ctx.flat_fine.h_thread_touched;
            fine_view.h_thread_topk_capacity = ctx.flat_fine.h_thread_topk_capacity;
            fine_view.h_thread_query_capacity = ctx.flat_fine.h_thread_query_capacity;

            double batch_fine_ms = time_cpu_or_gpu([&]() {
                fine_executor.submit({&ctx, ivfjl_cpu_fine_job, fine_view});
                fine_executor.wait();
            });
            fine_total_ms += batch_fine_ms;
            const auto fine_timing = fine_executor.latest_timing();
            if (collector) {
                collector->addStageTime("query_transform", batch_query_transform_ms);
                collector->addStageTime("query_preprocess", batch_query_preprocess_ms);
                collector->addStageTime("coarse_search", batch_coarse_ms);
                collector->addStageTime("coarse_d2h", batch_coarse_d2h_ms);
                collector->addStageTime("probe_metadata", batch_metadata_ms);
                collector->addStageTime("probe_metadata_d2h", metadata.d2h_ms);
                collector->addStageTime("cpu_fine", batch_fine_ms);
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
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (collector) {
            collector->stopTotal();
            collector->addField("query_transform_ms", collector->avgStageTimeMs("query_transform"));
            collector->addField("query_transform_total_ms", query_transform_total_ms);
            collector->addField("coarse_stage1_ms", collector->avgStageTimeMs("coarse_search"));
            collector->addField("coarse_gemm_ms", coarse_gemm_total_ms / std::max(1, n_batches));
            collector->addField("coarse_topk_ms", coarse_topk_total_ms / std::max(1, n_batches));
            collector->addField("coarse_d2h_ms", coarse_d2h_total_ms / std::max(1, n_batches));
            collector->addField("probe_metadata_d2h_ms", collector->avgStageTimeMs("probe_metadata_d2h"));
            collector->addField("probe_metadata_compact", 0.0);
            collector->addField("cpu_fine_query_prepare_ms", collector->avgStageTimeMs("cpu_fine_query_prepare"));
            collector->addField("cpu_fine_metadata_build_ms", collector->avgStageTimeMs("cpu_fine_metadata_build"));
            collector->addField("cpu_fine_scratch_clear_ms", collector->avgStageTimeMs("cpu_fine_scratch_clear"));
            collector->addField("cpu_fine_task_build_ms", collector->avgStageTimeMs("cpu_fine_task_build"));
            collector->addField("cpu_fine_scan_ms", collector->avgStageTimeMs("cpu_fine_scan"));
            collector->addField("cpu_fine_merge_ms", collector->avgStageTimeMs("cpu_fine_merge"));
            collector->addField("cpu_fine_runner_ms", collector->avgStageTimeMs("cpu_fine_runner"));
            collector->addField("cpu_fine_id_map_ms", collector->avgStageTimeMs("cpu_fine_id_map"));
            collector->addField("cpu_fine_candidate_vectors", (double)total_candidate_vectors);
            collector->addField("cpu_fine_avg_candidates_per_query",
                                arg.n_query > 0 ? (double)total_candidate_vectors / (double)arg.n_query : 0.0);
            collector->addField("cpu_fine_max_candidates_per_query", (double)max_candidates_per_query);
            collector->addField("cpu_fine_max_probe_list_size", (double)max_probe_list_size);
            collector->addField("transform_d_in", (double)arg.original_dim);
            collector->addField("transform_d_out", (double)arg.projected_dim);
            collector->addField("transform_seed", (double)arg.jl_seed);
            collector->finalizeCoreFields();
            collector->addField(
                "ivfjl_end_to_end_search_ms", collector->averageBatchTimeMs());
            collector->addField("transform", "orthogonal_projection");
        }
    } catch (...) {
        if (d_query_raw) cudaFree(d_query_raw);
        ivf_search_ctx_free_all(&ctx);
        throw;
    }

    if (d_query_raw) cudaFree(d_query_raw);
    ivf_search_ctx_free_all(&ctx);
}

void paraselect::search::ivfjl_search(
    const IVFJLSearchRequest& request,
    IVFSearchStats* stats) {
    validate_ivf_index_shape(request.index.coarse, request.index.lists);

    IVFJLSearchState state;
    state.h_queries_fp32 = request.queries.fp32;
    state.lists = request.index.lists;
    state.h_cluster_vectors = request.index.storage.projected_rowmajor;
    state.h_cluster_centers = request.index.coarse.centroids;
    state.h_cluster_norm_in = request.index.storage.projected_norms;
    state.n_query = request.queries.count;
    state.original_dim = request.queries.dim;
    state.projected_dim = request.index.coarse.dim;
    state.n_probes = request.params.nprobe;
    state.k = request.params.topk;
    state.query_batch_size = request.params.batch_size;
    state.distance_mode = request.params.distance;
    state.jl_seed = request.options.seed;
    state.fp16_coarse = request.params.fp16_coarse;
    state.cpu_batch_pipeline_overlap = request.options.overlap_cpu_pipeline;
    state.h_topk_index_out = request.output.indices;
    state.h_topk_dist_out = request.output.distances;
    state.h_coarse_cluster_ids_out = request.output.coarse_ids;
    state.cpu_fine_combo_name = request.options.cpu_fine_combo;
    ivfjl_search_impl(&state, stats);
}
