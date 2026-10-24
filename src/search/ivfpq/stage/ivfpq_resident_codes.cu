#include "search/ivfpq/stage/ivfpq_resident_codes.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_combos.h"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"
#include "search/ivfpq/stage/ivfpq_stages.cuh"
#include "search/ivfrespq/stage/ivfrespq_candidate_order.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace {

constexpr int kMaxPqM = 64;
constexpr int kPqK = 256;

}  // namespace

static void release_workspace(IVFPQResidentCodeContext& res) {
    if (res.d_queries) cudaFree(res.d_queries);
    if (res.d_top_nprobe) cudaFree(res.d_top_nprobe);
    if (res.d_adc_candidate_idx) cudaFree(res.d_adc_candidate_idx);
    if (res.d_adc_candidate_dist) cudaFree(res.d_adc_candidate_dist);
    paraselect::search::residual_pq_grouped_workspace_release(&res.adc_workspace);
    paraselect::search::pq_adc_threshold_workspace_release(&res.threshold_workspace);
    res.d_queries = nullptr;
    res.d_top_nprobe = nullptr;
    res.d_adc_candidate_idx = nullptr;
    res.d_adc_candidate_dist = nullptr;
    res.max_batch_n = 0;
    res.local_k = 0;
    res.query_prepared = 0;
}

void ivfpq_resident_codes_prepare_storage(IVFPQSearchContext* ctx) {
    if (!ctx) return;
    IVFPQResidentCodeContext& res = ctx->resident_codes;
    const auto& index = ctx->index;
    const auto& options = ctx->options;
    res.enabled = options.resident_codes ? 1 : 0;
    if (!res.enabled) {
        if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
            throw std::invalid_argument(
                "ivfpq threshold ADC requires resident PQ code storage");
        }
        return;
    }
    const int pq_m = index.storage.pq.subvectors;
    const int pq_k = index.storage.pq.codebook_size;
    const int dsub = index.coarse.dim / pq_m;
    if (pq_k != kPqK || pq_m <= 0 || pq_m > kMaxPqM ||
        (dsub != 2 && dsub != 4 && dsub != 8)) {
        throw std::invalid_argument("ivfpq resident codes: unsupported PQ/stage1 shape");
    }
    const auto* combo = paraselect::search::resolve_residual_pq_gpu_combo_ptr_or_throw(
        options.resident_combo.c_str());
    if (combo->impl != paraselect::search::ResidualPQGpuImpl::GroupedVectorChunk) {
        throw std::invalid_argument(
            "ivfpq resident codes: plain ADC requires grouped vector chunk combo");
    }
    if (!res.storage_prepared) {
        std::strncpy(res.combo_name, combo->name, sizeof(res.combo_name) - 1);
        res.combo_name[sizeof(res.combo_name) - 1] = '\0';
        CUDA_CHECK(cudaStreamCreate(&res.stream));
        CUDA_CHECK(cudaMalloc(
            &res.d_pq_codes,
            (size_t)index.lists.n_vectors * pq_m * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &res.d_cluster_offsets,
            ((size_t)index.lists.nlist + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(
            &res.d_cluster_counts,
            (size_t)index.lists.nlist * sizeof(int)));
        CUDA_CHECK(cudaMalloc(
            &res.d_pq_codebook,
            (size_t)pq_m * pq_k * dsub * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(
            res.d_pq_codes,
            index.storage.pq.codes,
            (size_t)index.lists.n_vectors * pq_m * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            res.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            res.d_cluster_offsets,
            ctx->cluster_offsets.data(),
            ((size_t)index.lists.nlist + 1) * sizeof(long long),
            cudaMemcpyHostToDevice,
            res.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            res.d_cluster_counts,
            ctx->cluster_counts.data(),
            (size_t)index.lists.nlist * sizeof(int),
            cudaMemcpyHostToDevice,
            res.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            res.d_pq_codebook,
            index.storage.pq.codebook,
            (size_t)pq_m * pq_k * dsub * sizeof(float),
            cudaMemcpyHostToDevice,
            res.stream));
        CUDA_SYNC_CHECK(res.stream, "ivfpq resident code storage prepare");
        res.storage_prepared = 1;
    } else if (std::strcmp(res.combo_name, combo->name) != 0) {
        throw std::logic_error("IVFPQ resident combo changed after index prepare");
    }

}

void ivfpq_resident_codes_prepare_workspace(IVFPQSearchContext* ctx) {
    if (!ctx) return;
    IVFPQResidentCodeContext& res = ctx->resident_codes;
    release_workspace(res);
    if (!res.enabled) return;

    const auto& index = ctx->index;
    const auto& params = ctx->params;
    const auto& options = ctx->options;
    const int pq_m = index.storage.pq.subvectors;
    const int pq_k = index.storage.pq.codebook_size;
    const int dsub = index.coarse.dim / pq_m;
    const auto* combo = paraselect::search::resolve_residual_pq_gpu_combo_ptr_or_throw(
        options.resident_combo.c_str());
    const int stage1_k =
        ctx->pq_stage1_k > 0 ? ctx->pq_stage1_k : params.topk;
    if (stage1_k <= 0 ||
        (options.adc_strategy ==
             paraselect::search::PQAdcSearchStrategy::Dense &&
         stage1_k > 512)) {
        throw std::invalid_argument("ivfpq resident codes: unsupported stage1 shape");
    }
    res.adc_spec = paraselect::search::make_residual_pq_gpu_spec_from_combo(
        *combo, params.topk, pq_m, pq_k);
    res.adc_spec.pq_d_sub = dsub;
    res.adc_spec.pq_d_sub_padded = dsub;
    res.adc_spec.pq_dim_padded = index.coarse.dim;
    res.adc_spec.search_strategy = options.adc_strategy;
    res.adc_spec.threshold = options.threshold;
    if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Dense) {
        if (stage1_k > res.adc_spec.global_k) {
            throw std::invalid_argument("ivfpq resident codes: stage1_k exceeds ADC candidate count");
        }
        res.local_k = res.adc_spec.global_k;
    } else {
        if (options.n_rerank <= 0 ||
            options.n_rerank > options.threshold.candidate_capacity ||
            stage1_k != options.n_rerank) {
            throw std::invalid_argument(
                "ivfpq threshold ADC requires 0 < n_rerank <= candidate_capacity");
        }
        // Plain IVFPQ has no separate ResPQ candidate-width option. Its
        // stage-1 output width is the requested rerank input count.
        res.adc_spec.threshold.candidate_k = options.n_rerank;
        res.local_k = options.threshold.candidate_capacity;
    }
    try {
        res.max_batch_n = ctx->query_batch_size;
        CUDA_CHECK(cudaMalloc(
            &res.d_queries,
            (size_t)res.max_batch_n * index.coarse.dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &res.d_top_nprobe,
            (size_t)res.max_batch_n * params.nprobe * sizeof(idx_t)));
        if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Dense) {
            CUDA_CHECK(cudaMalloc(&res.d_adc_candidate_idx,
                                  (size_t)res.max_batch_n * res.local_k * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&res.d_adc_candidate_dist,
                                  (size_t)res.max_batch_n * res.local_k * sizeof(float)));
        } else {
            paraselect::search::pq_adc_threshold_workspace_reserve(
                &res.threshold_workspace, res.max_batch_n,
                options.threshold.candidate_capacity, options.threshold.seed_probes);
        }
        paraselect::search::residual_pq_grouped_workspace_reserve(
            &res.adc_workspace,
            res.max_batch_n,
            params.nprobe,
            std::max(1, res.adc_spec.max_vector_chunks_per_probe),
            res.adc_spec.group_chunks,
            res.adc_spec.global_k);
        res.query_prepared = 1;
    } catch (...) {
        release_workspace(res);
        throw;
    }
}

void ivfpq_resident_codes_release(IVFPQSearchContext* ctx) {
    if (!ctx) return;
    IVFPQResidentCodeContext& res = ctx->resident_codes;
    release_workspace(res);
    if (res.d_pq_codes) cudaFree(res.d_pq_codes);
    if (res.d_cluster_offsets) cudaFree(res.d_cluster_offsets);
    if (res.d_cluster_counts) cudaFree(res.d_cluster_counts);
    if (res.d_pq_codebook) cudaFree(res.d_pq_codebook);
    if (res.stream) cudaStreamDestroy(res.stream);
    res = IVFPQResidentCodeContext{};
}

void ivfpq_fine_resident_codes_gpu(IVFPQSearchContext* ctx, int batch_start, int batch_n) {
    const auto& index = ctx->index;
    const auto& queries = ctx->queries;
    const auto& params = ctx->params;
    const auto& options = ctx->options;
    const float* pq_queries = options.transformed_queries
        ? options.transformed_queries
        : queries.fp32;
    IVFPQResidentCodeContext& res = ctx->resident_codes;
    if (!res.query_prepared || batch_n <= 0) return;
    const auto t0 = std::chrono::high_resolution_clock::now();
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_query = nullptr;
    cudaEvent_t ev_adc = nullptr;
    cudaEvent_t ev_prefix = nullptr;
    cudaEvent_t ev_d2h = nullptr;
    if (ctx->collector) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_query));
        CUDA_CHECK(cudaEventCreate(&ev_adc));
        CUDA_CHECK(cudaEventCreate(&ev_prefix));
        CUDA_CHECK(cudaEventCreate(&ev_d2h));
        CUDA_CHECK(cudaEventRecord(ev_start, res.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(res.d_queries,
                               pq_queries + (size_t)batch_start * queries.dim,
                               (size_t)batch_n * queries.dim * sizeof(float),
                               cudaMemcpyHostToDevice, res.stream));
    if (ctx->collector) CUDA_CHECK(cudaEventRecord(ev_query, res.stream));
    const idx_t* d_top_nprobe = res.d_top_nprobe;
    if (ctx->coarse.d_top_nprobe_batch_valid && ctx->coarse.d_top_nprobe_batch) {
        d_top_nprobe = ctx->coarse.d_top_nprobe_batch;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(res.d_top_nprobe,
                                   ctx->top_nprobe_batch.data(),
                                   (size_t)batch_n * params.nprobe * sizeof(idx_t),
                                   cudaMemcpyHostToDevice, res.stream));
    }
    const int stage1_k = ctx->pq_stage1_k > 0 ? ctx->pq_stage1_k : params.topk;
    if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
        pq_adc_threshold_search_gpu(
            paraselect::search::PQAdcDistanceMode::Plain,
            res.d_queries,
            nullptr,
            d_top_nprobe,
            res.d_pq_codes,
            res.d_cluster_offsets,
            res.d_cluster_counts,
            res.d_pq_codebook,
            batch_n,
            queries.dim,
            index.lists.nlist,
            params.nprobe,
            res.adc_spec,
            nullptr,
            nullptr,
            res.stream,
            &res.threshold_workspace,
            &res.adc_workspace);
        if (ctx->collector) CUDA_CHECK(cudaEventRecord(ev_adc, res.stream));
        if (options.order_candidate_ids && ctx->pq_rerank_enabled) {
            // Only permute already retained candidates; -1 padding stays at the end.
            ivfrespq_order_candidates_by_local_idx_gpu(
                res.threshold_workspace.d_candidate_idx, nullptr,
                batch_n, stage1_k, res.stream);
        }
        if (ctx->collector) CUDA_CHECK(cudaEventRecord(ev_prefix, res.stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            ctx->rerank_candidate_idx_batch.data(),
            (size_t)stage1_k * sizeof(int),
            res.threshold_workspace.d_candidate_idx,
            (size_t)res.threshold_workspace.candidate_capacity * sizeof(int),
            (size_t)stage1_k * sizeof(int),
            batch_n,
            cudaMemcpyDeviceToHost, res.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            ctx->rerank_candidate_count_batch.data(),
            res.threshold_workspace.d_candidate_count,
            (size_t)batch_n * sizeof(int), cudaMemcpyDeviceToHost, res.stream));
    } else {
        pq_adc_search_gpu(
            paraselect::search::PQAdcDistanceMode::Plain,
            res.d_queries,
            nullptr,
            d_top_nprobe,
            res.d_pq_codes,
            res.d_cluster_offsets,
            res.d_cluster_counts,
            res.d_pq_codebook,
            batch_n,
            queries.dim,
            index.lists.nlist,
            params.nprobe,
            res.adc_spec,
            nullptr,
            nullptr,
            res.d_adc_candidate_idx,
            res.d_adc_candidate_dist,
            res.stream,
            nullptr,
            &res.adc_workspace);
        if (ctx->collector) CUDA_CHECK(cudaEventRecord(ev_adc, res.stream));
        if (options.order_candidate_ids && ctx->pq_rerank_enabled) {
            // Only permute already retained candidates; -1 padding stays at the end.
            ivfrespq_order_candidates_by_local_idx_gpu(
                res.d_adc_candidate_idx, res.d_adc_candidate_dist,
                batch_n, stage1_k, res.stream, res.local_k);
        }
        if (ctx->collector) CUDA_CHECK(cudaEventRecord(ev_prefix, res.stream));
        if (ctx->pq_rerank_enabled) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                ctx->rerank_candidate_idx_batch.data(),
                (size_t)stage1_k * sizeof(int),
                res.d_adc_candidate_idx,
                (size_t)res.adc_spec.global_k * sizeof(int),
                (size_t)stage1_k * sizeof(int),
                batch_n,
                cudaMemcpyDeviceToHost,
                res.stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                ctx->topk_local_idx_batch.data(),
                (size_t)params.topk * sizeof(int),
                res.d_adc_candidate_idx,
                (size_t)res.adc_spec.global_k * sizeof(int),
                (size_t)params.topk * sizeof(int),
                batch_n,
                cudaMemcpyDeviceToHost,
                res.stream));
            CUDA_CHECK(cudaMemcpy2DAsync(
                ctx->output.distances + (size_t)batch_start * params.topk,
                (size_t)params.topk * sizeof(float),
                res.d_adc_candidate_dist,
                (size_t)res.adc_spec.global_k * sizeof(float),
                (size_t)params.topk * sizeof(float),
                batch_n,
                cudaMemcpyDeviceToHost,
                res.stream));
        }
    }
    if (ctx->collector) CUDA_CHECK(cudaEventRecord(ev_d2h, res.stream));
    CUDA_SYNC_CHECK(res.stream, "ivfpq resident codes scan");

    if (options.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
        ctx->threshold_candidate_total += std::accumulate(
            ctx->rerank_candidate_count_batch.begin(),
            ctx->rerank_candidate_count_batch.begin() + batch_n,
            0LL);
        ctx->threshold_query_total += batch_n;
        for (int q = 0; q < batch_n; ++q) {
            ctx->threshold_candidate_max = std::max(
                ctx->threshold_candidate_max,
                ctx->rerank_candidate_count_batch[(size_t)q]);
        }
    }

    float query_h2d_ms = 0.0f;
    float adc_ms = 0.0f;
    float prefix_ms = 0.0f;
    float d2h_ms = 0.0f;
    if (ctx->collector) {
        CUDA_CHECK(cudaEventElapsedTime(&query_h2d_ms, ev_start, ev_query));
        CUDA_CHECK(cudaEventElapsedTime(&adc_ms, ev_query, ev_adc));
        CUDA_CHECK(cudaEventElapsedTime(&prefix_ms, ev_adc, ev_prefix));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_prefix, ev_d2h));
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_query);
        cudaEventDestroy(ev_adc);
        cudaEventDestroy(ev_prefix);
        cudaEventDestroy(ev_d2h);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    long long scanned = 0;
    if (!ctx->coarse.d_top_nprobe_batch_valid) {
        for (int q = 0; q < batch_n; ++q) {
            for (int p = 0; p < params.nprobe; ++p) {
                const idx_t cluster =
                    ctx->top_nprobe_batch[(size_t)q * params.nprobe + p];
                if (cluster >= 0 && cluster < index.lists.nlist) {
                    scanned += ctx->cluster_counts.data()[cluster];
                }
            }
        }
        ctx->pq_scanned_codes_total += scanned;
        ctx->pq_adc_lookups_total += scanned * index.storage.pq.subvectors;
    }
    if (ctx->collector) {
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ctx->collector->addStageTime("ivfpq_resident_codes_gpu", ms);
        ctx->collector->addStageTime("ivfpq_adc_query_h2d", query_h2d_ms);
        ctx->collector->addStageTime("ivfpq_adc_kernel", adc_ms);
        ctx->collector->addStageTime("ivfpq_adc_prefix", prefix_ms);
        ctx->collector->addStageTime("ivfpq_adc_d2h", d2h_ms);
        ctx->collector->addField("ivfpq_resident_combo", res.combo_name);
        ctx->collector->addField("ivfpq_resident_local_k", (double)res.local_k);
        ctx->collector->addField("ivfpq_adc_global_k", (double)res.adc_spec.global_k);
        ctx->collector->addField("rpq_local_k", (double)res.local_k);
        ctx->collector->addField("rpq_global_k", (double)stage1_k);
        ctx->collector->addStageTime("pq_adc_scan", ms);
        ctx->collector->addStageTime("pq_adc_lut_scan", ms);
        ctx->collector->addStageTime("rpq_gpu_search", ms);
        ctx->collector->addStageTime("rpq_fine_gpu_wall", ms);
        ctx->collector->addField("pq_scanned_codes", (double)ctx->pq_scanned_codes_total);
        ctx->collector->addField("pq_adc_lookups", (double)ctx->pq_adc_lookups_total);
    }
}
