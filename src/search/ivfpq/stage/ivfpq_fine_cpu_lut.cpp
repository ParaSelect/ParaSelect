#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/stage/ivfpq_stages.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {

struct Candidate {
    float dist;
    int local_idx;
};

bool operator<(const Candidate& a, const Candidate& b) {
    return a.dist < b.dist;
}

}  // namespace

void ivfpq_fine_cpu_lut(IVFPQSearchContext* ctx, int batch_start, int batch_n) {
    const auto& index = ctx->index;
    const auto& queries = ctx->queries;
    const auto& params = ctx->params;
    const float* pq_queries = ctx->options.transformed_queries
        ? ctx->options.transformed_queries
        : queries.fp32;
    const int dim = queries.dim;
    const int pq_m = index.storage.pq.subvectors;
    const int pq_k = index.storage.pq.codebook_size;
    const int dsub = dim / pq_m;
    const int topk = params.topk;
    const int stage1_k = ctx->pq_stage1_k > 0 ? ctx->pq_stage1_k : topk;
    const auto t0 = std::chrono::high_resolution_clock::now();
    double lut_build_ms = 0.0;
    double adc_scan_ms = 0.0;
    long long scanned = 0;
    long long lookups = 0;
    std::vector<float> lut((size_t)pq_m * pq_k, 0.0f);
    std::fill(ctx->topk_local_idx_batch.begin(), ctx->topk_local_idx_batch.end(), -1);
    std::fill(ctx->rerank_candidate_idx_batch.begin(), ctx->rerank_candidate_idx_batch.end(), -1);

    for (int bq = 0; bq < batch_n; ++bq) {
        const int qid = batch_start + bq;
        const float* query = pq_queries + (size_t)qid * dim;
        const auto lut_t0 = std::chrono::high_resolution_clock::now();
        for (int m = 0; m < pq_m; ++m) {
            const float* qsub = query + (size_t)m * dsub;
            for (int code = 0; code < pq_k; ++code) {
                const float* center = index.storage.pq.codebook +
                    ((size_t)m * pq_k + code) * dsub;
                float dist = 0.0f;
                for (int d = 0; d < dsub; ++d) {
                    const float diff = qsub[d] - center[d];
                    dist += diff * diff;
                }
                lut[(size_t)m * pq_k + code] = dist;
            }
        }
        const auto lut_t1 = std::chrono::high_resolution_clock::now();
        lut_build_ms += std::chrono::duration<double, std::milli>(lut_t1 - lut_t0).count();

        std::priority_queue<Candidate> heap;
        const auto scan_t0 = std::chrono::high_resolution_clock::now();
        for (int p = 0; p < params.nprobe; ++p) {
            const idx_t cluster_id =
                ctx->top_nprobe_batch[(size_t)bq * params.nprobe + p];
            if (cluster_id < 0 || cluster_id >= index.lists.nlist) continue;
            const long long begin = ctx->cluster_offsets[(size_t)cluster_id];
            const long long end = ctx->cluster_offsets[(size_t)cluster_id + 1];
            for (long long local = begin; local < end; ++local) {
                const uint8_t* codes = index.storage.pq.codes + (size_t)local * pq_m;
                float dist = 0.0f;
                for (int m = 0; m < pq_m; ++m) {
                    dist += lut[(size_t)m * pq_k + codes[m]];
                }
                ++scanned;
                lookups += pq_m;
                if ((int)heap.size() < stage1_k) {
                    heap.push({dist, (int)local});
                } else if (dist < heap.top().dist) {
                    heap.pop();
                    heap.push({dist, (int)local});
                }
            }
        }
        const auto scan_t1 = std::chrono::high_resolution_clock::now();
        adc_scan_ms += std::chrono::duration<double, std::milli>(scan_t1 - scan_t0).count();

        std::vector<Candidate> sorted((size_t)heap.size());
        for (int i = (int)sorted.size() - 1; i >= 0; --i) {
            sorted[(size_t)i] = heap.top();
            heap.pop();
        }
        if (ctx->pq_rerank_enabled) {
            const size_t cand_base = (size_t)bq * stage1_k;
            const int n_out = std::min(stage1_k, (int)sorted.size());
            for (int i = 0; i < n_out; ++i) {
                ctx->rerank_candidate_idx_batch[cand_base + i] = sorted[(size_t)i].local_idx;
            }
        } else {
            const size_t out_base = (size_t)bq * topk;
            for (int i = 0; i < topk; ++i) {
                if (i >= (int)sorted.size()) {
                    ctx->topk_local_idx_batch[out_base + i] = -1;
                    ctx->output.distances[(size_t)qid * topk + i] =
                        std::numeric_limits<float>::max();
                } else {
                    ctx->topk_local_idx_batch[out_base + i] = sorted[(size_t)i].local_idx;
                    ctx->output.distances[(size_t)qid * topk + i] = sorted[(size_t)i].dist;
                }
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    ctx->pq_scanned_codes_total += scanned;
    ctx->pq_adc_lookups_total += lookups;
    if (ctx->collector) {
        ctx->collector->addStageTime("pq_lut_build", lut_build_ms);
        ctx->collector->addStageTime("pq_adc_scan", adc_scan_ms);
        ctx->collector->addStageTime("pq_adc_lut_scan", std::chrono::duration<double, std::milli>(t1 - t0).count());
        ctx->collector->addField("pq_scanned_codes", (double)ctx->pq_scanned_codes_total);
        ctx->collector->addField("pq_adc_lookups", (double)ctx->pq_adc_lookups_total);
    }
}
