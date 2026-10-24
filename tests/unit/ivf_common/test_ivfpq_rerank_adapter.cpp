#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/stage/ivfpq_stages.cuh"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits>

int main() {
    for (bool fp32 : {false, true}) {
        // Second batch: detect incorrect global query/output offsets.
        const int dim = 4, width = 8, topk = 3;
        std::vector<float> base(6 * dim), queries(4 * dim, 0), output(4 * topk, -7);
        std::vector<uint8_t> base_u8(6 * dim), queries_u8(4 * dim, 0);
        for (int r = 0; r < 6; ++r)
            for (int d = 0; d < dim; ++d)
                base[r * dim + d] = base_u8[r * dim + d] = r;
        for (int d = 0; d < dim; ++d) queries[2 * dim + d] = queries_u8[2 * dim + d] = 2;
        IVFPQSearchContext ctx;
        ctx.index.lists.n_vectors = 6;
        ctx.index.storage.rerank_fp32 = base.data();
        ctx.index.storage.rerank_u8 = base_u8.data();
        ctx.queries = {queries.data(), queries_u8.data(), 4, dim};
        ctx.params.topk = topk;
        ctx.output.distances = output.data();
        ctx.options.rerank_dtype = fp32 ? IVFExactRerankDType::FP32 : IVFExactRerankDType::U8;
        ctx.options.cpu_rerank_threads = 2;
        ctx.options.adc_strategy = paraselect::search::PQAdcSearchStrategy::Threshold;
        ctx.pq_rerank_enabled = 1;
        ctx.pq_stage1_k = width;
        ctx.rerank_candidate_idx_batch = {3, 2, 2, 1, -1, 999, 4, 5, 0, 1, 2, 3, 4, 5, 0, 0};
        // Preserve the legacy adapter contract: counts are capped to retained width.
        ctx.rerank_candidate_count_batch = {width + 5, 0};
        ctx.topk_local_idx_batch.resize(2 * topk);
        ivfpq_rerank_cpu(&ctx, 2, 2);
        const std::vector<int> expected = {2, 1, 3, -1, -1, -1};
        if (ctx.topk_local_idx_batch != expected) throw std::runtime_error("IDs/dedup/ties/count mismatch");
        for (int i = 0; i < 6; ++i) if (output[i] != -7) throw std::runtime_error("earlier batch overwritten");
        if (output[6] != 0 || output[7] != 4 || output[8] != 4) throw std::runtime_error("wrong query offset");
        for (int i = 9; i < 12; ++i)
            if (output[i] != std::numeric_limits<float>::max()) throw std::runtime_error("padding mismatch");
    }
    std::cout << "PASS: plain adapter batch offsets, count cap, duplicates, ties, U8/FP32\n";
}
