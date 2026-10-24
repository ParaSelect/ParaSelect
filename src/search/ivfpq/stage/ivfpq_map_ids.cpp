#include "search/ivfpq/ivfpq_search_context.h"
#include "search/ivfpq/stage/ivfpq_stages.cuh"

void ivfpq_map_ids(IVFPQSearchContext* ctx, int batch_start, int batch_n) {
    const auto& index = ctx->index;
    const auto& params = ctx->params;
    const auto& output = ctx->output;
    for (int bq = 0; bq < batch_n; ++bq) {
        const int qid = batch_start + bq;
        for (int i = 0; i < params.topk; ++i) {
            const int local = ctx->topk_local_idx_batch[(size_t)bq * params.topk + i];
            output.indices[(size_t)qid * params.topk + i] =
                local >= 0 ? index.lists.reordered_ids[local] : (idx_t)-1;
        }
    }
}
