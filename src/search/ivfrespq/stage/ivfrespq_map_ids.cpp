#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

static bool ivfrespq_debug_spacev_enabled() {
    const char* env = std::getenv("PARASELECT_IVFRESPQ_DEBUG_SPACEV");
    return env && *env && std::strcmp(env, "0") != 0;
}

static int ivfrespq_debug_batch_limit() {
    int limit = 4;
    if (const char* env = std::getenv("PARASELECT_IVFRESPQ_DEBUG_BATCH_LIMIT")) {
        if (*env) limit = std::atoi(env);
    }
    return std::max(0, limit);
}

static bool ivfrespq_debug_this_batch(int batch_start, int batch_n) {
    const int limit = ivfrespq_debug_batch_limit();
    return ivfrespq_debug_spacev_enabled() && limit > 0 && batch_start < limit * batch_n;
}

}  // namespace

void ivfrespq_map_ids_view(
    IVFRESPQSearchCtx* ctx,
    const IVFRESPQBatchView& view)
{
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int batch_start = view.batch_start;
    const int batch_n = view.batch_n;
    idx_t* batch_idx_out = view.h_topk_index_out;
    const int* local_idx = view.h_topk_local_idx_batch;
    const size_t n = (size_t)batch_n * arg.k;
    long long invalid_local = 0;
    long long invalid_global = 0;
    long long valid_global = 0;
    int min_local = std::numeric_limits<int>::max();
    int max_local = -1;
    idx_t min_global = std::numeric_limits<idx_t>::max();
    idx_t max_global = -1;
    for (size_t i = 0; i < n; ++i) {
        const int local = local_idx[i];
        if (local < 0) {
            ++invalid_local;
            batch_idx_out[i] = (idx_t)-1;
            continue;
        }
        min_local = std::min(min_local, local);
        max_local = std::max(max_local, local);
        const idx_t global = arg.host_rows ? idx_t(local)
            : fine.assets.h_reordered_indices[(size_t)local];
        if (global < 0) {
            ++invalid_global;
        } else {
            ++valid_global;
            min_global = std::min(min_global, global);
            max_global = std::max(max_global, global);
        }
        batch_idx_out[i] = global;
    }
    if (ivfrespq_debug_this_batch(batch_start, batch_n)) {
        std::cerr << "[IVFRESPQ_DEBUG_SPACEV] stage3_map"
                  << " batch_start=" << batch_start
                  << " batch_n=" << batch_n
                  << " topk=" << arg.k
                  << " invalid_local=" << invalid_local
                  << " invalid_global=" << invalid_global
                  << " min_local=" << (invalid_local == (long long)n ? -1 : min_local)
                  << " max_local=" << max_local
                  << " min_global=" << (valid_global ? min_global : (idx_t)-1)
                  << " max_global=" << (valid_global ? max_global : (idx_t)-1)
                  << "\n";
        const int rows = std::min(batch_n, 2);
        for (int q = 0; q < rows; ++q) {
            std::cerr << "[IVFRESPQ_DEBUG_SPACEV] stage3_map q=" << q << " local=";
            for (int k = 0; k < arg.k; ++k) {
                if (k) std::cerr << ",";
                std::cerr << local_idx[(size_t)q * arg.k + k];
            }
            std::cerr << " global=";
            for (int k = 0; k < arg.k; ++k) {
                if (k) std::cerr << ",";
                std::cerr << batch_idx_out[(size_t)q * arg.k + k];
            }
            std::cerr << "\n";
        }
    }
}
void ivfrespq_map_ids(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n)
{
    IVFRESPQBatchView view{};
    view.batch_start = batch_start;
    view.batch_n = batch_n;
    view.h_topk_index_out = ctx->arg.h_topk_index_out + (size_t)batch_start * ctx->arg.k;
    view.h_topk_local_idx_batch = ctx->fine.host.h_topk_local_idx_batch;
    ivfrespq_map_ids_view(ctx, view);
}
