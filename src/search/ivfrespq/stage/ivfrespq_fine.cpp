#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"

#include <stdexcept>

void ivfrespq_fine(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n)
{
    IVFRESPQFineRunnerFn runner = ctx->fine.runtime.fine_runner;
    if (!runner) {
        throw std::runtime_error("ivfrespq_fine: fine runner is not prepared");
    }
    runner(ctx, batch_start, batch_n);
}
