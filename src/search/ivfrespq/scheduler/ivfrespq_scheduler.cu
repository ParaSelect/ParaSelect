#include "search/ivfrespq/scheduler/ivfrespq_scheduler.cuh"
#include "search/ivfrespq/pipeline/ivfrespq_pipeline.cuh"

void ivfrespq_select_pipeline(IVFRESPQSearchCtx* ctx) {
    if (ctx && ctx->arg.candidate_only) {
        ivfrespq_pipeline_cpu(ctx);
        return;
    }
    const bool chunk_overlap = ctx && ctx->arg.cpu_batch_pipeline_overlap;
    if (chunk_overlap && (!ctx || ctx->fine.device.use_resident_rerank == 0)) {
        ivfrespq_pipeline_chunk_overlap(ctx);
        return;
    }
    ivfrespq_pipeline_cpu(ctx);
}
