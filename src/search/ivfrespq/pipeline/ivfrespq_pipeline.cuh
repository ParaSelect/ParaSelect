#ifndef PARASELECT_IVFRESPQ_PIPELINE_H
#define PARASELECT_IVFRESPQ_PIPELINE_H

#include "search/ivfrespq/ivfrespq_search_ctx.h"

void ivfrespq_pipeline_cpu(IVFRESPQSearchCtx* ctx);
void ivfrespq_pipeline_chunk_overlap(IVFRESPQSearchCtx* ctx);

#endif  // PARASELECT_IVFRESPQ_PIPELINE_H
