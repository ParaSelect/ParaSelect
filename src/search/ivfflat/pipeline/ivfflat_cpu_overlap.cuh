#ifndef PARASELECT_IVFFLAT_PIPELINE_CPU_OVERLAP_CUH
#define PARASELECT_IVFFLAT_PIPELINE_CPU_OVERLAP_CUH

#include "search/ivfflat/ivfflat_search_ctx.h"


void ivfflat_cpu_overlap(
    IVFSearchCtx* ctx,
    paraselect::search::IVFSearchStats* stats);

#endif  // PARASELECT_IVFFLAT_PIPELINE_CPU_OVERLAP_CUH
