#ifndef PARASELECT_IVFFLAT_PIPELINE_H
#define PARASELECT_IVFFLAT_PIPELINE_H

#pragma once

#include "search/ivfflat/ivfflat_search_ctx.h"
#include "search/ivfflat/ivfflat_search.cuh"

void ivfflat_pipeline(
    IVFSearchCtx* ctx,
    paraselect::search::IVFSearchStats* stats = nullptr
);

#endif
