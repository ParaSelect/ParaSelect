/**
 * Stage3 CPU ()
 *  combo/runner  CPU fine.
 */

#include "utils/pch.h"
#include "search/ivfflat/stage/ivfflat_stages.cuh"

#include <stdexcept>

void ivfflat_fine_cpu(
    IVFSearchCtx* ctx,
    const paraselect::cpu_fine::CpuFineBatchView& view)
{
    if (!ctx) {
        throw std::runtime_error("Stage3-CPU: invalid ctx");
    }
    const IVFSearchArg* arg = &ctx->arg;
    const paraselect::search::ivfflat_stage::CpuFineRunConfig cfg{
        view.n_query > 0 ? view.n_query : arg->n_query,
        arg->n_dim,
        arg->n_probes,
        arg->k,
        0,
    };
    paraselect::search::ivfflat_stage::run_cpu_fine_batch(&ctx->flat_fine, cfg, view);
}
