#include "search/kernels/coarse/coarse_prepare.h"
#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_prepare_internal.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_combos.h"
#include "utils/check_cuda_error.cuh"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cub/cub.cuh>

int residual_pq_candidate_source_count(
    int nprobe,
    const paraselect::search::ResidualPQGpuSearchSpec& spec)
{
    if (paraselect::search::residual_pq_gpu_impl_uses_probe_chunks(spec.impl)) {
        return (nprobe + spec.probes_per_chunk - 1) / spec.probes_per_chunk;
    }
    if (paraselect::search::residual_pq_gpu_impl_uses_vector_chunks(spec.impl)) {
        return nprobe * std::max(1, spec.max_vector_chunks_per_probe);
    }
    if (paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(spec.impl)) {
        return std::max(1, nprobe * std::max(1, spec.max_vector_chunks_per_probe));
    }
    return nprobe;
}

int residual_pq_max_vector_chunks_per_probe(
    const IVFResSearchState& arg,
    const paraselect::search::ResidualPQGpuSearchSpec& spec,
    const int* cluster_counts)
{
    if (!paraselect::search::residual_pq_gpu_impl_uses_vector_chunks(spec.impl) &&
        !paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(spec.impl)) {
        return 1;
    }
    if (spec.vector_chunk_size <= 0) {
        throw std::invalid_argument(
            "ivfrespq_prepare: residual PQ vector chunk size must be positive");
    }
    int max_count = 0;
    for (int cid = 0; cid < arg.lists.nlist; ++cid) {
        max_count = std::max(max_count, cluster_counts[cid]);
    }
    return std::max(1, (max_count + spec.vector_chunk_size - 1) / spec.vector_chunk_size);
}


void prepare_fine_scan_workspace(
    IVFRESPQSearchCtx* ctx,
    const paraselect::search::ResidualPQGpuSearchSpec& rpq,
    int nprobe,
    size_t threshold_workspace_bytes) {
    auto& fine = ctx->fine;
    auto& coarse = ctx->coarse;
    const auto& arg = ctx->arg;
    if (ivfrespq_uses_query_lut_seed(arg, rpq)) return;
    const int candidate_sources = residual_pq_candidate_source_count(nprobe, rpq);
    if (paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(rpq.impl)) {
        int grouped_workspace_max_chunks = rpq.max_vector_chunks_per_probe;
        paraselect::search::residual_pq_grouped_workspace_reserve(
        &fine.device.rpq_grouped_workspace,
        coarse.query_batch_size,
        nprobe,
        grouped_workspace_max_chunks,
        rpq.group_chunks,
        rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
            ? 512 : rpq.global_k,
        threshold_workspace_bytes);
    } else {
        const size_t candidate_count =
        ((size_t)coarse.query_batch_size * (size_t)candidate_sources *
         (size_t)rpq.local_k);
        CUDA_CHECK(cudaMalloc(&fine.device.d_probe_candidate_idx, candidate_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&fine.device.d_probe_candidate_dist, candidate_count * sizeof(float)));
    }
}
