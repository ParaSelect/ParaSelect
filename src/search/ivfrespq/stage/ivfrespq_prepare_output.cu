#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_prepare_internal.cuh"
#include "utils/check_cuda_error.cuh"
#include <algorithm>
#include <cstdlib>
#include <new>
#include <vector>

namespace {
static std::vector<int> make_compact_to_padded_map(
    const long long* compact_offsets,
    const int* counts,
    int nlist,
    int n_total_vectors)
{
    std::vector<int> map((size_t)n_total_vectors, -1);
    int padded = 0;
    for (int cid = 0; cid < nlist; ++cid) {
        const int count = counts[cid];
        const int compact = (int)compact_offsets[cid];
        for (int i = 0; i < count; ++i) {
            map[(size_t)compact + (size_t)i] = padded + i;
        }
        padded += ((count + 31) / 32) * 32;
    }
    return map;
}

static void prepare_pca_resources(IVFRESPQSearchCtx* ctx) {
    const auto& arg = ctx->arg;
    auto& fine = ctx->fine;
    auto& coarse = ctx->coarse;
        if (arg.pca_storage.enabled()) {
            fine.host.h_shortlist_candidate_idx = static_cast<int*>(std::malloc(
                static_cast<size_t>(coarse.query_batch_size) *
                arg.shortlist_count * sizeof(int)));
            if (!fine.host.h_shortlist_candidate_idx) throw std::bad_alloc();
            if (arg.pca_backend == IVFRefineBackend::Gpu || fine.device.use_resident_rerank) {
                CUDA_CHECK(cudaMalloc(
                    &fine.device.d_pca_candidate_idx,
                    static_cast<size_t>(coarse.query_batch_size) *
                        arg.shortlist_count * sizeof(int)));
            }
            if (arg.pca_backend == IVFRefineBackend::Gpu) {
                const size_t pca_values = static_cast<size_t>(arg.lists.n_vectors) *
                    arg.pca_storage.dim;
                CUDA_CHECK(cudaMalloc(&fine.device.d_pca_vectors, pca_values * sizeof(int8_t)));
                CUDA_CHECK(cudaMemcpyAsync(
                    fine.device.d_pca_vectors,
                    arg.pca_storage.vectors,
                    pca_values * sizeof(int8_t),
                    cudaMemcpyHostToDevice,
                    coarse.stream));
                fine.assets.pca_gpu_refiner =
                    std::make_shared<paraselect::search::IVFPCAGpuRefiner>(
                        arg.pca_storage,
                        fine.device.d_pca_vectors,
                        coarse.query_batch_size,
                        fine.runtime.candidate_width,
                        arg.shortlist_count);
            }
        }
}

}

void ivfrespq_prepare_candidate_output(IVFRESPQSearchCtx* ctx) {
    const auto& arg = ctx->arg;
    auto& fine = ctx->fine;
    auto& coarse = ctx->coarse;
    const auto& rpq = fine.assets.residual_pq_gpu_spec;
    const int nprobe = arg.n_probes;
    const int topk = arg.k;
        const size_t rerank_candidate_count =
            (size_t)coarse.query_batch_size * (size_t)fine.runtime.candidate_width;
        if (rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
            // Exact seed reuses this output; the global scan pool is separate.
            paraselect::search::pq_adc_threshold_workspace_reserve(
                &fine.device.rpq_threshold_workspace,
                coarse.query_batch_size,
                std::max(rpq.threshold.candidate_k, rpq.threshold.seed_rank),
                std::min(rpq.threshold.seed_probes, nprobe));
        } else {
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_rerank_candidate_idx,
                rerank_candidate_count * sizeof(int)));
            if (rpq.global_k > 256 || fine.device.use_resident_rerank || arg.candidate_only ||
                paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(rpq.impl)) {
                CUDA_CHECK(cudaMalloc(
                    &fine.device.d_rerank_candidate_dist,
                    rerank_candidate_count * sizeof(float)));
            }
        }
        if (!fine.device.use_resident_rerank || arg.pca_backend == IVFRefineBackend::Cpu) {
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&fine.host.h_rerank_candidate_idx),
                rerank_candidate_count * sizeof(int)));
        }
        prepare_pca_resources(ctx);
        if (fine.device.use_resident_rerank) {
            const std::vector<int> compact_to_padded = make_compact_to_padded_map(
                fine.host.h_cluster_offsets,
                ctx->cluster_counts.data(),
                arg.lists.nlist,
                arg.lists.n_vectors);
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_resident_compact_to_padded,
                (size_t)arg.lists.n_vectors * sizeof(int)));
            CUDA_CHECK(cudaMemcpyAsync(
                fine.device.d_resident_compact_to_padded,
                compact_to_padded.data(),
                compact_to_padded.size() * sizeof(int),
                cudaMemcpyHostToDevice,
                coarse.stream));
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_resident_topk_local_idx,
                (size_t)coarse.query_batch_size * (size_t)topk * sizeof(int)));
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_resident_topk_dist,
                (size_t)coarse.query_batch_size * (size_t)topk * sizeof(float)));
        }
}
