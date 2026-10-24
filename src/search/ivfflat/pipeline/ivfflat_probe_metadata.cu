#include "ivfflat_probe_metadata.cuh"

#include "ivf_cpu_compact_metadata.h"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace paraselect {
namespace cpu_fine {

CpuProbeMetadataResult build_cpu_probe_metadata(
    const idx_t* top_nprobe_batch,
    int n_query,
    int n_probes,
    int n_total_clusters,
    bool use_compact_metadata,
    int* h_cluster_query_offset,
    int* h_cluster_query_data,
    int* h_active_cluster_ids)
{
    if (!top_nprobe_batch || !h_cluster_query_offset || !h_cluster_query_data || !h_active_cluster_ids) {
        throw std::invalid_argument("build_cpu_probe_metadata: missing inputs");
    }
    if (n_query <= 0 || n_probes <= 0 || n_total_clusters <= 0) {
        throw std::invalid_argument("build_cpu_probe_metadata: invalid sizes");
    }

    CpuProbeMetadataResult result{};
    const size_t ncluster_offsets = (size_t)n_total_clusters + 1u;
    std::memset(h_cluster_query_offset, 0, ncluster_offsets * sizeof(int));

    for (int q = 0; q < n_query; ++q) {
        const size_t row = (size_t)q * (size_t)n_probes;
        for (int p = 0; p < n_probes; ++p) {
            const int cid = static_cast<int>(top_nprobe_batch[row + (size_t)p]);
            if (cid < 0 || cid >= n_total_clusters) continue;
            ++h_cluster_query_offset[cid + 1];
        }
    }
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        h_cluster_query_offset[cid + 1] += h_cluster_query_offset[cid];
    }

    if (use_compact_metadata) {
        std::vector<int> active_pos((size_t)n_total_clusters, -1);
        int active_count = 0;
        for (int cid = 0; cid < n_total_clusters; ++cid) {
            if (h_cluster_query_offset[cid] != h_cluster_query_offset[cid + 1]) {
                active_pos[(size_t)cid] = active_count;
                h_active_cluster_ids[active_count++] = cid;
            }
        }
        std::vector<int> compact_offsets((size_t)active_count + 1u, 0);
        for (int cid = 0; cid < n_total_clusters; ++cid) {
            const int pos = active_pos[(size_t)cid];
            if (pos >= 0) {
                compact_offsets[(size_t)pos + 1u] = h_cluster_query_offset[cid + 1] - h_cluster_query_offset[cid];
            }
        }
        for (int i = 0; i < active_count; ++i) {
            compact_offsets[(size_t)i + 1u] += compact_offsets[(size_t)i];
        }
        std::vector<int> write_pos((size_t)active_count, 0);
        for (int i = 0; i < active_count; ++i) {
            write_pos[(size_t)i] = compact_offsets[(size_t)i];
        }
        for (int q = 0; q < n_query; ++q) {
            const size_t row = (size_t)q * (size_t)n_probes;
            for (int p = 0; p < n_probes; ++p) {
                const int cid = static_cast<int>(top_nprobe_batch[row + (size_t)p]);
                if (cid < 0 || cid >= n_total_clusters) continue;
                const int pos = active_pos[(size_t)cid];
                if (pos < 0) continue;
                h_cluster_query_data[(size_t)write_pos[(size_t)pos]++] = q;
            }
        }
        std::memcpy(h_cluster_query_offset, compact_offsets.data(), ((size_t)active_count + 1u) * sizeof(int));
        result.active_cluster_count = active_count;
        return result;
    }

    ivf_cpu_fill_active_clusters_from_full_offsets(
        h_cluster_query_offset,
        n_total_clusters,
        h_active_cluster_ids,
        &result.active_cluster_count);
    std::vector<int> write_pos(h_cluster_query_offset, h_cluster_query_offset + n_total_clusters);
    for (int q = 0; q < n_query; ++q) {
        const size_t row = (size_t)q * (size_t)n_probes;
        for (int p = 0; p < n_probes; ++p) {
            const int cid = static_cast<int>(top_nprobe_batch[row + (size_t)p]);
            if (cid < 0 || cid >= n_total_clusters) continue;
            h_cluster_query_data[(size_t)write_pos[(size_t)cid]++] = q;
        }
    }
    return result;
}

CpuProbeMetadataResult transfer_cpu_probe_metadata(
    IVFSearchCtx* ctx,
    int pool_slot,
    int n_query,
    cudaStream_t stream,
    CpuProbeMetadataBuffers buffers,
    const char* error_prefix)
{
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFFlatFineCtx* fine = &ctx->flat_fine;
    IVFSearchArg* arg = &ctx->arg;
    CpuProbeMetadataResult result{};
    const char* prefix = error_prefix ? error_prefix : "pipeline";

    if (!buffers.h_cluster_query_offset || !buffers.h_cluster_query_data ||
        !buffers.h_active_cluster_ids) {
        throw std::runtime_error(std::string(prefix) + ": missing host metadata buffers");
    }

    const auto d2h_t0 = std::chrono::high_resolution_clock::now();
    if (fine->h_use_compact_metadata != 0) {
        ivfflat_metadata_compact(ctx, pool_slot);
        CHECK_CUDA_ERRORS;
        const int active_upper = std::min(n_query * arg->n_probes, arg->n_total_clusters);
        CUDA_CHECK(cudaMemcpyAsync(&result.active_cluster_count,
                                   coarse->d_active_cluster_count[pool_slot],
                                   sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        if (result.active_cluster_count < 0 || result.active_cluster_count > active_upper) {
            throw std::runtime_error(std::string(prefix) + ": invalid compact active cluster count");
        }
        CUDA_CHECK(cudaMemcpyAsync(buffers.h_active_cluster_ids,
                                   coarse->d_active_cluster_ids[pool_slot],
                                   (size_t)result.active_cluster_count * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(buffers.h_cluster_query_offset,
                                   coarse->d_compact_cluster_query_offset[pool_slot],
                                   (size_t)(result.active_cluster_count + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(buffers.h_cluster_query_data,
                                   coarse->d_compact_cluster_query_data[pool_slot],
                                   (size_t)n_query * (size_t)arg->n_probes * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
    } else {
        ivfflat_metadata(ctx, pool_slot);
        CHECK_CUDA_ERRORS;
        CUDA_CHECK(cudaMemcpyAsync(buffers.h_cluster_query_offset,
                                   coarse->d_pool_cluster_offset[pool_slot],
                                   (size_t)(arg->n_total_clusters + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;

        const int total_qp = buffers.h_cluster_query_offset[arg->n_total_clusters];
        if (total_qp >= 0 && total_qp <= n_query * arg->n_probes) {
            CUDA_CHECK(cudaMemcpyAsync(buffers.h_cluster_query_data,
                                       coarse->d_cluster_query_data[pool_slot],
                                       (size_t)total_qp * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            cudaStreamSynchronize(stream);
            CHECK_CUDA_ERRORS;
            ivf_cpu_fill_active_clusters_from_full_offsets(
                buffers.h_cluster_query_offset,
                arg->n_total_clusters,
                buffers.h_active_cluster_ids,
                &result.active_cluster_count);
        }
    }
    const auto d2h_t1 = std::chrono::high_resolution_clock::now();
    result.d2h_ms = std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();
    return result;
}

}  // namespace cpu_fine
}  // namespace paraselect
