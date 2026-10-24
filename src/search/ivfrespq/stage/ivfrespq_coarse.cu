#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "search/kernels/coarse/coarse.h"
#include "search/kernels/coarse/runtime/coarse_runtime.h"
#include "l2norm/l2norm.cuh"
#include "quantization/linear/linear_fp16.cuh"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>

namespace {

static bool ivfrespq_detailed_timing_enabled() {
    const char* env = std::getenv("PARASELECT_IVFRESPQ_DETAILED_TIMING");
    return env && env[0] == '1';
}

static bool ivfrespq_probe_load_skip_coarse_enabled() {
    const char* env = std::getenv("PARASELECT_RPQ_PROBE_LOAD_SKIP_COARSE");
    return env && env[0] == '1';
}

static const std::uint16_t* fp16_query_view_for(
    const IVFResSearchState& arg,
    const float* h_query,
    int n_query,
    int dim) {
    if (!arg.h_queries_fp16 || !arg.h_queries_fp32 || !h_query ||
        n_query <= 0 || dim <= 0) {
        return nullptr;
    }
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(arg.h_queries_fp32);
    const std::uintptr_t query = reinterpret_cast<std::uintptr_t>(h_query);
    const std::size_t total_bytes =
        static_cast<std::size_t>(arg.n_query) * dim * sizeof(float);
    if (query < base || query > base + total_bytes) return nullptr;
    const std::size_t byte_offset = query - base;
    if (byte_offset % sizeof(float) != 0) return nullptr;
    const std::size_t element_offset = byte_offset / sizeof(float);
    if (element_offset + static_cast<std::size_t>(n_query) * dim >
        static_cast<std::size_t>(arg.n_query) * dim) {
        return nullptr;
    }
    return arg.h_queries_fp16 + element_offset;
}

static paraselect::coarse::Request make_coarse_request(
    IVFRESPQSearchCtx* ctx,
    int n_query)
{
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;

    paraselect::coarse::Request run_args;
    run_args.fp16_coarse = arg.fp16_coarse;
    run_args.distance_mode = arg.distance_mode;
    run_args.n_query = n_query;
    run_args.dim = arg.n_dim;
    run_args.nlist = arg.lists.nlist;
    run_args.nprobe = arg.n_probes;
    run_args.d_query = coarse.d_query_batch;
    run_args.d_query_fp16 = coarse.d_query_fp16;
    run_args.d_query_norm = coarse.d_query_norm;
    run_args.d_cluster_centers = coarse.d_cluster_centers;
    run_args.d_cluster_centers_fp16 = coarse.d_cluster_centers_fp16;
    run_args.d_cluster_centers_norm = coarse.d_cluster_centers_norm;
    run_args.d_top_nprobe_dist = coarse.d_top_nprobe_dist;
    run_args.d_top_nprobe_index = coarse.d_top_nprobe_index;
    run_args.cublas_handle = coarse.cublas_handle;
    run_args.stream = coarse.stream;
    return run_args;
}

static void ivfrespq_preprocess_query(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query)
{
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    const int dim = arg.n_dim;
    const std::uint16_t* h_query_fp16 =
        fp16_query_view_for(arg, h_query, n_query, dim);
    if (arg.fp16_coarse && h_query_fp16) {
        // The coarse PCIe path is half precision end-to-end.  Decode once on
        // device because residual/fine stages still consume FP32 queries.
        CUDA_CHECK(cudaMemcpyAsync(
            coarse.d_query_fp16,
            h_query_fp16,
            static_cast<size_t>(n_query) * dim * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice,
            coarse.stream));
        ivf::LinearQuantizerFP16GPU quantizer(1.f, 0.f);
        quantizer.decode_async(
            coarse.d_query_fp16, n_query, dim, coarse.d_query_batch, coarse.stream);
        compute_l2_norm_gpu(
            coarse.d_query_batch,
            coarse.d_query_norm,
            n_query,
            dim,
            L2NORM_AUTO,
            coarse.stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
            coarse.d_query_batch,
            h_query,
            (size_t)n_query * dim * sizeof(float),
            cudaMemcpyHostToDevice,
            coarse.stream));
        compute_l2_norm_gpu(
            coarse.d_query_batch,
            coarse.d_query_norm,
            n_query,
            dim,
            L2NORM_AUTO,
            coarse.stream);
        if (arg.fp16_coarse) {
            ivf::LinearQuantizerFP16GPU quantizer(1.f, 0.f);
            quantizer.encode_async(
                coarse.d_query_batch, n_query, dim, coarse.d_query_fp16, coarse.stream);
        }
    }
}

// Debug/A-B hook: replace only the probe IDs after normal online coarse search.
// The file is intentionally fixed-size and must match this invocation exactly.
static bool ivfrespq_load_probe_ids_override(
    IVFRESPQSearchCtx* ctx,
    int n_query,
    int nprobe)
{
    const char* path = std::getenv("PARASELECT_RPQ_PROBE_LOAD_PATH");
    if (!path || !*path) return false;
    if (n_query <= 0 || nprobe <= 0) {
        throw std::runtime_error("probe load requires positive query/probe counts");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("failed to open probe load file: ") + path);
    }
    char magic[8] = {};
    std::uint32_t version = 0;
    std::uint32_t query_start = 0;
    std::uint32_t query_count = 0;
    std::uint32_t probe_count = 0;
    std::uint32_t flags = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&query_start), sizeof(query_start));
    in.read(reinterpret_cast<char*>(&query_count), sizeof(query_count));
    in.read(reinterpret_cast<char*>(&probe_count), sizeof(probe_count));
    in.read(reinterpret_cast<char*>(&flags), sizeof(flags));
    const bool has_offsets = std::memcmp(magic, "IAPRB003", 8) == 0 && version == 3;
    const bool has_distances =
        (std::memcmp(magic, "IAPRB002", 8) == 0 && version == 2) || has_offsets;
    const bool ids_only = std::memcmp(magic, "IAPRB001", 8) == 0 && version == 1;
    if (!in || (!has_distances && !ids_only && !has_offsets) ||
        query_count != static_cast<std::uint32_t>(n_query) ||
        probe_count != static_cast<std::uint32_t>(nprobe) || (flags & 1u) == 0) {
        throw std::runtime_error("probe load header does not match query/probe invocation");
    }
    (void)query_start;
    std::vector<idx_t> ids((size_t)n_query * (size_t)nprobe);
    in.read(reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(idx_t)));
    if (!in) throw std::runtime_error("probe load file is truncated");
    std::vector<float> distances;
    if (has_distances && (flags & 2u)) {
        distances.resize(ids.size());
        in.read(reinterpret_cast<char*>(distances.data()),
                static_cast<std::streamsize>(distances.size() * sizeof(float)));
        if (!in) throw std::runtime_error("probe load distance section is truncated");
    }
    if (has_offsets && (flags & 4u)) {
        std::vector<std::int64_t> offsets(ids.size());
        in.read(reinterpret_cast<char*>(offsets.data()),
                static_cast<std::streamsize>(offsets.size() * sizeof(std::int64_t)));
        if (!in) throw std::runtime_error("probe load offset section is truncated");
        for (std::int64_t offset : offsets) {
            if (offset < 0) throw std::runtime_error("probe load contains a negative list offset");
        }
    }
    for (idx_t id : ids) {
        if (id < 0 || id >= static_cast<idx_t>(ctx->arg.lists.nlist)) {
            throw std::runtime_error("probe load contains an out-of-range list ID");
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        ctx->coarse.d_top_nprobe_index,
        ids.data(),
        ids.size() * sizeof(idx_t),
        cudaMemcpyHostToDevice,
        ctx->coarse.stream));
    if (!distances.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(
            ctx->coarse.d_top_nprobe_dist,
            distances.data(),
            distances.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            ctx->coarse.stream));
    }

    // A coarse bypass is valid only when the payload supplies distances for
    // downstream threshold/fine stages. IDs-only remains a post-coarse A/B hook.
    const bool can_skip_coarse = has_distances && !distances.empty();
    return can_skip_coarse;
}

static void ivfrespq_coarse_impl(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    bool query_preloaded,
    int n_query,
    idx_t* h_top_nprobe_out,
    idx_t* d_top_nprobe_out,
    float* d_top_nprobe_dist_out,
    bool copy_top_nprobe_to_host,
    bool sync_when_device_only)
{
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    const IVFResSearchState& arg = ctx->arg;
    const int nprobe = arg.n_probes;
    const bool detailed_timing = ivfrespq_detailed_timing_enabled() && ctx->collector;

    if (!query_preloaded) {
        const auto prep_t0 = std::chrono::high_resolution_clock::now();
        ivfrespq_preprocess_query(ctx, h_query, n_query);
        if (detailed_timing) {
            CUDA_SYNC_CHECK(coarse.stream, "ivfrespq stage1 preprocess");
            const auto prep_t1 = std::chrono::high_resolution_clock::now();
            ctx->collector->addStageTime(
                "ivfrespq_coarse_query_preprocess",
                std::chrono::duration<double, std::milli>(prep_t1 - prep_t0).count());
        }
    } else if (detailed_timing) {
        ctx->collector->addStageTime("ivfrespq_coarse_query_preprocess", 0.0);
    }

    bool probe_payload_loaded = false;
    if (ivfrespq_probe_load_skip_coarse_enabled()) {
        probe_payload_loaded = ivfrespq_load_probe_ids_override(ctx, n_query, nprobe);
    }

    if (!probe_payload_loaded) {
        paraselect::coarse::Request run_args = make_coarse_request(ctx, n_query);
        paraselect::coarse::Timing timing;

        paraselect::coarse::search(run_args, *coarse.coarse_workspace, detailed_timing ? &timing : nullptr);
        if (detailed_timing) {
            ctx->collector->addStageTime("ivfrespq_coarse_init_topk", timing.init_topk_ms);
            ctx->collector->addStageTime("ivfrespq_coarse_tile_fill", timing.tile_fill_ms);
            ctx->collector->addStageTime("ivfrespq_coarse_gemm", timing.gemm_ms);
            ctx->collector->addStageTime("ivfrespq_coarse_topk", timing.topk_ms);
            ctx->collector->addStageTime("ivfrespq_coarse_merge", timing.merge_ms);
        }
    }

    if (!probe_payload_loaded) {
        (void)ivfrespq_load_probe_ids_override(ctx, n_query, nprobe);
    }

    if (d_top_nprobe_out) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_top_nprobe_out,
            coarse.d_top_nprobe_index,
            (size_t)n_query * nprobe * sizeof(idx_t),
            cudaMemcpyDeviceToDevice,
            coarse.stream));
        if (d_top_nprobe_dist_out) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_top_nprobe_dist_out,
                coarse.d_top_nprobe_dist,
                (size_t)n_query * nprobe * sizeof(float),
                cudaMemcpyDeviceToDevice,
                coarse.stream));
        }
        if (sync_when_device_only) {
            CUDA_SYNC_CHECK(coarse.stream, "ivfrespq stage1 coarse to device");
        }
        if (detailed_timing) {
            ctx->collector->addStageTime("ivfrespq_coarse_d2h_topnprobe", 0.0);
        }
    } else if (copy_top_nprobe_to_host) {
        const auto copy_t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(
            h_top_nprobe_out,
            coarse.d_top_nprobe_index,
            (size_t)n_query * nprobe * sizeof(idx_t),
            cudaMemcpyDeviceToHost,
            coarse.stream));
        CUDA_SYNC_CHECK(coarse.stream, "ivfrespq stage1 coarse");
        const auto copy_t1 = std::chrono::high_resolution_clock::now();
        if (detailed_timing) {
            ctx->collector->addStageTime(
                "ivfrespq_coarse_d2h_topnprobe",
                std::chrono::duration<double, std::milli>(copy_t1 - copy_t0).count());
        }
    } else {
        if (sync_when_device_only) {
            CUDA_SYNC_CHECK(coarse.stream, "ivfrespq stage1 coarse for gpu");
        }
        if (detailed_timing) {
            ctx->collector->addStageTime("ivfrespq_coarse_d2h_topnprobe", 0.0);
        }
    }
}

}  // namespace

void ivfrespq_coarse(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query,
    idx_t* h_top_nprobe_out)
{
    ivfrespq_coarse_impl(
        ctx, h_query, false, n_query, h_top_nprobe_out, nullptr, nullptr, true, true);
}

void ivfrespq_coarse_for_gpu(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query)
{
    ivfrespq_coarse_impl(
        ctx, h_query, false, n_query, nullptr, nullptr, nullptr, false, false);
}

void ivfrespq_coarse_to_device(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query,
    idx_t* d_top_nprobe_out,
    float* d_top_nprobe_dist_out)
{
    ivfrespq_coarse_impl(
        ctx, h_query, false, n_query, nullptr, d_top_nprobe_out,
        d_top_nprobe_dist_out, false, false);
}

void ivfrespq_coarse_preload_query_batch(
    IVFRESPQSearchCtx* ctx,
    const float* h_query,
    int n_query)
{
    ivfrespq_preprocess_query(ctx, h_query, n_query);
}

void ivfrespq_coarse_preloaded_to_device(
    IVFRESPQSearchCtx* ctx,
    int n_query,
    idx_t* d_top_nprobe_out,
    float* d_top_nprobe_dist_out)
{
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    const IVFResSearchState& arg = ctx->arg;
    const int dim = arg.n_dim;
    const int nprobe = arg.n_probes;
    const int sub_batch = coarse.query_tile > 0
        ? coarse.query_tile
        : n_query;

    float* saved_query = coarse.d_query_batch;
    float* saved_query_norm = coarse.d_query_norm;
    __half* saved_query_fp16 = coarse.d_query_fp16;

    for (int off = 0; off < n_query; off += sub_batch) {
        const int n_this = std::min(sub_batch, n_query - off);
        coarse.d_query_batch = saved_query + (size_t)off * dim;
        coarse.d_query_norm = saved_query_norm + off;
        if (saved_query_fp16) {
            coarse.d_query_fp16 = saved_query_fp16 + (size_t)off * dim;
        }
        ivfrespq_coarse_impl(
            ctx,
            nullptr,
            true,
            n_this,
            nullptr,
            d_top_nprobe_out + (size_t)off * nprobe,
            d_top_nprobe_dist_out
                ? d_top_nprobe_dist_out + (size_t)off * nprobe
                : nullptr,
            false,
            false);
    }

    coarse.d_query_batch = saved_query;
    coarse.d_query_norm = saved_query_norm;
    coarse.d_query_fp16 = saved_query_fp16;
}
