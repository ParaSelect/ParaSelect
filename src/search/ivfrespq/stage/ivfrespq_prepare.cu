#include "search/kernels/coarse/coarse_prepare.h"
#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>

#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "search/ivfrespq/stage/ivfrespq_prepare_internal.cuh"
#include "search/ivfrespq/stage/residual_pq_gpu_combos.h"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include "search/kernels/fine/gpu_fine/resident/pq_resident_rerank_combos.cuh"
#include "l2norm/l2norm.cuh"
#include "quantization/linear/prepare_fp16_host.h"
#include "cpu/utils/cpu_affinity.h"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>
#include <cub/cub.cuh>

namespace {

// Keep only one bounded host staging buffer; no duplicate device code or ID map.
template<class T>
static void upload_pq_rows(T* device,const T* host,int width,idx_t rows,
                           cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(device,host,size_t(rows)*width*sizeof(T),
                               cudaMemcpyHostToDevice,stream));
}

static int normalize_query_batch_size(int requested_batch_size, int n_query) {
    if (requested_batch_size <= 0 || requested_batch_size >= n_query) {
        return n_query;
    }
    return requested_batch_size;
}

static uint8_t quantize_u8_sat(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= 255.0f) return 255;
    return static_cast<uint8_t>(x + 0.5f);
}

static bool ivfrespq_resident_rerank_requested(const IVFResSearchState* arg) {
    return arg && arg->exact_storage.kind == IVFExactStorageKind::Resident;
}

static void fill_owned_query_u8(IVFRESPQSearchCtx* ctx) {
    const IVFResSearchState& arg = ctx->arg;
    if (arg.h_queries_u8) {
        ctx->fine.assets.h_queries_u8 = arg.h_queries_u8;
        return;
    }
    ctx->owned_query_u8.resize((size_t)arg.n_query * (size_t)arg.n_dim);
    for (size_t i = 0; i < ctx->owned_query_u8.size(); ++i) {
        ctx->owned_query_u8[i] = quantize_u8_sat(arg.h_queries_fp32[i]);
    }
    ctx->fine.assets.h_queries_u8 = ctx->owned_query_u8.data();
}

static void fill_owned_query_fp16(IVFRESPQSearchCtx* ctx) {
    IVFResSearchState& arg = ctx->arg;
    if (arg.h_queries_fp16) return;
    if (!arg.h_queries_fp32 || arg.n_query <= 0 || arg.n_dim <= 0) {
        throw std::invalid_argument(
            "ivfrespq_prepare: FP16 coarse query cache requires FP32 queries");
    }
    const size_t count = static_cast<size_t>(arg.n_query) * arg.n_dim;
    ctx->owned_query_fp16.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const __half value = __float2half_rn(arg.h_queries_fp32[i]);
        std::memcpy(&ctx->owned_query_fp16[i], &value, sizeof(value));
    }
    arg.h_queries_fp16 = ctx->owned_query_fp16.data();
}

static void fill_owned_base_u8(IVFRESPQSearchCtx* ctx) {
    const IVFResSearchState& arg = ctx->arg;
    if (arg.exact_storage.host.u8) {
        ctx->fine.assets.h_base_u8 = arg.exact_storage.host.u8;
        return;
    }
    ctx->owned_base_u8.resize((size_t)arg.lists.n_vectors * (size_t)arg.n_dim);
    if (!arg.exact_storage.host.fp32) {
        throw std::invalid_argument(
            "ivfrespq_prepare: host U8 rerank requires U8 or FP32 base storage");
    }
    for (size_t i = 0; i < ctx->owned_base_u8.size(); ++i) {
        ctx->owned_base_u8[i] = quantize_u8_sat(arg.exact_storage.host.fp32[i]);
    }
    ctx->fine.assets.h_base_u8 = ctx->owned_base_u8.data();
}

static void build_cluster_offsets(IVFRESPQSearchCtx* ctx) {
    const IVFResSearchState& arg = ctx->arg;
    ctx->fine.host.h_cluster_offsets = static_cast<long long*>(
        std::malloc((size_t)(arg.lists.nlist + 1) * sizeof(long long)));
    if (!ctx->fine.host.h_cluster_offsets) throw std::bad_alloc();

    for (int c = 0; c <= arg.lists.nlist; ++c) {
        ctx->fine.host.h_cluster_offsets[c] = static_cast<long long>(arg.lists.offsets[c]);
    }
}

static bool ivfrespq_gpu_lut_enabled(const IVFResSearchState& arg) {
    if (arg.fine_backend != IVFResPQFineBackend::GpuLutCpuScan) return false;
    if (arg.pq_M != 16 || arg.pq_K != 256 || arg.n_dim != 128) {
        throw std::invalid_argument(
            "ivfrespq_prepare: GPU LUT backend requires dim=128, M=16, K=256");
    }
    return true;
}

// dsub 16 (e.g. 1024D with M=64) is served by the query-LUT path only; the raw
// probe-residual ADC kernels keep their 2/4/8 instantiations and reject it.
static int pad_residual_pq_d_sub(int d_sub) {
    if (d_sub <= 0 || d_sub > 16) return 0;
    if (d_sub <= 2) return 2;
    if (d_sub <= 4) return 4;
    return d_sub <= 8 ? 8 : 16;
}

static bool residual_pq_gpu_enabled(const IVFResSearchState& arg) {
    if (arg.fine_backend != IVFResPQFineBackend::ResidualPQGpu) return false;
    if (arg.pq_K != 256 || arg.pq_M <= 0 || arg.n_dim % arg.pq_M != 0 ||
        pad_residual_pq_d_sub(arg.n_dim / arg.pq_M) == 0) {
        throw std::invalid_argument(
            "ivfrespq_prepare: selected residual PQ GPU backend does not support this PQ shape");
    }
    return true;
}



static bool env_enabled_default_off(const char* name) {
    const char* env = std::getenv(name);
    return env && std::strcmp(env, "1") == 0;
}

static bool phase_log_enabled_prepare() {
    const char* env = std::getenv("PARASELECT_PHASE_LOG");
    return env && *env && std::strcmp(env, "0") != 0;
}

static bool rpq_kernel_breakdown_enabled_prepare() {
    return env_enabled_default_off("PARASELECT_RPQ_KERNEL_BREAKDOWN") ||
           env_enabled_default_off("PARASELECT_PQ_BREAKDOWN");
}

static bool rpq_rerank_breakdown_enabled_prepare() {
    return env_enabled_default_off("PARASELECT_RPQ_RERANK_BREAKDOWN");
}

static paraselect::cpu_fine::CpuPqCodeStorageView prepare_pq_code_storage(
    const IVFResSearchState& arg,
    const IVFRESPQFineCtx& fine,
    const int* cluster_counts)
{
    if (arg.pq.code_storage) {
        return *arg.pq.code_storage;
    }
    return paraselect::cpu_fine::make_pq_row_major_storage(
        arg.pq.codes,
        fine.host.h_cluster_offsets,
        cluster_counts,
        arg.pq_M,
        arg.lists.nlist,
        arg.lists.n_vectors);
}

static std::vector<float> make_padded_pq_codebook(
    const float* codebook,
    int pq_m,
    int pq_k,
    int d_sub,
    int d_sub_padded)
{
    std::vector<float> padded((size_t)pq_m * (size_t)pq_k * (size_t)d_sub_padded, 0.0f);
    for (int m = 0; m < pq_m; ++m) {
        for (int k = 0; k < pq_k; ++k) {
            const float* src = codebook + ((size_t)m * pq_k + k) * d_sub;
            float* dst = padded.data() + ((size_t)m * pq_k + k) * d_sub_padded;
            std::memcpy(dst, src, (size_t)d_sub * sizeof(float));
        }
    }
    return padded;
}

static std::vector<float> make_padded_rowmajor_vectors(
    const float* vectors,
    int n,
    int pq_m,
    int d_sub,
    int d_sub_padded)
{
    std::vector<float> padded((size_t)n * (size_t)pq_m * (size_t)d_sub_padded, 0.0f);
    const int dim = pq_m * d_sub;
    const int padded_dim = pq_m * d_sub_padded;
    for (int row = 0; row < n; ++row) {
        for (int m = 0; m < pq_m; ++m) {
            const float* src = vectors + (size_t)row * dim + (size_t)m * d_sub;
            float* dst = padded.data() + (size_t)row * padded_dim + (size_t)m * d_sub_padded;
            std::memcpy(dst, src, (size_t)d_sub * sizeof(float));
        }
    }
    return padded;
}

static paraselect::search::ResidualPQGpuSearchSpec make_residual_pq_gpu_spec(
    const IVFResSearchState& arg,
    int topk)
{
    if (arg.gpu.thread_topk > 0 &&
        paraselect::search::normalize_residual_pq_thread_k(arg.gpu.thread_topk) !=
            arg.gpu.thread_topk) {
        throw std::invalid_argument(
            "ivfrespq_prepare: GPU thread top-k must be 4, 8, or 16");
    }
    if (arg.gpu.local_topk > 0 &&
        paraselect::search::normalize_residual_pq_local_k(
            arg.gpu.local_topk, topk) != arg.gpu.local_topk) {
        throw std::invalid_argument(
            "ivfrespq_prepare: unsupported GPU local top-k");
    }
    const int candidate_limit =
        arg.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
            ? arg.adc_threshold.candidate_capacity : 512;
    if (arg.gpu.candidate_count > 0 &&
        (arg.gpu.candidate_count < topk || arg.gpu.candidate_count > candidate_limit)) {
        throw std::invalid_argument(
            "ivfrespq_prepare: GPU candidate count is outside the supported range");
    }
    if (arg.cpu.rerank_threads < 0 || arg.cpu.rerank_prefetch < 0) {
        throw std::invalid_argument("ivfrespq_prepare: invalid CPU rerank configuration");
    }
    const int pq_d_sub = arg.n_dim / arg.pq_M;
    const paraselect::search::ResidualPQGpuCombo* combo =
        paraselect::search::resolve_residual_pq_gpu_combo_ptr_or_throw(
            arg.gpu.combo);
    const bool allow_override = arg.gpu.thread_topk > 0 ||
        arg.gpu.local_topk > 0 || arg.gpu.candidate_count > 0;
    paraselect::search::ResidualPQGpuSearchSpec spec =
        paraselect::search::make_residual_pq_gpu_spec_from_combo_with_overrides(
            *combo,
            topk,
            arg.pq_M,
            arg.pq_K,
            allow_override,
            arg.gpu.thread_topk,
            arg.gpu.local_topk,
            arg.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
                ? -1 : arg.gpu.candidate_count,
            -1,
            -1);
    spec.pq_d_sub = pq_d_sub;
    spec.scan_mode = spec.pq_d_sub <= 2
        ? paraselect::search::ResidualPQGpuScanMode::DirectCodebook
        : paraselect::search::ResidualPQGpuScanMode::SharedLut;
    spec.pq_d_sub_padded = pad_residual_pq_d_sub(spec.pq_d_sub);
    spec.pq_dim_padded = arg.pq_M * spec.pq_d_sub_padded;
    if (paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(spec.impl)) {
        if (const char* chunk_env = std::getenv("PARASELECT_RPQ_VECTOR_CHUNK_SIZE")) {
            char* end = nullptr;
            const long parsed = std::strtol(chunk_env, &end, 10);
            if (!end || *end != '\0' || parsed < 256 || parsed > 8192 ||
                (parsed & (parsed - 1)) != 0) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: PARASELECT_RPQ_VECTOR_CHUNK_SIZE must be a power of two in [256,8192]");
            }
            spec.vector_chunk_size = static_cast<int>(parsed);
        }
    }
    spec.search_strategy = arg.adc_strategy;
    spec.threshold = arg.adc_threshold;
    if (arg.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
        // Keep output capacity independent of the exact seed/rerank budget.
        spec.threshold.candidate_k = arg.gpu.candidate_count > 0
            ? arg.gpu.candidate_count : spec.global_k;
        spec.threshold.n_rerank = arg.pq_rerank_n > 0
            ? std::max(topk, arg.pq_rerank_n) : spec.threshold.candidate_k;
        if (spec.threshold.n_rerank > spec.threshold.candidate_k)
            throw std::invalid_argument("n_rerank exceeds GPU candidate output width");
    }
    if (arg.gpu.group_chunks != 0) {
        if (arg.gpu.group_chunks != 1 && arg.gpu.group_chunks != 2 &&
            arg.gpu.group_chunks != 4 && arg.gpu.group_chunks != 8) {
            throw std::invalid_argument(
                "ivfrespq_prepare: GPU group chunks must be 1, 2, 4, or 8");
        }
        spec.group_chunks = arg.gpu.group_chunks;
    }
    if (arg.gpu.query_tile <= 0 || arg.gpu.worker_blocks <= 0 ||
        arg.gpu.worker_blocks > 8192) {
        throw std::invalid_argument("ivfrespq_prepare: invalid GPU query tile or worker blocks");
    }
    spec.fine_query_tile = arg.gpu.query_tile;
    spec.worker_blocks = arg.gpu.worker_blocks;
    return spec;
}

static int residual_pq_candidate_width(
    const paraselect::search::ResidualPQGpuSearchSpec& spec) {
    switch (spec.search_strategy) {
        case paraselect::search::PQAdcSearchStrategy::Dense:
            return spec.global_k;
        case paraselect::search::PQAdcSearchStrategy::Threshold:
            return spec.threshold.candidate_k;
    }
    throw std::invalid_argument("ivfrespq_prepare: unknown ADC search strategy");
}

static void validate_residual_pq_gpu_query_spec(
    const IVFResSearchState& arg,
    const paraselect::search::ResidualPQGpuSearchSpec& spec)
{
    if (arg.n_probes * spec.local_k < arg.k) {
        throw std::invalid_argument(
            "ivfrespq_prepare: residual PQ GPU nprobe * local_k is smaller than topk");
    }
    const int block_threads = paraselect::search::residual_pq_gpu_probe_block_threads(spec.local_k);
    if (spec.thread_k * block_threads < spec.local_k) {
        throw std::invalid_argument(
            "ivfrespq_prepare: residual PQ GPU thread_k * block_threads is smaller than local_k");
    }
}

}  // namespace


void ivfrespq_prepare(IVFRESPQSearchCtx* ctx, const IVFResSearchState* user_arg) {
    ivfrespq_validate_arg(user_arg);
    ctx->arg = *user_arg;
    ctx->cluster_counts = derive_ivf_list_counts(ctx->arg.lists);

    IVFResSearchState& arg = ctx->arg;
    IVFRESPQCoarseCtx& coarse = ctx->coarse;
    if (arg.fp16_coarse) {
        fill_owned_query_fp16(ctx);
    }
    const auto codec_kind = arg.residual_codec;
    IVFRESPQFineCtx& fine = ctx->fine;
    fine.runtime.fixed_tile_rows = 512;
    if (arg.adc_strategy == paraselect::search::PQAdcSearchStrategy::Threshold &&
        arg.adc_threshold.scan_layout == paraselect::search::PQAdcThresholdScanLayout::ListCentric &&
        arg.adc_threshold.list_work_mode == paraselect::search::PQAdcListScanWorkMode::FixedTile) {
        if (const char* value = std::getenv("PARASELECT_RPQ_FIXED_TILE_ROWS")) {
            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || (parsed != 256 && parsed != 512)) {
                throw std::invalid_argument("PARASELECT_RPQ_FIXED_TILE_ROWS must be 256 or 512");
            }
            fine.runtime.fixed_tile_rows = static_cast<int>(parsed);
        }
    }
    const char* order_env = std::getenv("PARASELECT_RPQ_THRESHOLD_ORDER_IDS");
    fine.runtime.order_threshold_ids = order_env && std::strcmp(order_env, "1") == 0;

    if (arg.h_residual_rotation &&
        (codec_kind != paraselect::search::ResidualCodecKind::PQ ||
         arg.fine_backend != IVFResPQFineBackend::ResidualPQGpu ||
         arg.cpu_batch_pipeline_overlap)) {
        throw std::invalid_argument("residual OPQ requires the serial residual PQ GPU backend");
    }
    if (codec_kind != paraselect::search::ResidualCodecKind::PQ &&
        (arg.n_dim != 128 || arg.lists.n_vectors > 2147483647LL)) {
        throw std::invalid_argument("sign residual codec requires SIFT128 with <= INT_MAX rows");
    }
    if (codec_kind != paraselect::search::ResidualCodecKind::PQ &&
        (arg.fine_backend != paraselect::search::IVFResPQFineBackend::ResidualPQGpu ||
         arg.adc_strategy != paraselect::search::PQAdcSearchStrategy::Threshold ||
         arg.exact_storage.kind != paraselect::search::IVFExactStorageKind::Host ||
         !arg.exact_storage.host.u8)) {
        throw std::invalid_argument("sign residual codec requires GPU threshold and host U8 rerank");
    }
    const int dim = arg.n_dim;
    const int nlist = arg.lists.nlist;
    const int nprobe = arg.n_probes;
    const int topk = arg.k;

    coarse.h_cluster_sizes = ctx->cluster_counts.data();
    coarse.h_cluster_centers = arg.h_cluster_centers;
    coarse.query_batch_size = normalize_query_batch_size(arg.query_batch_size, arg.n_query);

    const bool residual_pq_gpu_is_enabled = residual_pq_gpu_enabled(arg);
    if (arg.candidate_only && !residual_pq_gpu_is_enabled) {
        throw std::invalid_argument(
            "ivfrespq_collect_candidates: residual PQ GPU candidate path is required");
    }
    paraselect::search::ResidualPQGpuSearchSpec planned_rpq_spec = {};
    if (residual_pq_gpu_is_enabled) {
        planned_rpq_spec = make_residual_pq_gpu_spec(arg, topk);
        planned_rpq_spec.adc_mode = arg.residual_adc_mode;
        planned_rpq_spec.max_vector_chunks_per_probe =
            residual_pq_max_vector_chunks_per_probe(
                arg, planned_rpq_spec, ctx->cluster_counts.data());
        validate_residual_pq_gpu_query_spec(arg, planned_rpq_spec);
        const int planned_candidate_width = residual_pq_candidate_width(planned_rpq_spec);
        if (planned_rpq_spec.search_strategy ==
            paraselect::search::PQAdcSearchStrategy::Threshold) {
            const int threshold_capacity = planned_rpq_spec.threshold.candidate_capacity;
            if (planned_rpq_spec.threshold.seed_probes <= 0 ||
                planned_rpq_spec.threshold.candidate_k <= 0 ||
                threshold_capacity < planned_rpq_spec.threshold.candidate_k ||
                threshold_capacity > arg.lists.n_vectors) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: threshold candidate capacity must satisfy "
                    "candidate_k <= capacity <= n_vectors");
            }
            if ((arg.exact_storage.kind == IVFExactStorageKind::File &&
                 !arg.pca_storage.enabled()) ||
                arg.file_block_selection.enabled()) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: threshold ADC file rerank requires PCA shortlist");
            }
            if (arg.cpu_batch_pipeline_overlap &&
                arg.exact_storage.kind != IVFExactStorageKind::Host) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: threshold overlap requires host exact storage");
            }
            if (!paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(
                    planned_rpq_spec.impl)) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: threshold ADC requires the grouped GPU implementation");
            }
            const bool list_centric = planned_rpq_spec.threshold.scan_layout ==
                paraselect::search::PQAdcThresholdScanLayout::ListCentric;
            if (planned_rpq_spec.threshold.scan_layout !=
                    paraselect::search::PQAdcThresholdScanLayout::PairCentric &&
                !list_centric) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: invalid threshold scan layout");
            }
            const bool fixed_tile_work =
                planned_rpq_spec.threshold.list_work_mode ==
                paraselect::search::PQAdcListScanWorkMode::FixedTile;
            if (planned_rpq_spec.threshold.list_work_mode !=
                    paraselect::search::PQAdcListScanWorkMode::Entry &&
                !fixed_tile_work) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: invalid list scan work mode");
            }
            if (fixed_tile_work && !list_centric) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: fixed-tile work requires list-centric scan");
            }
            const bool interleaved_rows =
                planned_rpq_spec.threshold.list_row_mode ==
                paraselect::search::PQAdcListScanRowMode::Interleaved;
            const bool query_row_tile4 =
                planned_rpq_spec.threshold.list_row_mode ==
                paraselect::search::PQAdcListScanRowMode::QueryRowTile4;
            if (planned_rpq_spec.threshold.list_row_mode !=
                    paraselect::search::PQAdcListScanRowMode::Serial &&
                !interleaved_rows && !query_row_tile4) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: invalid list scan row mode");
            }
            if (query_row_tile4 && (fixed_tile_work || planned_rpq_spec.pq_m != 16 ||
                                    planned_rpq_spec.adc_mode !=
                                        paraselect::search::ResidualPQAdcMode::QueryLut ||
                                    planned_rpq_spec.threshold.lut_read_mode !=
                                        paraselect::search::PQAdcLutReadMode::Shared)) {
                throw std::invalid_argument(
                    "ivfrespq_prepare: query_row_tile4 requires entry/M=16/pq_query_lut/shared");
            }
            // Scan layout owns only GPU workspace. The overlap scheduler serializes
            // GPU work and hands CPU rerank a separate pinned slot per batch.
            if (list_centric &&
                ((codec_kind == paraselect::search::ResidualCodecKind::PQ &&
                  planned_rpq_spec.adc_mode !=
                     paraselect::search::ResidualPQAdcMode::QueryLut) ||
                 (planned_rpq_spec.pq_m != 16 && planned_rpq_spec.pq_m != 20 &&
                  planned_rpq_spec.pq_m != 64))) {
                throw std::invalid_argument(
                "ivfrespq_prepare: list-centric threshold scan requires "
                    "query-LUT/QUERY_LUT_BIAS and pq_m=16, 20 or 64");
            }
        }
        if (arg.candidate_only && arg.candidate_capacity < planned_candidate_width) {
            throw std::invalid_argument(
                "ivfrespq_collect_candidates: output capacity is smaller than candidate count");
        }
        if (arg.file_block_selection.enabled() &&
            arg.file_block_selection.protected_candidate_rank > planned_rpq_spec.global_k) {
            throw std::invalid_argument(
                "ivfrespq_search: protected candidate rank exceeds candidate count");
        }
        const int rpq_candidate_sources =
            residual_pq_candidate_source_count(arg.n_probes, planned_rpq_spec);
        if (!paraselect::search::residual_pq_gpu_impl_uses_grouped_vector_chunks(
                planned_rpq_spec.impl)) {
            planned_rpq_spec.global_k = std::min(
                planned_rpq_spec.global_k,
                rpq_candidate_sources * planned_rpq_spec.local_k);
        }
    }


    coarse.query_tile = arg.query_tile > 0
        ? std::min(arg.query_tile, coarse.query_batch_size)
        : coarse.query_batch_size;
    coarse.coarse_workspace=paraselect::coarse::prepare_coarse(
        arg.fp16_coarse,arg.distance_mode,coarse.query_tile,dim,nlist,nprobe);
    std::fprintf(stderr,"[coarse] path=%s batch=%d k=%d workspace=%zu\n",
        paraselect::coarse::current::path_name(coarse.coarse_workspace->plan().path),
        coarse.query_tile,nprobe,coarse.coarse_workspace->workspace_bytes());
    arg.query_batch_size = coarse.query_batch_size;

    fine.assets.h_pq_codes = arg.pq.codes;
    fine.assets.h_pq_codebook = arg.pq.codebook;
    fine.assets.h_base_fp32 = nullptr;
    fine.assets.h_queries_fp32 = arg.h_queries_fp32;
    fine.assets.h_reordered_indices = arg.lists.reordered_ids;
    fine.assets.h_cluster_sizes = ctx->cluster_counts.data();
    fine.assets.h_cluster_centers = arg.h_cluster_centers;
    fine.device.d_resident_interleaved_vectors = arg.exact_storage.resident.fp32;
    fine.device.d_resident_interleaved_vectors_u8 = arg.exact_storage.resident.u8;
    fine.device.d_resident_interleaved_norms = arg.exact_storage.resident.norms;
    fine.device.d_resident_interleaved_reordered_indices = arg.exact_storage.resident.source_ids;
    fine.assets.file_storage = arg.exact_storage.file;
    if (arg.exact_storage.kind == IVFExactStorageKind::File) {
        fine.assets.file_reader = paraselect::search::make_ivf_file_reader(
            fine.assets.file_storage, arg.file_read);
    }
    fine.device.use_resident_rerank = ivfrespq_resident_rerank_requested(&arg) ? 1 : 0;
    if (fine.device.use_resident_rerank) {
        const char* combo_name = arg.resident_combo_name;
        if (!combo_name || !*combo_name) {
            throw std::invalid_argument("ivfrespq_prepare: resident combo is empty");
        }
        paraselect::pq_resident_fine::resolve_pq_resident_rerank_combo_ptr_or_throw(combo_name);
        std::strncpy(fine.device.resident_combo_name, combo_name, RESIDENT_FINE_COMBO_NAME_MAX_LEN - 1);
        fine.device.resident_combo_name[RESIDENT_FINE_COMBO_NAME_MAX_LEN - 1] = '\0';
    }
    fine.assets.pq_M = arg.pq_M;
    fine.assets.pq_K = arg.pq_K;
    fine.runtime.pq_rerank_n = (arg.pq_rerank_n > 0) ? arg.pq_rerank_n : std::max(topk, topk * 8);
    fine.runtime.pq_local_k = topk;
    fine.assets.rpq_gpu_combo = nullptr;
    fine.assets.residual_pq_gpu_spec = {};
    fine.runtime.pq_use_float_rerank = arg.pq_use_float_rerank;
    fine.runtime.use_residual_pq_gpu = residual_pq_gpu_is_enabled ? 1 : 0;
    if (fine.runtime.use_residual_pq_gpu) {
        fine.assets.residual_pq_gpu_spec = planned_rpq_spec;
        fine.runtime.candidate_width = residual_pq_candidate_width(planned_rpq_spec);
        fine.assets.rpq_gpu_combo =
            paraselect::search::resolve_residual_pq_gpu_combo_ptr_or_throw(
                arg.gpu.combo);
        fine.runtime.pq_local_k = fine.assets.residual_pq_gpu_spec.local_k;
    }
    if (arg.pca_storage.enabled() && arg.shortlist_count > fine.runtime.candidate_width) {
        throw std::invalid_argument(
            "ivfrespq_prepare: PCA shortlist exceeds candidate width");
    }
    fine.runtime.use_gpu_lut = (!fine.runtime.use_residual_pq_gpu && ivfrespq_gpu_lut_enabled(arg)) ? 1 : 0;
    fine.runtime.fine_runner = fine.runtime.use_residual_pq_gpu
        ? ivfrespq_fine_residual_pq_gpu
        : (fine.runtime.use_gpu_lut ? ivfrespq_fine_gpu_lut_cpu_scan : ivfrespq_fine_cpu_lut);
    fine.runtime.consume_probe_on_gpu = fine.runtime.use_residual_pq_gpu && arg.h_coarse_cluster_ids_out == nullptr;
    fine.runtime.reuse_query_on_gpu = fine.runtime.consume_probe_on_gpu && !arg.fp16_coarse;
    fine.runtime.cpu_config = arg.cpu;
    fine.runtime.cpu_fine_num_threads = arg.cpu.rerank_threads > 0
        ? arg.cpu.rerank_threads
        : paraselect::resolve_cpu_fine_thread_count();
    fine.runtime.phase_log_enabled = phase_log_enabled_prepare() ? 1 : 0;
    fine.runtime.rpq_kernel_breakdown = rpq_kernel_breakdown_enabled_prepare() ? 1 : 0;
    fine.runtime.rpq_rerank_breakdown = rpq_rerank_breakdown_enabled_prepare() ? 1 : 0;

    if (arg.exact_storage.kind == IVFExactStorageKind::Resident) {
        fine.assets.h_queries_u8 = arg.h_queries_u8;
    } else if (arg.exact_storage.kind == IVFExactStorageKind::File) {
        fine.assets.h_queries_u8 = arg.h_queries_u8;
        if (!fine.runtime.use_residual_pq_gpu) {
            throw std::invalid_argument(
                "ivfrespq_prepare: file exact storage requires the residual PQ GPU candidate path");
        }
    } else if (arg.exact_storage.kind == IVFExactStorageKind::Host) {
        fine.assets.h_base_fp32 = arg.exact_storage.host.fp32;
        if (arg.pq_use_float_rerank == 0) {
            fill_owned_query_u8(ctx);
            if (arg.exact_storage.host.u8) {
                fine.assets.h_base_u8 = arg.exact_storage.host.u8;
            } else {
                fill_owned_base_u8(ctx);
            }
        } else {
            fine.assets.h_queries_u8 = arg.h_queries_u8;
        }
    }
    build_cluster_offsets(ctx);
    if (codec_kind == paraselect::search::ResidualCodecKind::PQ) {
        fine.assets.pq_storage = prepare_pq_code_storage(arg, fine, ctx->cluster_counts.data());
    }

    if (!fine.runtime.consume_probe_on_gpu) {
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&fine.host.h_top_nprobe_batch),
            (size_t)coarse.query_batch_size * nprobe * sizeof(idx_t)));
    }
    fine.host.h_topk_local_idx_batch = static_cast<int*>(
        std::malloc((size_t)coarse.query_batch_size * topk * sizeof(int)));
    if ((!fine.runtime.consume_probe_on_gpu && !fine.host.h_top_nprobe_batch) || !fine.host.h_topk_local_idx_batch) {
        throw std::bad_alloc();
    }

    CUDA_CHECK(cudaStreamCreate(&coarse.stream));
    cublasStatus_t blas_status = cublasCreate(&coarse.cublas_handle);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("ivfrespq_prepare: cublasCreate failed");
    }
    cublasSetStream(coarse.cublas_handle, coarse.stream);
    CUDA_CHECK(cudaMalloc(&coarse.d_query_batch,
                          (size_t)coarse.query_batch_size * dim * sizeof(float)));
    const int query_workspace_n = fine.runtime.use_residual_pq_gpu
        ? coarse.query_batch_size
        : coarse.query_tile;
    CUDA_CHECK(cudaMalloc(&coarse.d_query_norm,
                          (size_t)query_workspace_n * sizeof(float)));
    if (arg.fp16_coarse) {
        CUDA_CHECK(cudaMalloc(&coarse.d_query_fp16,
                              (size_t)query_workspace_n * dim * sizeof(__half)));
    }
    CUDA_CHECK(cudaMalloc(&coarse.d_top_nprobe_dist,
                          (size_t)coarse.query_tile * nprobe * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coarse.d_top_nprobe_index,
                          (size_t)coarse.query_tile * nprobe * sizeof(idx_t)));
    ivfrespq_upload_coarse_centers(ctx);


    if (fine.runtime.use_gpu_lut || fine.runtime.use_residual_pq_gpu) {
        const int d_sub = dim / arg.pq_M;
        const int gpu_d_sub = fine.runtime.use_residual_pq_gpu
            ? fine.assets.residual_pq_gpu_spec.pq_d_sub_padded
            : d_sub;
        const size_t codebook_elems = (size_t)arg.pq_M * (size_t)arg.pq_K * (size_t)gpu_d_sub;
        if (fine.runtime.use_gpu_lut) {
            fine.host.pq_lut_batch_capacity =
                (size_t)coarse.query_batch_size * (size_t)nprobe *
                (size_t)arg.pq_M * (size_t)arg.pq_K;
        } else {
            fine.host.pq_lut_batch_capacity = 0;
        }
        if (codec_kind == paraselect::search::ResidualCodecKind::PQ) {
        if(arg.query_lut_codebook_fp16) {
            if(!fine.runtime.use_residual_pq_gpu || fine.runtime.use_gpu_lut ||
               arg.adc_strategy!=paraselect::search::PQAdcSearchStrategy::Threshold ||
               fine.assets.residual_pq_gpu_spec.adc_mode!=paraselect::search::ResidualPQAdcMode::QueryLut ||
               arg.adc_threshold.code_tile_consumer!=paraselect::search::PQAdcCodeTileConsumer::LutScalar)
                throw std::invalid_argument("FP16 codebook requires the query-LUT consumer");
            const auto padded=make_padded_pq_codebook(arg.pq.codebook,arg.pq_M,arg.pq_K,d_sub,gpu_d_sub);
            std::vector<__half> book(codebook_elems);
            for(size_t i=0;i<codebook_elems;++i){book[i]=__float2half_rn(padded[i]);
                if(!std::isfinite(__half2float(book[i])))throw std::invalid_argument("codebook exceeds FP16 range");}
            CUDA_CHECK(cudaMalloc(&fine.device.d_pq_codebook_fp16,codebook_elems*sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(fine.device.d_pq_codebook_fp16,book.data(),codebook_elems*sizeof(__half),cudaMemcpyHostToDevice));
        }else{
        CUDA_CHECK(cudaMalloc(&fine.device.d_pq_codebook, codebook_elems * sizeof(float)));
        if (fine.runtime.use_residual_pq_gpu && gpu_d_sub != d_sub) {
            const std::vector<float> padded_codebook = make_padded_pq_codebook(
                arg.pq.codebook, arg.pq_M, arg.pq_K, d_sub, gpu_d_sub);
            CUDA_CHECK(cudaMemcpy(
                fine.device.d_pq_codebook,
                padded_codebook.data(),
                codebook_elems * sizeof(float),
                cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                fine.device.d_pq_codebook,
                arg.pq.codebook,
                codebook_elems * sizeof(float),
                cudaMemcpyHostToDevice,
                coarse.stream));
        }
        }
        }
        if (fine.runtime.use_gpu_lut) {
            CUDA_CHECK(cudaMalloc(&fine.device.d_pq_lut_batch, fine.host.pq_lut_batch_capacity * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(
            &fine.device.d_top_nprobe_batch,
            (size_t)coarse.query_batch_size * (size_t)nprobe * sizeof(idx_t)));
        if (codec_kind == paraselect::search::ResidualCodecKind::PQ &&
            fine.assets.residual_pq_gpu_spec.adc_mode ==
            paraselect::search::ResidualPQAdcMode::QueryLut) {
            const size_t query_lut_entries =
                (size_t)coarse.query_batch_size * (size_t)arg.pq_M *
                paraselect::search::kPqQueryLutGlobalStride;
            if (arg.pq_distance_mode == paraselect::search::PQAdcDistanceMode::Residual) {
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_top_nprobe_dist_batch,
                (size_t)coarse.query_batch_size * (size_t)nprobe * sizeof(float)));
            }
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_query_codeword_dot_lut,
                query_lut_entries * sizeof(float)));
            if (arg.query_lut_quantization ==
                paraselect::search::PQAdcLutQuantizationMode::U8Affine) {
                CUDA_CHECK(cudaMalloc(
                    &fine.device.d_query_codeword_dot_lut_u8,
                    query_lut_entries * sizeof(uint8_t)));
                CUDA_CHECK(cudaMalloc(
                    &fine.device.d_query_lut_scale,
                    (size_t)coarse.query_batch_size * (size_t)arg.pq_M * sizeof(float)));
                CUDA_CHECK(cudaMalloc(
                    &fine.device.d_query_lut_lo_sum,
                    (size_t)coarse.query_batch_size * sizeof(float)));
                CUDA_CHECK(cudaMalloc(
                    &fine.device.d_query_lut_inflation,
                    (size_t)coarse.query_batch_size * sizeof(float)));
            }
            if (arg.query_lut_bias.u16) {
                CUDA_CHECK(cudaMalloc(
                    &fine.device.query_lut_bias_device.u16,
                    (size_t)arg.lists.n_vectors * sizeof(uint16_t)));
                upload_pq_rows(fine.device.query_lut_bias_device.u16, arg.query_lut_bias.u16, 1,
                    arg.lists.n_vectors, coarse.stream);
            } else if (arg.query_lut_bias.fp32) {
                CUDA_CHECK(cudaMalloc(
                    &fine.device.query_lut_bias_device.fp32,
                    (size_t)arg.lists.n_vectors * sizeof(float)));
                upload_pq_rows(fine.device.query_lut_bias_device.fp32, arg.query_lut_bias.fp32, 1,
                    arg.lists.n_vectors, coarse.stream);
            }
        }
    }
    if (fine.runtime.use_gpu_lut) {
        fine.host.h_pq_lut_batch = static_cast<float*>(
            std::aligned_alloc(64, fine.host.pq_lut_batch_capacity * sizeof(float)));
        if (!fine.host.h_pq_lut_batch) throw std::bad_alloc();
        CUDA_SYNC_CHECK(coarse.stream, "ivfrespq upload pq codebook");
    }
    if (fine.runtime.use_residual_pq_gpu) {
        CUDA_CHECK(cudaMalloc(
            &fine.device.d_query_padded,
            (size_t)coarse.query_batch_size *
                (size_t)fine.assets.residual_pq_gpu_spec.pq_dim_padded * sizeof(float)));
        if (arg.h_residual_rotation) {
            if (fine.assets.residual_pq_gpu_spec.pq_dim_padded != dim) {
                throw std::invalid_argument("residual OPQ currently requires unpadded PQ dimensions");
            }
            // Validate once before query execution. No rotation branch in ADC kernels.
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double dot = 0;
                    for (int d = 0; d < dim; ++d) {
                        dot += double(arg.h_residual_rotation[size_t(i) * dim + d]) *
                               arg.h_residual_rotation[size_t(j) * dim + d];
                    }
                    if (!std::isfinite(dot) || std::abs(dot - (i == j ? 1.0 : 0.0)) > 2e-4) {
                        throw std::invalid_argument("residual OPQ matrix is not orthogonal");
                    }
                }
            }
            ctx->residual_rotation_owner = std::make_unique<paraselect::transform::OPQTransform>(dim);
            auto& transform = *ctx->residual_rotation_owner;
            transform.set_matrix(std::vector<float>(arg.h_residual_rotation,
                arg.h_residual_rotation + size_t(dim) * dim));
            transform.prepare();
            fine.assets.residual_rotation = &transform;
            CUDA_CHECK(cudaMalloc(&fine.device.d_cluster_centers_padded, size_t(nlist) * dim * sizeof(float)));
            transform.apply_device(coarse.d_cluster_centers, fine.device.d_cluster_centers_padded,
                                   nlist, coarse.stream);
        } else if (fine.assets.residual_pq_gpu_spec.pq_dim_padded != dim) {
            const std::vector<float> padded_centers = make_padded_rowmajor_vectors(
                arg.h_cluster_centers,
                nlist,
                arg.pq_M,
                fine.assets.residual_pq_gpu_spec.pq_d_sub,
                fine.assets.residual_pq_gpu_spec.pq_d_sub_padded);
            CUDA_CHECK(cudaMalloc(
                &fine.device.d_cluster_centers_padded,
                (size_t)nlist *
                    (size_t)fine.assets.residual_pq_gpu_spec.pq_dim_padded * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(
                fine.device.d_cluster_centers_padded,
                padded_centers.data(),
                padded_centers.size() * sizeof(float),
                cudaMemcpyHostToDevice));
        }
        if (codec_kind == paraselect::search::ResidualCodecKind::PQ) {
        CUDA_CHECK(cudaMalloc(
            &fine.device.d_pq_codes,
            (size_t)arg.lists.n_vectors * (size_t)arg.pq_M * sizeof(uint8_t)));
        upload_pq_rows(fine.device.d_pq_codes, arg.pq.codes, arg.pq_M,
            arg.lists.n_vectors, coarse.stream);
        }
        CUDA_CHECK(cudaMalloc(
            &fine.device.d_cluster_offsets,
            (size_t)(arg.lists.nlist + 1) * sizeof(long long)));
        std::vector<long long> scan_offsets(fine.host.h_cluster_offsets,
            fine.host.h_cluster_offsets + arg.lists.nlist + 1);
        std::vector<int> scan_counts = ctx->cluster_counts;
        CUDA_CHECK(cudaMemcpy(fine.device.d_cluster_offsets, scan_offsets.data(),
            scan_offsets.size() * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&fine.device.d_cluster_counts, (size_t)arg.lists.nlist * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(fine.device.d_cluster_counts, scan_counts.data(),
            scan_counts.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (codec_kind != paraselect::search::ResidualCodecKind::PQ) {
            if (!fine.assets.h_base_u8 || fine.assets.residual_pq_gpu_spec.search_strategy !=
                    paraselect::search::PQAdcSearchStrategy::Threshold) {
                throw std::invalid_argument("alternative residual codec requires host U8 threshold search");
            }
            CUDA_CHECK(cudaStreamSynchronize(coarse.stream));
            ctx->residual_codec_owner = paraselect::search::make_residual_codec(codec_kind,
                {fine.assets.h_base_u8, arg.h_cluster_centers, fine.host.h_cluster_offsets,
                 static_cast<int>(arg.lists.n_vectors), nlist, dim, coarse.query_batch_size,
                 arg.turboquant.codes, arg.turboquant.scales});
            fine.device.residual_codec = ctx->residual_codec_owner.get();
        }
        const auto& rpq = fine.assets.residual_pq_gpu_spec;
        fine.runtime.adaptive_probe_padding = false;
        if (rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold &&
            rpq.threshold.scan_layout ==
                paraselect::search::PQAdcThresholdScanLayout::ListCentric) {
            const size_t max_probe_refs =
                (size_t)coarse.query_batch_size * (size_t)nprobe;
            if (max_probe_refs > static_cast<size_t>(INT_MAX)) {
                throw std::overflow_error(
                    "ivfrespq_prepare: list-centric probe schedule exceeds int32 capacity");
            }
            const auto schedule_strategy =
                paraselect::search::choose_ivf_probe_schedule_strategy(
                    paraselect::search::IVFProbeScheduleStrategy::Auto,
                    max_probe_refs,
                    nlist);
            fine.device.rpq_probe_schedule =
                std::make_shared<paraselect::search::IVFProbeScheduleBuilder>(
                    nlist, static_cast<int>(max_probe_refs), fine.device.d_cluster_counts,
                    schedule_strategy);
        }
        const size_t threshold_workspace_bytes =
            rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
            ? paraselect::search::pq_adc_threshold_workspace_required_bytes(
                coarse.query_batch_size, std::max(rpq.threshold.candidate_k, rpq.threshold.seed_rank),
                std::min(rpq.threshold.seed_probes, nprobe)) : 0;
        prepare_fine_scan_workspace(ctx, rpq, nprobe,
            threshold_workspace_bytes);
        ivfrespq_prepare_candidate_output(ctx);
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_lut_start));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_lut_stop));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_schedule_start));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_schedule_stop));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_tile_stop));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_gpu_start));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_gpu_stop));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_d2h_start));
        CUDA_CHECK(cudaEventCreate(&fine.runtime.ev_rpq_d2h_stop));
        CUDA_SYNC_CHECK(coarse.stream, "ivfrespq upload residual pq gpu data");
    }
    ivfrespq_prepare_cpu_pipeline(ctx);
}
