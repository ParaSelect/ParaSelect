#include "quantization/pca/pca_storage.h"
#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivfrespq/stage/ivfrespq_prepare_internal.cuh"
#include <climits>
#include <cmath>
#include <stdexcept>

void ivfrespq_validate_arg(const IVFResSearchState* arg) {
    if (!arg) throw std::invalid_argument("ivfrespq_search: arg is null");
    if (!arg->h_queries_fp32) throw std::invalid_argument("ivfrespq_search: h_queries_fp32 is null");
    validate_ivf_list_layout(arg->lists);
    if (arg->lists.n_vectors > INT_MAX) {
        throw std::invalid_argument("IVFResPQ currently requires n_vectors <= INT_MAX");
    }
    const IVFExactStorageView& exact = arg->exact_storage;
    const paraselect::quantization::PCAStorageView& pca = arg->pca_storage;
    const bool pca_requested = !pca.empty() || arg->shortlist_count != 0 ||
        arg->pca_backend != IVFRefineBackend::None;
    if (arg->host_rows) {
        if (!arg->lists.reordered_ids || exact.kind != IVFExactStorageKind::Host || exact.host.source_rows ||
            pca_requested ||
            arg->residual_codec != ResidualCodecKind::PQ ||
            (!arg->pq_use_float_rerank && !exact.host.u8) ||
            arg->fine_backend != IVFResPQFineBackend::ResidualPQGpu)
            throw std::invalid_argument("shared host rows require GPU candidates and direct host rerank");
        if (exact.host.count <= 0 || exact.host.count > INT_MAX)
            throw std::invalid_argument("shared host store exceeds supported row range");
        for (idx_t row = 0; row < arg->lists.n_vectors; ++row) {
            if (arg->host_rows[row] < 0 || arg->host_rows[row] >= exact.host.count ||
                arg->lists.reordered_ids[row] != arg->host_rows[row])
                throw std::invalid_argument("shared host rows must match valid original result IDs");
        }
    }
    if (exact.kind == IVFExactStorageKind::Host) {
        if ((!arg->host_rows && exact.host.count != arg->lists.n_vectors) || exact.host.dim != arg->n_dim) {
            throw std::invalid_argument("ivfrespq_search: host exact storage shape mismatch");
        }
        if (arg->pq_use_float_rerank != 0) {
            if (!exact.host.fp32) {
                throw std::invalid_argument("ivfrespq_search: fp32 rerank requires host fp32 storage");
            }
        } else if (!exact.host.u8 && !exact.host.fp32) {
            throw std::invalid_argument("ivfrespq_search: u8 rerank requires host u8 or fp32 storage");
        }
    } else if (exact.kind == IVFExactStorageKind::Resident) {
        if (exact.resident.count != arg->lists.n_vectors || exact.resident.dim != arg->n_dim) {
            throw std::invalid_argument("ivfrespq_search: resident exact storage shape mismatch");
        }
        if ((exact.resident.fp32 != nullptr) == (exact.resident.u8 != nullptr) ||
            !exact.resident.norms || !exact.resident.source_ids) {
            throw std::invalid_argument(
                "ivfrespq_search: resident rerank requires one vector dtype, norms, and reordered ids");
        }
    } else if (exact.kind == IVFExactStorageKind::File) {
        size_t exact_row_bytes = 0;
        switch (exact.file.dtype) {
            case IVFExactRerankDType::U8:
                if (arg->pq_use_float_rerank != 0 || !arg->h_queries_u8) {
                    throw std::invalid_argument(
                        "ivfrespq_search: u8 file storage requires u8 exact rerank and queries");
                }
                exact_row_bytes = static_cast<size_t>(arg->n_dim);
                break;
            case IVFExactRerankDType::FP32:
                if (arg->pq_use_float_rerank == 0 || !arg->h_queries_fp32) {
                    throw std::invalid_argument(
                        "ivfrespq_search: fp32 file storage requires fp32 exact rerank and queries");
                }
                exact_row_bytes =
                    static_cast<size_t>(arg->n_dim) * sizeof(float);
                break;
            case IVFExactRerankDType::Disabled:
                throw std::invalid_argument(
                    "ivfrespq_search: file exact storage dtype is disabled");
            default:
                throw std::invalid_argument(
                    "ivfrespq_search: unknown file exact storage dtype");
        }
        if (exact.file.fd < 0 || exact.file.count != arg->lists.n_vectors ||
            exact.file.dim != arg->n_dim || exact.file.block_size == 0 ||
            exact.file.row_stride < exact_row_bytes ||
            exact.file.payload_offset % exact.file.block_size != 0 ||
            exact.file.block_size % exact.file.row_stride != 0) {
            throw std::invalid_argument("ivfrespq_search: invalid file exact storage");
        }
        const IVFFileReadConfig& read = arg->file_read;
        if (read.read_block_bytes < exact.file.block_size ||
            read.read_block_bytes % exact.file.block_size != 0 ||
            read.read_block_bytes % exact.file.row_stride != 0 ||
            read.ring_count <= 0 || read.queue_depth_per_ring <= 0 ||
            (read.cache_capacity_bytes != 0 &&
             (read.cache_capacity_bytes < read.read_block_bytes ||
              read.cache_capacity_bytes % read.read_block_bytes != 0))) {
            throw std::invalid_argument("ivfrespq_search: invalid file read configuration");
        }
        if (pca_requested && read.cache_capacity_bytes != 0) {
            throw std::invalid_argument(
                "ivfrespq_search: PCA file backend requires file cache disabled");
        }
        if (!pca_requested && read.cache_capacity_bytes == 0) {
            throw std::invalid_argument(
                "ivfrespq_search: ResPQ file backend requires a positive file cache");
        }
    } else if (!arg->candidate_only) {
        throw std::invalid_argument("ivfrespq_search: exact storage backend is required");
    }
    if (pca_requested) {
        if (!pca.enabled() || !pca.projection || !pca.bias || !pca.scales ||
            pca.count != arg->lists.n_vectors || pca.source_dim != arg->n_dim ||
            pca.dim <= 0 || pca.dim > pca.source_dim) {
            throw std::invalid_argument("ivfrespq_search: invalid PCA storage");
        }
        if (arg->shortlist_count < arg->k) {
            throw std::invalid_argument(
                "ivfrespq_search: PCA shortlist count must cover topk");
        }
        if (arg->pca_backend != IVFRefineBackend::Cpu &&
            arg->pca_backend != IVFRefineBackend::Gpu) {
            throw std::invalid_argument(
                "ivfrespq_search: PCA refinement backend must be CPU or GPU");
        }
        if (arg->file_block_selection.enabled()) {
            throw std::invalid_argument(
                "ivfrespq_search: PCA shortlist does not support file block filtering");
        }
        if (arg->cpu_batch_pipeline_overlap) {
            throw std::invalid_argument(
                "ivfrespq_search: PCA shortlist does not support overlap");
        }
    }
    const paraselect::search::IVFFileBlockSelection& block_selection =
        arg->file_block_selection;
    if (block_selection.sparse_ref_threshold < 0 ||
        block_selection.protected_candidate_rank < 0) {
        throw std::invalid_argument(
            "ivfrespq_search: file block selection parameters must be non-negative");
    }
    const bool partial_block_selection =
        (block_selection.sparse_ref_threshold > 0) !=
        (block_selection.protected_candidate_rank > 0);
    if (partial_block_selection) {
        throw std::invalid_argument(
            "ivfrespq_search: file block selection requires both parameters");
    }
    if (block_selection.enabled() && exact.kind != IVFExactStorageKind::File) {
        throw std::invalid_argument(
            "ivfrespq_search: file block selection requires file exact storage");
    }
    if (!arg->h_cluster_centers) throw std::invalid_argument("ivfrespq_search: h_cluster_centers is null");
    if (!arg->lists.reordered_ids) throw std::invalid_argument("ivfrespq_search: h_reordered_indices is null");
    if (arg->residual_codec == paraselect::search::ResidualCodecKind::PQ && !arg->pq.codes) throw std::invalid_argument("ivfrespq_search: h_pq_codes is null");
    if (arg->residual_codec == paraselect::search::ResidualCodecKind::PQ && !arg->pq.codebook) throw std::invalid_argument("ivfrespq_search: h_pq_codebook is null");
    if (!arg->candidate_only && (!arg->h_topk_index_out || !arg->h_topk_dist_out)) {
        throw std::invalid_argument("ivfrespq_search: output buffers are null");
    }
    if (arg->candidate_only &&
        (!arg->h_candidate_local_ids_out || !arg->h_candidate_adc_distances_out ||
         arg->candidate_capacity <= 0)) {
        throw std::invalid_argument("ivfrespq_collect_candidates: invalid output view");
    }
    if (arg->n_query <= 0 || arg->n_dim <= 0 || arg->lists.nlist <= 0 ||
        arg->lists.n_vectors <= 0 || arg->n_probes <= 0 || arg->k <= 0) {
        throw std::invalid_argument("ivfrespq_search: invalid shape");
    }
    if (arg->n_probes > arg->lists.nlist) {
        throw std::invalid_argument("ivfrespq_search: n_probes exceeds n_total_cluster");
    }
    if (arg->k > arg->lists.n_vectors) {
        throw std::invalid_argument("ivfrespq_search: k exceeds n_total_vectors");
    }
    if (arg->pq_M <= 0 || arg->pq_K <= 0 || arg->pq_K > 256) {
        throw std::invalid_argument("ivfrespq_search: invalid PQ shape");
    }
    if ((arg->n_dim % arg->pq_M) != 0) {
        throw std::invalid_argument("ivfrespq_search: n_dim must be divisible by pq_M");
    }
    if (arg->distance_mode != (int)L2_DISTANCE) {
        throw std::invalid_argument("ivfrespq_search: residual PQ path currently supports L2 only");
    }
    const bool query_lut_mode =
        arg->residual_adc_mode ==
        paraselect::search::ResidualPQAdcMode::QueryLut;
    const bool plain_pq = arg->pq_distance_mode == PQAdcDistanceMode::Plain;
    if (arg->pq_distance_mode != PQAdcDistanceMode::Plain &&
        arg->pq_distance_mode != PQAdcDistanceMode::Residual)
        throw std::invalid_argument("invalid PQ quantizer mode");
    if (plain_pq && (arg->residual_codec != ResidualCodecKind::PQ ||
        arg->fine_backend != IVFResPQFineBackend::ResidualPQGpu || !query_lut_mode ||
        arg->adc_strategy != PQAdcSearchStrategy::Threshold || arg->h_residual_rotation ||
        arg->query_lut_bias.fp32 || arg->query_lut_bias.u16))
        throw std::invalid_argument("plain PQ requires GPU query-LUT threshold and plain assets without residual correction/rotation");
    if (arg->residual_codec == paraselect::search::ResidualCodecKind::PQ &&
        query_lut_mode && !plain_pq) {
        const bool has_query_lut_bias_fp32 = arg->query_lut_bias.fp32 != nullptr;
        const bool has_query_lut_bias_u16 = arg->query_lut_bias.u16 != nullptr;
        if (has_query_lut_bias_fp32 == has_query_lut_bias_u16 ||
            arg->query_lut_bias.count != arg->lists.n_vectors) {
            throw std::invalid_argument(
                "ivfrespq_search: pq_query_lut requires one row correction per reordered PQ row");
        }
        if (has_query_lut_bias_u16 && !(arg->query_lut_bias.scale > 0.0f)) {
            throw std::invalid_argument(
                "ivfrespq_search: QUERY_LUT_BIAS_U16 row correction requires a positive affine scale");
        }
        // FP16 controls GEMM inputs; coarse top-k distances remain FP32
        // and can be consumed directly by query-LUT/QUERY_LUT_BIAS ADC.
        if (arg->adc_strategy != paraselect::search::PQAdcSearchStrategy::Threshold) {
            throw std::invalid_argument(
                "ivfrespq_search: pq_query_lut currently supports threshold ADC only");
        }
    }
    // A row-correction asset may be present for cache reuse even when the
    // selected ADC mode is probe-residual. It is simply not consumed there.
}
