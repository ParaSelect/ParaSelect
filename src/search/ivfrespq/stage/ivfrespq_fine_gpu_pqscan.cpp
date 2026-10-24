#include "search/ivfrespq/ivfrespq_search_ctx.h"
#include "search/ivf_common/ivf_rerank_policy.h"
#include "search/ivfrespq/stage/ivfrespq_stages.cuh"
#include "search/ivfrespq/stage/ivfrespq_cpu_rerank.h"
#include "search/ivfrespq/stage/ivfrespq_candidate_order.cuh"
#include "search/ivfrespq/stage/ivfrespq_gpu_lut.cuh"
#include "search/ivf_common/ivf_pca_refine.h"
#include "search/ivf_common/ivf_host_rerank.h"
#include "search/ivf_common/file/ivf_file_rerank.h"
#include "search/ivfrespq/stage/residual_pq_gpu.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_adc_gpu.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include "search/kernels/fine/gpu_fine/resident/pq_resident_rerank.cuh"
#include "search/kernels/fine/gpu_fine/resident/pq_resident_rerank_combos.cuh"
#include "cpu/distance/distance_dispatch.h"
#include "cpu/utils/adaptive_thread_count.h"
#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/persistent_thread_pool.h"
#include "utils/check_cuda_error.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <thread>
#include <vector>
#include <omp.h>

namespace {

static void record_rpq_kernel_breakdown(
    paraselect::search::IVFSearchStatsCollector* collector,
    const ResidualPQGpuDebugTiming& timing,
    bool enabled) {
    if (!collector || !enabled) return;
    collector->addField("rpq_timing_clock", "cuda_event");
    collector->addStageTime("rpq_query_prepare", timing.query_prepare_ms);
    collector->addStageTime("rpq_seed_count", timing.seed_count_ms);
    collector->addStageTime("rpq_candidate_readback", timing.candidate_readback_ms);
    collector->addStageTime("rpq_query_lut", timing.query_lut_build_ms);
    collector->addStageTime("rpq_candidate_desc", timing.candidate_desc_ms);
    collector->addStageTime("rpq_pq_scan", timing.pq_scan_ms);
    collector->addStageTime("rpq_candidate_reduce", timing.candidate_reduce_ms);
    collector->addStageTime("rpq_candidate_cap_select", timing.candidate_cap_select_ms);
    collector->addStageTime("rpq_final_copy", timing.final_copy_ms);
    collector->addStageTime("rpq_seed_compact", timing.seed_compact_ms);
    collector->addStageTime("rpq_seed_adc", timing.seed_adc_ms);
    collector->addStageTime("rpq_threshold_policy", timing.threshold_policy_ms);
    collector->addStageTime("rpq_scan_setup", timing.scan_setup_ms);
    collector->addStageTime("rpq_scan_kernel", timing.scan_kernel_ms);
}

static void accumulate_file_rerank_stats(
    IVFRESPQSearchCtx* ctx,
    const paraselect::search::IVFRerankStats& stats) {
    if (!ctx || !ctx->collector) return;
    if (ctx->file_block_candidate_ref_histogram.size() <
        stats.block_candidate_ref_histogram.size()) {
        ctx->file_block_candidate_ref_histogram.resize(
            stats.block_candidate_ref_histogram.size(), 0);
    }
    for (size_t refs = 1; refs < stats.block_candidate_ref_histogram.size(); ++refs) {
        ctx->file_block_candidate_ref_histogram[refs] +=
            stats.block_candidate_ref_histogram[refs];
    }
    ctx->file_blocks_considered_total += stats.blocks_considered;
    ctx->file_blocks_skipped_total += stats.blocks_skipped;
}

#include "search/ivfrespq/stage/ivfrespq_debug.cuh"

static void debug_print_candidate_distance_compare(
    const char* tag,
    int batch_start,
    int batch_n,
    int global_k,
    int dim,
    const int* candidate_idx,
    const float* pq_dist,
    const uint8_t* base_u8,
    const uint8_t* query_u8,
    const float* base_fp32,
    const float* query_fp32,
    bool use_float_rerank) {
    if (!candidate_idx || !pq_dist) return;
    const auto fp32_l2 = paraselect::cpu_distance::select_fp32_l2_fn();
    const auto u8_l2 = paraselect::cpu_distance::select_u8_l2_fn();
    std::cerr << "[IVFRESPQ_DEBUG_SPACEV] " << tag
              << " batch_start=" << batch_start
              << " batch_n=" << batch_n
              << " cols=" << global_k
              << " use_float_rerank=" << (use_float_rerank ? 1 : 0)
              << "\n";
    const int rows = std::min(batch_n, 2);
    const int cols = std::min(global_k, 16);
    for (int q = 0; q < rows; ++q) {
        std::cerr << "[IVFRESPQ_DEBUG_SPACEV] " << tag << " q=" << q << " entries=";
        for (int k = 0; k < cols; ++k) {
            const int idx = candidate_idx[(size_t)q * global_k + k];
            const float pd = pq_dist[(size_t)q * global_k + k];
            float rd = std::numeric_limits<float>::max();
            if (idx >= 0) {
                const idx_t storage_idx = static_cast<idx_t>(idx);
                if (use_float_rerank && base_fp32 && query_fp32) {
                    rd = fp32_l2(
                        query_fp32 + (size_t)q * dim,
                        base_fp32 + (size_t)storage_idx * dim, dim);
                } else if (base_u8 && query_u8) {
                    rd = (float)u8_l2(
                        query_u8 + (size_t)q * dim,
                        base_u8 + (size_t)storage_idx * dim, dim);
                }
            }
            if (k) std::cerr << ";";
            std::cerr << idx << ":pq=" << pd << ":rerank=" << rd;
        }
        std::cerr << "\n";
    }
}

} // namespace

static const uint8_t* batch_query_u8_or_null(
    const IVFRESPQFineCtx& fine,
    int batch_start,
    int dim) {
    return fine.assets.h_queries_u8 ? fine.assets.h_queries_u8 + (size_t)batch_start * dim : nullptr;
}

static void fill_stage2_outputs(
    IVFRESPQFineCtx& fine,
    int batch_local_offset,
    int batch_n,
    int topk) {
    int* begin = fine.host.h_topk_local_idx_batch + (size_t)batch_local_offset * topk;
    std::fill(begin, begin + (size_t)batch_n * topk, -1);
}

static int resolve_special_m_fine_query_tile(int batch_n, int configured_tile) {
    const int tile = configured_tile;
    if (tile <= 0 || tile > batch_n) return batch_n;
    return std::max(1, tile);
}

static void ivfrespq_fine_residual_pq_gpu_tile(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n,
    int batch_local_offset,
    bool run_cpu_rerank) {
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    const int topk = arg.k;
    const int nprobe = arg.n_probes;
    const float* batch_query_fp32 = arg.h_queries_fp32 + (size_t)batch_start * dim;
    const uint8_t* batch_query_u8 = batch_query_u8_or_null(fine, batch_start, dim);
    float* batch_dist_out = arg.h_topk_dist_out
                                ? arg.h_topk_dist_out + (size_t)batch_start * topk
                                : nullptr;
    const int num_threads = fine.runtime.cpu_fine_num_threads;
    const idx_t* h_top_nprobe_tile = fine.host.h_top_nprobe_batch
                                         ? fine.host.h_top_nprobe_batch + (size_t)batch_local_offset * nprobe
                                         : nullptr;
    float* d_query_tile = fine.runtime.d_query_batch_valid
                              ? ctx->coarse.d_query_batch + (size_t)batch_local_offset * dim
                              : ctx->coarse.d_query_batch;
    idx_t* d_top_nprobe_tile = fine.runtime.d_top_nprobe_batch_valid
                                   ? fine.device.d_top_nprobe_batch + (size_t)batch_local_offset * nprobe
                                   : fine.device.d_top_nprobe_batch;
    int* h_topk_local_tile = fine.host.h_topk_local_idx_batch + (size_t)batch_local_offset * topk;

    if (!arg.candidate_only) {
        fill_stage2_outputs(fine, batch_local_offset, batch_n, topk);
    }
    paraselect::cpu_fine::PQResidCpuLUTTiming pq_timing;
    auto rpq = fine.assets.residual_pq_gpu_spec;
    const char* rpq_scan_mode_name =
        rpq.scan_mode == paraselect::search::ResidualPQGpuScanMode::DirectCodebook
            ? "direct"
            : "shared_lut";
    if (fine.runtime.phase_log_enabled || ivfrespq_debug_spacev_enabled()) {
        std::cerr << "[paraselect:ivfrespq_fine:phase]"
                  << " residual_pq_begin batch_n=" << batch_n
                  << " nprobe=" << nprobe
                  << " combo=" << (fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none")
                  << " local_k=" << fine.assets.residual_pq_gpu_spec.local_k
                  << " thread_k=" << fine.assets.residual_pq_gpu_spec.thread_k
                  << " global_k=" << fine.assets.residual_pq_gpu_spec.global_k
                  << " impl=" << (fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none")
                  << " scan_mode=" << rpq_scan_mode_name
                  << " pq_m=" << rpq.pq_m
                  << " dsub=" << rpq.pq_d_sub
                  << " dsub_padded=" << rpq.pq_d_sub_padded
                  << std::endl;
    }
    const int local_k = rpq.local_k;
    const int global_k = rpq.global_k;
    const int host_slot_row_stride = fine.runtime.candidate_width;
    int* h_rerank_candidate_tile = fine.host.h_rerank_candidate_idx
                                       ? fine.host.h_rerank_candidate_idx +
                                             (size_t)batch_local_offset * host_slot_row_stride
                                       : nullptr;
    const bool use_resident_rerank = fine.device.use_resident_rerank != 0;
    const auto gpu_t0 = std::chrono::high_resolution_clock::now();
    double rpq_d2h_ms = 0.0;
    double rpq_cpu_rerank_ms = 0.0;
    float rpq_gpu_search_ms = 0.0f;
    float rpq_d2h_event_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_gpu_start, ctx->coarse.stream));
    if (!fine.runtime.d_query_batch_valid) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_query_tile,
            batch_query_fp32,
            (size_t)batch_n * dim * sizeof(float),
            cudaMemcpyHostToDevice,
            ctx->coarse.stream));
    }
    if (!fine.runtime.d_top_nprobe_batch_valid) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_top_nprobe_tile,
            h_top_nprobe_tile,
            (size_t)batch_n * nprobe * sizeof(idx_t),
            cudaMemcpyHostToDevice,
            ctx->coarse.stream));
    }
    if (fine.assets.residual_rotation) {
        fine.assets.residual_rotation->apply_device(d_query_tile, fine.device.d_query_padded,
                                            batch_n, ctx->coarse.stream);
    } else {
        residual_pq_pad_queries_gpu(
            d_query_tile,
            batch_n,
            fine.assets.pq_M,
            rpq.pq_d_sub,
            rpq.pq_d_sub_padded,
            fine.device.d_query_padded,
            ctx->coarse.stream);
    }
    const float* d_rpq_queries = fine.device.d_query_padded;
    const float* d_rpq_centers = fine.device.d_cluster_centers_padded
                                     ? fine.device.d_cluster_centers_padded
                                     : ctx->coarse.d_cluster_centers;
    const int rpq_dim = rpq.pq_dim_padded;
    if (fine.device.residual_codec) {
        rpq.sign_codec = fine.device.residual_codec->prepare_queries(
            d_query_tile, batch_n, ctx->coarse.stream);
    }
    if (fine.runtime.phase_log_enabled) {
        std::cerr << "[paraselect:ivfrespq_fine:phase] residual_pq_kernel_call" << std::endl;
    }
    ResidualPQGpuDebugTiming debug_timing;
    const bool kernel_breakdown = fine.runtime.rpq_kernel_breakdown != 0;
    const bool rerank_breakdown = fine.runtime.rpq_rerank_breakdown != 0;
    if (fine.runtime.phase_log_enabled) {
        std::cerr << "[paraselect:ivfrespq_fine:phase] rpq_kernel_breakdown="
                  << (kernel_breakdown ? 1 : 0) << std::endl;
    }
    int* d_candidate_idx = fine.device.d_rerank_candidate_idx;
    float* d_candidate_dist = fine.device.d_rerank_candidate_dist;
    int* d_candidate_count = nullptr;
    paraselect::search::PQQueryLutView pq_query_lut;
    const paraselect::search::PQQueryLutView* pq_query_lut_ptr = nullptr;
    if (rpq.sign_codec.codes) {
        pq_query_lut.sign_codec = rpq.sign_codec;
        if (rpq.threshold.scan_layout == paraselect::search::PQAdcThresholdScanLayout::ListCentric)
            pq_query_lut_ptr = &pq_query_lut;
    } else if (rpq.adc_mode ==
        paraselect::search::ResidualPQAdcMode::QueryLut) {
        const bool plain_pq = arg.pq_distance_mode == paraselect::search::PQAdcDistanceMode::Plain;
        const auto lut_kind = plain_pq ? paraselect::search::PQQueryLutKind::PlainL2
                                       : paraselect::search::PQQueryLutKind::ResidualDot;
        if (!fine.device.d_query_codeword_dot_lut || (!plain_pq &&
            ((fine.device.query_lut_bias_device.fp32 == nullptr) ==
                (fine.device.query_lut_bias_device.u16 == nullptr) ||
             !fine.device.d_top_nprobe_dist_batch || !fine.runtime.d_top_nprobe_dist_batch_valid))) {
            throw std::logic_error("pq_query_lut runtime buffers are incomplete");
        }
        if (kernel_breakdown) CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_lut_start, ctx->coarse.stream));
        if(fine.device.d_pq_codebook_fp16)
            pq_build_query_lut_gpu(d_rpq_queries,fine.device.d_pq_codebook_fp16,batch_n,rpq.pq_m,
                rpq.pq_d_sub_padded,fine.device.d_query_codeword_dot_lut,ctx->coarse.stream,lut_kind);
        else
            pq_build_query_lut_gpu(d_rpq_queries,fine.device.d_pq_codebook,batch_n,rpq.pq_m,
                rpq.pq_d_sub_padded,fine.device.d_query_codeword_dot_lut,ctx->coarse.stream,lut_kind);
        if (ctx->arg.query_lut_quantization ==
            paraselect::search::PQAdcLutQuantizationMode::U8Affine) {
            pq_quantize_query_lut_u8_gpu(
                fine.device.d_query_codeword_dot_lut,
                batch_n,
                rpq.pq_m,
                fine.device.d_query_codeword_dot_lut_u8,
                fine.device.d_query_lut_scale,
                fine.device.d_query_lut_lo_sum,
                fine.device.d_query_lut_inflation,
                ctx->coarse.stream, lut_kind);
        }
        if (kernel_breakdown) CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_lut_stop, ctx->coarse.stream));
        pq_query_lut = {
            fine.device.d_query_codeword_dot_lut,
            plain_pq ? nullptr : fine.device.d_top_nprobe_dist_batch + (size_t)batch_local_offset * nprobe,
            {fine.device.query_lut_bias_device.fp32,
             fine.device.query_lut_bias_device.u16,
             ctx->arg.query_lut_bias.scale,
             ctx->arg.query_lut_bias.offset,
             ctx->arg.query_lut_bias.count},
            fine.device.d_query_codeword_dot_lut_u8,
            fine.device.d_query_lut_scale,
            fine.device.d_query_lut_lo_sum,
            fine.device.d_query_lut_inflation};
        pq_query_lut.kind = lut_kind;
        pq_query_lut_ptr = &pq_query_lut;
    }
    paraselect::search::IVFProbeScheduleResult probe_schedule_result;
    paraselect::search::IVFProbeTileScheduleResult probe_tile_schedule_result;
    const paraselect::search::IVFProbeScheduleView* probe_schedule = nullptr;
    const paraselect::search::IVFProbeTileScheduleView* probe_tile_schedule = nullptr;
    float probe_schedule_ms = 0.0f;
    float tile_schedule_ms = 0.0f;
    if (rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold &&
        rpq.threshold.scan_layout ==
            paraselect::search::PQAdcThresholdScanLayout::ListCentric) {
        if (!fine.device.rpq_probe_schedule) {
            throw std::logic_error("list-centric threshold probe schedule is not prepared");
        }
        if (kernel_breakdown) CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_schedule_start, ctx->coarse.stream));
        probe_schedule_result = fine.device.rpq_probe_schedule->build(
            {d_top_nprobe_tile, batch_n, nprobe}, nprobe, ctx->coarse.stream,
            fine.runtime.adaptive_probe_padding != 0);
        if (kernel_breakdown) CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_schedule_stop, ctx->coarse.stream));
        probe_schedule = &probe_schedule_result.view;
        if (ctx->collector) {
            const double entry_count =
                static_cast<double>(probe_schedule_result.view.entry_count);
            const double reference_count =
                static_cast<double>(probe_schedule_result.reference_count);
            ctx->collector->addField("rpq_probe_schedule_entries", entry_count);
            ctx->collector->addField("rpq_probe_schedule_strategy",
                paraselect::search::ivf_probe_schedule_strategy_name(probe_schedule_result.strategy));
            ctx->collector->addField(
                "rpq_probe_schedule_references", reference_count);
            ctx->collector->addField(
                "rpq_probe_schedule_avg_visitors",
                entry_count > 0.0 ? reference_count / entry_count : 0.0);
            ctx->collector->addField(
                "rpq_probe_schedule_scanned_vectors",
                static_cast<double>(probe_schedule_result.scanned_vectors));
        }
        if (rpq.threshold.list_work_mode ==
            paraselect::search::PQAdcListScanWorkMode::FixedTile) {
            probe_tile_schedule_result = fine.device.rpq_probe_schedule->build_tiles(
            probe_schedule_result.view, fine.runtime.fixed_tile_rows, ctx->coarse.stream);
            if (kernel_breakdown) CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_tile_stop, ctx->coarse.stream));
            probe_tile_schedule = &probe_tile_schedule_result.view;
        }
    }
    const uint8_t* scan_codes = fine.device.d_pq_codes;
    const long long* scan_offsets = fine.device.d_cluster_offsets;
    const int* scan_counts = fine.device.d_cluster_counts;
    if (rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold) {
        if (pq_query_lut_ptr && !rpq.sign_codec.codes) {
            pq_query_lut_threshold_search_gpu(
                *pq_query_lut_ptr, d_top_nprobe_tile, scan_codes,
                scan_offsets, scan_counts,
                batch_n, arg.lists.nlist, nprobe, rpq.pq_m, rpq.threshold,
                probe_schedule, probe_tile_schedule, ctx->coarse.stream,
                &fine.device.rpq_threshold_workspace,
                kernel_breakdown ? &debug_timing : nullptr,
                d_rpq_queries, fine.device.d_pq_codebook, rpq_dim);
        } else {
            pq_adc_threshold_search_gpu(
                arg.pq_distance_mode,
                d_rpq_queries,
                d_rpq_centers,
                d_top_nprobe_tile,
                fine.device.d_pq_codes,
                fine.device.d_cluster_offsets,
                fine.device.d_cluster_counts,
                fine.device.d_pq_codebook,
                batch_n,
                rpq_dim,
                arg.lists.nlist,
                nprobe,
                rpq,
                probe_schedule,
                probe_tile_schedule,
                ctx->coarse.stream,
                &fine.device.rpq_threshold_workspace,
                &fine.device.rpq_grouped_workspace,
                kernel_breakdown ? &debug_timing : nullptr);
        }
        d_candidate_idx = fine.device.rpq_threshold_workspace.d_candidate_idx;
        d_candidate_dist = fine.device.rpq_threshold_workspace.d_candidate_dist;
        d_candidate_count = fine.device.rpq_threshold_workspace.d_candidate_count;
    } else {
        residual_pq_search_gpu(
            d_rpq_queries,
            d_rpq_centers,
            d_top_nprobe_tile,
            fine.device.d_pq_codes,
            fine.device.d_cluster_offsets,
            fine.device.d_cluster_counts,
            fine.device.d_pq_codebook,
            batch_n,
            rpq_dim,
            arg.lists.nlist,
            nprobe,
            rpq,
            fine.device.d_probe_candidate_idx,
            fine.device.d_probe_candidate_dist,
            fine.device.d_rerank_candidate_idx,
            fine.device.d_rerank_candidate_dist,
            ctx->coarse.stream,
            kernel_breakdown ? &debug_timing : nullptr,
            &fine.device.rpq_grouped_workspace);
    }
    std::vector<int> threshold_candidate_counts;
    int device_row_stride = d_candidate_count
        ? fine.device.rpq_threshold_workspace.candidate_capacity : host_slot_row_stride;
    if (d_candidate_count) {
        threshold_candidate_counts.resize(batch_n);
        CUDA_CHECK(cudaMemcpyAsync(
            threshold_candidate_counts.data(),
            d_candidate_count,
            threshold_candidate_counts.size() * sizeof(int),
            cudaMemcpyDeviceToHost,
            ctx->coarse.stream));
        CUDA_SYNC_CHECK(ctx->coarse.stream, "residual pq threshold candidate counts");
        const int batch_max_valid = *std::max_element(
            threshold_candidate_counts.begin(), threshold_candidate_counts.end());
        if (std::any_of(threshold_candidate_counts.begin(), threshold_candidate_counts.end(),
                        [host_slot_row_stride](int count) {
                            return count < 0 || count > host_slot_row_stride;
                        })) {
            throw std::runtime_error(
                "residual PQ threshold returned an invalid candidate count");
        }
        ctx->threshold_candidate_total += std::accumulate(
            threshold_candidate_counts.begin(),
            threshold_candidate_counts.end(),
            0LL);
        ctx->threshold_query_total += batch_n;
        ctx->threshold_candidate_max =
            std::max(ctx->threshold_candidate_max, batch_max_valid);
    }
    const int n_rerank = paraselect::search::ivf_rerank_input_count(
        host_slot_row_stride, fine.runtime.pq_rerank_n, topk);
    int n_exact = arg.candidate_only ? host_slot_row_stride : n_rerank;
    // Serial GPU->host copies are compact (n_exact entries per row). The
    // overlap path preserves the fixed candidate slot stride so the CPU
    // consumer can run while another slot is being produced.
    const int host_candidate_stride = arg.cpu_batch_pipeline_overlap
        ? host_slot_row_stride : n_exact;
    // Reorder retained IDs for CPU locality; never select or drop candidates here.
    if ((rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Dense ||
         fine.runtime.order_threshold_ids) && !arg.file_block_selection.enabled()) {
        ivfrespq_order_candidates_by_local_idx_gpu(
            d_candidate_idx,
            d_candidate_dist,
            batch_n,
            n_exact,
            ctx->coarse.stream,
            device_row_stride);
    }
    paraselect::search::IVFPCARefineStats pca_debug;
    if (!arg.candidate_only && arg.pca_backend == IVFRefineBackend::Gpu) {
        if (!fine.assets.pca_gpu_refiner || !fine.device.d_pca_candidate_idx) {
            throw std::logic_error("GPU IVF PCA refiner was not prepared");
        }
        fine.assets.pca_gpu_refiner->run(
            d_query_tile,
            d_candidate_idx,
            batch_n,
            device_row_stride,
            n_exact,
            arg.shortlist_count,
            fine.device.d_pca_candidate_idx,
            &pca_debug,
            ctx->coarse.stream);
        d_candidate_idx = fine.device.d_pca_candidate_idx;
        d_candidate_dist = nullptr;
        d_candidate_count = nullptr;
        device_row_stride = arg.shortlist_count;
        n_exact = arg.shortlist_count;
    }
    double resident_rerank_ms = 0.0;
    const bool resident_on_device =
        use_resident_rerank && arg.pca_backend != IVFRefineBackend::Cpu;
    if (resident_on_device) {
        if (fine.runtime.phase_log_enabled) {
            std::cerr << "[paraselect:ivfrespq_fine:phase] resident_rerank_kernel_call" << std::endl;
        }
        const auto resident_t0 = std::chrono::high_resolution_clock::now();
        const auto* resident_combo =
            paraselect::pq_resident_fine::resolve_pq_resident_rerank_combo_ptr_or_throw(
                fine.device.resident_combo_name[0] ? fine.device.resident_combo_name : "full");
        if (resident_combo->kind != paraselect::pq_resident_fine::PQResidentRerankKernelKind::Full) {
            throw std::runtime_error("unsupported resident IVFRESPQ combo");
        }
        launch_pq_resident_rerank_full(
            topk,
            dim,
            d_query_tile,
            d_candidate_idx,
            n_exact,
            device_row_stride,
            fine.device.d_resident_interleaved_vectors,
            fine.device.d_resident_interleaved_vectors_u8,
            fine.device.d_resident_compact_to_padded,
            arg.lists.n_vectors,
            batch_n,
            fine.device.d_resident_topk_dist,
            fine.device.d_resident_topk_local_idx,
            ctx->coarse.stream);
        CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_gpu_stop, ctx->coarse.stream));
        const auto d2h_t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_d2h_start, ctx->coarse.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            h_topk_local_tile,
            fine.device.d_resident_topk_local_idx,
            (size_t)batch_n * topk * sizeof(int),
            cudaMemcpyDeviceToHost,
            ctx->coarse.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            batch_dist_out,
            fine.device.d_resident_topk_dist,
            (size_t)batch_n * topk * sizeof(float),
            cudaMemcpyDeviceToHost,
            ctx->coarse.stream));
        CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_d2h_stop, ctx->coarse.stream));
        CUDA_SYNC_CHECK(ctx->coarse.stream, "resident ivfrespq rerank");
        const auto resident_t1 = std::chrono::high_resolution_clock::now();
        resident_rerank_ms = std::chrono::duration<double, std::milli>(resident_t1 - resident_t0).count();
        rpq_d2h_ms = std::chrono::duration<double, std::milli>(resident_t1 - d2h_t0).count();
        if (fine.runtime.phase_log_enabled) {
            std::cerr << "[paraselect:ivfrespq_fine:phase] resident_rerank_done" << std::endl;
        }
    } else {
        if (fine.runtime.phase_log_enabled) {
            std::cerr << "[paraselect:ivfrespq_fine:phase] residual_pq_kernel_done" << std::endl;
        }
        CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_gpu_stop, ctx->coarse.stream));
    }
    const auto d2h_t0 = std::chrono::high_resolution_clock::now();
    if (!resident_on_device) {
        CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_d2h_start, ctx->coarse.stream));
        const int host_row_stride = arg.cpu_batch_pipeline_overlap
            ? host_slot_row_stride : n_exact;
        if (d_candidate_count || n_exact != device_row_stride ||
            host_row_stride != n_exact) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                h_rerank_candidate_tile,
                (size_t)host_row_stride * sizeof(int),
                d_candidate_idx,
                (size_t)device_row_stride * sizeof(int),
                (size_t)n_exact * sizeof(int),
                batch_n,
                cudaMemcpyDeviceToHost,
                ctx->coarse.stream));
        } else {
            if (n_exact != device_row_stride) {
                throw std::logic_error(
                    "compact IVF candidate buffer has inconsistent stride and width");
            }
            CUDA_CHECK(cudaMemcpyAsync(
                h_rerank_candidate_tile,
                d_candidate_idx,
                (size_t)batch_n * n_exact * sizeof(int),
                cudaMemcpyDeviceToHost,
                ctx->coarse.stream));
        }
    }
    const bool debug_batch = ivfrespq_debug_this_batch(batch_start, batch_n);
    std::vector<float> candidate_adc_dist;
    if (!resident_on_device &&
        (arg.candidate_only || debug_batch || arg.return_adc_distances) &&
        d_candidate_dist) {
        candidate_adc_dist.resize((size_t)batch_n * n_exact);
        if (d_candidate_count) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                candidate_adc_dist.data(),
                (size_t)n_exact * sizeof(float),
                d_candidate_dist,
                (size_t)device_row_stride * sizeof(float),
                (size_t)n_exact * sizeof(float),
                batch_n,
                cudaMemcpyDeviceToHost,
                ctx->coarse.stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                candidate_adc_dist.data(),
                d_candidate_dist,
                candidate_adc_dist.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                ctx->coarse.stream));
        }
    }
    if (!resident_on_device) {
        CUDA_CHECK(cudaEventRecord(fine.runtime.ev_rpq_d2h_stop, ctx->coarse.stream));
        CUDA_SYNC_CHECK(ctx->coarse.stream, "residual pq gpu fine");
    }
    if (arg.candidate_only) {
        if (candidate_adc_dist.empty()) {
            throw std::logic_error(
                "ivfrespq_collect_candidates: ADC distance output was not prepared");
        }
        for (int q = 0; q < batch_n; ++q) {
            const size_t src = (size_t)q * n_exact;
            const size_t dst = (size_t)(batch_start + q) * arg.candidate_capacity;
            std::copy_n(
                h_rerank_candidate_tile + src,
                n_exact,
                arg.h_candidate_local_ids_out + dst);
            std::copy_n(
                candidate_adc_dist.data() + src,
                n_exact,
                arg.h_candidate_adc_distances_out + dst);
        }
    }
    std::vector<idx_t> debug_top_nprobe;
    std::vector<float> debug_top_nprobe_dist;
    std::vector<long long> debug_top_nprobe_offsets;
    if (debug_batch) {
        if (!candidate_adc_dist.empty()) {
            debug_print_candidate_distance_compare(
                "stage2_gpu_vs_rerank_dist",
                batch_start,
                batch_n,
                n_exact,
                dim,
                h_rerank_candidate_tile,
                candidate_adc_dist.data(),
                fine.assets.h_base_u8,
                batch_query_u8,
                fine.assets.h_base_fp32,
                batch_query_fp32,
                fine.runtime.pq_use_float_rerank != 0);
        }
        const idx_t* top_nprobe_for_debug = fine.runtime.h_top_nprobe_batch_valid
                                                ? h_top_nprobe_tile
                                                : nullptr;
        if (!top_nprobe_for_debug && fine.runtime.d_top_nprobe_batch_valid) {
            debug_top_nprobe.resize((size_t)batch_n * nprobe);
            CUDA_CHECK(cudaMemcpy(
                debug_top_nprobe.data(),
                d_top_nprobe_tile,
                debug_top_nprobe.size() * sizeof(idx_t),
                cudaMemcpyDeviceToHost));
            top_nprobe_for_debug = debug_top_nprobe.data();
        }
        const float* top_nprobe_dist_for_debug = nullptr;
        if (d_top_nprobe_tile &&
            (fine.runtime.d_top_nprobe_dist_batch_valid || ctx->coarse.d_top_nprobe_dist)) {
            debug_top_nprobe_dist.resize((size_t)batch_n * nprobe);
            const float* d_top_nprobe_dist_tile =
                fine.runtime.d_top_nprobe_dist_batch_valid
                    ? fine.device.d_top_nprobe_dist_batch + (size_t)batch_local_offset * nprobe
                    : ctx->coarse.d_top_nprobe_dist + (size_t)batch_local_offset * nprobe;
            CUDA_CHECK(cudaMemcpy(
                debug_top_nprobe_dist.data(),
                d_top_nprobe_dist_tile,
                debug_top_nprobe_dist.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
            top_nprobe_dist_for_debug = debug_top_nprobe_dist.data();
        }
        const long long* top_nprobe_offsets_for_debug = nullptr;
        if (top_nprobe_for_debug && fine.host.h_cluster_offsets) {
            debug_top_nprobe_offsets.resize((size_t)batch_n * nprobe);
            for (int q = 0; q < batch_n; ++q) {
                for (int p = 0; p < nprobe; ++p) {
                    const idx_t cid = top_nprobe_for_debug[(size_t)q * nprobe + p];
                    debug_top_nprobe_offsets[(size_t)q * nprobe + p] =
                        (cid >= 0 && cid < static_cast<idx_t>(arg.lists.nlist))
                            ? fine.host.h_cluster_offsets[cid]
                            : -1;
                }
            }
            top_nprobe_offsets_for_debug = debug_top_nprobe_offsets.data();
        }
        debug_print_top_nprobe(
            "stage2_top_nprobe", batch_start, batch_n, nprobe, top_nprobe_for_debug,
            top_nprobe_dist_for_debug, top_nprobe_offsets_for_debug);
        debug_print_candidates(
            "stage2_after_gpu", batch_start, batch_n, n_exact,
            h_rerank_candidate_tile);
        if (top_nprobe_for_debug) {
            const RerankCandidateLocalityStats dbg = compute_rerank_candidate_locality(
                h_rerank_candidate_tile,
                top_nprobe_for_debug,
                fine.host.h_cluster_offsets,
                fine.assets.h_cluster_sizes,
                batch_n,
                nprobe,
                n_exact);
            std::cerr << "[IVFRESPQ_DEBUG_SPACEV] stage2_candidate_locality"
                      << " batch_start=" << batch_start
                      << " invalid=" << dbg.invalid
                      << " duplicates=" << dbg.duplicates
                      << " in_probe_range=" << dbg.in_probe_range
                      << " avg_adjacent_gap="
                      << (dbg.adjacent_gap_count
                              ? (double)dbg.adjacent_gap_sum / (double)dbg.adjacent_gap_count
                              : 0.0)
                      << " max_adjacent_gap=" << dbg.max_adjacent_gap
                      << " avg_probe_span=" << dbg.avg_probe_span
                      << " max_probe_span=" << dbg.max_probe_span
                      << "\n";
        }
    }
    if (fine.runtime.phase_log_enabled) {
        std::cerr << "[paraselect:ivfrespq_fine:phase] residual_pq_d2h_done" << std::endl;
    }
    CUDA_CHECK(cudaEventElapsedTime(&rpq_gpu_search_ms, fine.runtime.ev_rpq_gpu_start, fine.runtime.ev_rpq_gpu_stop));
    CUDA_CHECK(cudaEventElapsedTime(&rpq_d2h_event_ms, fine.runtime.ev_rpq_d2h_start, fine.runtime.ev_rpq_d2h_stop));
    const auto d2h_t1 = std::chrono::high_resolution_clock::now();
    rpq_d2h_ms = std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();
    const double rpq_fine_gpu_wall_ms =
        std::chrono::duration<double, std::milli>(d2h_t1 - gpu_t0).count();
    if (kernel_breakdown) {
        if (pq_query_lut_ptr && !rpq.sign_codec.codes) {
            CUDA_CHECK(cudaEventElapsedTime(&debug_timing.query_prepare_ms,
                fine.runtime.ev_rpq_gpu_start, fine.runtime.ev_rpq_lut_start));
            CUDA_CHECK(cudaEventElapsedTime(&debug_timing.query_lut_build_ms,
                fine.runtime.ev_rpq_lut_start, fine.runtime.ev_rpq_lut_stop));
        }
        if (probe_schedule) {
            CUDA_CHECK(cudaEventElapsedTime(&probe_schedule_ms,
                fine.runtime.ev_rpq_schedule_start, fine.runtime.ev_rpq_schedule_stop));
        }
        if (probe_tile_schedule) {
            CUDA_CHECK(cudaEventElapsedTime(&tile_schedule_ms,
                fine.runtime.ev_rpq_schedule_stop, fine.runtime.ev_rpq_tile_stop));
        }
    }
    record_rpq_kernel_breakdown(ctx->collector, debug_timing, kernel_breakdown);
    if (ctx->collector) {
        ctx->collector->addStageTime("rpq_probe_schedule", probe_schedule_ms);
        ctx->collector->addStageTime("rpq_tile_schedule", tile_schedule_ms);
    }

    // GPU accounting belongs to candidate production, before either rerank scheduler.
    const unsigned long long scanned_codes =
        rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
            ? fine.device.rpq_threshold_workspace.scanned_codes
            : fine.device.rpq_grouped_workspace.scanned_codes;
    const unsigned long long physical_code_rows =
        rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
            ? fine.device.rpq_threshold_workspace.physical_code_rows
            : scanned_codes;
    ctx->pq_scanned_codes_total += static_cast<long long>(scanned_codes);
    ctx->pq_physical_code_rows_total += static_cast<long long>(physical_code_rows);
    ctx->pq_adc_lookups_total +=
        static_cast<long long>(scanned_codes) * static_cast<long long>(rpq.pq_m);
    if (ctx->collector) {
        ctx->collector->addField(
            "rpq_adc_strategy",
            rpq.search_strategy == paraselect::search::PQAdcSearchStrategy::Threshold
                ? "threshold"
                : "dense");
        ctx->collector->addField(
            "rpq_threshold_scan_layout",
            rpq.threshold.scan_layout ==
                    paraselect::search::PQAdcThresholdScanLayout::ListCentric
                ? "list"
                : "pair");
        ctx->collector->addField(
            "rpq_list_scan_work_mode",
            rpq.threshold.list_work_mode ==
                    paraselect::search::PQAdcListScanWorkMode::FixedTile
                ? "tile"
                : "entry");
        ctx->collector->addField(
            "rpq_list_scan_row_mode",
            rpq.threshold.list_row_mode == paraselect::search::PQAdcListScanRowMode::Interleaved
                ? "interleaved"
                : rpq.threshold.list_row_mode == paraselect::search::PQAdcListScanRowMode::QueryRowTile4
                    ? "query_row_tile4" : "serial");
        ctx->collector->addField("rpq_codebook_precision", fine.device.d_pq_codebook_fp16 ? "fp16" : "fp32");
        ctx->collector->addField("rpq_code_tile_consumer",
            rpq.threshold.code_tile_consumer == paraselect::search::PQAdcCodeTileConsumer::U8Hybrid ? "u8_hybrid" : rpq.threshold.code_tile_consumer == paraselect::search::PQAdcCodeTileConsumer::Fp16Hybrid ? "fp16_hybrid" : "lut");
        ctx->collector->addField(
            "rpq_lut_read_mode",
            rpq.threshold.lut_read_mode ==
                    paraselect::search::PQAdcLutReadMode::Global
                ? "global"
                : "shared");
        ctx->collector->addField(
            "rpq_shared_lut_layout",
            rpq.threshold.shared_lut_layout ==
                    paraselect::search::PQAdcSharedLutLayout::Compact
                ? "compact"
                : "padded");
    }

    if (resident_on_device) {
        pq_timing.adc_lut_scan_ms = rpq_fine_gpu_wall_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime("pq_workspace_alloc", 0.0);
            ctx->collector->addStageTime("pq_adc_lut_scan", kernel_breakdown ? (double)debug_timing.pq_scan_ms : (double)rpq_gpu_search_ms);
            ctx->collector->addStageTime("pq_lut_d2h", (double)rpq_d2h_event_ms);
            ctx->collector->addStageTime("pq_adc_lookup", kernel_breakdown ? (double)debug_timing.pq_scan_ms : (double)rpq_gpu_search_ms);
            ctx->collector->addStageTime("pq_rerank", resident_rerank_ms);
            ctx->collector->addStageTime("rpq_gpu_search", (double)rpq_gpu_search_ms);
            ctx->collector->addStageTime("rpq_fine_gpu_wall", pq_timing.adc_lut_scan_ms);
            ctx->collector->addStageTime("rpq_d2h_ids", (double)rpq_d2h_event_ms);
            ctx->collector->addStageTime("rpq_resident_rerank", resident_rerank_ms);
            ctx->collector->addStageTime(
                "rpq_pca_query_transform", pca_debug.query_transform_ms);
            ctx->collector->addStageTime("rpq_pca_shortlist", pca_debug.select_ms);
            ctx->collector->addField("rpq_local_k", (double)local_k);
            ctx->collector->addField("rpq_thread_k", (double)rpq.thread_k);
            ctx->collector->addField("rpq_global_k", (double)global_k);
            ctx->collector->addField("rpq_combo_name", fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none");
            ctx->collector->addField("rpq_impl", fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none");
            ctx->collector->addField("rpq_candidate_k", (double)host_slot_row_stride);
            ctx->collector->addField("rpq_exact_rerank_n", (double)n_exact);
            ctx->collector->addField(
                "rpq_pca_dim", static_cast<double>(arg.pca_storage.dim));
            ctx->collector->addField(
                "rpq_shortlist_k", static_cast<double>(arg.shortlist_count));
            ctx->collector->addField(
                "rpq_pca_avg_input_candidates", pca_debug.avg_input_candidates);
            ctx->collector->addField("rpq_code_layout_tile", 0.0);
            ctx->collector->addField("rpq_code_layout", "rowmajor");
            ctx->collector->addField("ivfrespq_resident_combo", fine.device.resident_combo_name);
        }
        return;
    }
    if (!run_cpu_rerank) {
        pq_timing.adc_lut_scan_ms = rpq_fine_gpu_wall_ms;
        if (ctx->collector) {
            ctx->collector->addStageTime("pq_workspace_alloc", 0.0);
            ctx->collector->addStageTime("pq_adc_lut_scan", kernel_breakdown ? (double)debug_timing.pq_scan_ms : (double)rpq_gpu_search_ms);
            ctx->collector->addStageTime("pq_lut_d2h", (double)rpq_d2h_event_ms);
            ctx->collector->addStageTime("pq_adc_lookup", (double)rpq_gpu_search_ms);
            ctx->collector->addStageTime("rpq_gpu_search", (double)rpq_gpu_search_ms);
            ctx->collector->addStageTime("rpq_fine_gpu_wall", pq_timing.adc_lut_scan_ms);
            ctx->collector->addStageTime("rpq_d2h_ids", (double)rpq_d2h_event_ms);
            ctx->collector->addField("rpq_local_k", (double)local_k);
            ctx->collector->addField("rpq_thread_k", (double)rpq.thread_k);
            ctx->collector->addField("rpq_global_k", (double)global_k);
            ctx->collector->addField("rpq_combo_name", fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none");
            ctx->collector->addField("rpq_impl", fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none");
            ctx->collector->addField("rpq_candidate_k", (double)host_slot_row_stride);
            ctx->collector->addField("rpq_exact_rerank_n", (double)n_exact);
            ctx->collector->addField("rpq_code_layout_tile", 0.0);
            ctx->collector->addField("rpq_code_layout", "rowmajor");
        }
        return;
    }
    RerankCandidateLocalityStats candidate_locality;
    if (rerank_breakdown && fine.runtime.h_top_nprobe_batch_valid) {
        candidate_locality = compute_rerank_candidate_locality(
            h_rerank_candidate_tile,
            h_top_nprobe_tile,
            fine.host.h_cluster_offsets,
            fine.assets.h_cluster_sizes,
            batch_n,
            nprobe,
            n_exact);
    }
    paraselect::search::IVFRerankStats rerank_debug;
    const int* rerank_candidates = h_rerank_candidate_tile;
    int rerank_candidate_stride = host_candidate_stride;
    const auto rerank_t0 = std::chrono::high_resolution_clock::now();
    // Host rerank only collects per-candidate clocks in diagnostic mode.  The
    // regular query path must pass nullptr so the distance/top-k hot loop has
    // no steady_clock calls; pipeline rerank already follows this contract.
    auto* rerank_stats = rerank_breakdown ? &rerank_debug : nullptr;
    if (arg.pca_backend == IVFRefineBackend::Cpu) {
        paraselect::search::ivf_pca_refine_cpu(
            arg.pca_storage,
            batch_query_fp32,
            h_rerank_candidate_tile,
            batch_n,
            n_exact,
            arg.shortlist_count,
            num_threads,
            fine.host.h_shortlist_candidate_idx,
            &pca_debug);
        rerank_candidates = fine.host.h_shortlist_candidate_idx;
        n_exact = arg.shortlist_count;
        rerank_candidate_stride = arg.shortlist_count;
    }
    if (arg.exact_storage.kind == IVFExactStorageKind::Host) {
        ivf_rerank_host(
            arg.exact_storage.host,
            batch_query_u8,
            batch_query_fp32,
            fine.runtime.pq_use_float_rerank != 0,
            rerank_candidates,
            batch_n,
            n_exact,
            rerank_candidate_stride,
            topk,
            h_topk_local_tile,
            batch_dist_out,
            num_threads,
            fine.runtime.cpu_config.pin_rerank_threads,
            fine.runtime.cpu_config.rerank_prefetch,
            rerank_stats, nullptr, arg.host_rows,
            arg.host_rows ? arg.lists.n_vectors : 0);
    } else if (arg.exact_storage.kind == IVFExactStorageKind::File) {
        paraselect::search::ivf_file_rerank(
            *fine.assets.file_reader,
            fine.assets.file_storage,
            batch_query_u8,
            batch_query_fp32,
            rerank_candidates,
            batch_n,
            n_exact,
            topk,
            h_topk_local_tile,
            batch_dist_out,
            arg.file_block_selection,
            &rerank_debug);
    } else if (arg.exact_storage.kind == IVFExactStorageKind::Resident) {
        if (!fine.device.d_pca_candidate_idx) {
            throw std::logic_error("resident exact refinement input was not prepared");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            fine.device.d_pca_candidate_idx,
            rerank_candidates,
            static_cast<size_t>(batch_n) * n_exact * sizeof(int),
            cudaMemcpyHostToDevice,
            ctx->coarse.stream));
        launch_pq_resident_rerank_full(
            topk,
            dim,
            d_query_tile,
            fine.device.d_pca_candidate_idx,
            n_exact,
            n_exact,
            fine.device.d_resident_interleaved_vectors,
            fine.device.d_resident_interleaved_vectors_u8,
            fine.device.d_resident_compact_to_padded,
            arg.lists.n_vectors,
            batch_n,
            fine.device.d_resident_topk_dist,
            fine.device.d_resident_topk_local_idx,
            ctx->coarse.stream);
        CUDA_CHECK(cudaMemcpyAsync(
            h_topk_local_tile,
            fine.device.d_resident_topk_local_idx,
            static_cast<size_t>(batch_n) * topk * sizeof(int),
            cudaMemcpyDeviceToHost,
            ctx->coarse.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            batch_dist_out,
            fine.device.d_resident_topk_dist,
            static_cast<size_t>(batch_n) * topk * sizeof(float),
            cudaMemcpyDeviceToHost,
            ctx->coarse.stream));
        CUDA_SYNC_CHECK(ctx->coarse.stream, "resident exact refinement after CPU PCA");
    } else {
        throw std::logic_error("IVFResPQ exact rerank backend was not prepared");
    }
    if (arg.return_adc_distances && !candidate_adc_dist.empty()) {
        for (int q = 0; q < batch_n; ++q) {
            const int* candidates =
                h_rerank_candidate_tile + (size_t)q * n_rerank;
            const float* adc =
                candidate_adc_dist.data() + (size_t)q * n_rerank;
            int* out_idx = h_topk_local_tile + (size_t)q * topk;
            float* out_dist = batch_dist_out + (size_t)q * topk;
            for (int j = 0; j < topk; ++j) {
                const int id = out_idx[j];
                float adc_dist = std::numeric_limits<float>::infinity();
                for (int c = 0; c < n_rerank; ++c) {
                    if (candidates[c] == id) {
                        adc_dist = adc[c];
                        break;
                    }
                }
                out_dist[j] = adc_dist;
            }
        }
    }
    if (ivfrespq_debug_this_batch(batch_start, batch_n)) {
        debug_print_candidates(
            "rerank_local_topk", batch_start, batch_n, topk, h_topk_local_tile);
    }
    if (fine.runtime.phase_log_enabled) {
        std::cerr << "[paraselect:ivfrespq_fine:phase] residual_pq_rerank_done" << std::endl;
    }
    const auto rerank_t1 = std::chrono::high_resolution_clock::now();
    rpq_cpu_rerank_ms =
        std::chrono::duration<double, std::milli>(rerank_t1 - rerank_t0).count();
    const double rerank_dist_topk_ms = rerank_debug.distance_ms + rerank_debug.topk_ms;
    const double rerank_other_ms = std::max(
        0.0,
        rpq_cpu_rerank_ms - rerank_debug.plan_ms - rerank_debug.read_ms -
            rerank_dist_topk_ms - pca_debug.query_transform_ms - pca_debug.select_ms);
    if (arg.exact_storage.kind == IVFExactStorageKind::File) {
        accumulate_file_rerank_stats(ctx, rerank_debug);
    }
    pq_timing.adc_lut_scan_ms = rpq_fine_gpu_wall_ms;
    if (ctx->collector) {
        ctx->collector->addStageTime("pq_workspace_alloc", 0.0);
        ctx->collector->addStageTime("pq_adc_lut_scan", kernel_breakdown ? (double)debug_timing.pq_scan_ms : (double)rpq_gpu_search_ms);
        ctx->collector->addStageTime("pq_residual", 0.0);
        ctx->collector->addStageTime(
            "pq_lut_build",
            kernel_breakdown ? (double)debug_timing.candidate_desc_ms : 0.0);
        ctx->collector->addStageTime("pq_lut_d2h", (double)rpq_d2h_event_ms);
        ctx->collector->addStageTime("pq_adc_lookup", (double)rpq_gpu_search_ms);
        ctx->collector->addStageTime(
            "pq_heap_push",
            kernel_breakdown ? (double)(debug_timing.candidate_reduce_ms + debug_timing.final_copy_ms) : 0.0);
        ctx->collector->addStageTime("pq_rerank", rpq_cpu_rerank_ms);
        ctx->collector->addStageTime("rpq_gpu_search", (double)rpq_gpu_search_ms);
        ctx->collector->addStageTime("rpq_fine_gpu_wall", pq_timing.adc_lut_scan_ms);
        ctx->collector->addStageTime("rpq_probe_topk", kernel_breakdown ? (double)debug_timing.pq_scan_ms : 0.0);
        ctx->collector->addStageTime(
            "rpq_global_reduce",
            kernel_breakdown ? (double)debug_timing.global_reduce_ms : 0.0);
        ctx->collector->addStageTime("rpq_d2h_ids", (double)rpq_d2h_event_ms);
        ctx->collector->addStageTime("rpq_cpu_rerank", rpq_cpu_rerank_ms);
        ctx->collector->addStageTime(
            "rpq_pca_query_transform", pca_debug.query_transform_ms);
        ctx->collector->addStageTime(
            "rpq_pca_shortlist", pca_debug.select_ms);
        if (rerank_breakdown) {
            ctx->collector->addStageTime("rpq_cpu_rerank_sort", rerank_debug.plan_ms);
            ctx->collector->addStageTime("rpq_cpu_rerank_dist_heap", rerank_dist_topk_ms);
            ctx->collector->addStageTime("rpq_cpu_rerank_other", rerank_other_ms);
        }
        ctx->collector->addStageTime("rpq_file_plan", rerank_debug.plan_ms);
        ctx->collector->addStageTime("rpq_file_read", rerank_debug.read_ms);
        ctx->collector->addStageTime("rpq_file_distance", rerank_debug.distance_ms);
        ctx->collector->addStageTime("rpq_file_topk", rerank_debug.topk_ms);
        ctx->collector->addField("pq_heap_pushes", 0.0);
    }
    if (ctx->collector) {
        const int configured_rerank_threads =
            paraselect::search::ivf_rerank_thread_count(
                std::max(1, num_threads > 0 ? num_threads : omp_get_max_threads()),
                batch_n);
        ctx->collector->addField("rpq_local_k", (double)local_k);
        ctx->collector->addField("rpq_thread_k", (double)rpq.thread_k);
        ctx->collector->addField("rpq_global_k", (double)global_k);
        ctx->collector->addField(
            "rpq_combo_name",
            fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none");
        ctx->collector->addField(
            "rpq_impl",
            fine.assets.rpq_gpu_combo ? fine.assets.rpq_gpu_combo->name : "none");
        ctx->collector->addField("rpq_candidate_k", (double)host_slot_row_stride);
        ctx->collector->addField("rpq_exact_rerank_n", (double)n_exact);
        ctx->collector->addField(
            "rpq_shortlist_k", static_cast<double>(arg.shortlist_count));
        ctx->collector->addField(
            "rpq_pca_dim", static_cast<double>(arg.pca_storage.dim));
        ctx->collector->addField(
            "rpq_pca_avg_input_candidates", pca_debug.avg_input_candidates);
        ctx->collector->addField(
            "rpq_pca_threads", static_cast<double>(pca_debug.threads));
        ctx->collector->addField("rpq_desc_ms", kernel_breakdown ? (double)debug_timing.desc_ms : 0.0);
        ctx->collector->addField("rpq_group_merge_ms", kernel_breakdown ? (double)debug_timing.group_merge_ms : 0.0);
        ctx->collector->addField("rpq_final_merge_ms", kernel_breakdown ? (double)debug_timing.final_merge_ms : 0.0);
        ctx->collector->addField("rpq_code_layout_tile", 0.0);
        ctx->collector->addField("rpq_code_layout", "rowmajor");
        ctx->collector->addField(
            "rpq_cpu_rerank_avg_valid_candidates", rerank_debug.avg_valid_candidates);
        ctx->collector->addField(
            "rpq_cpu_rerank_avg_adjacent_gap", rerank_debug.avg_adjacent_gap);
        ctx->collector->addField(
            "rpq_cpu_rerank_max_adjacent_gap", rerank_debug.max_adjacent_gap);
        ctx->collector->addField(
            "rpq_cpu_rerank_threads",
            (double)(rerank_debug.threads > 0 ? rerank_debug.threads : configured_rerank_threads));
        ctx->collector->addField(
            "rpq_cpu_rerank_pinned",
            (double)(rerank_debug.threads > 0
                         ? rerank_debug.pinned
                         : (fine.runtime.cpu_config.pin_rerank_threads ? 1 : 0)));
        ctx->collector->addField(
            "rpq_cpu_rerank_persistent", (double)rerank_debug.persistent);
        if (rerank_breakdown) {
            ctx->collector->addField("rpq_cpu_rerank_sort_ms", rerank_debug.plan_ms);
            ctx->collector->addField("rpq_cpu_rerank_dist_heap_ms", rerank_dist_topk_ms);
            ctx->collector->addField("rpq_cpu_rerank_other_ms", rerank_other_ms);
            ctx->collector->addField("rpq_candidate_invalid", (double)candidate_locality.invalid);
            ctx->collector->addField("rpq_candidate_duplicates", (double)candidate_locality.duplicates);
            ctx->collector->addField("rpq_candidate_in_probe_range", (double)candidate_locality.in_probe_range);
            ctx->collector->addField(
                "rpq_candidate_avg_adjacent_gap",
                candidate_locality.adjacent_gap_count > 0
                    ? (double)candidate_locality.adjacent_gap_sum /
                          (double)candidate_locality.adjacent_gap_count
                    : 0.0);
            ctx->collector->addField("rpq_candidate_max_adjacent_gap", (double)candidate_locality.max_adjacent_gap);
            ctx->collector->addField("rpq_candidate_avg_probe_span", candidate_locality.avg_probe_span);
            ctx->collector->addField("rpq_candidate_max_probe_span", (double)candidate_locality.max_probe_span);
            std::cout << "[paraselect:ivfrespq_rerank_cpu:breakdown]"
                      << " batch_n=" << batch_n
                      << " dim=" << dim
                      << " global_k=" << global_k
                      << " topk=" << topk
                      << " threads=" << rerank_debug.threads
                      << " wall_ms=" << rpq_cpu_rerank_ms
                      << " sort_ms=" << rerank_debug.plan_ms
                      << " read_ms=" << rerank_debug.read_ms
                      << " dist_heap_ms=" << rerank_dist_topk_ms
                      << " other_ms=" << rerank_other_ms
                      << " pinned=" << rerank_debug.pinned
                      << " persistent=" << rerank_debug.persistent
                      << " avg_valid_candidates=" << rerank_debug.avg_valid_candidates
                      << " avg_adjacent_gap=" << rerank_debug.avg_adjacent_gap
                      << " max_adjacent_gap=" << rerank_debug.max_adjacent_gap
                      << " candidate_invalid=" << candidate_locality.invalid
                      << " candidate_duplicates=" << candidate_locality.duplicates
                      << " candidate_in_probe_range=" << candidate_locality.in_probe_range
                      << " candidate_avg_adjacent_gap="
                      << (candidate_locality.adjacent_gap_count > 0
                              ? (double)candidate_locality.adjacent_gap_sum /
                                    (double)candidate_locality.adjacent_gap_count
                              : 0.0)
                      << " candidate_max_adjacent_gap=" << candidate_locality.max_adjacent_gap
                      << " candidate_avg_probe_span=" << candidate_locality.avg_probe_span
                      << " candidate_max_probe_span=" << candidate_locality.max_probe_span
                      << std::endl;
        }
        ctx->collector->addField("rpq_file_unique_blocks", (double)rerank_debug.unique_blocks);
        ctx->collector->addField("rpq_file_bytes_read", (double)rerank_debug.bytes_read);
        ctx->collector->addField(
            "rpq_file_read_block_bytes", (double)rerank_debug.read_block_bytes);
        ctx->collector->addField("rpq_file_io_rings", (double)rerank_debug.io_rings);
        ctx->collector->addField(
            "rpq_file_io_depth_per_ring", (double)rerank_debug.io_depth_per_ring);
        ctx->collector->addField("rpq_file_distance_threads", (double)rerank_debug.threads);
        ctx->collector->addField(
            "rpq_file_distance_evaluations",
            (double)rerank_debug.distance_evaluations);
    }
}

void ivfrespq_fine_residual_pq_gpu(
    IVFRESPQSearchCtx* ctx,
    int batch_start,
    int batch_n) {
    if (!ctx || batch_n <= 0) return;
    const IVFRESPQFineCtx& fine = ctx->fine;
    const int pq_m = fine.assets.residual_pq_gpu_spec.pq_m;
    const int tile = pq_m == 20
        ? resolve_special_m_fine_query_tile(
              batch_n, fine.assets.residual_pq_gpu_spec.fine_query_tile)
        : batch_n;
    if (fine.runtime.phase_log_enabled && tile < batch_n) {
        std::cerr << "[paraselect:ivfrespq_fine:phase]"
                  << " residual_pq_m20_tiled"
                  << " batch_n=" << batch_n
                  << " pq_m=" << pq_m
                  << " fine_query_tile=" << tile
                  << std::endl;
    }
    for (int off = 0; off < batch_n; off += tile) {
        const int tile_n = std::min(tile, batch_n - off);
        ivfrespq_fine_residual_pq_gpu_tile(
            ctx, batch_start + off, tile_n, off, !ctx->arg.candidate_only);
    }
}

void ivfrespq_fine_residual_pq_gpu_candidates(
    IVFRESPQSearchCtx* ctx,
    const IVFRESPQBatchView& view) {
    if (!ctx || view.batch_n <= 0) return;
    const IVFRESPQFineCtx& fine = ctx->fine;
    const int pq_m = fine.assets.residual_pq_gpu_spec.pq_m;
    const int tile = pq_m == 20
        ? resolve_special_m_fine_query_tile(
              view.batch_n, fine.assets.residual_pq_gpu_spec.fine_query_tile)
        : view.batch_n;
    for (int off = 0; off < view.batch_n; off += tile) {
        const int tile_n = std::min(tile, view.batch_n - off);
        ivfrespq_fine_residual_pq_gpu_tile(
            ctx,
            view.batch_start + off,
            tile_n,
            off,
            false);
    }
}

void ivfrespq_rerank_residual_pq_candidates_cpu(
    IVFRESPQSearchCtx* ctx,
    const IVFRESPQBatchView& view) {
    if (!ctx || view.batch_n <= 0) return;
    const IVFResSearchState& arg = ctx->arg;
    IVFRESPQFineCtx& fine = ctx->fine;
    const int dim = arg.n_dim;
    const int topk = arg.k;
    const int nprobe = arg.n_probes;
    const int host_row_stride = fine.runtime.candidate_width;
    const int n_rerank = paraselect::search::ivf_rerank_input_count(
        host_row_stride, fine.runtime.pq_rerank_n, topk);
    const int candidate_stride = ctx->arg.cpu_batch_pipeline_overlap
        ? host_row_stride : n_rerank;
    const int num_threads = fine.runtime.cpu_fine_num_threads;
    const int pq_m = fine.assets.residual_pq_gpu_spec.pq_m;
    const int tile = pq_m == 20
        ? resolve_special_m_fine_query_tile(
              view.batch_n, fine.assets.residual_pq_gpu_spec.fine_query_tile)
        : view.batch_n;
    for (int off = 0; off < view.batch_n; off += tile) {
        const int tile_n = std::min(tile, view.batch_n - off);
        const uint8_t* batch_query_u8 = view.h_queries_u8
                                            ? view.h_queries_u8 + (size_t)off * dim
                                            : nullptr;
        const float* batch_query_fp32 = view.h_queries_fp32 + (size_t)off * dim;
        int* h_candidates =
            view.h_rerank_candidate_idx + (size_t)off * host_row_stride;
        int* h_topk_local = view.h_topk_local_idx_batch + (size_t)off * topk;
        float* h_dist_out = view.h_topk_dist_out + (size_t)off * topk;
        if (arg.exact_storage.kind == IVFExactStorageKind::Host) {
            ivf_rerank_host(
                arg.exact_storage.host, batch_query_u8, batch_query_fp32,
                fine.runtime.pq_use_float_rerank != 0, h_candidates, tile_n, n_rerank,
                candidate_stride, topk, h_topk_local, h_dist_out, num_threads,
                fine.runtime.cpu_config.pin_rerank_threads,
                fine.runtime.cpu_config.rerank_prefetch,
                nullptr, nullptr,
                arg.host_rows,
                arg.host_rows ? arg.lists.n_vectors : 0);
        } else if (arg.exact_storage.kind == IVFExactStorageKind::File) {
            paraselect::search::IVFRerankStats rerank_stats;
            paraselect::search::ivf_file_rerank(
                *fine.assets.file_reader,
                fine.assets.file_storage, batch_query_u8, batch_query_fp32,
                h_candidates, tile_n, n_rerank,
                topk, h_topk_local, h_dist_out,
                arg.file_block_selection, &rerank_stats);
            accumulate_file_rerank_stats(ctx, rerank_stats);
        } else {
            throw std::logic_error("IVFResPQ pipeline rerank backend was not prepared");
        }
    }
}

void ivfrespq_prewarm_residual_pq_rerank_cpu(
    IVFRESPQSearchCtx* ctx,
    int batch_n) {
    if (!ctx) return;
    const IVFRESPQFineCtx& fine = ctx->fine;
    const int max_threads = std::max(
        1,
        fine.runtime.cpu_fine_num_threads > 0 ? fine.runtime.cpu_fine_num_threads : omp_get_max_threads());
    if (ctx->arg.pca_backend == IVFRefineBackend::Cpu) {
        paraselect::search::ivf_pca_refine_cpu_prewarm(max_threads, batch_n);
    }
    if (ctx->arg.exact_storage.kind == IVFExactStorageKind::Host) {
        ivf_prewarm_host_rerank(
            max_threads, batch_n, fine.runtime.cpu_config.pin_rerank_threads);
    } else if (ctx->arg.exact_storage.kind == IVFExactStorageKind::File) {
        paraselect::search::ivf_file_rerank_prewarm(fine.assets.file_storage);
    }
}
