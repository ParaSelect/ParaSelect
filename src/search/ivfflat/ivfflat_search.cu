#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
#include "utils/pch.h"
#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/ivfflat/pipeline/ivfflat_pipeline.cuh"
#include "search/ivfflat/ivfflat_search_ctx.h"
#include "layout/ivf_interleaved_layout.cuh"
#include "quantization/linear/prepare_fp16_host.h"
#include "search/kernels/topk/fusion/fusion_dist_topk.cuh"
#include "search/kernels/fine/gpu_fine/gemv.cuh"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"
#include "search/kernels/topk/warpsort/warpsort_utils.cuh"
#include "dataset/dataset.cuh"
#include "utils/cudatimer.h"
#include "l2norm/l2norm.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <future>
#include <exception>

#if PARASELECT_ENABLE_CUVS
#include <raft/core/resources.hpp>
#endif

// CPU L2
#include "cpu/l2norm_precompute.h"

using namespace paraselect::warpsort_utils;
using namespace paraselect::search;

static int compute_coarse_sub_batch_size(
    int requested_batch_size,
    int n_query,
    int n_total_clusters,
    int n_probes,
    int dim)
{
    int query_batch_size = requested_batch_size;
    if (query_batch_size <= 0 || query_batch_size >= n_query) {
        query_batch_size = n_query;
    }

    constexpr long long kGiB = 1024LL * 1024 * 1024;
    constexpr long long kMaxGemmElems = 1900000000LL;
    constexpr int kAlign = 32;

    size_t gpu_free = 0, gpu_total = 0;
    if (cudaMemGetInfo(&gpu_free, &gpu_total) != cudaSuccess) {
        return query_batch_size;
    }

    const long long gemm_elems = (long long)query_batch_size * n_total_clusters;
    const long long headroom = std::max(4LL * kGiB, (long long)gpu_total / 8);
    long long usable = (long long)gpu_free - headroom;
    if (usable < (long long)gpu_free / 5) usable = (long long)gpu_free / 5;
    usable = usable * 9 / 10;

    const long long bytes_per_q = 2LL * n_total_clusters * sizeof(float)
                                + (long long)n_probes * (sizeof(float) + sizeof(int))
                                + (long long)dim * sizeof(float) + 4096;
    int max_sub_mem = (int)std::min((long long)query_batch_size, usable / std::max(bytes_per_q, 1LL));
    int max_sub_int = (int)(kMaxGemmElems / std::max(n_total_clusters, 1));
    int max_sub_raw = std::min(max_sub_mem, max_sub_int);
    if (max_sub_raw < 1) max_sub_raw = 1;

    if (gemm_elems > kMaxGemmElems || query_batch_size > max_sub_raw) {
        int max_sub = max_sub_raw;
        if (max_sub >= kAlign) max_sub = (max_sub / kAlign) * kAlign;
        if (max_sub < 1) max_sub = 1;
        std::fprintf(stderr,
                     "[WARNING] coarse OOM guard triggered: query_tile_size=%d "
                     "(query_batch_size=%d, n_query=%d, nlist=%d, nprobe=%d)\n",
                     max_sub, query_batch_size, n_query, n_total_clusters, n_probes);
        return max_sub;
    }
    return query_batch_size;
}

// Helper to delete cuvs_res (declared in ivfflat_search_ctx.h)
void ivfflat_search_delete_cuvs_res(raft::resources* res) {
#if PARASELECT_ENABLE_CUVS
    delete res;
#else
    if (res) std::terminate();
#endif
}

// -----------------------------------------------------------------------------
//  ivfflat_search: Cpu  Resident fine
// -----------------------------------------------------------------------------
static void ivfflat_search_impl(
    const float* h_queries,
    IVFListLayoutView lists,
    const float* h_cluster_vectors,
    const float* h_cluster_centers,
    int n_query,
    int n_dim,
    int n_probes,
    int k,
    int distance_mode,
    int query_batch_size,
    bool fp16_coarse,
    idx_t* h_topk_index_out,
    float* h_topk_dist_out,
    int* h_coarse_cluster_ids_out,
    const float* h_cluster_norm_in,
    const uint8_t* h_queries_u8,
    const uint8_t* h_cluster_vectors_u8,
    IVFSearchStats* stats,
    const char* cpu_fine_combo_name,
    const float* h_cluster_vectors_aosoa,
    const long long* h_cluster_offsets_aosoa,
    int h_cluster_aosoa_tile,
    bool cpu_batch_pipeline_overlap,
    const float* d_resident_interleaved_vectors,
    const uint8_t* d_resident_interleaved_vectors_u8,
    const float* d_resident_interleaved_norms,
    const idx_t* d_resident_interleaved_reordered_indices,
    IVFFlatFineKind fine_kind,
    const char* resident_combo_name,
    int resident_chunk_blocks,
    IVFFlatThresholdParams threshold
) {
    const std::vector<int> cluster_counts = derive_ivf_list_counts(lists);
    const int* h_cluster_sizes = cluster_counts.data();
    const idx_t* h_reordered_indices = lists.reordered_ids;
    const int n_total_cluster = lists.nlist;
    if (lists.n_vectors > INT_MAX) {
        throw std::invalid_argument("IVFFlat currently requires n_vectors <= INT_MAX");
    }
    const int n_total_vectors = static_cast<int>(lists.n_vectors);
    if (h_cluster_centers == nullptr) {
        fprintf(stderr, "[ERROR] ivfflat_search: clusters not found\n");
        throw std::invalid_argument("clusters must be provided");
    }
    if (h_queries == nullptr) {
        fprintf(stderr, "[ERROR] ivfflat_search: queries not found\n");
        throw std::invalid_argument("queries must be provided");
    }

    static bool printed_ivfflat_search_debug = false;
    if (!printed_ivfflat_search_debug && std::getenv("PARASELECT_DEBUG_IVFFLAT_SEARCH_INPUTS")) {
        fprintf(stderr,
                "[ivfflat_search:debug] combo=%s nq=%d dim=%d ncluster=%d ntotal=%d nprobe=%d batch=%d q_u8=%p base_u8=%p base_fp32=%p\n",
                cpu_fine_combo_name ? cpu_fine_combo_name : "",
                n_query,
                n_dim,
                n_total_cluster,
                n_total_vectors,
                n_probes,
                query_batch_size,
                (const void*)h_queries_u8,
                (const void*)h_cluster_vectors_u8,
                (const void*)h_cluster_vectors);
        printed_ivfflat_search_debug = true;
    }

    DistanceType dist_type = (distance_mode == 0) ? L2_DISTANCE : COSINE_DISTANCE;
    IVFSearchCtx ctx = {};
    ivf_search_ctx_zero(&ctx);

    const bool has_resident_fp32 = d_resident_interleaved_vectors != nullptr;
    const bool has_resident_u8 = d_resident_interleaved_vectors_u8 != nullptr;
    const bool has_any_resident_storage = has_resident_fp32 || has_resident_u8 ||
        d_resident_interleaved_norms != nullptr ||
        d_resident_interleaved_reordered_indices != nullptr;
    const bool resident = fine_kind == IVFFlatFineKind::Resident;
    if (resident && (has_resident_fp32 == has_resident_u8 || !d_resident_interleaved_norms || !d_resident_interleaved_reordered_indices)) {
        throw std::invalid_argument("resident IVFFlat requires exactly one vector storage, norms, and reordered ids");
    }
    if (!resident && has_any_resident_storage) {
        throw std::invalid_argument("CPU IVFFlat must not receive resident storage");
    }
    if (has_resident_u8 && h_queries_u8 == nullptr) {
        throw std::invalid_argument("u8 resident IVFFlat requires u8 queries");
    }
    const paraselect::cpu_fine::CpuFineKernelCombo* resolved_combo = nullptr;
    const char* resolved_combo_name = cpu_fine_combo_name;
    if (!resident) {
        resolved_combo = paraselect::cpu_fine::resolve_cpu_fine_combo_ptr_or_throw(cpu_fine_combo_name);
        resolved_combo_name = paraselect::cpu_fine::cpu_fine_combo_name_from_preset(
            resolved_combo->layout, resolved_combo->kernel, resolved_combo->schedule_policy);
        if (!resolved_combo_name) resolved_combo_name = cpu_fine_combo_name;
        if (resolved_combo->layout == CPU_FINE_LAYOUT_FP32_ROW_MAJOR && h_cluster_vectors == nullptr) {
            throw std::invalid_argument("fp32 row-major cluster vectors must be provided");
        }
        if (resolved_combo->layout == CPU_FINE_LAYOUT_U8_ROW_MAJOR && h_cluster_vectors_u8 == nullptr) {
            throw std::invalid_argument("u8 row-major cluster vectors must be provided");
        }
        if (resolved_combo->layout == CPU_FINE_LAYOUT_FP32_AOSOA &&
            (h_cluster_vectors_aosoa == nullptr || h_cluster_offsets_aosoa == nullptr)) {
            throw std::invalid_argument("AoSoA cluster vectors and offsets must be provided");
        }
    }

    // d_cluster_size  prepare
    ctx.flat_coarse.d_cluster_size = nullptr;
    try {
        CUDATimer timer_try_block("ivfflat_search without dataset transmission(alloc + pipeline + copy)", false);
        cudaStreamCreate(&ctx.flat_coarse.stream_0);
        cudaStreamCreate(&ctx.flat_coarse.stream_1);

        /*  */
        for (int i = 0; i < 2; ++i) {
            cudaEventCreate(&ctx.flat_coarse.ev_coarse_done[i]);
        }
        cudaEventCreate(&ctx.flat_coarse.ev_fine_done);

        if (query_batch_size <= 0 || query_batch_size >= n_query) {
            query_batch_size = n_query;
        }
        const int coarse_sub_batch_size = compute_coarse_sub_batch_size(
            query_batch_size, n_query, n_total_cluster, n_probes, n_dim);
        IVFSearchArg& arg_ref = ctx.arg;
        arg_ref.n_query = n_query;
        arg_ref.query_batch_size = query_batch_size;
        arg_ref.n_dim = n_dim;
        arg_ref.n_total_clusters = n_total_cluster;
        arg_ref.n_total_vectors = n_total_vectors;
        arg_ref.n_probes = n_probes;
        arg_ref.k = k;
        arg_ref.distance_mode = (int)dist_type;
        if (resident) {
            arg_ref.cpu_fine_combo_name[0] = '\0';
            if (!resident_combo_name || !resident_combo_name[0]) {
                throw std::invalid_argument("resident IVFFlat combo is empty");
            }
            std::strncpy(
                arg_ref.resident_combo_name,
                resident_combo_name,
                RESIDENT_FINE_COMBO_NAME_MAX_LEN - 1);
            arg_ref.resident_combo_name[RESIDENT_FINE_COMBO_NAME_MAX_LEN - 1] =
                '\0';
            arg_ref.resident_chunk_blocks = resident_chunk_blocks;
        } else {
            strncpy(arg_ref.cpu_fine_combo_name, resolved_combo_name, CPU_FINE_COMBO_NAME_MAX_LEN - 1);
            arg_ref.cpu_fine_combo_name[CPU_FINE_COMBO_NAME_MAX_LEN - 1] = '\0';
        }
        arg_ref.max_probe_slots = 0;
        arg_ref.fp16_coarse = fp16_coarse;
        arg_ref.fine_kind = fine_kind;
        arg_ref.threshold = threshold;
        arg_ref.cpu_batch_pipeline_overlap = cpu_batch_pipeline_overlap;

        ctx.flat_coarse.h_cluster_sizes = h_cluster_sizes;
        ctx.flat_coarse.h_cluster_centers = h_cluster_centers;
        ctx.flat_coarse.h_queries = h_queries;
        ctx.flat_coarse.h_queries_u8 = h_queries_u8;
        ctx.flat_coarse.h_topk_dist_out = h_topk_dist_out;
        ctx.flat_coarse.h_topk_index_out = h_topk_index_out;
        ctx.flat_coarse.coarse_sub_batch_size = coarse_sub_batch_size;

        ctx.flat_fine.h_cluster_sizes = h_cluster_sizes;
        ctx.flat_fine.h_queries = h_queries;
        ctx.flat_fine.h_queries_u8 = h_queries_u8;
        ctx.flat_fine.h_topk_dist_out = h_topk_dist_out;
        ctx.flat_fine.h_topk_index_out = h_topk_index_out;
        ctx.flat_fine.h_cluster_vectors = h_cluster_vectors;
        ctx.flat_fine.h_cluster_vectors_aosoa = h_cluster_vectors_aosoa;
        ctx.flat_fine.h_cluster_vectors_u8 = h_cluster_vectors_u8;
        ctx.flat_fine.h_reordered_indices = h_reordered_indices;
        ctx.flat_fine.d_resident_interleaved_vectors = d_resident_interleaved_vectors;
        ctx.flat_fine.d_resident_interleaved_vectors_u8 = d_resident_interleaved_vectors_u8;
        ctx.flat_fine.d_resident_interleaved_norms = d_resident_interleaved_norms;
        ctx.flat_fine.d_resident_interleaved_reordered_indices = d_resident_interleaved_reordered_indices;
        ctx.flat_fine.resident_interleaved_u8 = has_resident_u8 ? 1 : 0;
        ctx.flat_fine.h_cluster_centers = h_cluster_centers;
        ctx.flat_fine.h_cluster_offsets_aosoa_ll = h_cluster_offsets_aosoa;
        ctx.flat_fine.h_cluster_aosoa_tile = h_cluster_aosoa_tile;
        ctx.flat_fine.cached_combo = resolved_combo;

        // Row-major fp32 kernels use precomputed norms; AoSoA computes L2 directly from its layout.
        ctx.flat_fine.h_cluster_norm = nullptr;
        if (!resident && dist_type == L2_DISTANCE && resolved_combo->layout == CPU_FINE_LAYOUT_FP32_ROW_MAJOR) {
            if (h_cluster_norm_in == nullptr) {
                throw std::invalid_argument("no precompute l2norm");
            }
            ctx.flat_fine.h_cluster_norm = (float*)h_cluster_norm_in;
        }


        ivfflat_pipeline(&ctx, stats);

        //  CUDA
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        std::fprintf(
            stderr,
            "[ivfflat_search][exception] what=%s cpu_fine_combo_name=%s "
            "n_query=%d dim=%d nlist=%d nprobe=%d topk=%d batch=%d\n",
            e.what(), cpu_fine_combo_name,
            n_query, n_dim, n_total_cluster, n_probes, k, query_batch_size);
        ivf_search_ctx_free_all(&ctx);
        throw std::runtime_error(std::string("ivfflat_search failed: ") + e.what());
    } catch (...) {
        std::fprintf(
            stderr,
            "[ivfflat_search][exception] unknown cpu_fine_combo_name=%s "
            "n_query=%d dim=%d nlist=%d nprobe=%d topk=%d batch=%d\n",
            cpu_fine_combo_name,
            n_query, n_dim, n_total_cluster, n_probes, k, query_batch_size);
        ivf_search_ctx_free_all(&ctx);
        throw std::runtime_error("ivfflat_search failed: non-std exception");
    }

    ivf_search_ctx_free_all(&ctx);
}


void paraselect::search::ivfflat_search(
    const IVFFlatSearchRequest& request,
    IVFSearchStats* stats) {
    validate_ivf_index_shape(request.index.coarse, request.index.lists);
    if (request.queries.dim != request.index.coarse.dim) {
        throw std::invalid_argument("IVFFlat query and index dimensions differ");
    }
    const auto& storage = request.index.storage;
    ivfflat_search_impl(
        request.queries.fp32,
        request.index.lists,
        storage.rowmajor_fp32,
        request.index.coarse.centroids,
        request.queries.count,
        request.queries.dim,
        request.params.nprobe,
        request.params.topk,
        request.params.distance,
        request.params.batch_size,
        request.params.fp16_coarse,
        request.output.indices,
        request.output.distances,
        request.output.coarse_ids,
        storage.norms,
        request.queries.u8,
        storage.rowmajor_u8,
        stats,
        request.options.cpu_fine_combo,
        storage.aosoa_fp32,
        storage.aosoa_offsets,
        storage.aosoa_tile,
        request.options.overlap_cpu_pipeline,
        storage.resident_fp32,
        storage.resident_u8,
        storage.resident_norms,
        storage.resident_ids,
        request.options.fine_kind,
        request.options.resident_combo,
        request.options.resident_chunk_blocks,
        request.options.threshold);
}
