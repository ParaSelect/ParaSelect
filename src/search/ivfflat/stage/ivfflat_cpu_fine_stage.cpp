#include "search/ivfflat/stage/ivfflat_stages.cuh"

#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/numa_cluster_owner.h"
#include "search/ivfflat/pipeline/ivf_cpu_compact_metadata.h"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_query_prepare.h"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"
#include "search/kernels/fine/cpu_fine/types/cpu_ivf_storage_aosoa.h"
#include "utils/nsight_nvtx.h"
#include "utils/perf_control.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace paraselect::search::ivfflat_stage {
namespace {

inline auto fine_now() {
    return std::chrono::high_resolution_clock::now();
}

inline double fine_elapsed_ms(
    std::chrono::high_resolution_clock::time_point a,
    std::chrono::high_resolution_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline bool cpu_fine_debug_enabled() {
    const char* env = std::getenv("PARASELECT_DEBUG_CPU_FINE");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

paraselect::cpu_fine::CpuIvfStorageView make_storage_view(
    CpuFineVectorLayout layout,
    const CpuFinePrepareConfig& cfg,
    const IVFFlatFineCtx* fine)
{
    if (!fine->h_cluster_offsets_ll) {
        throw std::runtime_error("ivfflat fine stage: missing h_cluster_offsets_ll before storage view cache");
    }
    if (layout == CPU_FINE_LAYOUT_FP32_ROW_MAJOR) {
        if (!fine->h_cluster_vectors) {
            throw std::runtime_error("ivfflat fine stage: missing row-major cluster vectors");
        }
        return paraselect::cpu_fine::make_fp32_row_major_storage(
            fine->h_cluster_vectors,
            fine->h_cluster_offsets_ll,
            fine->h_cluster_sizes,
            cfg.n_dim,
            cfg.n_total_clusters,
            cfg.n_total_vectors);
    }
    if (layout == CPU_FINE_LAYOUT_FP32_AOSOA) {
        if (!fine->h_cluster_vectors_aosoa || !fine->h_cluster_offsets_aosoa_ll ||
            fine->h_cluster_aosoa_tile <= 0) {
            throw std::runtime_error("ivfflat fine stage: missing AoSoA vectors/offsets/tile");
        }
        return paraselect::cpu_fine::make_fp32_aosoa_storage(
            fine->h_cluster_vectors_aosoa,
            fine->h_cluster_offsets_aosoa_ll,
            fine->h_cluster_offsets_ll,
            fine->h_cluster_sizes,
            fine->h_cluster_aosoa_tile,
            cfg.n_dim,
            cfg.n_total_clusters,
            cfg.n_total_vectors);
    }
    if (layout == CPU_FINE_LAYOUT_U8_ROW_MAJOR) {
        if (!fine->h_cluster_vectors_u8) {
            throw std::runtime_error("ivfflat fine stage: missing u8 row-major vectors");
        }
        return paraselect::cpu_fine::make_u8_row_major_storage(
            fine->h_cluster_vectors_u8,
            fine->h_cluster_offsets_ll,
            fine->h_cluster_sizes,
            cfg.n_dim,
            cfg.n_total_clusters,
            cfg.n_total_vectors);
    }
    throw std::runtime_error("ivfflat fine stage: unsupported CPU fine vector layout");
}

void map_local_idx_to_global(
    int current_n_querys,
    int topk,
    const int* h_topk_local_idx,
    const idx_t* h_reordered_indices,
    idx_t* h_topk_index_out)
{
    if (!h_topk_index_out) return;
    const size_t n_out = (size_t)current_n_querys * (size_t)topk;
    for (size_t i = 0; i < n_out; ++i) {
        const int pos = h_topk_local_idx[i];
        h_topk_index_out[i] = (pos >= 0) ? h_reordered_indices[(size_t)pos] : (idx_t)-1;
    }
}

}  // namespace

void prepare_cpu_fine_stage(IVFFlatFineCtx* fine, const CpuFinePrepareConfig& cfg) {
    if (!fine) throw std::invalid_argument("prepare_cpu_fine_stage: fine is null");
    if (!cfg.h_cluster_sizes || !cfg.h_cluster_offsets || !cfg.h_reordered_indices) {
        throw std::invalid_argument("prepare_cpu_fine_stage: cluster metadata is missing");
    }
    if (cfg.query_batch_size <= 0 || cfg.n_dim <= 0 || cfg.n_total_clusters <= 0 ||
        cfg.n_total_vectors <= 0 || cfg.n_probes <= 0 || cfg.topk <= 0) {
        throw std::invalid_argument("prepare_cpu_fine_stage: invalid size configuration");
    }
    const int max_cpu_threads = paraselect::resolve_cpu_fine_thread_count();
    fine->h_cluster_sizes = cfg.h_cluster_sizes;
    fine->h_cluster_vectors = cfg.h_cluster_vectors;
    fine->h_cluster_vectors_aosoa = cfg.h_cluster_vectors_aosoa;
    fine->h_cluster_vectors_u8 = cfg.h_cluster_vectors_u8;
    fine->h_reordered_indices = cfg.h_reordered_indices;
    fine->h_cluster_offsets_aosoa_ll = cfg.h_cluster_offsets_aosoa_ll;
    fine->h_cluster_aosoa_tile = cfg.h_cluster_aosoa_tile;

    const auto* combo = paraselect::cpu_fine::resolve_cpu_fine_combo_ptr_or_throw(
        cfg.cpu_fine_combo_name);
    if (!paraselect::cpu_fine::cpu_fine_combo_supported_by_host(*combo)) {
        throw std::runtime_error("prepare_cpu_fine_stage: selected CPU fine combo is not supported by this host");
    }
    fine->cached_combo = combo;
    const bool compact_metadata_capable = combo->cluster_compact_metadata != nullptr;
    fine->h_use_compact_metadata =
        (cfg.allow_precomputed_metadata &&
         ivf_cpu_should_use_compact_metadata(
             cfg.query_batch_size,
             cfg.n_probes,
             cfg.n_total_clusters,
             compact_metadata_capable)) ? 1 : 0;
    if (!cfg.allow_precomputed_metadata) {
        fine->cached_cluster_runner = combo->cluster_scratch;
        fine->cached_cluster_metadata_mode = CPU_FINE_CLUSTER_METADATA_SCRATCH;
    } else if (fine->h_use_compact_metadata != 0) {
        fine->cached_cluster_runner = combo->cluster_compact_metadata;
        fine->cached_cluster_metadata_mode = CPU_FINE_CLUSTER_METADATA_COMPACT;
    } else {
        fine->cached_cluster_runner = combo->cluster_metadata;
        fine->cached_cluster_metadata_mode = CPU_FINE_CLUSTER_METADATA_FULL;
    }
    if (!fine->cached_cluster_runner) {
        throw std::runtime_error("prepare_cpu_fine_stage: selected CPU fine combo has no runner for metadata mode");
    }

    CUDA_CHECK(cudaMallocHost(&fine->h_top_nprobe_batch,
                              (size_t)cfg.query_batch_size * (size_t)cfg.n_probes * sizeof(idx_t)));
    CUDA_CHECK(cudaMallocHost(&fine->h_cluster_query_offset_batch,
                              (size_t)(cfg.n_total_clusters + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&fine->h_cluster_query_data_batch,
                              (size_t)cfg.query_batch_size * (size_t)cfg.n_probes * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&fine->h_active_cluster_ids_batch,
                              (size_t)cfg.query_batch_size * (size_t)cfg.n_probes * sizeof(int)));
    fine->h_active_cluster_count_batch = 0;
    fine->h_topk_local_idx_batch = static_cast<int*>(
        std::malloc((size_t)cfg.query_batch_size * (size_t)cfg.topk * sizeof(int)));
    fine->h_cluster_offsets_ll = static_cast<long long*>(
        std::malloc((size_t)(cfg.n_total_clusters + 1) * sizeof(long long)));
    fine->h_query_u8_tmp = static_cast<uint8_t*>(
        std::malloc((size_t)cfg.query_batch_size * (size_t)cfg.n_dim * sizeof(uint8_t)));
    fine->h_thread_topk_capacity =
        (size_t)max_cpu_threads * (size_t)cfg.query_batch_size * (size_t)cfg.topk;
    fine->h_thread_query_capacity = (size_t)max_cpu_threads * (size_t)cfg.query_batch_size;
    fine->h_thread_dists = static_cast<float*>(
        std::malloc(fine->h_thread_topk_capacity * sizeof(float)));
    fine->h_thread_idxs = static_cast<int*>(
        std::malloc(fine->h_thread_topk_capacity * sizeof(int)));
    fine->h_thread_sizes = static_cast<int*>(
        std::malloc(fine->h_thread_query_capacity * sizeof(int)));
    fine->h_thread_touched = static_cast<uint8_t*>(
        std::malloc(fine->h_thread_query_capacity * sizeof(uint8_t)));
    const int numa_nodes_for_boundaries = std::max(
        1,
        std::min(paraselect::get_numa_node_count(), max_cpu_threads));
    const std::vector<int> numa_boundaries = paraselect::numa::build_cluster_owner_boundaries_by_counts(
        fine->h_cluster_sizes,
        cfg.n_total_clusters,
        numa_nodes_for_boundaries);
    fine->h_numa_cluster_boundaries = static_cast<int*>(
        std::malloc(numa_boundaries.size() * sizeof(int)));
    fine->h_numa_cluster_boundary_count = static_cast<int>(numa_boundaries.size());
    if (fine->h_numa_cluster_boundaries) {
        std::memcpy(
            fine->h_numa_cluster_boundaries,
            numa_boundaries.data(),
            numa_boundaries.size() * sizeof(int));
    }
    if (!fine->h_topk_local_idx_batch || !fine->h_cluster_offsets_ll || !fine->h_query_u8_tmp ||
        !fine->h_thread_dists || !fine->h_thread_idxs || !fine->h_thread_sizes ||
        !fine->h_thread_touched || !fine->h_numa_cluster_boundaries) {
        throw std::bad_alloc();
    }
    for (int i = 0; i <= cfg.n_total_clusters; ++i) {
        fine->h_cluster_offsets_ll[i] = static_cast<long long>(cfg.h_cluster_offsets[i]);
    }
    fine->cached_storage_view = make_storage_view(combo->layout, cfg, fine);
    fine->cached_query_prepare = paraselect::cpu_fine::query_prepare_for_layout(combo->layout);
    if (!fine->cached_query_prepare) {
        throw std::runtime_error("prepare_cpu_fine_stage: selected CPU fine combo has no query prepare function");
    }
}

void bind_cpu_fine_batch(
    paraselect::cpu_fine::CpuFineBatchView* view,
    const CpuFineBatchBindConfig& cfg)
{
    if (!view) throw std::invalid_argument("bind_cpu_fine_batch: view is null");
    if (cfg.n_query <= 0 || cfg.topk <= 0 || !cfg.h_top_nprobe_batch ||
        !cfg.h_topk_local_idx_batch || !cfg.h_thread_sizes || !cfg.h_thread_touched) {
        throw std::invalid_argument("bind_cpu_fine_batch: invalid batch scratch");
    }
    std::memset(cfg.h_topk_local_idx_batch, -1, (size_t)cfg.n_query * (size_t)cfg.topk * sizeof(int));
    std::memset(cfg.h_thread_sizes, 0, (size_t)cfg.n_query == 0 ? 0 : cfg.h_thread_query_capacity * sizeof(int));
    std::memset(cfg.h_thread_touched, 0, cfg.h_thread_query_capacity * sizeof(uint8_t));

    std::memset(view, 0, sizeof(*view));
    view->n_query = cfg.n_query;
    view->h_queries = cfg.h_queries;
    view->h_queries_u8 = cfg.h_queries_u8;
    view->h_topk_dist_out = cfg.h_topk_dist_out;
    view->h_topk_index_out = cfg.h_topk_index_out;
    view->h_top_nprobe_batch = cfg.h_top_nprobe_batch;
    view->h_cluster_query_offset_batch = cfg.h_cluster_query_offset_batch;
    view->h_cluster_query_data_batch = cfg.h_cluster_query_data_batch;
    view->h_active_cluster_ids_batch = cfg.h_active_cluster_ids_batch;
    view->active_cluster_count = cfg.active_cluster_count;
    view->h_topk_local_idx_batch = cfg.h_topk_local_idx_batch;
    view->h_query_u8_tmp = cfg.h_query_u8_tmp;
    view->h_thread_dists = cfg.h_thread_dists;
    view->h_thread_idxs = cfg.h_thread_idxs;
    view->h_thread_sizes = cfg.h_thread_sizes;
    view->h_thread_touched = cfg.h_thread_touched;
    view->h_thread_topk_capacity = cfg.h_thread_topk_capacity;
    view->h_thread_query_capacity = cfg.h_thread_query_capacity;
}

void build_cpu_fine_metadata(
    paraselect::cpu_fine::CpuFineBatchView* view,
    const CpuFineMetadataBuildConfig& cfg)
{
    if (!view) throw std::invalid_argument("build_cpu_fine_metadata: view is null");
    if (cfg.n_query <= 0 || cfg.n_probes <= 0 || cfg.n_total_clusters <= 0 ||
        !view->h_top_nprobe_batch || !view->h_cluster_query_offset_batch ||
        !view->h_cluster_query_data_batch || !view->h_active_cluster_ids_batch) {
        throw std::invalid_argument("build_cpu_fine_metadata: invalid inputs");
    }

    const size_t ncluster_offsets = (size_t)cfg.n_total_clusters + 1u;
    std::memset(view->h_cluster_query_offset_batch, 0, ncluster_offsets * sizeof(int));

    const idx_t* top_nprobe = view->h_top_nprobe_batch;
    int* offsets = view->h_cluster_query_offset_batch;
    int* data = view->h_cluster_query_data_batch;
    int* active = view->h_active_cluster_ids_batch;

    for (int q = 0; q < cfg.n_query; ++q) {
        const size_t row = (size_t)q * (size_t)cfg.n_probes;
        for (int p = 0; p < cfg.n_probes; ++p) {
            const int cid = static_cast<int>(top_nprobe[row + (size_t)p]);
            if (cid < 0 || cid >= cfg.n_total_clusters) continue;
            ++offsets[cid + 1];
        }
    }
    for (int cid = 0; cid < cfg.n_total_clusters; ++cid) {
        offsets[cid + 1] += offsets[cid];
    }

    if (cfg.use_compact_metadata) {
        std::vector<int> active_pos((size_t)cfg.n_total_clusters, -1);
        int active_count = 0;
        for (int cid = 0; cid < cfg.n_total_clusters; ++cid) {
            if (offsets[cid] != offsets[cid + 1]) {
                active_pos[(size_t)cid] = active_count;
                active[active_count++] = cid;
            }
        }
        view->active_cluster_count = active_count;

        std::vector<int> compact_offsets((size_t)active_count + 1u, 0);
        for (int cid = 0; cid < cfg.n_total_clusters; ++cid) {
            const int pos = active_pos[(size_t)cid];
            if (pos >= 0) {
                compact_offsets[(size_t)pos + 1u] = offsets[cid + 1] - offsets[cid];
            }
        }
        for (int i = 0; i < active_count; ++i) {
            compact_offsets[(size_t)i + 1u] += compact_offsets[(size_t)i];
        }
        std::vector<int> write_pos((size_t)active_count, 0);
        for (int i = 0; i < active_count; ++i) {
            write_pos[(size_t)i] = compact_offsets[(size_t)i];
        }
        for (int q = 0; q < cfg.n_query; ++q) {
            const size_t row = (size_t)q * (size_t)cfg.n_probes;
            for (int p = 0; p < cfg.n_probes; ++p) {
                const int cid = static_cast<int>(top_nprobe[row + (size_t)p]);
                if (cid < 0 || cid >= cfg.n_total_clusters) continue;
                const int pos = active_pos[(size_t)cid];
                if (pos < 0) continue;
                data[(size_t)write_pos[(size_t)pos]++] = q;
            }
        }
        std::memcpy(offsets, compact_offsets.data(), ((size_t)active_count + 1u) * sizeof(int));
        return;
    }

    ivf_cpu_fill_active_clusters_from_full_offsets(offsets, cfg.n_total_clusters, active, &view->active_cluster_count);
    std::vector<int> write_pos(offsets, offsets + cfg.n_total_clusters);
    for (int q = 0; q < cfg.n_query; ++q) {
        const size_t row = (size_t)q * (size_t)cfg.n_probes;
        for (int p = 0; p < cfg.n_probes; ++p) {
            const int cid = static_cast<int>(top_nprobe[row + (size_t)p]);
            if (cid < 0 || cid >= cfg.n_total_clusters) continue;
            data[(size_t)write_pos[(size_t)cid]++] = q;
        }
    }
}

void run_cpu_fine_batch(
    const IVFFlatFineCtx* fine,
    const CpuFineRunConfig& cfg,
    const paraselect::cpu_fine::CpuFineBatchView& view)
{
    NVTX_PUSH("IVFFlat_Fine_CPU");
    if (!fine) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: fine is null");
    }
    if (!fine->h_reordered_indices || !fine->h_cluster_sizes) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: missing reordered indices or cluster sizes");
    }
    const int current_n_querys = view.n_query > 0 ? view.n_query : cfg.n_query;
    const idx_t* h_top_nprobe = view.h_top_nprobe_batch;
    if (!h_top_nprobe) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: missing h_top_nprobe_batch scratch");
    }

    const int num_threads = paraselect::resolve_cpu_fine_thread_count();
    paraselect::cpu_fine::thread_stats_detail::clear_latest_fine_timing();
    if (!fine->cached_combo || !fine->cached_query_prepare || !fine->cached_cluster_runner) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: missing cached fine runner state");
    }

    int* h_topk_local_idx = view.h_topk_local_idx_batch;
    const size_t n_topk = (size_t)current_n_querys * (size_t)cfg.topk;
    if (!h_topk_local_idx) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: missing h_topk_local_idx_batch scratch");
    }
    auto outer_timing = paraselect::cpu_fine::thread_stats_detail::get_latest_fine_timing();
    const auto scratch_outer_t0 = fine_now();
    std::fill(h_topk_local_idx, h_topk_local_idx + n_topk, -1);
    outer_timing.scratch_clear_ms += fine_elapsed_ms(scratch_outer_t0, fine_now());
    paraselect::cpu_fine::thread_stats_detail::record_latest_fine_timing(outer_timing);

    const size_t v3_heap_elems = (size_t)num_threads * (size_t)current_n_querys * (size_t)cfg.topk;
    const size_t v3_touch_elems = (size_t)num_threads * (size_t)current_n_querys;
    if (!view.h_thread_dists || !view.h_thread_idxs || !view.h_thread_sizes || !view.h_thread_touched ||
        view.h_thread_topk_capacity < v3_heap_elems ||
        view.h_thread_query_capacity < v3_touch_elems) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: missing V3 thread scratch");
    }

    const auto query_t0 = fine_now();
    const float* query_ptr = fine->cached_query_prepare(&view, current_n_querys, cfg.n_dim);
    auto query_timing = paraselect::cpu_fine::thread_stats_detail::get_latest_fine_timing();
    query_timing.query_prepare_ms += fine_elapsed_ms(query_t0, fine_now());
    paraselect::cpu_fine::thread_stats_detail::record_latest_fine_timing(query_timing);
    if (!query_ptr) {
        NVTX_POP();
        throw std::runtime_error("run_cpu_fine_batch: selected query buffer unavailable for cached runner");
    }

    const bool debug_cpu_fine = cpu_fine_debug_enabled();
    if (debug_cpu_fine) {
        const bool compact = fine->cached_cluster_metadata_mode == CPU_FINE_CLUSTER_METADATA_COMPACT;
        const int offset_tail_index = compact ? view.active_cluster_count : fine->cached_storage_view.n_total_clusters;
        const int offset_tail = view.h_cluster_query_offset_batch && offset_tail_index >= 0
            ? view.h_cluster_query_offset_batch[offset_tail_index]
            : -1;
        const long long cluster_tail = fine->cached_storage_view.u8_row_major_offsets
            ? fine->cached_storage_view.u8_row_major_offsets[fine->cached_storage_view.n_total_clusters]
            : (fine->cached_storage_view.row_major_offsets
                ? fine->cached_storage_view.row_major_offsets[fine->cached_storage_view.n_total_clusters]
                : -1LL);
        const long long cluster_last = fine->cached_storage_view.u8_row_major_offsets && fine->cached_storage_view.n_total_clusters > 0
            ? fine->cached_storage_view.u8_row_major_offsets[fine->cached_storage_view.n_total_clusters - 1]
            : -1LL;
        const int cluster_last_count = fine->cached_storage_view.cluster_counts && fine->cached_storage_view.n_total_clusters > 0
            ? fine->cached_storage_view.cluster_counts[fine->cached_storage_view.n_total_clusters - 1]
            : -1;
        std::fprintf(stderr,
            "[CPUFineDebug] before runner tag=%s layout=%d metadata_mode=%d n_query=%d n_dim=%d nlist=%d nprobe=%d topk=%d threads=%d active=%d offset_tail_index=%d offset_tail=%d storage_vectors=%d cluster_last=%lld cluster_last_count=%d cluster_tail=%lld query=%p top_nprobe=%p offsets=%p data=%p active_ids=%p local_idx=%p out_idx=%p\n",
            fine->cached_combo && fine->cached_combo->nvtx_tag ? fine->cached_combo->nvtx_tag : "<null>",
            fine->cached_combo ? (int)fine->cached_combo->layout : -1,
            (int)fine->cached_cluster_metadata_mode,
            current_n_querys,
            cfg.n_dim,
            fine->cached_storage_view.n_total_clusters,
            cfg.n_probes,
            cfg.topk,
            num_threads,
            view.active_cluster_count,
            offset_tail_index,
            offset_tail,
            fine->cached_storage_view.n_total_vectors,
            cluster_last,
            cluster_last_count,
            cluster_tail,
            (const void*)query_ptr,
            (const void*)h_top_nprobe,
            (const void*)view.h_cluster_query_offset_batch,
            (const void*)view.h_cluster_query_data_batch,
            (const void*)view.h_active_cluster_ids_batch,
            (const void*)h_topk_local_idx,
            (const void*)view.h_topk_index_out);
        std::fflush(stderr);
    }

    NVTX_PUSH(fine->cached_combo->nvtx_tag);
    {
        const paraselect::profiling::PerfControlScope perf_scope;
        const auto runner_t0 = fine_now();
        const paraselect::cpu_fine::CpuFineRunArgs run_args{
            &fine->cached_storage_view,
            query_ptr,
            h_top_nprobe,
            &view,
            cfg.n_probes,
            cfg.topk,
            cfg.distance_mode,
            num_threads,
            fine->h_numa_cluster_boundaries,
            fine->h_numa_cluster_boundary_count};
        auto runner = fine->cached_cluster_runner;
        const long long scanned = runner(run_args);
        if (debug_cpu_fine) {
            std::fprintf(stderr, "[CPUFineDebug] after runner scanned=%lld\n", scanned);
            std::fflush(stderr);
        }
        auto timing = paraselect::cpu_fine::thread_stats_detail::get_latest_fine_timing();
        timing.runner_ms += fine_elapsed_ms(runner_t0, fine_now());
        paraselect::cpu_fine::thread_stats_detail::record_latest_fine_timing(timing);
    }
    NVTX_POP();

    if (debug_cpu_fine) {
        int min_idx = fine->cached_storage_view.n_total_vectors;
        int max_idx = -1;
        size_t invalid_idx = 0;
        const size_t total_idx = (size_t)current_n_querys * (size_t)cfg.topk;
        size_t first_invalid = (size_t)-1;
        int first_invalid_value = -1;
        for (size_t i = 0; i < total_idx; ++i) {
            const int pos = h_topk_local_idx[i];
            if (pos < 0) continue;
            min_idx = std::min(min_idx, pos);
            max_idx = std::max(max_idx, pos);
            if (pos >= fine->cached_storage_view.n_total_vectors) {
                if (first_invalid == (size_t)-1) {
                    first_invalid = i;
                    first_invalid_value = pos;
                }
                ++invalid_idx;
            }
        }
        std::fprintf(stderr,
            "[CPUFineDebug] before id_map min_idx=%d max_idx=%d invalid_idx=%zu first_invalid=%zu first_invalid_value=%d n_total_vectors=%d out_idx=%p reordered=%p\n",
            min_idx,
            max_idx,
            invalid_idx,
            first_invalid,
            first_invalid_value,
            fine->cached_storage_view.n_total_vectors,
            (const void*)view.h_topk_index_out,
            (const void*)fine->h_reordered_indices);
        std::fflush(stderr);
        if (invalid_idx != 0) {
            throw std::runtime_error("run_cpu_fine_batch: topk local index out of range before id map");
        }
    }

    const auto map_t0 = fine_now();
    map_local_idx_to_global(
        current_n_querys,
        cfg.topk,
        h_topk_local_idx,
        fine->h_reordered_indices,
        view.h_topk_index_out);
    auto timing = paraselect::cpu_fine::thread_stats_detail::get_latest_fine_timing();
    timing.id_map_ms += fine_elapsed_ms(map_t0, fine_now());
    paraselect::cpu_fine::thread_stats_detail::record_latest_fine_timing(timing);
    NVTX_POP();
}

}  // namespace paraselect::search::ivfflat_stage
