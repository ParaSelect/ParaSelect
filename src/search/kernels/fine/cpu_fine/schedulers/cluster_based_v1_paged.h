#ifndef PARASELECT_CLUSTER_BASED_V1_PAGED_H
#define PARASELECT_CLUSTER_BASED_V1_PAGED_H

#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_v1_common.h"
#include "utils/pch.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <type_traits>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_v1_paged_detail {

static constexpr int kPrefetchAhead = 8;

inline cluster_based_v1_common_detail::SchedulerWorkspace& paged_scheduler_workspace() {
    thread_local cluster_based_v1_common_detail::SchedulerWorkspace workspace;
    return workspace;
}

template <typename Scalar>
struct PageTask {
    int cid = -1;
    int qlist_start = 0;
    int qlist_end = 0;
    int lane_begin = 0;
    int lane_end = 0;
    int phys_page = -1;
    const Scalar* page_ptr = nullptr;
};

template <typename Scalar>
inline std::vector<PageTask<Scalar>>& paged_page_tasks() {
    thread_local std::vector<PageTask<Scalar>> tasks;
    return tasks;
}

template <typename Scalar>
inline void build_page_tasks_for_metadata(
    std::vector<PageTask<Scalar>>* tasks,
    int metadata_compact,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* cluster_query_offset_ptr,
    const int* h_cluster_counts,
    const int* h_cluster_page_offsets,
    const int* h_page_ids,
    const Scalar* h_pages,
    int page_vecs,
    int dim,
    int n_total_clusters) {
    tasks->clear();
    auto append_cluster = [&](int cid) {
        const int qbegin = cluster_query_offset_ptr[cid];
        const int qend = cluster_query_offset_ptr[cid + 1];
        const int count = h_cluster_counts[cid];
        if (qbegin == qend || count <= 0) return;
        const int page_begin = h_cluster_page_offsets[(size_t)cid];
        for (int lane = 0; lane < count; lane += page_vecs) {
            const int logical_page = lane / page_vecs;
            const int slot_end = std::min(count - lane, page_vecs);
            const int phys_page = h_page_ids[(size_t)page_begin + (size_t)logical_page];
            const Scalar* page_ptr = h_pages + (size_t)phys_page * (size_t)page_vecs * (size_t)dim;
            tasks->push_back(PageTask<Scalar>{cid, qbegin, qend, lane, lane + slot_end, phys_page, page_ptr});
        }
    };
    if (metadata_compact || (h_active_cluster_ids && active_cluster_count >= 0)) {
        for (int i = 0; i < active_cluster_count; ++i) {
            const int cid = h_active_cluster_ids[i];
            if (cid >= 0 && cid < n_total_clusters) append_cluster(cid);
        }
    } else {
        for (int cid = 0; cid < n_total_clusters; ++cid) append_cluster(cid);
    }
}

template <typename Scalar>
inline const Scalar* paged_candidate_ptr(
    const Scalar* h_pages,
    const int* h_cluster_page_offsets,
    const int* h_page_ids,
    int page_vecs,
    int dim,
    int cid,
    int lane) {
    const int logical_page = lane / page_vecs;
    const int slot = lane - logical_page * page_vecs;
    const int phys_page = h_page_ids[(size_t)h_cluster_page_offsets[(size_t)cid] + (size_t)logical_page];
    return h_pages + ((size_t)phys_page * (size_t)page_vecs + (size_t)slot) * (size_t)dim;
}

template <bool Masked, typename Evaluator, typename PushLocal>
inline long long scan_u8_page_task(
    const PageTask<uint8_t>& task, const uint8_t* alive, int page_vecs,
    const uint8_t* queries, const int* query_list, int dim,
    int live_count, PushLocal& push_local) {
    const int query_count = task.qlist_end - task.qlist_start;
    const int lane_begin = task.lane_begin;
    const int vector_count = task.lane_end - lane_begin;
    const uint8_t* vectors =
        task.page_ptr + (size_t)(lane_begin % page_vecs) * (size_t)dim;
    long long operations = 0;
    int qi = 0;
    for (; qi + 4 <= query_count; qi += 4) {
        const int q0 = query_list[qi + 0];
        const int q1 = query_list[qi + 1];
        const int q2 = query_list[qi + 2];
        const int q3 = query_list[qi + 3];
        const uint8_t* pq0 = queries + (size_t)q0 * dim;
        const uint8_t* pq1 = queries + (size_t)q1 * dim;
        const uint8_t* pq2 = queries + (size_t)q2 * dim;
        const uint8_t* pq3 = queries + (size_t)q3 * dim;
        for (int vi = 0; vi < std::min(kPrefetchAhead, vector_count); ++vi)
            Evaluator::prefetch_candidate(vectors + (size_t)vi * dim, dim);
        for (int vi = 0; vi < vector_count; ++vi) {
            const int next = vi + kPrefetchAhead;
            if (next < vector_count)
                Evaluator::prefetch_candidate(vectors + (size_t)next * dim, dim);
            const int lane = lane_begin + vi;
            const int slot = lane - (lane / page_vecs) * page_vecs;
            const int index = task.phys_page * page_vecs + slot;
            if constexpr (Masked)
                if (!alive[(size_t)index]) continue;
            const uint8_t* vector = vectors + (size_t)vi * dim;
            int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            Evaluator::distance_q4(pq0, pq1, pq2, pq3, vector, dim,
                                   d0, d1, d2, d3);
            push_local(q0, (float)d0, index);
            push_local(q1, (float)d1, index);
            push_local(q2, (float)d2, index);
            push_local(q3, (float)d3, index);
        }
        operations += (long long)live_count * dim * 4LL;
    }
    for (; qi < query_count; ++qi) {
        const int qid = query_list[qi];
        const uint8_t* query = queries + (size_t)qid * dim;
        for (int vi = 0; vi < std::min(kPrefetchAhead, vector_count); ++vi)
            Evaluator::prefetch_candidate(vectors + (size_t)vi * dim, dim);
        for (int vi = 0; vi < vector_count; ++vi) {
            const int next = vi + kPrefetchAhead;
            if (next < vector_count)
                Evaluator::prefetch_candidate(vectors + (size_t)next * dim, dim);
            const int lane = lane_begin + vi;
            const int slot = lane - (lane / page_vecs) * page_vecs;
            const int index = task.phys_page * page_vecs + slot;
            if constexpr (Masked)
                if (!alive[(size_t)index]) continue;
            const int32_t distance = Evaluator::distance_q1(
                query, vectors + (size_t)vi * dim, dim);
            push_local(qid, (float)distance, index);
        }
        operations += (long long)live_count * dim;
    }
    return operations;
}

template <typename Scalar, typename Evaluator>
inline long long cluster_based_v1_paged_impl(
    const Scalar* h_pages,
    const uint8_t* h_alive,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_cluster_page_offsets,
    const int* h_cluster_page_counts,
    const int* h_page_ids,
    int page_vecs,
    const Scalar* h_query,
    const idx_t* h_coarse_cluster_ids,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    int metadata_compact,
    const int* h_numa_cluster_owner,
    int numa_cluster_owner_count,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched) {
    (void)distance_mode;
    (void)h_cluster_page_counts;
    (void)h_numa_cluster_owner;
    (void)numa_cluster_owner_count;
    using Accumulator = decltype(Evaluator::distance_q1(h_query, h_pages, dim));
    if (num_threads <= 0) num_threads = paraselect::resolve_cpu_fine_thread_count();
    num_threads = paraselect::resolve_adaptive_thread_count(
        num_threads, n_query * std::max(1, n_probes));

    auto now = cluster_based_v1_common_detail::scheduler_now;
    auto elapsed_ms = [](auto a, auto b) {
        return cluster_based_v1_common_detail::scheduler_elapsed_ms(a, b);
    };
    auto now_s = cluster_based_v1_common_detail::scheduler_now_s;
    thread_stats_detail::FineTimingSnapshot timing;

    auto& workspace = paged_scheduler_workspace();
    const int* cluster_query_offset_ptr = h_cluster_query_offsets;
    const int* cluster_query_list_ptr = h_cluster_query_list;
    auto metadata_t0 = now();
    cluster_based_v1_common_detail::prepare_cluster_query_metadata(
        &workspace, h_coarse_cluster_ids, n_query, n_probes, n_total_clusters,
        &cluster_query_offset_ptr, &cluster_query_list_ptr);
    timing.metadata_build_ms = elapsed_ms(metadata_t0, now());

    const int T = num_threads;
    const size_t touch_elems = (size_t)T * (size_t)n_query;
    auto scratch_t0 = now();
    std::fill(scratch_thread_touched, scratch_thread_touched + touch_elems, (uint8_t)0);
    timing.scratch_clear_ms = elapsed_ms(scratch_t0, now());

    auto& tasks = paged_page_tasks<Scalar>();
    auto task_t0 = now();
    build_page_tasks_for_metadata(
        &tasks, metadata_compact, h_active_cluster_ids, active_cluster_count,
        cluster_query_offset_ptr, h_cluster_counts, h_cluster_page_offsets, h_page_ids,
        h_pages, page_vecs, dim, n_total_clusters);
    timing.task_build_ms = elapsed_ms(task_t0, now());

    struct InternalDebugStats {
        long long cluster_query_refs = 0;
        long long task_query_refs = 0;
        long long task_vectors = 0;
        long long q4_groups = 0;
        long long q4_candidate_comps = 0;
        long long q1_refs = 0;
        long long q1_candidate_comps = 0;
        int nonempty_clusters = 0;
        int cluster_qlist_le1 = 0;
        int cluster_qlist_le3 = 0;
        int max_cluster_queries = 0;
        int max_cluster_count = 0;
        int task_qlist_le1 = 0;
        int task_qlist_le3 = 0;
        int max_task_queries = 0;
        int max_task_vectors = 0;
    };
    const bool debug_internal = std::getenv("PARASELECT_DEBUG_U8_PAGED_INTERNAL") != nullptr ||
                                std::getenv("PARASELECT_DEBUG_CPU_FINE_TASKS") != nullptr;
    InternalDebugStats dbg{};
    if (debug_internal) {
        for (int c = 0; c < n_total_clusters; ++c) {
            const int qrefs = cluster_query_offset_ptr[c + 1] - cluster_query_offset_ptr[c];
            if (qrefs > 0) {
                ++dbg.nonempty_clusters;
                if (qrefs <= 1) ++dbg.cluster_qlist_le1;
                if (qrefs <= 3) ++dbg.cluster_qlist_le3;
            }
            dbg.cluster_query_refs += qrefs;
            dbg.max_cluster_queries = std::max(dbg.max_cluster_queries, qrefs);
            dbg.max_cluster_count = std::max(dbg.max_cluster_count, h_cluster_counts[c]);
        }
        for (const auto& task : tasks) {
            const int qrefs = task.qlist_end - task.qlist_start;
            const int vecs = task.lane_end - task.lane_begin;
            dbg.task_query_refs += qrefs;
            dbg.task_vectors += vecs;
            dbg.q4_groups += (long long)(qrefs / 4);
            dbg.q4_candidate_comps += (long long)(qrefs / 4) * (long long)vecs * 4LL;
            dbg.q1_refs += (long long)(qrefs % 4);
            dbg.q1_candidate_comps += (long long)(qrefs % 4) * (long long)vecs;
            if (qrefs > 0 && qrefs <= 1) ++dbg.task_qlist_le1;
            if (qrefs > 0 && qrefs <= 3) ++dbg.task_qlist_le3;
            dbg.max_task_queries = std::max(dbg.max_task_queries, qrefs);
            dbg.max_task_vectors = std::max(dbg.max_task_vectors, vecs);
        }
    }

    const bool debug_cpu = std::getenv("PARASELECT_DEBUG_U8_PAGED_CPU") != nullptr;
    const bool pin_physical = std::getenv("PARASELECT_DEBUG_U8_PAGED_PIN_PHYSICAL") != nullptr;
    if (debug_cpu) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int affinity_count = 0;
        char cpus_buf[1024];
        int pos = 0;
        cpus_buf[0] = '\0';
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (!CPU_ISSET(cpu, &mask)) continue;
                ++affinity_count;
                if (pos < (int)sizeof(cpus_buf) - 16) {
                    pos += std::snprintf(cpus_buf + pos, sizeof(cpus_buf) - (size_t)pos,
                                         "%s%d", pos == 0 ? "" : ",", cpu);
                }
            }
        }
        std::fprintf(stderr,
                     "[paged_cpu] process_affinity_count=%d cpus=%s pin_physical=%d pthread_self=%lu sched_cpu=%d\n",
                     affinity_count, cpus_buf, pin_physical ? 1 : 0,
                     (unsigned long)pthread_self(), sched_getcpu());
    }

    const bool print_thread_stats = thread_stats_detail::cpu_fine_thread_stats_enabled();
    const bool collect_thread_stats = true;
    std::vector<int> debug_worker_cpus(debug_cpu ? (size_t)T : 0u, -1);
    std::vector<unsigned long> debug_worker_threads(debug_cpu ? (size_t)T : 0u, 0);
    auto& thread_stats = workspace.thread_stats;
    thread_stats.assign((size_t)T, thread_stats_detail::ThreadStat{});
    auto& thread_fma = workspace.thread_fma;
    thread_fma.assign((size_t)T, 0);

    auto scan_t0 = now();
    auto& pool = paraselect::cpu_fine::persistent_thread_pool_for_threads(T);
    std::atomic<int> next_task(0);
    pool.run([&](int tid) {
        if (pin_physical) {
            paraselect::pin_thread_to_physical_core(tid);
        }
        if (debug_cpu && tid >= 0 && tid < T) {
            debug_worker_cpus[(size_t)tid] = sched_getcpu();
            debug_worker_threads[(size_t)tid] = (unsigned long)pthread_self();
        }
        thread_stats_detail::ThreadStat local_stat;
        if (collect_thread_stats) local_stat.start_s = now_s();
        float* my_dists = scratch_thread_dists + (size_t)tid * (size_t)n_query * (size_t)topk;
        int* my_idxs = scratch_thread_idxs + (size_t)tid * (size_t)n_query * (size_t)topk;
        int* my_sizes = scratch_thread_sizes + (size_t)tid * (size_t)n_query;
        uint8_t* my_touched = scratch_thread_touched + (size_t)tid * (size_t)n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopK h;
            h.dists = my_dists + (size_t)qid * (size_t)topk;
            h.idxs = my_idxs + (size_t)qid * (size_t)topk;
            if (!my_touched[qid]) {
                my_touched[qid] = 1;
                my_sizes[qid] = 0;
            }
            h.size = my_sizes[qid];
            h.push(d, gidx, topk);
            my_sizes[qid] = h.size;
        };

        auto process_task = [&](const PageTask<Scalar>& task) {
            const int cid = task.cid;
            const int qlist_size = task.qlist_end - task.qlist_start;
            const int shard_begin = task.lane_begin;
            const int shard_end = task.lane_end;
            const int shard_count = shard_end - shard_begin;
            if (shard_count <= 0) return;
            int live_shard_count = shard_count;
            if (h_alive) {
                live_shard_count = 0;
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int lane = shard_begin + vi;
                    const int slot = lane - (lane / page_vecs) * page_vecs;
                    const int gidx = task.phys_page * page_vecs + slot;
                    live_shard_count += h_alive[(size_t)gidx] != 0;
                }
                if (live_shard_count == 0) return;
            }
            const int* qlist = cluster_query_list_ptr + task.qlist_start;
            const Scalar* shard_ptr = task.page_ptr + (size_t)(shard_begin % page_vecs) * (size_t)dim;
            if (collect_thread_stats) {
                local_stat.tasks += 1;
                local_stat.query_refs += qlist_size;
                local_stat.vectors += shard_count;
            }

            if constexpr (std::is_same_v<Scalar, uint8_t>) {
                const long long operations = h_alive
                    ? scan_u8_page_task<true, Evaluator>(
                          task, h_alive, page_vecs, h_query, qlist, dim,
                          live_shard_count, push_local)
                    : scan_u8_page_task<false, Evaluator>(
                          task, nullptr, page_vecs, h_query, qlist, dim,
                          live_shard_count, push_local);
                thread_fma[(size_t)tid] += operations;
                if (collect_thread_stats) local_stat.fmas += operations;
                return;
            }

            int qi = 0;
            for (; qi + 4 <= qlist_size; qi += 4) {
                const int q0 = qlist[qi + 0];
                const int q1 = qlist[qi + 1];
                const int q2 = qlist[qi + 2];
                const int q3 = qlist[qi + 3];
                const Scalar* pq0 = h_query + (size_t)q0 * (size_t)dim;
                const Scalar* pq1 = h_query + (size_t)q1 * (size_t)dim;
                const Scalar* pq2 = h_query + (size_t)q2 * (size_t)dim;
                const Scalar* pq3 = h_query + (size_t)q3 * (size_t)dim;
                const int pf_end = std::min(kPrefetchAhead, shard_count);
                for (int vi = 0; vi < pf_end; ++vi) {
                    Evaluator::prefetch_candidate(shard_ptr + (size_t)vi * (size_t)dim, dim);
                }
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < shard_count) {
                        Evaluator::prefetch_candidate(shard_ptr + (size_t)pf_vi * (size_t)dim, dim);
                    }
                    const int lane = shard_begin + vi;
                    const int slot = lane - (lane / page_vecs) * page_vecs;
                    const int gidx = task.phys_page * page_vecs + slot;
                    if (h_alive && !h_alive[(size_t)gidx]) continue;
                    const Scalar* v = shard_ptr + (size_t)vi * (size_t)dim;
                    Accumulator d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    Evaluator::distance_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    push_local(q0, (float)d0, gidx);
                    push_local(q1, (float)d1, gidx);
                    push_local(q2, (float)d2, gidx);
                    push_local(q3, (float)d3, gidx);
                }
                const long long fma = (long long)live_shard_count * (long long)dim * 4LL;
                thread_fma[(size_t)tid] += fma;
                if (collect_thread_stats) local_stat.fmas += fma;
            }
            for (; qi < qlist_size; ++qi) {
                const int qid = qlist[qi];
                const Scalar* q = h_query + (size_t)qid * (size_t)dim;
                const int pf_end = std::min(kPrefetchAhead, shard_count);
                for (int vi = 0; vi < pf_end; ++vi) {
                    Evaluator::prefetch_candidate(shard_ptr + (size_t)vi * (size_t)dim, dim);
                }
                for (int vi = 0; vi < shard_count; ++vi) {
                    const int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < shard_count) {
                        Evaluator::prefetch_candidate(shard_ptr + (size_t)pf_vi * (size_t)dim, dim);
                    }
                    const int lane = shard_begin + vi;
                    const int slot = lane - (lane / page_vecs) * page_vecs;
                    const int gidx = task.phys_page * page_vecs + slot;
                    if (h_alive && !h_alive[(size_t)gidx]) continue;
                    const Scalar* v = shard_ptr + (size_t)vi * (size_t)dim;
                    const Accumulator d = Evaluator::distance_q1(q, v, dim);
                    push_local(qid, (float)d, gidx);
                }
                const long long fma = (long long)live_shard_count * (long long)dim;
                thread_fma[(size_t)tid] += fma;
                if (collect_thread_stats) local_stat.fmas += fma;
            }
        };

        const int task_count = static_cast<int>(tasks.size());
        while (true) {
            const int task_id = next_task.fetch_add(1, std::memory_order_relaxed);
            if (task_id >= task_count) break;
            process_task(tasks[(size_t)task_id]);
        }
        if (collect_thread_stats) {
            local_stat.end_s = now_s();
            thread_stats[(size_t)tid] = local_stat;
        }
    });
    long long total_fma = 0;
    for (long long fma : thread_fma) total_fma += fma;
    timing.scan_ms = elapsed_ms(scan_t0, now());
    if (debug_cpu) {
        int min_cpu = 1000000;
        int max_cpu = -1;
        int valid_cpus = 0;
        std::vector<int> used;
        used.reserve(debug_worker_cpus.size());
        for (int cpu : debug_worker_cpus) {
            if (cpu < 0) continue;
            ++valid_cpus;
            min_cpu = std::min(min_cpu, cpu);
            max_cpu = std::max(max_cpu, cpu);
            if (std::find(used.begin(), used.end(), cpu) == used.end()) used.push_back(cpu);
        }
        std::sort(used.begin(), used.end());
        char worker_buf[2048];
        int pos = 0;
        worker_buf[0] = '\0';
        for (int tid = 0; tid < T && tid < 32; ++tid) {
            if (pos < (int)sizeof(worker_buf) - 32) {
                pos += std::snprintf(worker_buf + pos, sizeof(worker_buf) - (size_t)pos,
                                     "%s%d:%d", pos == 0 ? "" : ",", tid,
                                     debug_worker_cpus[(size_t)tid]);
            }
        }
        char used_buf[2048];
        int used_pos = 0;
        used_buf[0] = '\0';
        for (size_t i = 0; i < used.size() && i < 128; ++i) {
            if (used_pos < (int)sizeof(used_buf) - 16) {
                used_pos += std::snprintf(used_buf + used_pos, sizeof(used_buf) - (size_t)used_pos,
                                          "%s%d", used_pos == 0 ? "" : ",", used[i]);
            }
        }
        std::fprintf(stderr,
                     "[paged_cpu] worker_valid=%d unique_cpus=%zu min_cpu=%d max_cpu=%d "
                     "tid_cpu_first32=%s used_cpus=%s thread0=%lu thread_last=%lu\n",
                     valid_cpus, used.size(), min_cpu == 1000000 ? -1 : min_cpu, max_cpu,
                     worker_buf, used_buf,
                     debug_worker_threads.empty() ? 0ul : debug_worker_threads.front(),
                     debug_worker_threads.empty() ? 0ul : debug_worker_threads.back());
    }
    if (print_thread_stats) {
        thread_stats_detail::print_thread_stats_summary("paged_cluster_dynamic", thread_stats);
    }

    auto merge_t0 = now();
    cluster_based_v1_common_detail::merge_thread_topk(
        pool, T, n_query, topk,
        scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes, scratch_thread_touched,
        h_topk_dist, h_topk_local_idx);
    timing.merge_ms = elapsed_ms(merge_t0, now());
    if (debug_internal) {
        const long long total_candidate_comps = dbg.q4_candidate_comps + dbg.q1_candidate_comps;
        const double q1_frac = total_candidate_comps > 0
                                   ? (double)dbg.q1_candidate_comps / (double)total_candidate_comps
                                   : 0.0;
        std::fprintf(stderr,
                     "[paged_internal] nq=%d nprobe=%d T=%d page_vecs=%d "
                     "metadata_ms=%.3f scratch_ms=%.3f task_ms=%.3f scan_ms=%.3f merge_ms=%.3f "
                     "tasks=%zu nonempty_clusters=%d cluster_qrefs=%lld task_qrefs=%lld "
                     "cluster_qlist_le1=%d cluster_qlist_le3=%d max_cluster_q=%d max_cluster_count=%d "
                     "task_qlist_le1=%d task_qlist_le3=%d max_task_q=%d max_task_vecs=%d "
                     "task_vectors=%lld q4_groups=%lld q4_comps=%lld q1_refs=%lld q1_comps=%lld q1_frac=%.6f total_fma=%lld\n",
                     n_query, n_probes, T, page_vecs,
                     timing.metadata_build_ms, timing.scratch_clear_ms, timing.task_build_ms,
                     timing.scan_ms, timing.merge_ms, tasks.size(), dbg.nonempty_clusters,
                     dbg.cluster_query_refs, dbg.task_query_refs, dbg.cluster_qlist_le1,
                     dbg.cluster_qlist_le3, dbg.max_cluster_queries, dbg.max_cluster_count,
                     dbg.task_qlist_le1, dbg.task_qlist_le3, dbg.max_task_queries,
                     dbg.max_task_vectors, dbg.task_vectors, dbg.q4_groups,
                     dbg.q4_candidate_comps, dbg.q1_refs, dbg.q1_candidate_comps,
                     q1_frac, total_fma);
    }
    cluster_based_v1_common_detail::accumulate_latest_fine_timing(timing);
    return total_fma;
}

template <typename Scalar, typename Evaluator>
inline long long cluster_based_v1_paged_with_scratch(
    const Scalar* h_pages,
    const uint8_t* h_alive,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_cluster_page_offsets,
    const int* h_cluster_page_counts,
    const int* h_page_ids,
    int page_vecs,
    const Scalar* h_query,
    const idx_t* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched) {
    return cluster_based_v1_paged_impl<Scalar, Evaluator>(
        h_pages, h_alive, h_cluster_offsets, h_cluster_counts, h_cluster_page_offsets, h_cluster_page_counts,
        h_page_ids, page_vecs, h_query, h_coarse_cluster_ids, nullptr, nullptr, nullptr, -1, 0,
        nullptr, 0, n_query, dim, n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist, scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes,
        scratch_thread_touched);
}

template <typename Scalar, typename Evaluator>
inline long long cluster_based_v1_paged_with_metadata(
    const Scalar* h_pages,
    const uint8_t* h_alive,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_cluster_page_offsets,
    const int* h_cluster_page_counts,
    const int* h_page_ids,
    int page_vecs,
    const Scalar* h_query,
    const int* h_cluster_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* h_numa_cluster_owner,
    int numa_cluster_owner_count,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched) {
    return cluster_based_v1_paged_impl<Scalar, Evaluator>(
        h_pages, h_alive, h_cluster_offsets, h_cluster_counts, h_cluster_page_offsets, h_cluster_page_counts,
        h_page_ids, page_vecs, h_query, nullptr, h_cluster_query_offsets, h_cluster_query_list,
        h_active_cluster_ids, active_cluster_count, 0, h_numa_cluster_owner, numa_cluster_owner_count,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist, scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes,
        scratch_thread_touched);
}

template <typename Scalar, typename Evaluator>
inline long long cluster_based_v1_paged_with_compact_metadata(
    const Scalar* h_pages,
    const uint8_t* h_alive,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_cluster_page_offsets,
    const int* h_cluster_page_counts,
    const int* h_page_ids,
    int page_vecs,
    const Scalar* h_query,
    const int* h_compact_query_offsets,
    const int* h_cluster_query_list,
    const int* h_active_cluster_ids,
    int active_cluster_count,
    const int* h_numa_cluster_owner,
    int numa_cluster_owner_count,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    float* scratch_thread_dists,
    int* scratch_thread_idxs,
    int* scratch_thread_sizes,
    uint8_t* scratch_thread_touched) {
    return cluster_based_v1_paged_impl<Scalar, Evaluator>(
        h_pages, h_alive, h_cluster_offsets, h_cluster_counts, h_cluster_page_offsets, h_cluster_page_counts,
        h_page_ids, page_vecs, h_query, nullptr, h_compact_query_offsets, h_cluster_query_list,
        h_active_cluster_ids, active_cluster_count, 1, h_numa_cluster_owner, numa_cluster_owner_count,
        n_query, dim, n_total_clusters, n_probes, topk, distance_mode, num_threads,
        h_topk_local_idx, h_topk_dist, scratch_thread_dists, scratch_thread_idxs, scratch_thread_sizes,
        scratch_thread_touched);
}

} // namespace cluster_based_v1_paged_detail
} // namespace cpu_fine
} // namespace paraselect

#endif // PARASELECT_CLUSTER_BASED_V1_PAGED_H
