#ifndef PARASELECT_CPU_FINE_THREAD_STATS_H
#define PARASELECT_CPU_FINE_THREAD_STATS_H

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

namespace paraselect {
namespace cpu_fine {
namespace thread_stats_detail {



struct FineTimingSnapshot {
    double query_prepare_ms = 0.0;
    double metadata_build_ms = 0.0;
    double scratch_clear_ms = 0.0;
    double task_build_ms = 0.0;
    double scan_ms = 0.0;
    double merge_ms = 0.0;
    double runner_ms = 0.0;
    double id_map_ms = 0.0;
};

inline FineTimingSnapshot& latest_fine_timing_storage() {
    static FineTimingSnapshot timing;
    return timing;
}

static inline void clear_latest_fine_timing() {
    latest_fine_timing_storage() = FineTimingSnapshot{};
}

static inline void record_latest_fine_timing(const FineTimingSnapshot& timing) {
    latest_fine_timing_storage() = timing;
}

static inline FineTimingSnapshot get_latest_fine_timing() {
    return latest_fine_timing_storage();
}

struct NumaNodeStatSnapshot {
    int present = 0;
    int threads = 0;
    double wall_ms = 0.0;
    double sum_thread_ms = 0.0;
    double max_thread_ms = 0.0;
    long long tasks = 0;
    long long query_refs = 0;
    long long vectors = 0;
    long long fmas = 0;
};

static constexpr int kMaxNumaStatNodes = 8;

inline NumaNodeStatSnapshot* latest_numa_node_stats_storage() {
    static NumaNodeStatSnapshot stats[kMaxNumaStatNodes];
    return stats;
}

inline int& latest_numa_node_stats_count_storage() {
    static int count = 0;
    return count;
}

static inline void clear_latest_numa_node_stats() {
    std::memset(latest_numa_node_stats_storage(), 0,
                sizeof(NumaNodeStatSnapshot) * (size_t)kMaxNumaStatNodes);
    latest_numa_node_stats_count_storage() = 0;
}

static inline int get_latest_numa_node_stats(NumaNodeStatSnapshot* out, int max_nodes) {
    if (!out || max_nodes <= 0) return 0;
    const int count = std::min(latest_numa_node_stats_count_storage(), max_nodes);
    for (int i = 0; i < count; ++i) {
        out[i] = latest_numa_node_stats_storage()[i];
    }
    return count;
}

struct ThreadStat {
    double start_s = 0.0;
    double end_s = 0.0;
    long long tasks = 0;
    long long query_refs = 0;
    long long vectors = 0;
    long long fmas = 0;
    int node_id = -1;
};

static inline bool cpu_fine_thread_stats_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("PARASELECT_CPU_FINE_THREAD_STATS");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static inline double percentile_sorted(const std::vector<double>& vals, double p) {
    if (vals.empty()) return 0.0;
    const double clamped = std::max(0.0, std::min(1.0, p));
    const size_t idx = static_cast<size_t>(clamped * static_cast<double>(vals.size() - 1));
    return vals[idx];
}

static inline long long percentile_sorted_ll(const std::vector<long long>& vals, double p) {
    if (vals.empty()) return 0;
    const double clamped = std::max(0.0, std::min(1.0, p));
    const size_t idx = static_cast<size_t>(clamped * static_cast<double>(vals.size() - 1));
    return vals[idx];
}


static inline void record_numa_node_stats_summary(
    const char* tag,
    const std::vector<ThreadStat>& stats,
    bool print_stats)
{
    clear_latest_numa_node_stats();
    if (stats.empty()) return;
    (void)tag;
    int max_node = -1;
    for (const auto& s : stats) max_node = std::max(max_node, s.node_id);
    if (max_node < 0) return;

    struct NodeAgg {
        int threads = 0;
        long long tasks = 0;
        long long query_refs = 0;
        long long vectors = 0;
        long long fmas = 0;
        double min_start = 0.0;
        double max_end = 0.0;
        double sum_thread_ms = 0.0;
        double max_thread_ms = 0.0;
        bool seen = false;
    };
    std::vector<NodeAgg> nodes((size_t)max_node + 1u);
    for (const auto& s : stats) {
        if (s.node_id < 0) continue;
        auto& n = nodes[(size_t)s.node_id];
        n.threads += 1;
        n.tasks += s.tasks;
        n.query_refs += s.query_refs;
        n.vectors += s.vectors;
        n.fmas += s.fmas;
        const double thread_ms = (s.end_s - s.start_s) * 1.0e3;
        n.sum_thread_ms += thread_ms;
        n.max_thread_ms = std::max(n.max_thread_ms, thread_ms);
        if (!n.seen) {
            n.min_start = s.start_s;
            n.max_end = s.end_s;
            n.seen = true;
        } else {
            n.min_start = std::min(n.min_start, s.start_s);
            n.max_end = std::max(n.max_end, s.end_s);
        }
    }

    latest_numa_node_stats_count_storage() = std::min<int>((int)nodes.size(), kMaxNumaStatNodes);
    for (size_t node = 0; node < nodes.size() && node < (size_t)kMaxNumaStatNodes; ++node) {
        const auto& n = nodes[node];
        if (!n.seen) continue;
        auto& dst = latest_numa_node_stats_storage()[node];
        dst.present = 1;
        dst.threads = n.threads;
        dst.wall_ms = (n.max_end - n.min_start) * 1.0e3;
        dst.sum_thread_ms = n.sum_thread_ms;
        dst.max_thread_ms = n.max_thread_ms;
        dst.tasks = n.tasks;
        dst.query_refs = n.query_refs;
        dst.vectors = n.vectors;
        dst.fmas = n.fmas;
        if (print_stats) {
            std::fprintf(
                stderr,
                "[CpuFineNumaNodeStats] tag=%s node=%zu threads=%d wall_ms=%.3f "
                "sum_thread_ms=%.3f max_thread_ms=%.3f tasks=%lld qrefs=%lld vectors=%lld fmas=%lld\n",
                tag ? tag : "unknown",
                node,
                n.threads,
                dst.wall_ms,
                n.sum_thread_ms,
                n.max_thread_ms,
                n.tasks,
                n.query_refs,
                n.vectors,
                n.fmas);
        }
    }
}

static inline void print_numa_node_stats_summary(
    const char* tag,
    const std::vector<ThreadStat>& stats)
{
    record_numa_node_stats_summary(tag, stats, cpu_fine_thread_stats_enabled());
}

static inline void print_thread_stats_summary(
    const char* tag,
    const std::vector<ThreadStat>& stats)
{
    if (stats.empty()) {
        clear_latest_numa_node_stats();
        return;
    }
    bool has_numa_node = false;
    for (const auto& s : stats) {
        if (s.node_id >= 0) {
            has_numa_node = true;
            break;
        }
    }
    if (!has_numa_node) clear_latest_numa_node_stats();
    if (!cpu_fine_thread_stats_enabled()) return;

    double global_start = stats.front().start_s;
    double global_end = stats.front().end_s;
    for (const auto& s : stats) {
        global_start = std::min(global_start, s.start_s);
        global_end = std::max(global_end, s.end_s);
    }

    std::vector<double> duration_us;
    std::vector<double> tail_idle_us;
    std::vector<long long> tasks;
    std::vector<long long> query_refs;
    std::vector<long long> vectors;
    std::vector<long long> fmas;
    duration_us.reserve(stats.size());
    tail_idle_us.reserve(stats.size());
    tasks.reserve(stats.size());
    query_refs.reserve(stats.size());
    vectors.reserve(stats.size());
    fmas.reserve(stats.size());

    long long sum_tasks = 0;
    long long sum_query_refs = 0;
    long long sum_vectors = 0;
    long long sum_fmas = 0;
    for (const auto& s : stats) {
        const double dur = (s.end_s - s.start_s) * 1.0e6;
        const double tail = (global_end - s.end_s) * 1.0e6;
        duration_us.push_back(dur);
        tail_idle_us.push_back(tail);
        tasks.push_back(s.tasks);
        query_refs.push_back(s.query_refs);
        vectors.push_back(s.vectors);
        fmas.push_back(s.fmas);
        sum_tasks += s.tasks;
        sum_query_refs += s.query_refs;
        sum_vectors += s.vectors;
        sum_fmas += s.fmas;
    }

    std::sort(duration_us.begin(), duration_us.end());
    std::sort(tail_idle_us.begin(), tail_idle_us.end());
    std::sort(tasks.begin(), tasks.end());
    std::sort(query_refs.begin(), query_refs.end());
    std::sort(vectors.begin(), vectors.end());
    std::sort(fmas.begin(), fmas.end());

    std::fprintf(
        stderr,
        "[CpuFineThreadStats] tag=%s threads=%zu wall_ms=%.3f "
        "dur_us[min/p50/p90/max]=%.0f/%.0f/%.0f/%.0f "
        "tail_us[p50/p90/max]=%.0f/%.0f/%.0f "
        "tasks[min/p50/p90/max,sum]=%lld/%lld/%lld/%lld/%lld "
        "qrefs[min/p50/p90/max,sum]=%lld/%lld/%lld/%lld/%lld "
        "vectors[min/p50/p90/max,sum]=%lld/%lld/%lld/%lld/%lld "
        "fmas[min/p50/p90/max,sum]=%lld/%lld/%lld/%lld/%lld\n",
        tag ? tag : "unknown",
        stats.size(),
        (global_end - global_start) * 1.0e3,
        duration_us.front(),
        percentile_sorted(duration_us, 0.50),
        percentile_sorted(duration_us, 0.90),
        duration_us.back(),
        percentile_sorted(tail_idle_us, 0.50),
        percentile_sorted(tail_idle_us, 0.90),
        tail_idle_us.back(),
        tasks.front(),
        percentile_sorted_ll(tasks, 0.50),
        percentile_sorted_ll(tasks, 0.90),
        tasks.back(),
        sum_tasks,
        query_refs.front(),
        percentile_sorted_ll(query_refs, 0.50),
        percentile_sorted_ll(query_refs, 0.90),
        query_refs.back(),
        sum_query_refs,
        vectors.front(),
        percentile_sorted_ll(vectors, 0.50),
        percentile_sorted_ll(vectors, 0.90),
        vectors.back(),
        sum_vectors,
        fmas.front(),
        percentile_sorted_ll(fmas, 0.50),
        percentile_sorted_ll(fmas, 0.90),
        fmas.back(),
        sum_fmas);
}

}  // namespace thread_stats_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_THREAD_STATS_H
