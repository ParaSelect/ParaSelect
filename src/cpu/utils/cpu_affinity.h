#pragma once

#include <thread>
#include <vector>
#include <atomic>
#include <tuple>
#include <functional>
#include <algorithm>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

namespace paraselect {

inline int get_numa_node_count();

/**
 * ()
 *
 *  /proc/cpuinfo  core id ,
 *  std::thread::hardware_concurrency() / 2.
 */
inline int get_physical_core_count() {
#if defined(__linux__)
    FILE* fp = std::fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        int processor = -1;
        int core_id = -1;
        int physical_id = -1;
        std::vector<std::pair<int,int>> cores; // (physical_id, core_id)
        while (std::fgets(line, sizeof(line), fp)) {
            if (std::strncmp(line, "processor", 9) == 0) {
                char* p = std::strchr(line, ':');
                if (p) processor = std::atoi(p + 1);
            } else if (std::strncmp(line, "core id", 7) == 0) {
                char* p = std::strchr(line, ':');
                if (p) core_id = std::atoi(p + 1);
            } else if (std::strncmp(line, "physical id", 11) == 0) {
                char* p = std::strchr(line, ':');
                if (p) physical_id = std::atoi(p + 1);
            } else if (line[0] == '\n' && processor >= 0) {
                if (core_id >= 0 && physical_id >= 0) {
                    bool exists = false;
                    for (const auto& c : cores) {
                        if (c.first == physical_id && c.second == core_id) {
                            exists = true; break;
                        }
                    }
                    if (!exists) cores.emplace_back(physical_id, core_id);
                }
                processor = -1;
                core_id = -1;
                physical_id = -1;
            }
        }
        std::fclose(fp);
        if (!cores.empty()) return static_cast<int>(cores.size());
    }
#endif
    unsigned int logical_cores = std::thread::hardware_concurrency();
    if (logical_cores == 0) logical_cores = 4u;
    return static_cast<int>(logical_cores);
}

inline int parse_positive_env_int(const char* name) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) return 0;
    return static_cast<int>(parsed);
}

inline int default_worker_thread_count() {
    const int physical = get_physical_core_count();
    if (physical > 4) return std::min(64, physical - 4);
    if (physical > 0) return physical;
    const unsigned int logical = std::thread::hardware_concurrency();
    if (logical > 4) return static_cast<int>(logical - 4);
    if (logical > 0) return static_cast<int>(logical);
    return 1;
}

inline int resolve_cpu_fine_thread_count() {
    const int explicit_threads = parse_positive_env_int("PARASELECT_CPU_FINE_THREADS");
    if (explicit_threads > 0) return explicit_threads;

    const int omp_threads = parse_positive_env_int("OMP_NUM_THREADS");
    if (omp_threads > 0) return omp_threads;

    static const int default_threads = default_worker_thread_count();
    return default_threads;
}

/**
 *  ID()
 *  /proc/cpuinfo  (physical_id, core_id) → processor
 */
inline std::vector<int> get_physical_core_logical_ids() {
    std::vector<int> result;
#if defined(__linux__)
    FILE* fp = std::fopen("/proc/cpuinfo", "r");
    if (!fp) return result;

    char line[256];
    int processor = -1;
    int physical_id = -1;
    int core_id = -1;

    // key: physical_id * 1000 + core_id, value: first processor id
    std::vector<std::pair<int, int>> core_map;  // (uid, processor)

    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "processor", 9) == 0) {
            char* p = std::strchr(line, ':');
            if (p) processor = std::atoi(p + 1);
        } else if (std::strncmp(line, "physical id", 11) == 0) {
            char* p = std::strchr(line, ':');
            if (p) physical_id = std::atoi(p + 1);
        } else if (std::strncmp(line, "core id", 7) == 0) {
            char* p = std::strchr(line, ':');
            if (p) core_id = std::atoi(p + 1);
        } else if (line[0] == '\n' && processor >= 0 && physical_id >= 0 && core_id >= 0) {
            int uid = physical_id * 1000 + core_id;
            //
            bool exists = false;
            for (const auto& kv : core_map) {
                if (kv.first == uid) { exists = true; break; }
            }
            if (!exists) {
                core_map.emplace_back(uid, processor);
            }
            processor = -1;
            physical_id = -1;
            core_id = -1;
        }
    }
    std::fclose(fp);

    //  uid ,
    std::sort(core_map.begin(), core_map.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& kv : core_map) {
        result.push_back(kv.second);
    }
#endif
    return result;
}


inline void append_cpu_list_range(std::vector<int>* out, int begin, int end) {
    if (!out || begin < 0 || end < begin) return;
    for (int cpu = begin; cpu <= end; ++cpu) {
        out->push_back(cpu);
    }
}

inline std::vector<int> parse_cpu_list(const char* text) {
    std::vector<int> cpus;
    if (!text) return cpus;

    const char* p = text;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') ++p;
        if (!*p) break;

        char* end = nullptr;
        long begin = std::strtol(p, &end, 10);
        if (end == p || begin < 0) break;
        p = end;

        long last = begin;
        if (*p == '-') {
            ++p;
            last = std::strtol(p, &end, 10);
            if (end == p || last < begin) break;
            p = end;
        }
        append_cpu_list_range(&cpus, static_cast<int>(begin), static_cast<int>(last));

        while (*p && *p != ',') ++p;
    }

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

inline std::vector<int> read_numa_node_cpu_ids(int node_id) {
    std::vector<int> cpus;
#if defined(__linux__)
    char path[256];
    std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);
    FILE* fp = std::fopen(path, "r");
    if (!fp) return cpus;

    char buf[1024];
    if (std::fgets(buf, sizeof(buf), fp)) {
        cpus = parse_cpu_list(buf);
    }
    std::fclose(fp);
#else
    (void)node_id;
#endif
    return cpus;
}

inline std::vector<int> get_numa_node_physical_cpu_ids(int node_id) {
    std::vector<int> node_cpus = read_numa_node_cpu_ids(node_id);
    if (node_cpus.empty()) return node_cpus;

    const std::set<int> node_cpu_set(node_cpus.begin(), node_cpus.end());
    std::vector<int> physical_cpus;
#if defined(__linux__)
    FILE* fp = std::fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        int processor = -1;
        int physical_id = -1;
        int core_id = -1;
        std::set<std::pair<int, int>> seen_cores;

        auto maybe_add_cpu = [&]() {
            if (processor < 0 || physical_id < 0 || core_id < 0) return;
            if (node_cpu_set.find(processor) == node_cpu_set.end()) return;
            const std::pair<int, int> key(physical_id, core_id);
            if (seen_cores.insert(key).second) {
                physical_cpus.push_back(processor);
            }
        };

        while (std::fgets(line, sizeof(line), fp)) {
            if (std::strncmp(line, "processor", 9) == 0) {
                char* p = std::strchr(line, ':');
                if (p) processor = std::atoi(p + 1);
            } else if (std::strncmp(line, "physical id", 11) == 0) {
                char* p = std::strchr(line, ':');
                if (p) physical_id = std::atoi(p + 1);
            } else if (std::strncmp(line, "core id", 7) == 0) {
                char* p = std::strchr(line, ':');
                if (p) core_id = std::atoi(p + 1);
            } else if (line[0] == '\n') {
                maybe_add_cpu();
                processor = -1;
                physical_id = -1;
                core_id = -1;
            }
        }
        maybe_add_cpu();
        std::fclose(fp);
    }
#endif

    if (physical_cpus.empty()) {
        physical_cpus = node_cpus;
    }
    std::sort(physical_cpus.begin(), physical_cpus.end());
    physical_cpus.erase(std::unique(physical_cpus.begin(), physical_cpus.end()), physical_cpus.end());
    return physical_cpus;
}

inline std::vector<std::vector<int>> get_numa_node_physical_cpu_ids_all() {
    const int nodes = get_numa_node_count();
    std::vector<std::vector<int>> per_node((size_t)std::max(0, nodes));
    for (int node = 0; node < nodes; ++node) {
        per_node[(size_t)node] = get_numa_node_physical_cpu_ids(node);
    }
    return per_node;
}

/**
 *  CPU
 * @param core_id  ID
 */
inline void pin_thread_to_core(int core_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)core_id;
#endif
}

/**
 * (,)
 * @param thread_idx (0 ~ n_logical_cores-1)
 * @deprecated  pin_thread_to_physical_core
 */
inline void pin_thread_to_logical_core(int thread_idx) {
    static const std::vector<int> physical_ids = get_physical_core_logical_ids();
    if (thread_idx >= 0 && thread_idx < static_cast<int>(physical_ids.size())) {
        pin_thread_to_core(physical_ids[thread_idx]);
    }
}

/**
 * (,)
 * @param thread_idx (0 ~ n_physical_cores-1)
 */
inline void pin_thread_to_physical_core(int thread_idx) {
    static const std::vector<int> physical_ids = get_physical_core_logical_ids();
    if (thread_idx >= 0 && thread_idx < static_cast<int>(physical_ids.size())) {
        pin_thread_to_core(physical_ids[thread_idx]);
    }
}

/**
 *  CPU
 * @param func
 * @param core_id  ID
 * @param args
 * @return std::thread
 */
// :create_pinned_thread  C++17 std::apply,
// , C++17
/*
template<typename Func, typename... Args>
std::thread create_pinned_thread(int core_id, Func&& func, Args&&... args) {
    return std::thread([core_id, func, args...]() mutable {
        pin_thread_to_core(core_id);
        func(args...);
    });
}
*/

/**
 * ,()
 * @param n_tasks
 * @param func , void(int task_id, int core_id)
 * @deprecated  parallel_on_physical_cores
 */
template<typename Func>
void parallel_on_logical_cores(int n_tasks, Func&& func) {
    if (n_tasks <= 0) return;

    int n_cores = get_physical_core_count();
    int n_threads = std::min(n_tasks, n_cores);

    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    std::atomic<int> next_task{0};

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            pin_thread_to_physical_core(t);
            while (true) {
                int task_id = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_id >= n_tasks) break;
                func(task_id, t);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

/**
 * ,(,)
 * @param n_tasks
 * @param func , void(int task_id, int core_id)
 */
template<typename Func>
void parallel_on_physical_cores(int n_tasks, Func&& func) {
    if (n_tasks <= 0) return;

    int n_cores = get_physical_core_count();
    int n_threads = std::min(n_tasks, n_cores);

    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    std::atomic<int> next_task{0};

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            pin_thread_to_physical_core(t);
            while (true) {
                int task_id = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_id >= n_tasks) break;
                func(task_id, t);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

/**
 * ,()
 * @param n_items
 * @param func , void(int start, int end, int core_id)
 * @deprecated  parallel_chunk_on_physical_cores
 */
template<typename Func>
void parallel_chunk_on_logical_cores(int n_items, Func&& func) {
    if (n_items <= 0) return;

    int n_cores = get_physical_core_count();
    int n_threads = std::min(n_items, n_cores);

    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    int chunk_size = (n_items + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(n_items, start + chunk_size);
        if (start >= end) break;

        threads.emplace_back([=, &func]() {
            pin_thread_to_physical_core(t);
            func(start, end, t);
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

/**
 * ,(,)
 * @param n_items
 * @param func , void(int start, int end, int core_id)
 */
template<typename Func>
void parallel_chunk_on_physical_cores(int n_items, Func&& func) {
    if (n_items <= 0) return;

    int n_cores = get_physical_core_count();
    int n_threads = std::min(n_items, n_cores);

    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    int chunk_size = (n_items + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(n_items, start + chunk_size);
        if (start >= end) break;

        threads.emplace_back([=, &func]() {
            pin_thread_to_physical_core(t);
            func(start, end, t);
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

// ---------------------------------------------------------------------------
// NUMA
// ---------------------------------------------------------------------------

/**
 *  NUMA
 */
inline int get_numa_node_count() {
#if defined(__linux__)
    //  /sys  NUMA
    FILE* fp = fopen("/sys/devices/system/node/online", "r");
    if (fp) {
        char buf[256];
        if (fgets(buf, sizeof(buf), fp)) {
            //  "0-1"  2
            int start, end;
            if (sscanf(buf, "%d-%d", &start, &end) == 2) {
                fclose(fp);
                return end - start + 1;
            }
        }
        fclose(fp);
    }
#endif
    return 1;  //  NUMA
}

/**
 *  NUMA  ID
 *  NUMA
 */
inline int get_numa_node_first_core(int node_id) {
#if defined(__linux__)
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);
    FILE* fp = fopen(path, "r");
    if (fp) {
        char buf[256];
        if (fgets(buf, sizeof(buf), fp)) {
            int first_core;
            if (sscanf(buf, "%d", &first_core) == 1) {
                fclose(fp);
                return first_core;
            }
        }
        fclose(fp);
    }
#endif
    return node_id * 20;  //  20
}

/**
 *  NUMA
 * @param node_id NUMA  ID
 * @param core_in_node
 */
inline void pin_thread_to_numa_core(int node_id, int core_in_node) {
    static const std::vector<std::vector<int>> numa_physical_ids =
        get_numa_node_physical_cpu_ids_all();
    if (node_id >= 0 && node_id < static_cast<int>(numa_physical_ids.size())) {
        const std::vector<int>& cpus = numa_physical_ids[(size_t)node_id];
        if (!cpus.empty()) {
            const int pos = core_in_node >= 0
                ? core_in_node % static_cast<int>(cpus.size())
                : 0;
            pin_thread_to_core(cpus[(size_t)pos]);
            return;
        }
    }
    const int core_id = get_numa_node_first_core(node_id) + std::max(0, core_in_node);
    pin_thread_to_core(core_id);
}

} // namespace paraselect
