#ifndef PARASELECT_UTILS_ADAPTIVE_THREAD_COUNT_H
#define PARASELECT_UTILS_ADAPTIVE_THREAD_COUNT_H

#include <algorithm>
#include <cstdlib>

namespace paraselect {

inline int parse_env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) return fallback;
    if (parsed < 0) return fallback;
    return static_cast<int>(parsed);
}

inline bool adaptive_thread_count_enabled() {
    const char* value = std::getenv("PARASELECT_ADAPTIVE_THREADS");
    return !value || value[0] == '\0' || std::atoi(value) != 0;
}

inline int resolve_adaptive_thread_count(
    int max_threads,
    int task_count,
    int thread_group_size = 1)
{
    max_threads = std::max(1, max_threads);
    thread_group_size = std::max(1, std::min(thread_group_size, max_threads));
    if (!adaptive_thread_count_enabled()) return max_threads;

    const int override_threads = parse_env_int_or_default("PARASELECT_ADAPTIVE_THREADS_OVERRIDE", 0);
    if (override_threads > 0) {
        int selected = std::max(thread_group_size, std::min(max_threads, override_threads));
        if (thread_group_size > 1) {
            selected = ((selected + thread_group_size - 1) / thread_group_size) * thread_group_size;
            selected = std::min(max_threads, selected);
        }
        return selected;
    }

    if (task_count <= 0 || task_count >= max_threads) return max_threads;

    int selected = std::max(thread_group_size, task_count);
    if (thread_group_size > 1) {
        selected = ((selected + thread_group_size - 1) / thread_group_size) * thread_group_size;
    }
    return std::min(max_threads, selected);
}

}  // namespace paraselect

#endif  // PARASELECT_UTILS_ADAPTIVE_THREAD_COUNT_H
