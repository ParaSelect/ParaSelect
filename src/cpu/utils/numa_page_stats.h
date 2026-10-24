#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "cpu/utils/cpu_affinity.h"

namespace paraselect {
namespace numa {

inline bool page_stats_enabled() {
    const char* v = std::getenv("PARASELECT_NUMA_PAGE_STATS");
    return v && v[0] && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0;
}

inline bool find_vma_start_for_address(uintptr_t addr, uintptr_t* out_start) {
#if defined(__linux__)
    FILE* fp = std::fopen("/proc/self/maps", "r");
    if (!fp) return false;
    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        if (std::sscanf(line, "%llx-%llx", &start, &end) == 2) {
            if (addr >= static_cast<uintptr_t>(start) && addr < static_cast<uintptr_t>(end)) {
                std::fclose(fp);
                *out_start = static_cast<uintptr_t>(start);
                return true;
            }
        }
    }
    std::fclose(fp);
#else
    (void)addr;
    (void)out_start;
#endif
    return false;
}

inline bool print_page_distribution_from_numa_maps(const char* label, const void* ptr, size_t bytes) {
#if defined(__linux__)
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t target_vma_start = 0;
    if (!find_vma_start_for_address(begin, &target_vma_start)) return false;

    FILE* fp = std::fopen("/proc/self/numa_maps", "r");
    if (!fp) return false;

    const int nodes = std::max(1, get_numa_node_count());
    std::vector<unsigned long long> counts((size_t)nodes, 0);
    unsigned long long total = 0;
    bool found = false;

    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        char* save = nullptr;
        char* token = strtok_r(line, " \t\n", &save);
        if (!token) continue;
        char* parse_end = nullptr;
        const uintptr_t vma_start = static_cast<uintptr_t>(std::strtoull(token, &parse_end, 16));
        if (parse_end == token || vma_start != target_vma_start) continue;

        while ((token = strtok_r(nullptr, " \t\n", &save)) != nullptr) {
            if (token[0] != 'N') continue;
            char* eq = std::strchr(token, '=');
            if (!eq) continue;
            *eq = '\0';
            const int node = std::atoi(token + 1);
            const unsigned long long pages = std::strtoull(eq + 1, nullptr, 10);
            if (node >= 0 && node < nodes) {
                counts[(size_t)node] += pages;
                total += pages;
            }
        }
        found = total > 0;
        break;
    }
    std::fclose(fp);

    if (!found) return false;
    std::fprintf(stderr, "[numa_page_stats] %s numa_maps_vma=0x%llx pages=%llu bytes=%zu",
                 label ? label : "buffer",
                 static_cast<unsigned long long>(target_vma_start), total, bytes);
    for (int node = 0; node < nodes; ++node) {
        std::fprintf(stderr, " node%d=%llu", node, counts[(size_t)node]);
    }
    std::fprintf(stderr, "\n");
    return true;
#else
    (void)label;
    (void)ptr;
    (void)bytes;
    return false;
#endif
}

inline void print_page_distribution(const char* label, const void* ptr, size_t bytes, size_t max_pages = 4096) {
    if (!page_stats_enabled() || !ptr || bytes == 0) return;
#if defined(__linux__) && defined(SYS_move_pages)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t page_begin = begin & ~(static_cast<uintptr_t>(page_size) - 1u);
    const uintptr_t end = begin + bytes;
    const size_t total_pages = (end <= page_begin) ? 0 : (size_t)((end - page_begin + (uintptr_t)page_size - 1u) / (uintptr_t)page_size);
    if (total_pages == 0) return;

    const size_t sample_pages = std::min(total_pages, std::max<size_t>(1, max_pages));
    std::vector<void*> pages(sample_pages);
    std::vector<int> status(sample_pages, -1);
    for (size_t i = 0; i < sample_pages; ++i) {
        const size_t page_idx = (sample_pages == total_pages)
            ? i
            : (i * total_pages) / sample_pages;
        pages[i] = reinterpret_cast<void*>(page_begin + (uintptr_t)page_idx * (uintptr_t)page_size);
    }

    long rc = syscall(SYS_move_pages, 0, (unsigned long)sample_pages, pages.data(), nullptr, status.data(), 0);
    if (rc != 0) {
        if (errno == EPERM || errno == EACCES) {
            if (print_page_distribution_from_numa_maps(label, ptr, bytes)) return;
        }
        std::fprintf(stderr, "[numa_page_stats] %s move_pages failed errno=%d %s\n",
                     label ? label : "buffer", errno, std::strerror(errno));
        return;
    }

    const int nodes = std::max(1, get_numa_node_count());
    std::vector<size_t> counts((size_t)nodes, 0);
    size_t unknown = 0;
    for (int s : status) {
        if (s >= 0 && s < nodes) counts[(size_t)s]++;
        else unknown++;
    }
    std::fprintf(stderr, "[numa_page_stats] %s sampled_pages=%zu total_pages=%zu",
                 label ? label : "buffer", sample_pages, total_pages);
    for (int node = 0; node < nodes; ++node) {
        std::fprintf(stderr, " node%d=%zu", node, counts[(size_t)node]);
    }
    if (unknown) std::fprintf(stderr, " unknown=%zu", unknown);
    std::fprintf(stderr, "\n");
#else
    (void)max_pages;
    if (!print_page_distribution_from_numa_maps(label, ptr, bytes)) {
        (void)label;
        (void)ptr;
        (void)bytes;
    }
#endif
}

}  // namespace numa
}  // namespace paraselect
