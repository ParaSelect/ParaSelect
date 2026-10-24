#include "cpu/utils/cpu_affinity.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

bool contains_cpu(const std::vector<int>& cpus, int cpu) {
    return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
}

}  // namespace

int main() {
    const int nodes = paraselect::get_numa_node_count();
    const auto node_physical_cpus = paraselect::get_numa_node_physical_cpu_ids_all();

    std::printf("[numa_smoke] nodes=%d\n", nodes);
    if (nodes <= 0 || node_physical_cpus.empty()) {
        std::fprintf(stderr, "[numa_smoke] ERROR: no NUMA topology found\n");
        return 1;
    }

    for (int node = 0; node < nodes; ++node) {
        const auto& cpus = node_physical_cpus[(size_t)node];
        std::printf("[numa_smoke] node=%d physical_cpus=%zu list=", node, cpus.size());
        for (size_t i = 0; i < cpus.size(); ++i) {
            std::printf("%s%d", i == 0 ? "" : ",", cpus[i]);
        }
        std::printf("\n");
        if (cpus.empty()) {
            std::fprintf(stderr, "[numa_smoke] ERROR: node %d has no physical CPU list\n", node);
            return 1;
        }
    }

    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    for (int node = 0; node < nodes; ++node) {
        const int checks = std::min<int>(32, node_physical_cpus[(size_t)node].size());
        for (int local = 0; local < checks; ++local) {
            workers.emplace_back([&, node, local]() {
                paraselect::pin_thread_to_numa_core(node, local);
#if defined(__linux__)
                const int cpu = sched_getcpu();
#else
                const int cpu = -1;
#endif
                const bool ok = contains_cpu(node_physical_cpus[(size_t)node], cpu);
                std::printf(
                    "[numa_smoke] target_node=%d local=%d running_cpu=%d ok=%d\n",
                    node, local, cpu, ok ? 1 : 0);
                if (!ok) failures.fetch_add(1, std::memory_order_relaxed);
            });
        }
    }

    for (auto& worker : workers) worker.join();

    if (failures.load(std::memory_order_relaxed) != 0) {
        std::fprintf(stderr, "[numa_smoke] ERROR: affinity failures=%d\n", failures.load());
        return 1;
    }
    std::printf("[numa_smoke] PASS\n");
    return 0;
}
