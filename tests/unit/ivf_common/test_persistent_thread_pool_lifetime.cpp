#include "cpu/utils/persistent_thread_pool.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

namespace {

std::size_t task_count() {
    return static_cast<std::size_t>(std::distance(
        std::filesystem::directory_iterator("/proc/self/task"),
        std::filesystem::directory_iterator{}));
}

}  // namespace

int main() {
    constexpr int kPoolThreads = 4;
    constexpr int kOuterThreads = 20;
    const std::size_t baseline = task_count();

    for (int iteration = 0; iteration < kOuterThreads; ++iteration) {
        std::atomic<int> completed{0};
        std::thread outer([&]() {
            auto& pool = paraselect::persistent_thread_pool_for_threads(kPoolThreads);
            pool.run([&](int) { completed.fetch_add(1, std::memory_order_relaxed); });
        });
        outer.join();
        if (completed.load(std::memory_order_relaxed) != kPoolThreads) {
            std::cerr << "thread-pool job did not run on every worker\n";
            return 1;
        }
    }

    for (int attempt = 0; attempt < 20 && task_count() != baseline; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const std::size_t final_count = task_count();
    if (final_count != baseline) {
        std::cerr << "worker threads outlived their thread-local owner: baseline="
                  << baseline << " final=" << final_count << '\n';
        return 1;
    }
    return 0;
}
