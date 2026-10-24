#include "search/ivfrespq/stage/residual_pq_grouped_planner.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

long long ceil_div(long long value, long long divisor) {
    return value <= 0 ? 0 : (value + divisor - 1) / divisor;
}

int brute_choose(
    const std::vector<int>& chunks,
    int begin,
    int group_chunks,
    int global_k,
    const paraselect::search::ResidualPQGroupedWorkspaceLimits& limits) {
    int count = 1;
    while (begin + count < static_cast<int>(chunks.size())) {
        const int candidate_count = count + 1;
        long long total_chunks = 0;
        std::array<long long, 64> rounds{};
        for (int i = 0; i < candidate_count; ++i) {
            const long long query_chunks = std::max(0, chunks[begin + i]);
            total_chunks += query_chunks;
            long long groups = ceil_div(query_chunks, group_chunks);
            for (int round = 0; round < 64 && groups > 0; ++round) {
                rounds[round] += groups;
                const long long next = ceil_div(groups, group_chunks);
                if (next == groups) break;
                groups = next;
            }
        }
        long long max_groups = 0;
        long long max_a = 0;
        long long max_b = 0;
        for (int round = 0; round < 64 && rounds[round] > 0; ++round) {
            max_groups = std::max(max_groups, rounds[round]);
            if ((round & 1) == 0) {
                max_a = std::max(max_a, rounds[round] * global_k);
            } else {
                max_b = std::max(max_b, rounds[round] * global_k);
            }
        }
        if (total_chunks > limits.chunk_capacity || max_groups > limits.group_capacity ||
            max_a > limits.merge_a_item_capacity || max_b > limits.merge_b_item_capacity ||
            total_chunks > std::numeric_limits<int>::max() ||
            max_groups > std::numeric_limits<int>::max() ||
            max_a > std::numeric_limits<int>::max() ||
            max_b > std::numeric_limits<int>::max()) {
            break;
        }
        count = candidate_count;
    }
    return count;
}

}  // namespace

int main() {
    using paraselect::search::ResidualPQGroupedWorkspaceLimits;
    using paraselect::search::choose_residual_pq_grouped_query_tile;

    const ResidualPQGroupedWorkspaceLimits roomy{
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()};
    const std::vector<int> zeros(8192, 0);
    if (choose_residual_pq_grouped_query_tile(
            zeros.data(), zeros.size(), 0, 8, 512, roomy) != 8192) {
        std::cerr << "zero-work queries should share one tile\n";
        return 1;
    }

    std::mt19937 rng(20260914);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 2048);
        std::vector<int> chunks(n);
        for (int& value : chunks) value = static_cast<int>(rng() % 4096);
        const int begin = static_cast<int>(rng() % n);
        const int group_chunks = 2 + static_cast<int>(rng() % 15);
        const int global_k = (rng() & 1U) ? 64 : 512;
        const ResidualPQGroupedWorkspaceLimits limits{
            1 + static_cast<int>(rng() % 2000000),
            1 + static_cast<int>(rng() % 300000),
            1 + static_cast<int>(rng() % 200000000),
            1 + static_cast<int>(rng() % 200000000)};
        const int expected = brute_choose(chunks, begin, group_chunks, global_k, limits);
        const int actual = choose_residual_pq_grouped_query_tile(
            chunks.data(), n, begin, group_chunks, global_k, limits);
        if (actual != expected) {
            std::cerr << "planner mismatch trial=" << trial
                      << " expected=" << expected << " actual=" << actual << '\n';
            return 1;
        }
    }
    return 0;
}
