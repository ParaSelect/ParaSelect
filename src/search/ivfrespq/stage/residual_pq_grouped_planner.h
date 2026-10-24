#ifndef PARASELECT_SEARCH_IVFRESPQ_STAGE_RESIDUAL_PQ_GROUPED_PLANNER_H
#define PARASELECT_SEARCH_IVFRESPQ_STAGE_RESIDUAL_PQ_GROUPED_PLANNER_H

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace paraselect::search {

struct ResidualPQGroupedWorkspaceLimits {
    int chunk_capacity = 0;
    int group_capacity = 0;
    int merge_a_item_capacity = 0;
    int merge_b_item_capacity = 0;
};

inline int choose_residual_pq_grouped_query_tile(
    const int* query_chunks,
    int n_query,
    int query_begin,
    int group_chunks,
    int global_k,
    const ResidualPQGroupedWorkspaceLimits& limits) {
    if (!query_chunks || n_query <= 0 || query_begin < 0 || query_begin >= n_query ||
        group_chunks <= 0 || global_k <= 0) {
        throw std::invalid_argument("invalid residual PQ grouped planner input");
    }

    constexpr int kMaxMergeRounds = 64;
    std::array<long long, kMaxMergeRounds> round_group_sums{};
    long long total_chunks = 0;
    long long max_groups = 0;
    long long max_a_items = 0;
    long long max_b_items = 0;

    auto add_query = [&](int chunks) {
        const long long query_chunk_count = std::max(0, chunks);
        total_chunks += query_chunk_count;
        long long groups =
            (query_chunk_count + group_chunks - 1) / group_chunks;
        for (int round = 0; round < kMaxMergeRounds && groups > 0; ++round) {
            round_group_sums[round] += groups;
            const long long round_sum = round_group_sums[round];
            max_groups = std::max(max_groups, round_sum);
            if ((round & 1) == 0) {
                max_a_items = std::max(max_a_items, round_sum * global_k);
            } else {
                max_b_items = std::max(max_b_items, round_sum * global_k);
            }
            const long long next = (groups + group_chunks - 1) / group_chunks;
            if (next == groups) break;
            groups = next;
        }
    };

    auto fits = [&] {
        const long long int_max = std::numeric_limits<int>::max();
        return total_chunks <= int_max && max_groups <= int_max &&
               max_a_items <= int_max && max_b_items <= int_max &&
               total_chunks <= limits.chunk_capacity &&
               max_groups <= limits.group_capacity &&
               max_a_items <= limits.merge_a_item_capacity &&
               max_b_items <= limits.merge_b_item_capacity;
    };

    add_query(query_chunks[query_begin]);
    int count = 1;
    while (query_begin + count < n_query) {
        add_query(query_chunks[query_begin + count]);
        if (!fits()) break;
        ++count;
    }
    return count;
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFRESPQ_STAGE_RESIDUAL_PQ_GROUPED_PLANNER_H
