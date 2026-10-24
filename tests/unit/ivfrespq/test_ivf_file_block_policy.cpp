#include "ivf_file_block_policy_eval.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

using paraselect::tests::ivf_file_policy::Candidate;
using paraselect::tests::ivf_file_policy::Result;

bool near(double actual, double expected, double tolerance = 1e-12) {
    if (std::abs(actual - expected) <= tolerance) return true;
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    return false;
}

const Result& find_result(
    const std::vector<Result>& results,
    int threshold,
    int rank,
    uint64_t cached_block_limit) {
    for (const Result& result : results) {
        if (result.params.selection.sparse_ref_threshold == threshold &&
            result.params.selection.protected_candidate_rank == rank &&
            result.params.cached_block_limit == cached_block_limit) {
            return result;
        }
    }
    throw std::runtime_error("missing IVF file block policy scan point");
}

}  // namespace

int main() {
    const std::vector<Candidate> candidates = {
        {0, 0, 0, 0},
        {0, 1, 1, -1},
        {0, 2, 2, 1},
        {0, 3, 3, -1},
        {1, 0, 0, 0},
        {1, 1, 1, -1},
        {1, 4, 2, -1},
        {1, 5, 3, 1},
    };
    const std::vector<int> thresholds = {1, 2};
    const std::vector<int> protected_ranks = {1, 3, 4};
    const std::vector<uint64_t> cached_block_limits = {0, 3};
    const std::vector<Result> results = paraselect::tests::ivf_file_policy::scan(
        candidates, 2, 2, 2, thresholds, protected_ranks, cached_block_limits);

    if (results.size() != 12) return 1;
    for (const Result& result : results) {
        if (!near(result.baseline_recall, 1.0)) return 1;
    }

    const Result& dense_only = find_result(results, 1, 1, 0);
    if (!near(dense_only.filtered_recall, 0.5) ||
        !near(dense_only.retained_block_rate, 2.0 / 6.0) ||
        !near(dense_only.retained_candidate_rate, 0.5)) {
        return 1;
    }

    const Result& rank_three = find_result(results, 1, 3, 0);
    if (!near(rank_three.filtered_recall, 0.75) ||
        !near(rank_three.retained_block_rate, 4.0 / 6.0) ||
        !near(rank_three.retained_candidate_rate, 0.75)) {
        return 1;
    }

    const Result& strict = find_result(results, 2, 1, 0);
    if (!near(strict.filtered_recall, 0.5) ||
        !near(strict.retained_block_rate, 1.0 / 6.0) ||
        !near(strict.retained_candidate_rate, 0.25)) {
        return 1;
    }

    const Result& cached = find_result(results, 2, 1, 3);
    if (!near(cached.filtered_recall, 0.75) ||
        !near(cached.cache_block_rate, 0.5) ||
        !near(cached.retained_block_rate, 0.5) ||
        !near(cached.retained_candidate_rate, 5.0 / 8.0)) {
        return 1;
    }

    const Result& all_ranked = find_result(results, 2, 4, 0);
    if (!near(all_ranked.filtered_recall, 1.0) ||
        !near(all_ranked.retained_block_rate, 1.0) ||
        !near(all_ranked.retained_candidate_rate, 1.0)) {
        return 1;
    }

    for (uint64_t cache : cached_block_limits) {
        for (int threshold : thresholds) {
            for (size_t rank = 1; rank < protected_ranks.size(); ++rank) {
                const Result& previous =
                    find_result(results, threshold, protected_ranks[rank - 1], cache);
                const Result& current =
                    find_result(results, threshold, protected_ranks[rank], cache);
                if (current.filtered_recall < previous.filtered_recall ||
                    current.retained_block_rate < previous.retained_block_rate) {
                    return 1;
                }
            }
        }
    }
    for (uint64_t cache : cached_block_limits) {
        for (int rank : protected_ranks) {
            const Result& loose = find_result(results, 1, rank, cache);
            const Result& strict_threshold = find_result(results, 2, rank, cache);
            if (strict_threshold.filtered_recall > loose.filtered_recall ||
                strict_threshold.retained_block_rate > loose.retained_block_rate) {
                return 1;
            }
        }
    }
    for (int threshold : thresholds) {
        for (int rank : protected_ranks) {
            const Result& cold = find_result(results, threshold, rank, 0);
            const Result& warm = find_result(results, threshold, rank, 3);
            if (warm.filtered_recall < cold.filtered_recall ||
                warm.retained_block_rate < cold.retained_block_rate) {
                return 1;
            }
        }
    }
    return 0;
}
