#ifndef PARASELECT_TESTS_COMMON_IVF_FILE_BLOCK_POLICY_EVAL_H
#define PARASELECT_TESTS_COMMON_IVF_FILE_BLOCK_POLICY_EVAL_H

#include "search/ivf_common/file/ivf_file_block_selection.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace paraselect::tests::ivf_file_policy {

struct Candidate {
    int query = 0;
    uint64_t block = 0;
    int pq_rank = 0;
    int gt_rank = -1;
};

struct Params {
    paraselect::search::IVFFileBlockSelection selection;
    uint64_t cached_block_limit = 0;
};

struct Result {
    Params params;
    double baseline_recall = 0.0;
    double filtered_recall = 0.0;
    double cache_block_rate = 0.0;
    double retained_miss_block_rate = 0.0;
    double retained_block_rate = 0.0;
    double retained_candidate_rate = 0.0;
    long long total_blocks = 0;
    long long total_candidates = 0;
    long long cached_blocks = 0;
    long long miss_blocks = 0;
    long long retained_miss_blocks = 0;
    long long retained_blocks = 0;
    long long retained_candidates = 0;
};

namespace detail {

struct Block {
    uint64_t id = 0;
    int refs = 0;
    int best_pq_rank = std::numeric_limits<int>::max();
    std::vector<size_t> gt_slots;
};

struct PreparedCandidates {
    int query_count = 0;
    int recall_k = 0;
    long long baseline_hits = 0;
    long long total_blocks = 0;
    long long total_candidates = 0;
    std::vector<std::vector<Block>> batches;
};

inline PreparedCandidates prepare_candidates(
    const std::vector<Candidate>& candidates,
    int query_count,
    int batch_size,
    int recall_k) {
    if (query_count <= 0 || batch_size <= 0 || recall_k <= 0) {
        throw std::invalid_argument("invalid IVF file block policy evaluation shape");
    }
    for (const Candidate& candidate : candidates) {
        if (candidate.query < 0 || candidate.query >= query_count || candidate.pq_rank < 0 ||
            candidate.gt_rank < -1 || candidate.gt_rank >= recall_k) {
            throw std::invalid_argument("invalid IVF file block policy candidate");
        }
    }

    PreparedCandidates prepared;
    prepared.query_count = query_count;
    prepared.recall_k = recall_k;
    prepared.total_candidates = static_cast<long long>(candidates.size());
    std::vector<uint8_t> baseline_gt(static_cast<size_t>(query_count) * recall_k, 0);

    const int batch_count = (query_count + batch_size - 1) / batch_size;
    prepared.batches.resize(batch_count);
    std::vector<std::unordered_map<uint64_t, size_t>> block_positions(batch_count);
    for (const Candidate& candidate : candidates) {
        const int batch = candidate.query / batch_size;
        auto& positions = block_positions[batch];
        auto& blocks = prepared.batches[batch];
        const auto [position, inserted] = positions.emplace(candidate.block, blocks.size());
        if (inserted) blocks.push_back({candidate.block});
        Block& block = blocks[position->second];
        ++block.refs;
        block.best_pq_rank = std::min(block.best_pq_rank, candidate.pq_rank);
        if (candidate.gt_rank >= 0) {
            const size_t gt_slot =
                static_cast<size_t>(candidate.query) * recall_k + candidate.gt_rank;
            baseline_gt[gt_slot] = 1;
            block.gt_slots.push_back(gt_slot);
        }
    }

    for (auto& blocks : prepared.batches) {
        prepared.total_blocks += static_cast<long long>(blocks.size());
        for (Block& block : blocks) {
            std::sort(block.gt_slots.begin(), block.gt_slots.end());
            block.gt_slots.erase(
                std::unique(block.gt_slots.begin(), block.gt_slots.end()),
                block.gt_slots.end());
        }
    }
    prepared.baseline_hits =
        std::count(baseline_gt.begin(), baseline_gt.end(), static_cast<uint8_t>(1));
    return prepared;
}

inline Result evaluate_prepared(const PreparedCandidates& prepared, const Params& params) {
    Result result;
    result.params = params;
    result.total_blocks = prepared.total_blocks;
    result.total_candidates = prepared.total_candidates;
    std::vector<uint8_t> filtered_gt(
        static_cast<size_t>(prepared.query_count) * prepared.recall_k, 0);
    long long cached_blocks = 0;
    long long uncached_blocks = 0;
    long long retained_miss_blocks = 0;
    long long retained_blocks = 0;
    long long retained_candidates = 0;

    for (const auto& blocks : prepared.batches) {
        for (const Block& block : blocks) {
            const bool cached = block.id < params.cached_block_limit;
            const bool keep = paraselect::search::ivf_file_block_selected(
                params.selection,
                cached,
                static_cast<size_t>(block.refs),
                block.best_pq_rank);
            cached_blocks += cached ? 1 : 0;
            uncached_blocks += cached ? 0 : 1;
            retained_blocks += keep ? 1 : 0;
            retained_miss_blocks += keep && !cached ? 1 : 0;
            retained_candidates += keep ? block.refs : 0;
            if (keep) {
                for (size_t gt_slot : block.gt_slots) filtered_gt[gt_slot] = 1;
            }
        }
    }

    const double recall_denominator =
        static_cast<double>(prepared.query_count) * prepared.recall_k;
    result.baseline_recall = prepared.baseline_hits / recall_denominator;
    result.filtered_recall =
        std::count(filtered_gt.begin(), filtered_gt.end(), static_cast<uint8_t>(1)) /
        recall_denominator;
    result.cache_block_rate = result.total_blocks > 0
        ? static_cast<double>(cached_blocks) / result.total_blocks
        : 0.0;
    result.retained_miss_block_rate = uncached_blocks > 0
        ? static_cast<double>(retained_miss_blocks) / uncached_blocks
        : 0.0;
    result.retained_block_rate = result.total_blocks > 0
        ? static_cast<double>(retained_blocks) / result.total_blocks
        : 0.0;
    result.retained_candidate_rate = result.total_candidates > 0
        ? static_cast<double>(retained_candidates) / result.total_candidates
        : 0.0;
    result.cached_blocks = cached_blocks;
    result.miss_blocks = uncached_blocks;
    result.retained_miss_blocks = retained_miss_blocks;
    result.retained_blocks = retained_blocks;
    result.retained_candidates = retained_candidates;
    return result;
}

}  // namespace detail

inline Result evaluate(
    const std::vector<Candidate>& candidates,
    int query_count,
    int batch_size,
    int recall_k,
    const Params& params) {
    return detail::evaluate_prepared(
        detail::prepare_candidates(candidates, query_count, batch_size, recall_k), params);
}

inline std::vector<Result> scan(
    const std::vector<Candidate>& candidates,
    int query_count,
    int batch_size,
    int recall_k,
    const std::vector<int>& sparse_ref_thresholds,
    const std::vector<int>& protected_candidate_ranks,
    const std::vector<uint64_t>& cached_block_limits) {
    std::vector<Result> results;
    results.reserve(
        sparse_ref_thresholds.size() * protected_candidate_ranks.size() *
        cached_block_limits.size());
    for (int threshold : sparse_ref_thresholds) {
        if (threshold <= 0) {
            throw std::invalid_argument("IVF file block policy thresholds must be positive");
        }
    }
    for (int rank : protected_candidate_ranks) {
        if (rank <= 0) {
            throw std::invalid_argument("IVF file block policy ranks must be positive");
        }
    }
    const auto prepared =
        detail::prepare_candidates(candidates, query_count, batch_size, recall_k);
    for (uint64_t cached_block_limit : cached_block_limits) {
        for (int threshold : sparse_ref_thresholds) {
            for (int rank : protected_candidate_ranks) {
                results.push_back(detail::evaluate_prepared(
                    prepared, {{threshold, rank}, cached_block_limit}));
            }
        }
    }
    return results;
}

}  // namespace paraselect::tests::ivf_file_policy

#endif  // PARASELECT_TESTS_COMMON_IVF_FILE_BLOCK_POLICY_EVAL_H
