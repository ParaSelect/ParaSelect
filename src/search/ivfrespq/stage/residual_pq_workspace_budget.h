#ifndef PARASELECT_SEARCH_IVFRESPQ_STAGE_RESIDUAL_PQ_WORKSPACE_BUDGET_H
#define PARASELECT_SEARCH_IVFRESPQ_STAGE_RESIDUAL_PQ_WORKSPACE_BUDGET_H

#include "search/ivfrespq/stage/residual_pq_grouped_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace paraselect::search {

struct PQAdcThresholdWorkspaceShape {
    int max_queries = 0;
    int candidate_capacity = 0;
    int seed_probes = 0;
    int pool_capacity = 0;
    std::size_t sort_temp_bytes = 0;
};

namespace workspace_budget_detail {

inline std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error("residual PQ workspace byte count overflow");
    }
    return lhs + rhs;
}

inline std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error("residual PQ workspace byte count overflow");
    }
    return lhs * rhs;
}

}  // namespace workspace_budget_detail

inline std::size_t pq_adc_threshold_workspace_required_bytes(
    const PQAdcThresholdWorkspaceShape& shape) {
    if (shape.max_queries <= 0 || shape.candidate_capacity <= 0 ||
        shape.seed_probes <= 0 || shape.pool_capacity <= 0) {
        throw std::invalid_argument("invalid PQ threshold workspace shape");
    }

    using workspace_budget_detail::checked_add;
    using workspace_budget_detail::checked_mul;
    const std::size_t queries = static_cast<std::size_t>(shape.max_queries);
    const std::size_t candidates = static_cast<std::size_t>(shape.candidate_capacity);
    const std::size_t seed_probes = static_cast<std::size_t>(shape.seed_probes);
    const std::size_t pool = static_cast<std::size_t>(shape.pool_capacity);

    std::size_t bytes = 0;
    bytes = checked_add(bytes, checked_mul(
        checked_mul(queries, seed_probes), sizeof(std::int64_t)));
    bytes = checked_add(bytes, checked_mul(
        pool, sizeof(int) + sizeof(float) + sizeof(int)));
    bytes = checked_add(bytes, sizeof(int) + 2 * sizeof(unsigned long long));
    bytes = checked_add(bytes, checked_mul(
        checked_mul(queries, candidates), sizeof(int) + sizeof(float)));
    bytes = checked_add(bytes, checked_mul(
        queries, sizeof(int) + sizeof(float) + sizeof(unsigned long long)));
    bytes = checked_add(bytes, sizeof(int));
    bytes = checked_add(bytes, checked_mul(
        pool, 2 * sizeof(std::uint64_t) + sizeof(int)));
    bytes = checked_add(bytes, checked_mul(queries + 1, sizeof(int)));
    return checked_add(bytes, shape.sort_temp_bytes);
}

struct ResidualPQGroupedWorkspaceShape {
    int group_chunks = 0;
    int global_k = 0;
    std::size_t chunk_descriptor_bytes = 0;
    std::size_t group_descriptor_bytes = 0;
};

struct ResidualPQGroupedWorkspacePlan {
    std::size_t workspace_budget_bytes = 0;
    ResidualPQGroupedWorkspaceLimits limits;
};

inline std::size_t residual_pq_grouped_safety_bytes(
    std::size_t free_bytes, std::size_t reserved_bytes) {
    constexpr std::size_t minimum = 64ULL * 1024ULL * 1024ULL;
    constexpr std::size_t maximum = 1024ULL * 1024ULL * 1024ULL;
    const std::size_t uncommitted = free_bytes > reserved_bytes
        ? free_bytes - reserved_bytes : 0;
    return std::clamp(uncommitted / 4, minimum, maximum);
}

inline ResidualPQGroupedWorkspacePlan plan_residual_pq_grouped_workspace(
    std::size_t free_bytes,
    std::size_t reserved_after_grouped_bytes,
    std::size_t safety_bytes,
    const ResidualPQGroupedWorkspaceShape& shape) {
    if (shape.group_chunks <= 0 || shape.global_k <= 0 ||
        shape.chunk_descriptor_bytes == 0 || shape.group_descriptor_bytes == 0) {
        throw std::invalid_argument("invalid residual PQ grouped workspace shape");
    }

    const std::size_t unavailable = workspace_budget_detail::checked_add(
        reserved_after_grouped_bytes, safety_bytes);
    if (free_bytes <= unavailable) return {};

    const std::size_t group_chunks = static_cast<std::size_t>(shape.group_chunks);
    const std::size_t global_k = static_cast<std::size_t>(shape.global_k);
    std::size_t per_group_bytes = workspace_budget_detail::checked_mul(
        group_chunks, shape.chunk_descriptor_bytes);
    per_group_bytes = workspace_budget_detail::checked_add(
        per_group_bytes, shape.group_descriptor_bytes);
    per_group_bytes = workspace_budget_detail::checked_add(
        per_group_bytes,
        workspace_budget_detail::checked_mul(
            2 * (sizeof(float) + sizeof(int)), global_k));

    ResidualPQGroupedWorkspacePlan plan;
    plan.workspace_budget_bytes = free_bytes - unavailable;
    std::size_t group_capacity = plan.workspace_budget_bytes / per_group_bytes;
    const std::size_t int_capacity =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    group_capacity = std::min(group_capacity, int_capacity);
    group_capacity = std::min(group_capacity, int_capacity / group_chunks);
    group_capacity = std::min(group_capacity, int_capacity / global_k);
    if (group_capacity == 0) return plan;

    plan.limits.group_capacity = static_cast<int>(group_capacity);
    plan.limits.chunk_capacity = static_cast<int>(group_capacity * group_chunks);
    plan.limits.merge_a_item_capacity = static_cast<int>(group_capacity * global_k);
    plan.limits.merge_b_item_capacity = plan.limits.merge_a_item_capacity;
    return plan;
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFRESPQ_STAGE_RESIDUAL_PQ_WORKSPACE_BUDGET_H
