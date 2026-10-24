#include "search/ivfrespq/stage/residual_pq_workspace_budget.h"

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

bool expect_equal(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) return true;
    std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
    return false;
}

}  // namespace

int main() {
    using paraselect::search::PQAdcThresholdWorkspaceShape;
    using paraselect::search::plan_residual_pq_grouped_workspace;
    using paraselect::search::pq_adc_threshold_workspace_required_bytes;

    const PQAdcThresholdWorkspaceShape threshold{
        8192,
        8192,
        32,
        1 << 24,
        256ULL * 1024ULL * 1024ULL};
    const std::size_t expected_threshold =
        8192ULL * 32ULL * sizeof(long long) +
        (1ULL << 24) * (sizeof(int) + sizeof(float) + sizeof(int)) +
        sizeof(int) + 2 * sizeof(unsigned long long) +
        8192ULL * 8192ULL * (sizeof(int) + sizeof(float)) +
        8192ULL * (sizeof(int) + sizeof(float) + sizeof(unsigned long long)) +
        sizeof(int) +
        (1ULL << 24) * (2 * sizeof(unsigned long long) + sizeof(int)) +
        8193ULL * sizeof(int) +
        256ULL * 1024ULL * 1024ULL;
    if (!expect_equal(
            pq_adc_threshold_workspace_required_bytes(threshold),
            expected_threshold,
            "threshold workspace bytes")) {
        return 1;
    }

    const auto plan = plan_residual_pq_grouped_workspace(
        8 * kGiB, 2 * kGiB, kGiB, {8, 512, 16, 12});
    if (!expect_equal(plan.workspace_budget_bytes, 5 * kGiB, "grouped budget")) return 1;
    if (plan.limits.group_capacity <= 0 ||
        plan.limits.chunk_capacity != plan.limits.group_capacity * 8 ||
        plan.limits.merge_a_item_capacity != plan.limits.group_capacity * 512 ||
        plan.limits.merge_b_item_capacity != plan.limits.group_capacity * 512) {
        std::cerr << "grouped capacities are inconsistent\n";
        return 1;
    }

    const auto exhausted = plan_residual_pq_grouped_workspace(
        3 * kGiB, 2 * kGiB, kGiB, {8, 512, 16, 12});
    if (exhausted.workspace_budget_bytes != 0 || exhausted.limits.group_capacity != 0) {
        std::cerr << "exhausted memory must not produce a grouped allocation\n";
        return 1;
    }

    // Observed SPACEV1.4B: required future buffers fit, a fixed 1 GiB margin does not.
    constexpr std::size_t spacev_free = 1601241088;
    constexpr std::size_t spacev_reserved = 741714099;
    const auto spacev = plan_residual_pq_grouped_workspace(
        spacev_free, spacev_reserved,
        paraselect::search::residual_pq_grouped_safety_bytes(spacev_free, spacev_reserved),
        {8, 512, 16, 12});
    if (spacev.limits.group_capacity <= 0 ||
        spacev.workspace_budget_bytes + spacev_reserved >= spacev_free) {
        std::cerr << "SPACEV buffers must fit while preserving allocator headroom\n";
        return 1;
    }
    if (!expect_equal(paraselect::search::residual_pq_grouped_safety_bytes(
            8 * kGiB, 2 * kGiB), kGiB, "ample-memory safety margin")) return 1;
    const auto too_small = plan_residual_pq_grouped_workspace(
        spacev_reserved + (32ULL << 20), spacev_reserved,
        paraselect::search::residual_pq_grouped_safety_bytes(
            spacev_reserved + (32ULL << 20), spacev_reserved), {8, 512, 16, 12});
    if (too_small.limits.group_capacity != 0) {
        std::cerr << "tiny uncommitted budget must not permit allocation\n";
        return 1;
    }

    bool overflow_rejected = false;
    try {
        (void)pq_adc_threshold_workspace_required_bytes({
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<std::size_t>::max()});
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    if (!overflow_rejected) {
        std::cerr << "workspace byte overflow must be rejected\n";
        return 1;
    }
    return 0;
}
