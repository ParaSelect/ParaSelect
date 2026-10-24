#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace paraselect::search {

inline constexpr int PQDistanceHistogramMaxBins = 8192;

// A compact query-level histogram interface.  The producer is intentionally
// separate from threshold policy and ADC layout.  Bins are monotonic distance
// upper bounds; selecting an upper edge is conservative for admission.
struct PQDistanceHistogramView {
    const std::uint32_t* counts = nullptr;  // query-major: [query, bin]
    const float* upper_edges = nullptr;     // shared, monotonically increasing
    int bins = 0;
};

inline int histogram_quantile_bin(
    const PQDistanceHistogramView& histogram,
    int query,
    std::uint64_t target_rank) noexcept {
    if (!histogram.counts || !histogram.upper_edges || histogram.bins <= 0 ||
        query < 0 || target_rank == 0) {
        return -1;
    }
    std::uint64_t cumulative = 0;
    for (int bin = 0; bin < histogram.bins; ++bin) {
        cumulative += histogram.counts[static_cast<std::size_t>(query) * histogram.bins + bin];
        if (cumulative >= target_rank) return bin;
    }
    return -1;
}

inline float histogram_quantile_threshold(
    const PQDistanceHistogramView& histogram,
    int query,
    std::uint64_t target_rank) noexcept {
    const int bin = histogram_quantile_bin(histogram, query, target_rank);
    return bin >= 0 ? histogram.upper_edges[bin] : INFINITY;
}

#if defined(__CUDACC__)

// Shared bin convention for both ADC producers; the upper edge admits the score.
__device__ inline int pq_distance_histogram_bin(float distance, int bins,
                                               float log_min, float log_step) {
    const float log_distance = log2f(fmaxf(distance, exp2f(log_min)));
    int bin = min(bins - 1, max(0, static_cast<int>(
        floorf((log_distance - log_min) / log_step))));
    while (bin < bins - 1 && distance > exp2f(log_min + float(bin + 1) * log_step)) ++bin;
    return bin;
}
#endif

}  // namespace paraselect::search
