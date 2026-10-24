#pragma once
#include "search/ivf_common/ivf_probe_schedule.cuh"

namespace paraselect::search::detail {
// Five TC shapes (32..64, step 8), followed by unchanged LUT entries.
constexpr int CodewordBuckets = 6;
constexpr int CodewordMinQueries = 25;
// Two passes share this routing: null output counts exact storage; the second
// pass starts cursors at exclusive bucket offsets and emits the same groups.
// Adjacent entries for a list must have contiguous reference ranges.
static __global__ void split_codeword_schedule(IVFProbeScheduleView input,
    int* clusters, int* starts, int* counts, int* cursors,
    const int* list_rows, unsigned long long* physical_rows) {
    constexpr int buckets=CodewordBuckets;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= input.entry_count) return;
    int cluster = input.clusters[e];
    if (e && input.clusters[e - 1] == cluster) return;
    int end = e, refs = 0;
    while (end < input.entry_count && input.clusters[end] == cluster) {
        if (input.starts[end] != input.starts[e] + refs || input.counts[end] <= 0) {
            atomicExch(cursors + buckets, 1);
            return;
        }
        refs += input.counts[end++];
    }
    if (refs >= CodewordMinQueries) {
        // Balance eight-query units across groups, never leaving a tiny tail.
        int body=refs;
        int groups = (body + 63) / 64, units = (body + 7) / 8, first = 0;
        for (int g = 0; g < groups; ++g) {
            int n = min(8 * (units / groups + (g < units % groups)), body - first);
            int bucket = ((n + 7) / 8 * 8 - 32) / 8;
            int slot = atomicAdd(cursors + bucket, 1);
            if (clusters) {
                clusters[slot] = cluster;
                starts[slot] = input.starts[e] + first;
                counts[slot] = n;
            }
            first += n;
        }
        if (clusters && physical_rows)
            atomicAdd(physical_rows, static_cast<unsigned long long>(list_rows[cluster]) * groups);
    } else {
        int slot = atomicAdd(cursors + buckets - 1, end - e);
        if (clusters) for (int i = e; i < end; ++i) {
            clusters[slot + i - e] = cluster;
            starts[slot + i - e] = input.starts[i];
            counts[slot + i - e] = input.counts[i];
        }
    }
}
} // namespace paraselect::search::detail
