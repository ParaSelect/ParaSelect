#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_RERANK_STATS_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_RERANK_STATS_H

#include <vector>

namespace paraselect::search {

struct IVFRerankStats {
    double plan_ms = 0.0;
    double read_ms = 0.0;
    double distance_ms = 0.0;
    double topk_ms = 0.0;
    double other_ms = 0.0;
    double avg_valid_candidates = 0.0;
    double avg_adjacent_gap = 0.0;
    double max_adjacent_gap = 0.0;
    long long unique_blocks = 0;
    long long bytes_read = 0;
    long long read_block_bytes = 0;
    long long cache_hits = 0;
    long long cache_misses = 0;
    long long bytes_reused = 0;
    long long distance_evaluations = 0;
    long long blocks_considered = 0;
    long long blocks_skipped = 0;
    std::vector<long long> block_candidate_ref_histogram;
    int io_rings = 0;
    int io_depth_per_ring = 0;
    int threads = 0;
    int pinned = 0;
    int persistent = 1;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_RERANK_STATS_H
