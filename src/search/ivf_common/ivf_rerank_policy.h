#ifndef PARASELECT_SEARCH_IVF_COMMON_RERANK_POLICY_H
#define PARASELECT_SEARCH_IVF_COMMON_RERANK_POLICY_H

#include <algorithm>

namespace paraselect::search {

inline int ivf_rerank_thread_count(int max_threads, int batch_n) {
    max_threads = std::max(1, max_threads);
    batch_n = std::max(1, batch_n);
    return std::min(max_threads, batch_n);
}

inline int ivf_rerank_input_count(
    int gpu_candidate_count,
    int requested_rerank_count,
    int topk) {
    if (gpu_candidate_count <= 0) return 0;
    if (requested_rerank_count <= 0) return gpu_candidate_count;
    return std::min(
        gpu_candidate_count,
        std::max(std::max(1, topk), requested_rerank_count));
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_RERANK_POLICY_H
