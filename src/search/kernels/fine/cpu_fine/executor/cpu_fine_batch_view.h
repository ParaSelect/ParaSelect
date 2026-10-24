#ifndef PARASELECT_CPU_FINE_BATCH_VIEW_H
#define PARASELECT_CPU_FINE_BATCH_VIEW_H

#include "utils/pch.h"

#include <cstddef>

namespace paraselect {
namespace cpu_fine {

struct CpuFineBatchView {
    int n_query = 0;

    const float* h_queries = nullptr;
    const uint8_t* h_queries_u8 = nullptr;
    float* h_topk_dist_out = nullptr;
    idx_t* h_topk_index_out = nullptr;

    idx_t* h_top_nprobe_batch = nullptr;
    int* h_cluster_query_offset_batch = nullptr;
    int* h_cluster_query_data_batch = nullptr;
    int* h_active_cluster_ids_batch = nullptr;
    int active_cluster_count = 0;

    int* h_topk_local_idx_batch = nullptr;
    uint8_t* h_query_u8_tmp = nullptr;
    float* h_thread_dists = nullptr;
    int* h_thread_idxs = nullptr;
    int* h_thread_sizes = nullptr;
    uint8_t* h_thread_touched = nullptr;
    std::size_t h_thread_topk_capacity = 0;
    std::size_t h_thread_query_capacity = 0;
};

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_BATCH_VIEW_H
