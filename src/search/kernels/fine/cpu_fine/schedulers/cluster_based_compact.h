#ifndef PARASELECT_CLUSTER_BASED_COMPACT_H
#define PARASELECT_CLUSTER_BASED_COMPACT_H

#include "search/kernels/fine/cpu_fine/schedulers/cluster_based_sharding.h"

#include <stdexcept>
#include <vector>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_compact_detail {

inline void build_cluster_tasks_active_compact(
    std::vector<cluster_based_sharding_detail::ClusterTask>* tasks,
    const int* active_cluster_ids,
    int active_cluster_count,
    const int* compact_query_offsets,
    const int* cluster_counts,
    int n_total_clusters,
    int shard_tile)
{
    if (!tasks || !active_cluster_ids || !compact_query_offsets || !cluster_counts ||
        active_cluster_count < 0 || n_total_clusters <= 0) {
        throw std::invalid_argument("build_cluster_tasks_active_compact requires valid metadata");
    }
    if (shard_tile <= 0) shard_tile = 1;

    tasks->clear();
    tasks->reserve((size_t)active_cluster_count);
    for (int i = 0; i < active_cluster_count; ++i) {
        const int cid = active_cluster_ids[i];
        if (cid < 0 || cid >= n_total_clusters) {
            throw std::runtime_error("build_cluster_tasks_active_compact received invalid cluster id");
        }
        const int begin = compact_query_offsets[i];
        const int end = compact_query_offsets[i + 1];
        const int count = cluster_counts[cid];
        cluster_based_sharding_detail::append_cluster_task_shards(
            tasks, cid, begin, end, count, shard_tile);
    }
}

}  // namespace cluster_based_compact_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_COMPACT_H
