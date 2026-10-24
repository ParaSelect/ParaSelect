#ifndef PARASELECT_CLUSTER_BASED_SHARDING_H
#define PARASELECT_CLUSTER_BASED_SHARDING_H

#include <algorithm>
#include <vector>
#include <stdexcept>

namespace paraselect {
namespace cpu_fine {
namespace cluster_based_sharding_detail {

// Keep this aligned with current AoSoA policy.
static constexpr int kHugeClusterShardMinGroups = 1024;
static constexpr int kHugeClusterShardTargetGroups = 512;

struct ClusterTask {
    int cid;
    int qlist_start;
    int qlist_end;
    int lane_begin;
    int lane_end;
};

inline void append_cluster_task_shards(
    std::vector<ClusterTask>* tasks,
    int cid,
    int qlist_start,
    int qlist_end,
    int count,
    int shard_tile)
{
    if (qlist_start == qlist_end || count <= 0) return;
    const int groups = (count + shard_tile - 1) / shard_tile;
    if (groups < kHugeClusterShardMinGroups) {
        tasks->push_back({cid, qlist_start, qlist_end, 0, count});
        return;
    }

    const int shard_groups = std::max(1, kHugeClusterShardTargetGroups);
    for (int gb = 0; gb < groups; gb += shard_groups) {
        const int ge = std::min(gb + shard_groups, groups);
        const int lane_begin = gb * shard_tile;
        const int lane_end = std::min(ge * shard_tile, count);
        if (lane_begin < lane_end) {
            tasks->push_back({cid, qlist_start, qlist_end, lane_begin, lane_end});
        }
    }
}

inline void build_cluster_tasks(
    std::vector<ClusterTask>* tasks,
    const int* cluster_query_offset_ptr,
    const int* cluster_counts,
    int n_total_clusters,
    int shard_tile)
{
    if (!tasks || !cluster_query_offset_ptr || !cluster_counts || n_total_clusters <= 0) return;
    if (shard_tile <= 0) shard_tile = 1;

    tasks->clear();
    tasks->reserve(n_total_clusters);
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        const int begin = cluster_query_offset_ptr[cid];
        const int end = cluster_query_offset_ptr[cid + 1];
        if (begin == end) continue;
        const int count = cluster_counts[cid];
        if (count <= 0) continue;

        append_cluster_task_shards(tasks, cid, begin, end, count, shard_tile);
    }
}

inline void build_cluster_tasks_active(
    std::vector<ClusterTask>* tasks,
    const int* active_cluster_ids,
    int active_cluster_count,
    const int* cluster_query_offset_ptr,
    const int* cluster_counts,
    int n_total_clusters,
    int shard_tile)
{
    if (!tasks || !active_cluster_ids || !cluster_query_offset_ptr || !cluster_counts ||
        active_cluster_count < 0 || n_total_clusters <= 0) {
        throw std::invalid_argument("build_cluster_tasks_active requires valid metadata");
    }
    if (shard_tile <= 0) shard_tile = 1;

    tasks->clear();
    tasks->reserve((size_t)active_cluster_count);
    for (int i = 0; i < active_cluster_count; ++i) {
        const int cid = active_cluster_ids[i];
        if (cid < 0 || cid >= n_total_clusters) {
            throw std::runtime_error("build_cluster_tasks_active received invalid cluster id");
        }
        const int begin = cluster_query_offset_ptr[cid];
        const int end = cluster_query_offset_ptr[cid + 1];
        const int count = cluster_counts[cid];
        append_cluster_task_shards(tasks, cid, begin, end, count, shard_tile);
    }
}

}  // namespace cluster_based_sharding_detail
}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CLUSTER_BASED_SHARDING_H
