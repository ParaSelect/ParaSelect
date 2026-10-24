#ifndef PARASELECT_IVF_CPU_COMPACT_METADATA_H
#define PARASELECT_IVF_CPU_COMPACT_METADATA_H

#include "utils/pch.h"

inline bool ivf_cpu_should_use_compact_metadata(
    int query_batch_size,
    int n_probes,
    int n_total_clusters,
    bool compact_metadata_capable)
{
    return compact_metadata_capable &&
           (long long)query_batch_size * (long long)n_probes < (long long)n_total_clusters / 2LL;
}


inline void ivf_cpu_fill_active_clusters_from_full_offsets(
    const int* h_cluster_query_offsets,
    int n_total_clusters,
    int* h_active_cluster_ids,
    int* h_active_cluster_count)
{
    int active_count = 0;
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        if (h_cluster_query_offsets[cid] != h_cluster_query_offsets[cid + 1]) {
            h_active_cluster_ids[active_count++] = cid;
        }
    }
    *h_active_cluster_count = active_count;
}


#endif  // PARASELECT_IVF_CPU_COMPACT_METADATA_H
