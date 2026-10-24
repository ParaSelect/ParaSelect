#ifndef PARASELECT_DATASET_METADATA_CUH
#define PARASELECT_DATASET_METADATA_CUH

#include "dataset/ownership_helper.h"
#include "utils/pch.h"

#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <stdexcept>

namespace paraselect {
namespace dataset {

struct DatasetMetadataBuffers {
    idx_t* reordered_indices = nullptr;
    idx_t* cluster_offsets = nullptr;
    int* cluster_counts = nullptr;
    float* centroids = nullptr;

    void release() {
        if (reordered_indices) {
            std::free(reordered_indices);
            reordered_indices = nullptr;
        }
        if (cluster_offsets) {
            std::free(cluster_offsets);
            cluster_offsets = nullptr;
        }
        if (cluster_counts) {
            std::free(cluster_counts);
            cluster_counts = nullptr;
        }
        if (centroids) {
            std::free(centroids);
            centroids = nullptr;
        }
    }
};


inline DatasetMetadataBuffers copy_cluster_metadata_without_reordered_indices(
    const float* h_centroids,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int dim,
    int n_clusters)
{
    if (!h_centroids || !h_cluster_offsets || !h_cluster_counts || dim <= 0 || n_clusters <= 0) {
        throw std::invalid_argument("copy_cluster_metadata_without_reordered_indices: invalid arguments");
    }

    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;

    const size_t offsets_bytes = ((size_t)n_clusters + 1u) * sizeof(idx_t);
    const size_t counts_bytes = (size_t)n_clusters * sizeof(int);
    const size_t centroids_bytes = (size_t)n_clusters * (size_t)dim * sizeof(float);

    Handle offsets(memalign(64, offsets_bytes), offsets_bytes, Owner::StdMalloc);
    Handle counts(memalign(64, counts_bytes), counts_bytes, Owner::StdMalloc);
    Handle centroids(memalign(64, centroids_bytes), centroids_bytes, Owner::StdMalloc);
    if (!offsets.data || !counts.data || !centroids.data) {
        throw std::bad_alloc();
    }

    std::memcpy(offsets.data, h_cluster_offsets, offsets_bytes);
    std::memcpy(counts.data, h_cluster_counts, counts_bytes);
    std::memcpy(centroids.data, h_centroids, centroids_bytes);

    DatasetMetadataBuffers buffers{};
    paraselect::ownership::take_to(offsets, buffers.cluster_offsets);
    paraselect::ownership::take_to(counts, buffers.cluster_counts);
    paraselect::ownership::take_to(centroids, buffers.centroids);
    return buffers;
}

inline DatasetMetadataBuffers copy_cluster_metadata(
    const idx_t* h_reordered_indices,
    const float* h_centroids,
    const idx_t* h_cluster_offsets,
    const int* h_cluster_counts,
    int n_total_vectors,
    int dim,
    int n_clusters)
{
    if (!h_reordered_indices || !h_centroids || !h_cluster_offsets || !h_cluster_counts ||
        n_total_vectors <= 0 || dim <= 0 || n_clusters <= 0) {
        throw std::invalid_argument("copy_cluster_metadata: invalid arguments");
    }

    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;

    const size_t reordered_bytes = (size_t)n_total_vectors * sizeof(idx_t);
    const size_t offsets_bytes = ((size_t)n_clusters + 1u) * sizeof(idx_t);
    const size_t counts_bytes = (size_t)n_clusters * sizeof(int);
    const size_t centroids_bytes = (size_t)n_clusters * (size_t)dim * sizeof(float);

    Handle reordered(memalign(64, reordered_bytes), reordered_bytes, Owner::StdMalloc);
    Handle offsets(memalign(64, offsets_bytes), offsets_bytes, Owner::StdMalloc);
    Handle counts(memalign(64, counts_bytes), counts_bytes, Owner::StdMalloc);
    Handle centroids(memalign(64, centroids_bytes), centroids_bytes, Owner::StdMalloc);
    if (!reordered.data || !offsets.data || !counts.data || !centroids.data) {
        throw std::bad_alloc();
    }

    std::memcpy(reordered.data, h_reordered_indices, reordered_bytes);
    std::memcpy(offsets.data, h_cluster_offsets, offsets_bytes);
    std::memcpy(counts.data, h_cluster_counts, counts_bytes);
    std::memcpy(centroids.data, h_centroids, centroids_bytes);

    DatasetMetadataBuffers buffers{};
    paraselect::ownership::take_to(reordered, buffers.reordered_indices);
    paraselect::ownership::take_to(offsets, buffers.cluster_offsets);
    paraselect::ownership::take_to(counts, buffers.cluster_counts);
    paraselect::ownership::take_to(centroids, buffers.centroids);
    return buffers;
}

}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_DATASET_METADATA_CUH
