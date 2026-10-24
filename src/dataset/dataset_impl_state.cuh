#ifndef PARASELECT_DATASET_IMPL_STATE_CUH
#define PARASELECT_DATASET_IMPL_STATE_CUH

inline size_t ClusterDataset::get_reordered_data_size() const {
    if (vector_dim <= 0) return 0;
    if (!reordered_data) return 0;
    return (size_t)n_total_vectors * vector_dim;
}

inline const paraselect::dataset::ReorderedRowMajorStorage& ClusterDataset::get_row_major_storage() const {
    return row_major_storage;
}
inline const paraselect::dataset::DatasetAosoaStorage& ClusterDataset::get_aosoa_storage() const {
    return aosoa_storage;
}

inline const float* ClusterDataset::get_row_major_fp32() const { return reordered_data; }
inline const uint8_t* ClusterDataset::get_row_major_u8() const { return reordered_data_u8; }
inline const float* ClusterDataset::get_aosoa_fp32() const { return aosoa_storage.data; }
inline const long long* ClusterDataset::get_aosoa_offsets() const { return aosoa_storage.offsets; }
inline int ClusterDataset::get_aosoa_tile() const { return aosoa_storage.tile; }
inline size_t ClusterDataset::get_aosoa_data_size() const { return aosoa_storage.size(cluster_info.k); }
inline const paraselect::dataset::GpuResidentInterleavedStorage& ClusterDataset::get_gpu_resident_interleaved() const { return gpu_resident_interleaved; }
inline bool ClusterDataset::has_gpu_resident_interleaved() const { return gpu_resident_interleaved.is_valid(); }
inline paraselect::ownership::BufferOwner ClusterDataset::row_major_owner() const { return reordered_data_owner_; }

inline void ClusterDataset::release() {
    paraselect::dataset::release_rowmajor_storage(&row_major_storage);
    aosoa_storage.release();
    gpu_resident_interleaved.release();

    if (reordered_indices && owns_cluster_metadata) {
        std::free(reordered_indices);
    }
    reordered_indices = nullptr;
    if (centroids && owns_cluster_metadata) {
        std::free(centroids);
    }
    centroids = nullptr;
    if (cluster_info.offsets && owns_cluster_metadata) {
        std::free(cluster_info.offsets);
    }
    cluster_info.offsets = nullptr;
    if (cluster_info.counts && owns_cluster_metadata) {
        std::free(cluster_info.counts);
    }
    cluster_info.counts = nullptr;
    if (vector_l2_norm) {
        std::free(vector_l2_norm);
        vector_l2_norm = nullptr;
    }

    cluster_info.k = 0;
    n_total_vectors = 0;
    vector_dim = 0;
    owns_cluster_metadata = true;
}

inline bool ClusterDataset::is_valid() const {
    return (reordered_data != nullptr || reordered_data_u8 != nullptr || aosoa_storage.is_valid() || gpu_resident_interleaved.is_valid()) &&
           reordered_indices != nullptr &&
           centroids != nullptr &&
           cluster_info.offsets != nullptr &&
           cluster_info.counts != nullptr &&
           cluster_info.k > 0 &&
           n_total_vectors > 0 &&
           vector_dim > 0;
}

inline int ClusterDataset::get_n_clusters() const { return cluster_info.k; }

#endif
