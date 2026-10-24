#ifndef PARASELECT_DATASET_IMPL_CTOR_POLICY_CUH
#define PARASELECT_DATASET_IMPL_CTOR_POLICY_CUH

inline ClusterDataset::ClusterDataset()
    : row_major_storage(),
      aosoa_storage(),
      reordered_data(row_major_storage.fp32),
      reordered_data_u8(row_major_storage.u8),
      reordered_data_owner_(row_major_storage.fp32_owner),
      reordered_indices(nullptr),
      centroids(nullptr),
      cluster_info{nullptr, nullptr, 0},
      n_total_vectors(0),
      vector_dim(0),
      vector_l2_norm(nullptr),
      owns_cluster_metadata(true) {}

#endif
