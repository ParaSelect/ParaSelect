#ifndef CLUSTER_DATASET_CUH
#define CLUSTER_DATASET_CUH

#include "clustering/clustering.cuh"
#include "dataset_aosoa.cuh"
#include "dataset_metadata.cuh"
#include "dataset_reordered_storage.cuh"
#include "gpu_resident_interleaved_storage.cuh"
#include <cstdlib>
#include <string>

/**
 * Cluster
 *  ClusterInfo ,.
 *  ClusteringInterface ().
 */
struct ClusterDataset {
    paraselect::dataset::ReorderedRowMajorStorage row_major_storage;
    paraselect::dataset::DatasetAosoaStorage aosoa_storage;
    paraselect::dataset::GpuResidentInterleavedStorage gpu_resident_interleaved;

    /* : storage  */
    float*& reordered_data;      // = row_major_storage.fp32
    uint8_t*& reordered_data_u8; // = row_major_storage.u8
    paraselect::ownership::BufferOwner& reordered_data_owner_; // = row_major_storage.fp32_owner

    idx_t* reordered_indices;     // [n_total_vectors] compact ( reordered_data )
    float* centroids;           // [cluster_info.k, vector_dim] (CPU)
    ClusterInfo cluster_info;   // cluster(offsetscounts,kcluster)
    int n_total_vectors;        //
    int vector_dim;             //
    /* L2 norm (,) -  row-major  */
    float* vector_l2_norm;      // [n_total_vectors]  L2 norm (||x||^2), CPU
    bool owns_cluster_metadata;

    ClusterDataset();

    /**
     * ClusterDataset(K-means),(in-place)
     * :
     * @param train_sample_ratio  ,1.0=(),0.1=10%
     * @param train_max_samples   ,0=()
     */
    void init_with_kmeans(
        int n_total_vectors,
        int vector_dim,
        int n_clusters,
        float* h_objective,
        float* h_data = nullptr,
        int kmeans_iters = 20,
        bool use_minibatch = false,
        DistanceType distance_mode = COSINE_DISTANCE,
        unsigned int seed = 1234,
        int batch_size = (1 << 20),
        int device_id = 0,
        paraselect::ownership::BufferOwner input_owner = paraselect::ownership::BufferOwner::Borrowed,
        float train_sample_ratio = 1.0f,             //
        int train_max_samples = 0,                   //
        KMeansTrainStrategy train_strategy = KMeansTrainStrategy::FullLloyd,
        int progressive_stage1_samples = 0,
        int progressive_stage2_samples = 0,
        int fast_ivf_train_samples = 0,
        KMeansFinalAssignStrategy final_assign_strategy = KMeansFinalAssignStrategy::Exact,
        bool cluster_only = false,
        const std::string& cache_dataset_name = "",
        const std::string& cache_build_recipe = "",
        bool use_cache = true,
        KMeansReorderPolicy reorder_policy = KMeansReorderPolicy::Standard,
        const std::string& raw_base_path = ""
    );

    /**  compact reordered_data  */
    size_t get_reordered_data_size() const;

    const paraselect::dataset::ReorderedRowMajorStorage& get_row_major_storage() const;
    const paraselect::dataset::DatasetAosoaStorage& get_aosoa_storage() const;
    const float* get_row_major_fp32() const;
    const uint8_t* get_row_major_u8() const;
    const float* get_aosoa_fp32() const;
    const long long* get_aosoa_offsets() const;
    int get_aosoa_tile() const;
    size_t get_aosoa_data_size() const;
    const paraselect::dataset::GpuResidentInterleavedStorage& get_gpu_resident_interleaved() const;
    bool has_gpu_resident_interleaved() const;
    paraselect::ownership::BufferOwner row_major_owner() const;

private:
    void _compute_vector_l2_norm(const float* row_source, bool numa_first_touch_row_major);

public:
    void init_from_cluster_metadata(
        const idx_t* h_reordered_indices,
        const float* h_centroids,
        const idx_t* h_cluster_offsets,
        const int* h_cluster_counts,
        int n_total_vectors_,
        int vector_dim_,
        int n_clusters_
    );

    void attach_cluster_metadata(
        const idx_t* h_reordered_indices,
        const float* h_centroids,
        const idx_t* h_cluster_offsets,
        const int* h_cluster_counts,
        int n_total_vectors_,
        int vector_dim_,
        int n_clusters_);

    void set_reordered_fp32_owned(
        float* h_reordered_data,
        paraselect::ownership::BufferOwner data_owner,
        bool numa_first_touch_row_major = false,
        bool print_numa_distribution = false
    );

    void set_reordered_u8_owned(
        uint8_t* h_reordered_data_u8,
        paraselect::ownership::BufferOwner data_owner,
        bool print_numa_distribution = false
    );

    void set_aosoa_owned(
        float* h_aosoa_data,
        paraselect::ownership::BufferOwner data_owner,
        long long* h_aosoa_offsets,
        paraselect::ownership::BufferOwner offsets_owner,
        int tile,
        bool print_numa_distribution = false
    );

    void set_gpu_resident_interleaved_owned(
        paraselect::dataset::GpuResidentInterleavedStorage& storage
    );

public:
    /**
     * ClusterDataset
     */
    void release();

    /**
     * ClusterDataset
     */
    bool is_valid() const;

    /**
     * cluster
     */
    int get_n_clusters() const;

};

#include "dataset_impl_ctor_policy.cuh"
#include "dataset_impl_builders.cuh"
#include "dataset_impl_init.cuh"
#include "dataset_impl_state.cuh"

#endif // CLUSTER_DATASET_CUH
