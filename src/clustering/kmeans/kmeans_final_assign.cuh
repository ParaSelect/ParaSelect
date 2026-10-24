#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_FINAL_ASSIGN_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_FINAL_ASSIGN_CUH

#include "clustering/kmeans/kmeans.cuh"
#include "clustering/kmeans/kmeans_assignment_workspace.h"

enum class GpuCentroidAssignmentCombo {
    ExactGemm = 0,
    CagraTopkSelect = 1,
};

struct GpuCentroidAssignmentConfig {
    KMeansFinalAssignStrategy strategy = KMeansFinalAssignStrategy::Exact;
    GpuCentroidAssignmentCombo combo = GpuCentroidAssignmentCombo::ExactGemm;
    int device_id = 0;
    int max_gpus = 8;
    int batch_size = 1 << 15;
    int centroid_tile = 4096;
    int cagra_graph_degree = 32;
    int cagra_intermediate_graph_degree = 64;
    int cagra_topk_centroids = 8;
    int cagra_itopk_size = 96;
    int cagra_search_width = 1;
    int cagra_max_iterations = 0;
    int cagra_team_size = 0;
    int cagra_thread_block_size = 0;
    float cagra_index_scale = 1.0f / 255.0f;
    KMeansCagraBuildAlgo cagra_build_algo = KMeansCagraBuildAlgo::Auto;
};

struct GpuCentroidAssignmentWorkspace {
    int device_id = -1;
    int max_batch = 0;
    int assign_capacity = 0;
    int max_k = 0;
    int dim = 0;
    int centroid_tile = 0;
    StreamEnv env;
    cublasHandle_t handle = nullptr;
    float* d_cnorm2 = nullptr;
    float* d_data_buf = nullptr;
    idx_t* d_assign = nullptr;
};

bool assign_vectors_to_centroids_gpu(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    idx_t* d_assign);

bool gpu_centroid_assignment_workspace_prepare(
    GpuCentroidAssignmentWorkspace& ws,
    const GpuCentroidAssignmentConfig& cfg,
    int max_batch,
    int k,
    int dim);

void gpu_centroid_assignment_workspace_release(GpuCentroidAssignmentWorkspace& ws);

bool assign_vectors_to_centroids_gpu_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    GpuCentroidAssignmentWorkspace& ws);

// Always rebuilds from the supplied centers, including in-place content changes.
// Search requires this explicit preparation and never builds/resizes implicitly.
bool gpu_centroid_assignment_cagra_workspace_build(
    GpuCentroidAssignmentCagraWorkspace& ws,
    const GpuCentroidAssignmentConfig& cfg,
    const float* d_centroids,
    int k,
    int dim,
    int max_batch);

void gpu_centroid_assignment_cagra_workspace_release(GpuCentroidAssignmentCagraWorkspace& ws);

bool assign_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* h_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign);

bool assign_u8_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const uint8_t* h_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign);

bool assign_device_u8_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const uint8_t* d_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign);

bool assign_device_vectors_to_centroids_gpu_cagra_with_workspace(
    const GpuCentroidAssignmentConfig& cfg,
    const KMeansCase& kmeans_cfg,
    const float* d_data,
    const float* d_centroids,
    GpuCentroidAssignmentCagraWorkspace& ws,
    idx_t* d_assign);

#endif // PARASELECT_CUDA_CLUSTERING_KMEANS_FINAL_ASSIGN_CUH
