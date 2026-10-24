#ifndef PGVECTOR_CUDA_KMEANS_CUH
#define PGVECTOR_CUDA_KMEANS_CUH

#include "utils/pch.h"
#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans_policy.cuh"
#include "clustering/utils.cuh"
#include <cub/cub.cuh>

// ============================================================
// Config
// ============================================================
enum KMeansInitPolicy {
    /** n*k_eff  CPU , kmeans++  */
    KMEANS_INIT_AUTO_THRESHOLD = 0,
    KMEANS_INIT_FORCE_KMEANSPP = 1,
    KMEANS_INIT_FORCE_RANDOM = 2,
};

struct KMeansInitStats {
    int mode;        // 0=kmeans++, 1=random_fallback
    double init_ms;  // wall-clock initialization time in milliseconds
    int n;
    int k_eff;
};

struct KMeansLloydStats {
    int n = 0;
    int dim = 0;
    int k = 0;
    int iters = 0;
    int batch_size = 0;
    int total_batches = 0;
    int ktile = 0;
    bool use_flash_assign = false;
    bool timing_valid = false;
    double total_ms = 0.0;
    double init_ms = 0.0;
    double batch_ms = 0.0;
    double update_ms = 0.0;
    double final_assign_ms = 0.0;
    double objective_ms = 0.0;
    float last_iter_objective = 0.0f;
    float final_objective = 0.0f;
};

struct KMeansCase {
    int n;          // number of vectors
    int dim;        // vector dimension
    int k;          // number of clusters
    int iters;      // Lloyd iterations
    int minibatch_iters;      // Minibatch iters
    int seed;       // random seed
    DistanceType dist;  //  pch.h  DistanceType
    KMeansComputePrecision precision = KMeansComputePrecision::Tf32;
    KMeansInitPolicy init_policy = KMEANS_INIT_AUTO_THRESHOLD;
    bool use_flash_assign = false;
    bool collect_timing = false;
};

// ============================================================
// StreamEnv:  buffer,
// ============================================================
struct StreamEnv {
    float* d_xnorm2;      // [B]
    float* d_best_dist2;  // [B]
    idx_t*   d_best_idx;    // [B]
    float* d_dot;         // [B * Ktile]

    // Sort-by-cluster (allocate_accum_sort )
    idx_t*   d_sorted_keys;    // [B]  cluster id(SortPairs  key)
    idx_t*   d_sorted_vals;    // [B] (SortPairs  value)
    idx_t*   d_sample_idx;     // [B]  [0,1,...,B-1](SortPairs  value)
    int*   d_unique_ids;     // [B] RunLengthEncode : cluster id
    int*   d_run_counts;     // [B] RunLengthEncode :
    int*   d_run_offsets;    // [B+1] ExclusiveSum :
    int*   d_num_runs;       // [1]  RunLengthEncode : cluster (device)
    int*   h_num_runs;       // [1]  pinned host , cudaMemcpyAsync
    void*  d_cub_tmp;        // CUB
    size_t cub_tmp_bytes;    // d_cub_tmp

    StreamEnv()
        : d_xnorm2(nullptr), d_best_dist2(nullptr), d_best_idx(nullptr), d_dot(nullptr),
          d_sorted_keys(nullptr), d_sorted_vals(nullptr), d_sample_idx(nullptr),
          d_unique_ids(nullptr), d_run_counts(nullptr), d_run_offsets(nullptr),
          d_num_runs(nullptr), h_num_runs(nullptr), d_cub_tmp(nullptr), cub_tmp_bytes(0) {}

    void allocate(int B, int Ktile, int dot_B = -1);
    /**  sort-by-cluster ;k  CUB  key  */
    void allocate_accum_sort(int B, int k);
    void free();
};

// ============================================================
// GPU Kernels Declaration
// ============================================================

/**
 * Kernel:
 *
 * @param centroids : [k, dim]
 * @param accum : [k, dim]
 * @param counts : [k]
 * @param k
 * @param dim
 */
__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
);

/**
 * Kernel: Minibatch()
 *
 * @param centroids : [k, dim]
 * @param accum :minibatch [k, dim]
 * @param counts :minibatch [k]
 * @param total_counts : [k]()
 * @param k
 * @param dim
 */
__global__ void kernel_update_centroids_minibatch(
    float* __restrict__ centroids,      // [k, dim] (in/out)
    const float* __restrict__ accum,    // [k, dim]
    const int* __restrict__ counts,     // [k]
    int* __restrict__ total_counts,      // [k] (in/out)
    int k, int dim
);

/**
 * Kernel: (best_dist2 = INF, best_idx = 0)
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    idx_t* __restrict__ best_idx,
    int n
);

/**
 * Kernel:  GEMM (col-major dotT)
 */
__global__ void kernel_update_best_from_dotT(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] centroid
    int curB,
    int curK,
    int cbase,                            // centroid
    bool cnorm_is_l2,
    idx_t* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
);

/**
 * Kernel: ( GEMM )
 */
__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const idx_t* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
);

/**
 * Kernel: Sort-by-cluster ( atomic )
 *
 *  Block  cluster , float4 .
 *  Block , accum.
 *
 * @param data4          , float4 [n, dim/4]
 * @param sorted_indices  [curB](SortPairs  value)
 * @param cluster_offsets  sorted_indices  [num_active+1]
 * @param active_clusters  cluster id  [num_active]
 * @param accum            [k, dim]
 * @param counts           [k]
 * @param num_active       cluster
 * @param dim_v4          dim / 4
 */
__global__ void kernel_segmented_accum_v2(
    const float4* __restrict__ data4,
    const idx_t*   __restrict__ sorted_indices,
    const int*   __restrict__ cluster_offsets,
    const int*   __restrict__ active_clusters,
    float*       __restrict__ accum,
    int*         __restrict__ counts,
    int num_active, int dim_v4
);

/**
 * Kernel: (block reduce)
 */
__global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  //  float
    int n
);

// ============================================================
// GPU KMeans Runner
// ============================================================

/**
 *  assignment ()
 *  Final Assignment Pass, d_assign  d_centroids
 *
 * @param cfg KMeans
 * @param h_data  [n, dim] row-major
 * @param d_assign : [n]
 * @param d_centroids : [k, dim]()
 * @param d_cnorm2 :centroid  [k]()
 * @param env StreamEnv: buffer
 * @param d_data_buffer  [B, dim](,/)
 * @param stream CUDA
 * @param handle cuBLAS handle
 * @param B batch
 * @param Ktile centroid
 * @param n
 * @param dim
 * @param k
 */
struct KMeansAssignmentTiming {
    double gemm_seconds = 0;
    double select_seconds = 0;
};

void perform_assignment_only(
    const KMeansCase& cfg,
    const float* h_data,
    idx_t* d_assign,
    const float* d_centroids,
    float* d_cnorm2,
    StreamEnv& env,
    float* d_data_buffer,
    cudaStream_t stream,
    cublasHandle_t handle,
    int B,
    int Ktile,
    int n,
    int dim,
    int k,
    float* d_objective_sum = nullptr,
    const float* d_precomputed_cnorm2 = nullptr,
    bool precomputed_cnorm_is_l2 = false,
    KMeansAssignmentTiming* timing = nullptr
);

/**
 * Lloyd : cuBLAS , GEMM/assign .
 *  K-means , create + with_workspace + destroy
 *  gpu_kmeans_lloyd  cudaMalloc/cublasCreate.
 */
struct KMeansLloydWorkspace;

KMeansLloydWorkspace* kmeans_lloyd_workspace_create(int max_n, int max_k, int dim);
void kmeans_lloyd_workspace_destroy(KMeansLloydWorkspace* ws);

/**
 *  Lloyd( cfg.n<=max_n, cfg.k<=max_k, cfg.dim==dim).
 */
void gpu_kmeans_lloyd_with_workspace(
    KMeansLloydWorkspace* ws,
    const KMeansCase& cfg,
    const float* h_data,
    idx_t* d_assign,
    float* d_centroids,
    float* h_objective
);

/**
 * GPU KMeans Lloyd (GEMM )
 *
 *  GEMM , K
 *  l2norm , cublas GEMM
 *
 * @param cfg KMeans
 * @param h_data  [n, dim] row-major (float),
 * @param d_assign : [n]
 * @param d_centroids : [k, dim] float(,),
 * @param h_objective :()
 */
void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] row-major,
    idx_t* d_assign,                 // [n]
    float* d_centroids,             // [k, dim] float (in/out),
    float* h_objective               // scalar (sum dist2)
);

/**
 * GPU KMeans Minibatch (GEMM )
 *
 * minibatch,
 *
 *
 * @param cfg KMeans
 * @param h_data  [n, dim] row-major (float)
 * @param d_assign : [n] (,minibatchassign)
 * @param d_centroids : [k, dim] float(,),
 * @param h_objective :()
 */
void gpu_kmeans_minibatch(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] row-major,
    idx_t* d_assign,                 // [n] ()
    float* d_centroids,             // [k, dim] float (in/out),
    float* h_objective               // scalar (sum dist2) ()
);

// ============================================================
// CPU Initialization Functions
// ============================================================

/**
 * : k
 *
 *  Fisher-Yates ,k
 *  CPU  GPU
 *
 * @param cfg KMeans ( seed)
 * @param data  [n, dim] row-major
 * @param out_centroids : [k, dim] row-major
 */
__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
);

/**
 *  init_centroids_by_sampling ().
 */
__host__ KMeansInitStats get_last_kmeans_init_stats();

__host__ KMeansLloydStats get_last_kmeans_lloyd_stats();

// ============================================================
// Vector Reordering After Clustering
// (ClusterInfo  clustering/clustering.cuh)
// ============================================================

/**
 * GPU Kernel: cluster
 */
__global__ void kernel_compute_cluster_indices(
    const int* __restrict__ assign,      // [n]
    int* __restrict__ cluster_indices,   // [n] :cluster
    int* __restrict__ cluster_counts,    // [k] :cluster()
    int n, int k
);

/**
 * GPU Kernel: Exclusive scan ()
 *  offsets[i] = sum(counts[0..i-1])
 *
 * @param counts :cluster [k]
 * @param offsets :cluster [k]
 * @param k cluster
 */
__global__ void kernel_exclusive_scan(
    const int* __restrict__ counts,   // [k]
    int* __restrict__ offsets,        // [k]
    int k
);

/**
 * GPU Kernel: assign
 *
 * @param data_in : [n, dim] row-major
 * @param assign : [n],assign[i] icluster
 * @param cluster_offsets :cluster [k]
 * @param cluster_indices :cluster [n]
 * @param data_out : [n, dim] row-major
 * @param n
 * @param dim
 */
__global__ void kernel_reorder_vectors_by_cluster(
    const float* __restrict__ data_in,   // [n, dim]
    const int* __restrict__ assign,      // [n]
    const int* __restrict__ cluster_offsets,  // [k]
    const int* __restrict__ cluster_indices,   // [n]
    float* __restrict__ data_out,        // [n, dim]
    int n, int dim
);

/**
 * GPU:(,)
 *
 * cluster,
 * ,GPU
 *
 * @param cfg KMeans
 * @param h_data_in : [n, dim] row-major(pageablepinned memory)
 * @param h_assign : [n]
 * @param h_data_out : [n, dim] row-major
 * @param h_cluster_info :cluster(offsetscounts),nullptr
 * @param device_id GPUID
 * @param B batch( 1<<20 = 1M)
 * @param stream CUDA,0
 */
void gpu_reorder_vectors_by_cluster(
    const KMeansCase& cfg,
    const float* h_data_in,    // [n,dim] CPU
    const int*   h_assign,     // [n] CPU
    float*       h_data_out,   // [n,dim] CPU
    ClusterInfo* h_cluster_info, // optional host output
    int device_id = 0,
    int B = (1 << 20),         // batch size (e.g. 1<<20)
    cudaStream_t stream = 0
);



/**
 * IVF K-means:(,)
 *
 * K-means,:
 * 1.
 * 2.
 * 3.
 *
 * @param cfg KMeans
 * @param h_data : [n, dim] row-major (pageable memory)
 * @param d_centroids : [k, dim] (,)
 * @param h_assign : [n],nullptr
 * @param h_cluster_sizes : [k],nullptr
 * @param use_minibatch Minibatch(true=Minibatch, false=Lloyd)
 * @param device_id GPUID
 * @param h_objective :,nullptr
 * @return true,false
 */
bool ivf_kmeans_clustering(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] CPU input
    float* d_centroids,            // [k, dim] GPU (in/out)
    idx_t* h_assign,                  // [n] CPU output (optional)
    int* h_cluster_sizes,           // [k] CPU output (optional)
    bool use_minibatch = false,    // true for minibatch, false for Lloyd
    int device_id = 0,
    float* h_objective = nullptr,  // optional output
    idx_t** d_assign_out = nullptr   // [out]  assign , host
);

/**
 * IVF K-means:K-means
 *
 * :
 * 1. K-means(LloydMinibatch)
 * 2. permutation(cluster)
 * 3. ()
 * 4. ()
 *
 * @param cfg KMeans
 * @param h_data_in : [n, dim] row-major (pageable memory)
 * @param h_data_out : [n, dim] row-major (pageable memory),
 * @param d_centroids : [k, dim] (,)
 * @param h_cluster_info :cluster(offsetscounts),nullptr
 * @param use_minibatch Minibatch(true=Minibatch, false=Lloyd)
 * @param device_id GPUID
 * @param batch_size permutationbatch( 1<<20)
 * @param h_objective :,nullptr
 * @param h_indices_in : [n], [0, 1, 2, ..., n-1],nullptr
 * @param h_indices_out : [n],,nullptr
 * @return true,false
 */
// ------------------------------------------------------------
// Out-of-place version (default, better performance)
// ------------------------------------------------------------
bool ivf_kmeans(
    const KMeansCase& cfg,
    const float* h_data_in,        // [n, dim] CPU input
    float* h_data_out,             // [n, dim] CPU output (must be pre-allocated)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch = false,    // true for minibatch, false for Lloyd
    int device_id = 0,
    int batch_size = (1 << 20),   // batch size for permutation building
    float* h_objective = nullptr, // optional output
    const idx_t* h_indices_in = nullptr,  // [n] CPU ()
    idx_t* h_indices_out = nullptr,       // [n] CPU ()
    KMeansReorderPolicy reorder_policy = KMeansReorderPolicy::Standard
);

// ------------------------------------------------------------
// In-place version (memory efficient, O(1) extra space)
// ------------------------------------------------------------
bool ivf_kmeans_inplace(
    const KMeansCase& cfg,
    float* h_data,                 // [n, dim] CPU in/out (in-place reorder)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch = false,    // true for minibatch, false for Lloyd
    int device_id = 0,
    int batch_size = (1 << 20),   // batch size for permutation building
    float* h_objective = nullptr, // optional output
    idx_t* h_indices = nullptr      // [n] CPU (,in-place reorder)
);

/**
 * ClusterInfo
 */
void free_cluster_info(ClusterInfo* info, bool is_device);

#endif // PGVECTOR_CUDA_KMEANS_CUH
