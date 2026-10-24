#ifndef PARASELECT_CUDA_CLUSTERING_FLASH_KMEANS_FLASH_KMEANS_ASSIGN_V8_CUH
#define PARASELECT_CUDA_CLUSTERING_FLASH_KMEANS_FLASH_KMEANS_ASSIGN_V8_CUH

#include "utils/pch.h"

namespace paraselect {
namespace clustering {
namespace kmeans {

void launch_flash_assign_min_gemm_v8(
    const float* d_data,
    const float* d_centroids,
    const float* d_xnorm2,
    const float* d_cnorm2,
    int n,
    int k,
    int dim,
    int n_tiles,
    int point_tile,
    int centroid_tile,
    int cutlass_k_tile,
    int cutlass_stages,
    int shared_reduce_min_tiles,
    unsigned long long* d_partial_packed,
    cudaStream_t stream);

}  // namespace kmeans
}  // namespace clustering
}  // namespace paraselect

#endif  // PARASELECT_CUDA_CLUSTERING_FLASH_KMEANS_FLASH_KMEANS_ASSIGN_V8_CUH
