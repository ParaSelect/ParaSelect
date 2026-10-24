#ifndef PARASELECT_CUDA_CLUSTERING_KMEANS_FLASH_KMEANS_ASSIGN_CONFIG_CUH
#define PARASELECT_CUDA_CLUSTERING_KMEANS_FLASH_KMEANS_ASSIGN_CONFIG_CUH

namespace paraselect {
namespace clustering {
namespace kmeans {

struct FlashKMeansAssignConfig {
    int point_tile = 128;
    int centroid_tile = 256;
    int cutlass_k_tile = 32;
    int cutlass_stages = 2;
    int shared_reduce_min_tiles = 32768;
    int small_reduce_max_tiles = 8;
    int warp_size = 32;
    int variant = 8;          // 8=CUTLASS mainloop + fused tile partial-min()
    bool collect_timing = true;
};

inline FlashKMeansAssignConfig default_flash_kmeans_assign_config(int dim) {
    (void)dim;
    return FlashKMeansAssignConfig{};
}

}  // namespace kmeans
}  // namespace clustering
}  // namespace paraselect

#endif  // PARASELECT_CUDA_CLUSTERING_KMEANS_FLASH_KMEANS_ASSIGN_CONFIG_CUH
