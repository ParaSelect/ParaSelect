#pragma once
#include <cuda_runtime_api.h>

namespace paraselect::coarse {
// Distribute eight-value groups across the row, using every fetched sector.
// This estimates a threshold only; full counts still establish exact top-k.
#ifdef __CUDACC__
__host__ __device__
#endif
inline int coarse_histogram_sample_index(int i, int n, int samples) {
    if (n == samples || samples < 8) return int((static_cast<unsigned long long>(i) * n) / samples);
    const int groups = samples / 8;
    return int((static_cast<unsigned long long>(i / 8) * n) / groups) + i % 8;
}

#ifdef __CUDACC__
__device__ __forceinline__ float coarse_l2_distance(float dot,float query_norm,float centroid_norm) {
    // Fix contraction order: all histogram/selection passes must classify
    // a distance identically, including values exactly at a bucket edge.
    const float norm_sum=__fmaf_rn(centroid_norm,centroid_norm,__fmul_rn(query_norm,query_norm));
    return __fsub_rn(norm_sum,__fmul_rn(2.f,dot));
}
#endif
// Shared distance distribution for seed estimation and exact coarse selection.
// Workspace is caller-owned; bucket edges are inclusive upper bounds.
struct CoarseHistogramQuerySubset {
    const int* ids = nullptr;
    const int* count = nullptr;
};
struct CoarseDistanceHistogram {
    static constexpr int bins = 256;
    unsigned* counts = nullptr; // [queries,bins]
    float* minimum = nullptr;
    float* maximum = nullptr;
    float* lower = nullptr;    // lower bound (exclusive)
    float* upper = nullptr;    // current threshold (inclusive)
    int* selected_counts = nullptr;
    int* status = nullptr;
};
void coarse_distance_histogram_build(const float* distances, int queries, int nlist,
                                CoarseDistanceHistogram workspace, cudaStream_t stream);
// Parallel range/histogram passes for large centroid rows.
void coarse_distance_histogram_build_tiled(const float* distances, int queries, int nlist,
                                          CoarseDistanceHistogram workspace, cudaStream_t stream);
// Reads GEMM scores without materializing L2 distances. Norm inputs are
// L2 norms, not squared norms. Range and histogram use identical arithmetic.
void coarse_distance_histogram_build_l2_tiled(const float* scores, const float* query_norm,
    const float* centroid_norm, int queries, int nlist,
    CoarseDistanceHistogram workspace, cudaStream_t stream);
// Samples only the binning range; all distances participate in the counts.
// Outer buckets cover (-inf, first edge] and (last finite edge, +inf).
void coarse_distance_histogram_select_l2(const float* scores, const float* query_norm,
    const float* centroid_norm, int queries, int nlist, int target,
    CoarseDistanceHistogram workspace, cudaStream_t stream, const int* complete = nullptr,
    CoarseHistogramQuerySubset subset = {});
// Estimate only: callers must verify the full candidate count before selection.
void coarse_distance_histogram_estimate_l2(const float* scores, const float* query_norm,
    const float* centroid_norm, int queries, int nlist, int target,
    CoarseDistanceHistogram workspace, cudaStream_t stream);
// Exact selection: lower excludes the boundary bin, upper includes it.
void coarse_distance_histogram_boundary(int queries, int target,
                                       CoarseDistanceHistogram workspace, cudaStream_t stream);
}
