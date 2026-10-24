#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace paraselect::coarse {
// Largest scores, sorted descending; caller provides aligned FP32 input and outputs.
void adaptive_threshold_topk_fp32(const float* scores, int batch, int length, int stride,
                     int k, float* values, int64_t* indices, cudaStream_t stream);
// Exact ascending selector for an already materialized distance pool.  The
// input pool is scratch and may be negated in place; output values are
// restored to ascending distances.
void adaptive_threshold_topk_ascending(float* distances, int batch, int length, int stride,
                     int k, float* values, int64_t* indices, cudaStream_t stream);
// Exact selector for a pool that already stores negative distances.  It avoids
// a full-pool sign conversion; output values are restored to ascending L2.
void adaptive_threshold_topk_negative_distance(const float* negative_distances,
                     int batch, int length, int stride, int k,
                     float* values, int64_t* indices, cudaStream_t stream);
// Consumes GEMM workspace in place; output is ascending squared L2 distance.
void adaptive_threshold_topk_l2(float* inner, const float* query_norm, const float* centroid_norm,
                   int batch, int length, int k, float* distances, int64_t* indices,
                   cudaStream_t stream);
void adaptive_threshold_topk_fused(float* inner, const float* query_norm, const float* centroid_norm,
                        int batch, int length, int k, float* distances, int64_t* indices,
                        cudaStream_t stream);

// FP16 producer-consumer direct path for small K.  It computes each distance
// tile and immediately feeds the warp-local selector; no inner-product matrix
// is materialized.  Returns false when the shape is outside the specialized
// K<=64 path, allowing the caller to use the regular direct reducer.
bool adaptive_direct_fp16(const __half* queries, const __half* centers,
                          const float* query_norm, const float* centroid_norm,
                          int batch, int dim, int length, int k,
                          float* scratch, int scratch_stride,
                          float* distances, int64_t* indices,
                          cudaStream_t stream);
}
