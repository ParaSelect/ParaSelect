#pragma once

#include "utils/pch.h"
#include <cuda_fp16.h>
#include <cstddef>
#include "search/kernels/coarse/coarse_policy.h"

namespace paraselect::coarse {

inline bool adaptive_threshold_topk_gemm_l2_fp16_supported(
    int batch, int dim, int nlist, int k) {
    return fused_gemm_shape_supported(batch,dim,nlist,k);
}

void prepare_adaptive_threshold_topk_gemm_l2_fp16();

// Conservative capacity valid for this batch and all smaller tail batches.
// Returns a count of floats (scores and packed IDs), not bytes.
size_t adaptive_threshold_topk_gemm_l2_fp16_workspace_floats(
    int batch, int nlist, int k);

// Exact selection of scores computed from FP16 dot products and supplied FP32
// L2 norms (not squared norms). Not equivalent to full-FP32 distances.
// Inputs/norms must be finite, with representable intermediate scores; pointers
// to half matrices must be 32-byte aligned. No query-time allocation occurs.
// Top1 ties choose the smallest centroid ID. For top8/16/32, equal-distance
// membership/order is unspecified. Output distances are ascending.
// Returns false before launch for unsupported shapes, null/misaligned inputs,
// or insufficient scratch.
bool adaptive_threshold_topk_gemm_l2_fp16(
    const __half* queries,
    const __half* centers,
    const float* query_norm,
    const float* center_norm,
    int batch,
    int dim,
    int nlist,
    int k,
    float* workspace,
    size_t workspace_floats,
    float* distances,
    idx_t* ids,
    cudaStream_t stream);

// Variant for an explicitly packed center view. The packed layout consists of
// one 8-row x 16-dimension center tile per 128 half values, packed in the
// MMA lane fragment order. This is opt-in; the default wrapper above retains
// the row-major layout contract.
bool adaptive_threshold_topk_gemm_l2_fp16_with_layout(
    const __half* queries,
    const __half* centers,
    const float* query_norm,
    const float* center_norm,
    int batch,
    int dim,
    int nlist,
    int k,
    float* workspace,
    size_t workspace_floats,
    float* distances,
    idx_t* ids,
    bool centers_tiled,
    cudaStream_t stream);

}  // namespace paraselect::coarse
