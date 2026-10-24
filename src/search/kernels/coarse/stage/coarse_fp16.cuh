#ifndef COARSE_FP16_CUH
#define COARSE_FP16_CUH

#include "utils/pch.h"
#include <cuda_fp16.h>

/**
 * Internal FP16 GEMM primitive. Production callers use coarse::search.
 */
void coarse_stage_fp16_gemm(
    const __half* d_query_fp16,
    const __half* d_centers_fp16,
    int n_query,
    int n_dim,
    int n_total_clusters,
    float* d_inner_product_f32,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);

#endif
