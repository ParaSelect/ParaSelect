/**
 *  fp16():
 *  - :half query/centers  cublasGemmEx,CUDA_R_16F ,CUBLAS_COMPUTE_32F , float ;
 *     CUBLAS_GEMM_DEFAULT_TENSOR_OP  Tensor Core; half  L2/cos.
 *  - GEMM  float, coarse runner  FP32 norm  L2 top-k policy;
 *     PARASELECT_FP16_COARSE=1 + PARASELECT_COARSE_STRATEGY=auto.
 */
#include "coarse_fp16.cuh"
#include "utils/pch.h"
#include "utils/utils.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

void coarse_stage_fp16_gemm(
    const __half* d_query_fp16,
    const __half* d_centers_fp16,
    int n_query,
    int n_dim,
    int n_total_clusters,
    float* d_inner_product_f32,
    cublasHandle_t cublas_handle,
    cudaStream_t stream)
{
    if (cublasSetStream(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("coarse fp16 failed to set cuBLAS stream");
    }
    const float alpha = 1.f;
    const float beta = 0.f;

    const cublasStatus_t gemm_st = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n_total_clusters,
        n_query,
        n_dim,
        &alpha,
        d_centers_fp16,
        CUDA_R_16F,
        n_dim,
        d_query_fp16,
        CUDA_R_16F,
        n_dim,
        &beta,
        d_inner_product_f32,
        CUDA_R_32F,
        n_total_clusters,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (gemm_st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("coarse fp16 GEMM failed");
    }

    CUDA_LAUNCH_CHECK("coarse fp16 GEMM");
}
