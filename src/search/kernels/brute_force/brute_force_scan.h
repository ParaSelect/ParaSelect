#ifndef BRUTE_FORCE_SCAN_H
#define BRUTE_FORCE_SCAN_H

#include "utils/pch.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

enum class BruteForceGemmMathMode {
    kCudaCore = 0,
    kTensorCore = 1,
};

/**
 * Brute force scan ()-
 *
 * @param d_query_fp16  [n_query, n_dim] (FP16)
 * @param d_data_fp16  [n_data, n_dim] (FP16)
 * @param d_query_norm_f32  L2  [n_query] (FP32)
 * @param d_data_norm_f32  L2  [n_data] (FP32)
 * @param n_query
 * @param n_data
 * @param n_dim
 * @param k  topk
 * @param distance_mode  (L2_DISTANCE  COSINE_DISTANCE)
 * @param d_topk_dist :topk  [n_query, k] (FP32)
 * @param d_topk_index :topk  [n_query, k]
 * @param stream CUDA
 */
void brute_force_scan_gemmex_topk(
    const __half* d_query_fp16,
    const __half* d_data_fp16,
    const float* d_query_norm_f32,
    const float* d_data_norm_f32,
    int n_query,
    int n_data,
    int n_dim,
    int k,
    int distance_mode,
    float* d_topk_dist,
    idx_t* d_topk_index,
    cudaStream_t stream);

/**
 * ( FP32->FP16 )
 *
 * @param h_query  [n_query, n_dim] (FP32)
 * @param h_data  [n_data, n_dim] (FP32)
 * @param n_query
 * @param n_data
 * @param n_dim
 * @param k  topk
 * @param distance_mode  (L2_DISTANCE  COSINE_DISTANCE)
 * @param h_topk_dist : topk  [n_query, k] (FP32)
 * @param h_topk_index : topk  [n_query, k]
 */
void brute_force_scan(
    const float* h_query,
    const float* h_data,
    int n_query,
    int n_data,
    int n_dim,
    int k,
    int distance_mode,
    float* h_topk_dist,
    idx_t* h_topk_index);

/**
 * ()-
 *
 * @param d_query_f32  [n_query, n_dim] (FP32)
 * @param d_data_f32  [n_data, n_dim] (FP32)
 * @param n_query
 * @param n_data
 * @param n_dim
 * @param k  topk
 * @param distance_mode  (L2_DISTANCE  COSINE_DISTANCE)
 * @param d_topk_dist : topk  [n_query, k] (FP32)
 * @param d_topk_index : topk  [n_query, k]
 * @param enable_timing  CUDATimer
 */
void brute_force_scan_device_data(
    const float* d_query_f32,
    const float* d_data_f32,
    int n_query,
    int n_data,
    int n_dim,
    int k,
    int distance_mode,
    float* d_topk_dist,
    idx_t* d_topk_index,
    bool enable_timing = true);

/**
 *  -
 * chunks,
 *
 * @param h_query  [n_query, n_dim] (FP32)
 * @param h_data  [n_data, n_dim] (FP32)
 * @param n_query
 * @param n_data
 * @param n_dim
 * @param k  topk
 * @param distance_mode  (L2_DISTANCE  COSINE_DISTANCE)
 * @param h_topk_dist : topk  [n_query, k] (FP32)
 * @param h_topk_index : topk  [n_query, k]
 * @param max_chunk_size_mb (MB),0
 * @param enable_timing
 * @param gemm_math_mode GEMM math mode: CUDA-core pedantic FP32 or TF32 Tensor Core
 */
void brute_force_scan_chunked(
    const float* h_query,
    const float* h_data,
    int n_query,
    int n_data,
    int n_dim,
    int k,
    int distance_mode,
    float* h_topk_dist,
    idx_t* h_topk_index,
    size_t max_chunk_size_mb = 0,
    bool enable_timing = true,
    BruteForceGemmMathMode gemm_math_mode = BruteForceGemmMathMode::kCudaCore);

#endif // BRUTE_FORCE_SCAN_H
