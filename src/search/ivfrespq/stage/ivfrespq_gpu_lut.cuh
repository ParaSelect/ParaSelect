#ifndef PARASELECT_IVFRESPQ_GPU_LUT_CUH
#define PARASELECT_IVFRESPQ_GPU_LUT_CUH

#include "utils/pch.h"

#include <cuda_runtime.h>

void ivfrespq_build_lut_gpu(
    const float* d_queries,
    const float* d_centroids,
    const idx_t* d_top_nprobe,
    const float* d_codebook,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int pq_M,
    int pq_K,
    float* d_lut,
    cudaStream_t stream);

#endif  // PARASELECT_IVFRESPQ_GPU_LUT_CUH
