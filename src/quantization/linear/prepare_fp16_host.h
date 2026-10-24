#ifndef PREPARE_FP16_HOST_H
#define PREPARE_FP16_HOST_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace ivf {

/**
 * Host  fp32  fp16  device( host ).
 *  query  d_*_fp16 , ivf_search fp16 .
 *  cudaFree(d_fp16_out).
 */
void convert_fp32_to_fp16_upload(
    const float* h_fp32,
    int n,
    int dim,
    __half** d_fp16_out
);

/**
 *  device buffer, fp32  fp16 .
 * d_fp16_buf  >= n * dim * sizeof(__half).
 */
void convert_fp32_to_fp16_inplace(
    const float* h_fp32,
    int n,
    int dim,
    __half* d_fp16_buf
);

} // namespace ivf

#endif
