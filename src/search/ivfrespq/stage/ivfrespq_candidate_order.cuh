#ifndef PARASELECT_IVFRESPQ_CANDIDATE_ORDER_CUH
#define PARASELECT_IVFRESPQ_CANDIDATE_ORDER_CUH

#include <cuda_runtime.h>

void ivfrespq_order_candidates_by_local_idx_gpu(
    int* d_candidate_idx,
    float* d_candidate_dist,
    int batch_n,
    int candidate_k,
    cudaStream_t stream,
    int candidate_stride = 0);

#endif  // PARASELECT_IVFRESPQ_CANDIDATE_ORDER_CUH
