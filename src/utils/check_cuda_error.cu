/**
 * CUDA Error Check Utilities Implementation
 *
 * Implementation of CUDA error checking functions.
 */

#include "pch.h"
#include "utils/check_cuda_error.cuh"

void _check_cuda_last_error(const char *file, int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "[CUDA Last Error]: %s ---- Location: %s:%d\n",
                cudaGetErrorString(err), file, line);

        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
