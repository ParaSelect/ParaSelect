#ifndef CUDA_CHECK_UTILS_CUH
#define CUDA_CHECK_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * ============================
 * CUDA ERROR CHECK UTILITIES
 * ============================
 *
 * :
 *
 * 1. CUDA_CHECK(call)
 *    →  CUDA Runtime API(cudaMemcpy / cudaMalloc / cudaStreamSynchronize )
 *
 * 2. CUDA_LAUNCH_CHECK()
 *    →  kernel launch , launch/config
 *
 * 3. CUDA_SYNC_CHECK(stream, tag)
 *    →  stream,()
 *
 */


/* ============================================================
 * ()
 * ============================================================
 *
 * [ 1] CUDA API  CUDA_CHECK
 *
 *    CUDA_CHECK(cudaMemcpy(...));
 *    CUDA_CHECK(cudaMalloc(...));
 *    CUDA_CHECK(cudaStreamSynchronize(...));
 *
 *
 * [ 2] kernel launch  CUDA_LAUNCH_CHECK
 *
 *    kernel<<<...>>>(...);
 *    CUDA_LAUNCH_CHECK("kernel<<<...>>>");
 *
 *
 * [ 3]debug
 *
 *    CUDA_SYNC_CHECK(stream, "after kernel A");
 *
 *
 * [ 4] cudaGetLastError() ""
 *
 *    ❌ :cudaMemcpy(...); CHECK_CUDA_ERRORS;
 *    ✅ :CUDA_CHECK(cudaMemcpy(...));
 *
 *
 * [ 5] stream  stream sync, device sync
 *
 * ============================================================ */
void _check_cuda_last_error(const char *file, int line);
#define CHECK_CUDA_ERRORS _check_cuda_last_error(__FILE__, __LINE__);/*CUDA*/

/* ============================================================
 * 1.  CUDA Runtime API()
 * ============================================================ */
inline void _cuda_check(cudaError_t err, const char* expr, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr,
                "\n[CUDA API ERROR]\n"
                "  Expr   : %s\n"
                "  File   : %s:%d\n"
                "  Code   : %d\n"
                "  Name   : %s\n"
                "  Desc   : %s\n\n",
                expr, file, line,
                (int)err,
                cudaGetErrorName(err),
                cudaGetErrorString(err));

        cudaDeviceReset();
        abort();
    }
}

#define CUDA_CHECK(call) \
    do { \
        _cuda_check((call), #call, __FILE__, __LINE__); \
    } while (0)


/* ============================================================
 * 2.  Kernel Launch( launch/config )
 * ============================================================ */
inline void _cuda_launch_check(const char* file, int line, const char* kernel)
{
    cudaError_t err = cudaGetLastError();  //  launch/config

    if (err != cudaSuccess) {
        fprintf(stderr,
                "\n[CUDA KERNEL LAUNCH ERROR]\n"
                "  Kernel : %s\n"
                "  File   : %s:%d\n"
                "  Name   : %s\n"
                "  Desc   : %s\n\n",
                kernel, file, line,
                cudaGetErrorName(err),
                cudaGetErrorString(err));

        cudaDeviceReset();
        abort();
    }
}

#define CUDA_LAUNCH_CHECK(kernel_call_str) \
    _cuda_launch_check(__FILE__, __LINE__, kernel_call_str)


/* ============================================================
 * 3.  Kernel ( → )
 * ============================================================ */
inline void _cuda_sync_check(cudaStream_t stream,
                            const char* file,
                            int line,
                            const char* tag)
{
    cudaError_t err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        int dev = -1;
        cudaGetDevice(&dev);

        fprintf(stderr,
                "\n[CUDA ASYNC EXECUTION ERROR]\n"
                "  Tag    : %s\n"
                "  Device : %d\n"
                "  Stream : %p\n"
                "  File   : %s:%d\n"
                "  Name   : %s\n"
                "  Desc   : %s\n\n",
                tag,
                dev,
                (void*)stream,
                file,
                line,
                cudaGetErrorName(err),
                cudaGetErrorString(err));

        cudaDeviceReset();
        abort();
    }
}

/**
 * Debug
 *
 * :
 * CUDA_SYNC_CHECK(stream, "after stage X");
 */
#define CUDA_SYNC_CHECK(stream, tag) \
    _cuda_sync_check(stream, __FILE__, __LINE__, tag)


/* ============================================================
 * 4. Debug ()
 * ============================================================ */
#ifdef CUDA_DEBUG_SYNC

#define CUDA_DEBUG_SYNC_CHECK(stream, tag) \
    CUDA_SYNC_CHECK(stream, tag)

#else

#define CUDA_DEBUG_SYNC_CHECK(stream, tag) \
    do {} while (0)

#endif


#endif
