/**
 * @file nsight_nvtx.h
 *
 * Nsight Systems / NVTX (`USE_NVTX`  CMake `PARASELECT_USE_NVTX` ; no-op).
 * IVF Stage,, `#if USE_NVTX`.
 *
 * - NVTX_PUSH / NVTX_POP:(, **CPU ** )
 * - NVTX_MARK:(, nvtxMarkA)
 * - NVTX_NAME_CUDA_STREAM / NVTX_NAME_CUDA_EVENT: **CUDA stream / event** ;
 *   Nsight Systems  **CUDA HW**  stream , **H2D  kernel  stream **.
 *    `nvToolsExt`;`nvToolsExtCudaRt.h`  CUDA Toolkit( `nvtx3/` , include ).
 *   :`nvtxNameCudaStreamA` / `nvtxNameCudaEventA` **** `name` ,****(),.
 */
#ifndef PARASELECT_NSIGHT_NVTX_H
#define PARASELECT_NSIGHT_NVTX_H

#if defined(USE_NVTX) && USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)
#if defined(__CUDACC__) || defined(__CUDA__)
/*  stream : Nsight  CUDA GPU → Streams, Timeline  stream
 * : nvtx3  CUDA Runtime , nvToolsExt.h  */
#include <nvtx3/nvToolsExtCudaRt.h>
#define NVTX_NAME_CUDA_STREAM(stream, name) nvtxNameCudaStreamA((stream), (name))
#define NVTX_NAME_CUDA_EVENT(ev, name) nvtxNameCudaEventA((ev), (name))
#else
#define NVTX_NAME_CUDA_STREAM(stream, name) ((void)0)
#define NVTX_NAME_CUDA_EVENT(ev, name) ((void)0)
#endif
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP() ((void)0)
#define NVTX_MARK(name) ((void)0)
#define NVTX_NAME_CUDA_STREAM(stream, name) ((void)0)
#define NVTX_NAME_CUDA_EVENT(ev, name) ((void)0)
#endif

#endif /* PARASELECT_NSIGHT_NVTX_H */
