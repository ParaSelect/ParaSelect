// pch.h
#ifndef PCH_H
#define PCH_H

// Standard library headers (must come before CUDA headers for Thrust compatibility)
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <limits.h>
#include <climits>
#include <limits>
#include <cstddef>
#include <cstdint>
#include "utils/types.h"

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <optional>
#include "cudatimer.h"

// Standard library headers
#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <time.h>
#include "utils/check_cuda_error.cuh"
#include "utils/nsight_nvtx.h"

//  CUDA ()
#ifndef ENABLE_CUDA_TIMING
#define ENABLE_CUDA_TIMING 0
#endif

#ifndef PARASELECT_DISTANCE_TYPE_DEFINED
#define PARASELECT_DISTANCE_TYPE_DEFINED
enum DistanceType {
    L2_DISTANCE = 0,
    COSINE_DISTANCE = 1
};
#endif

enum CpuFineVectorLayout {
    CPU_FINE_LAYOUT_UNSPECIFIED = 0,
    CPU_FINE_LAYOUT_FP32_ROW_MAJOR = 1,
    CPU_FINE_LAYOUT_FP32_AOSOA = 2,
    CPU_FINE_LAYOUT_U8_ROW_MAJOR = 3,
    CPU_FINE_LAYOUT_U8_PAGED = 4,
    CPU_FINE_LAYOUT_FP32_PAGED = 5,
};

enum CpuFineKernelKind {
    CPU_FINE_KERNEL_UNSPECIFIED = 0,
    CPU_FINE_KERNEL_FP32_AUTO = 1,
    CPU_FINE_KERNEL_FP32_AOSOA_V5 = 2,
    CPU_FINE_KERNEL_FP32_AOSOA_V6 = 3,
    CPU_FINE_KERNEL_U8 = 4,
};

enum CpuFineSchedulePolicy {
    CPU_FINE_SCHEDULE_POLICY_UNSPECIFIED = 0,
    CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC = 1,
    CPU_FINE_SCHEDULE_POLICY_CLUSTER_DYNAMIC_NUMA = 3,
};

// FINE_SEARCH_STRATEGY removed - only CPU_FP32_FINE_SEARCH is supported now

/**
 * IVF_DEBUG_INDEX_RANGE:  Stage3/Stage4
 *
 * ( 1,):
 *   - 1: , g_ivf_debug_index_range
 *   - 0: ,()
 *
 * :
 *   extern bool g_ivf_debug_index_range;
 *   g_ivf_debug_index_range = true;   //
 *   g_ivf_debug_index_range = false;  // ()
 *
 * (,):
 *   IVF_DEBUG_INDEX_RANGE=1 ./your_binary
 */
#ifndef IVF_DEBUG_INDEX_RANGE
#define IVF_DEBUG_INDEX_RANGE 1
#endif

#if IVF_DEBUG_INDEX_RANGE
#ifdef __cplusplus
extern bool g_ivf_debug_index_range;
#define IVF_DBG_IDX_RANGE (g_ivf_debug_index_range)
#endif
#else
#define IVF_DBG_IDX_RANGE (false)
#endif

#endif //PCH_H
