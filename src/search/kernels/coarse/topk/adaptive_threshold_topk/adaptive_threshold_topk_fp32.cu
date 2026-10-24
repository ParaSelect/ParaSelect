#include "adaptive_threshold_topk_fp32.h"
#include "validation.h"
#include "small_topk.cuh"
#include "utils/pch.h"
#include "cuda_kernels/v3_fp32/topk_select.cuh"
#include <stdexcept>
namespace paraselect::coarse {
// The TMA selector is tuned for wide rows and large K.  For the small-K
// adaptive cases a single warp can keep the complete candidate set in
// registers and sort one compact shared tile, avoiding the score transform
// and TMA/radix passes.
template<int K> void launch(const TopkSelectArgs& args) {
    // 16 values/thread; compact candidate and two-stage TMA buffers fit SM120.
    using Config = TopkSelectConfig<float,int64_t,true,false,true,K,128,1,2048,2048,2>;
    using Kernel = topk_select_fp32::TopkSelectKernelFP32<Config>;
    static_assert(sizeof(typename Kernel::SharedMemoryPlanFP32) <= 99*1024);
    topk_select_fp32::run_topk_select_kernel<Config>(args);
}
// Large k selects exact 2048-item blocks within SM120's shared-memory limit.
// A private score copy preserves the caller's input and stream ordering.
__global__ void exclude_selected(float* scores, const int64_t* ids,
                                 int batch, int length, int output_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * 2048) {
        const int q = i / 2048;
        scores[size_t(q) * length + ids[size_t(q) * output_stride + i % 2048]] = -INFINITY;
    }
}
// Masked entries tie with genuine -inf inputs. Repair only that degenerate
// suffix, choosing distinct original -inf IDs not retained by the first pass.
__global__ void repair_negative_infinity_ids(const float* scores, int length, int stride,
                                            int k, const float* values, int64_t* ids) {
    const int q = blockIdx.x;
    if (threadIdx.x != 0 || values[size_t(q)*k+k-1] != -INFINITY) return;
    int next = 0;
    for (int rank = 2048; rank < k; ++rank) {
        if (values[size_t(q)*k+rank] != -INFINITY) continue;
        for (; next < length; ++next) {
            if (scores[size_t(q)*stride+next] != -INFINITY) continue;
            bool used = false;
            for (int j = 0; j < 2048; ++j)
                used |= ids[size_t(q)*k+j] == next;
            if (!used) break;
        }
        ids[size_t(q)*k+rank] = next++;
    }
}
void adaptive_threshold_topk_fp32(const float* scores,int batch,int length,int stride,int k,
                     float* values,int64_t* indices,cudaStream_t stream) {
    detail::validate_topk_input(scores, batch, length, stride, k, values, indices);
    if (small_topk::launch_scores_if_supported(scores, batch, length, stride, k,
                                               values, indices, stream)) {
        CUDA_LAUNCH_CHECK("adaptive small-K score selector");
        return;
    }
    int device=0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    TopkSelectArgs a{};
    a.batch_size=batch; a.vocab_size=length; a.topk=k;
    a.input=const_cast<float*>(scores); a.output_value=values; a.output_index=indices;
    a.stride_input_batch=stride; a.stride_output_value_batch=k; a.stride_output_index_batch=k;
    a.sorted_value=true; a.return_value=true; a.idx_oob_fill_value=-1;
    a.value_oob_fill_value=-INFINITY; a.abort_when_nan_found=true;
    a.shared_memory_size_per_sm=prop.sharedMemPerBlockOptin; a.stream=stream;
    if (k > 2048) {
        float* remaining = nullptr;
        CUDA_CHECK(cudaMallocAsync(&remaining, size_t(batch) * stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemcpy2DAsync(remaining, size_t(stride) * sizeof(float),
            scores, size_t(stride) * sizeof(float), size_t(length) * sizeof(float),
            batch, cudaMemcpyDeviceToDevice, stream));
        a.topk = 2048;
        launch<2048>(a);
        a.input = remaining;
        for (int start = 2048; start < k; start += 2048) {
            exclude_selected<<<(batch * 2048 + 255) / 256, 256, 0, stream>>>(
                remaining, indices + start - 2048, batch, stride, k);
            a.topk = std::min(2048, k - start);
            a.output_value = values + start;
            a.output_index = indices + start;
            launch<2048>(a);
        }
        repair_negative_infinity_ids<<<batch,32,0,stream>>>(
            scores, length, stride, k, values, indices);
        CUDA_CHECK(cudaFreeAsync(remaining, stream));
        return;
    }
    if (k <= 128) launch<128>(a);
    else if (k <= 256) launch<256>(a);
    else if (k <= 512) launch<512>(a);
    else if (k <= 1024) launch<1024>(a);
    else launch<2048>(a);
}

__global__ void negate_selector_pool(float* values, size_t total) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += size_t(blockDim.x) * gridDim.x) {
        values[i] = -values[i];
    }
}

__global__ void restore_selector_values(float* values, size_t total) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += size_t(blockDim.x) * gridDim.x) {
        values[i] = -values[i];
    }
}

void adaptive_threshold_topk_ascending(float* distances, int batch, int length, int stride,
                     int k, float* values, int64_t* indices, cudaStream_t stream) {
    detail::validate_topk_input(distances, batch, length, stride, k, values, indices);
    const size_t total = size_t(batch) * stride;
    negate_selector_pool<<<std::min<size_t>(65535, (total + 255) / 256), 256, 0, stream>>>(
        distances, total);
    adaptive_threshold_topk_fp32(distances, batch, length, stride, k, values, indices, stream);
    restore_selector_values<<<std::min<size_t>(65535, (size_t(batch) * k + 255) / 256), 256, 0, stream>>>(
        values, size_t(batch) * k);
}

void adaptive_threshold_topk_negative_distance(const float* negative_distances,
                     int batch, int length, int stride, int k,
                     float* values, int64_t* indices, cudaStream_t stream) {
    detail::validate_topk_input(negative_distances, batch, length, stride, k, values, indices);
    adaptive_threshold_topk_fp32(negative_distances, batch, length, stride, k,
                                  values, indices, stream);
    restore_selector_values<<<std::min<size_t>(65535, (size_t(batch) * k + 255) / 256), 256, 0, stream>>>(
        values, size_t(batch) * k);
}

__global__ void make_negative_l2(float* inner,const float* qn,const float* cn,int n,size_t total) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(blockDim.x)*gridDim.x) {
        float q=qn[i/n], c=cn[i%n];
        inner[i]=-(q*q+c*c-2.0f*inner[i]);
    }
}
__global__ void restore_l2(float* values,size_t total) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(blockDim.x)*gridDim.x)
        values[i]=-values[i];
}
void adaptive_threshold_topk_l2(float* inner,const float* qn,const float* cn,int batch,int n,int k,
                   float* distances,int64_t* ids,cudaStream_t stream) {
    detail::validate_l2_input(inner, qn, cn, batch, n, k, distances, ids);
    if (small_topk::launch_l2_if_supported(inner, qn, cn, batch, n, k,
                                          distances, ids, stream)) {
        CUDA_LAUNCH_CHECK("adaptive small-K L2 selector");
        restore_l2<<<(size_t(batch)*k+255)/256,256,0,stream>>>(
            distances, size_t(batch)*k);
        return;
    }
    make_negative_l2<<<std::min(size_t(65535),(size_t(batch)*n+255)/256),256,0,stream>>>(inner,qn,cn,n,size_t(batch)*n);
    adaptive_threshold_topk_fp32(inner,batch,n,n,k,distances,ids,stream);
    restore_l2<<<(size_t(batch)*k+255)/256,256,0,stream>>>(distances,size_t(batch)*k);
}

bool adaptive_direct_fp16(const __half* queries, const __half* centers,
                          const float* qn, const float* cn,
                          int batch, int dim, int length, int k,
                          float* scratch, int scratch_stride,
                          float* distances, int64_t* ids, cudaStream_t stream) {
    if (!queries || !centers || !qn || !cn || !distances || !ids ||
        !scratch || scratch_stride <= 0 ||
        batch <= 0 || dim != 128 || length <= 0 || k <= 0 || k > length) {
        return false;
    }
    if (!small_topk::launch_l2_vectors_if_supported(
            queries, centers, qn, cn, batch, dim, length, k,
            scratch, scratch_stride, distances, ids, stream)) {
        return false;
    }
    CUDA_LAUNCH_CHECK("adaptive direct FP16 producer-consumer selector");
    return true;
}
}
