/**
 * Fusion Distance Top-K Launcher - Capacity 512/1024/2048
 *
 * Uses HeapSort-based fused kernel for large k values (256 < k <= 2048).
 */

#include "fusion_dist_topk_heapsort_impl.cuh"
#include "fusion_dist_topk_launcher.cuh"

namespace paraselect {
namespace fusion_dist_topk_heapsort {

// Explicit instantiations for kernel templates
template __global__ void fusion_cos_topk_heapsort_kernel<true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_heapsort_kernel<false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_cos_topk_heapsort_kernel<true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_cos_topk_heapsort_kernel<false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);

template __global__ void fusion_l2_topk_heapsort_kernel<true, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_heapsort_kernel<false, float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*);
template __global__ void fusion_l2_topk_heapsort_kernel<true, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);
template __global__ void fusion_l2_topk_heapsort_kernel<false, float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*);

} // namespace fusion_dist_topk_heapsort

namespace fusion_dist_topk_warpsort {

using namespace fusion_dist_topk_heapsort;

// Host launch functions for 256 < k <= 1024 using HeapSort
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort_large(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream)
{
    dim3 grid(batch_size);
    dim3 block(kBlockThreads);

    size_t smem_bytes =
        sizeof(T) * kWarpsPerBlock * k +
        sizeof(IdxT) * kWarpsPerBlock * k +
        sizeof(T) * kWarpsPerBlock * kWarpSize_ +
        sizeof(IdxT) * kWarpsPerBlock * kWarpSize_;

    if (select_min) {
        cudaFuncSetAttribute(
            fusion_cos_topk_heapsort_kernel<true, T, IdxT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));

        fusion_cos_topk_heapsort_kernel<true, T, IdxT>
            <<<grid, block, smem_bytes, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
    } else {
        cudaFuncSetAttribute(
            fusion_cos_topk_heapsort_kernel<false, T, IdxT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));

        fusion_cos_topk_heapsort_kernel<false, T, IdxT>
            <<<grid, block, smem_bytes, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
    }
    return cudaGetLastError();
}

template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort_large(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream)
{
    dim3 grid(batch_size);
    dim3 block(kBlockThreads);

    size_t smem_bytes =
        sizeof(T) * kWarpsPerBlock * k +
        sizeof(IdxT) * kWarpsPerBlock * k +
        sizeof(T) * kWarpsPerBlock * kWarpSize_ +
        sizeof(IdxT) * kWarpsPerBlock * kWarpSize_;

    if (select_min) {
        cudaFuncSetAttribute(
            fusion_l2_topk_heapsort_kernel<true, T, IdxT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));

        fusion_l2_topk_heapsort_kernel<true, T, IdxT>
            <<<grid, block, smem_bytes, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
    } else {
        cudaFuncSetAttribute(
            fusion_l2_topk_heapsort_kernel<false, T, IdxT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));

        fusion_l2_topk_heapsort_kernel<false, T, IdxT>
            <<<grid, block, smem_bytes, stream>>>(
                d_query_norm, d_data_norm, d_inner_product,
                batch_size, len, k, output_vals, output_idx);
    }
    return cudaGetLastError();
}

// Explicit instantiations for host functions
template cudaError_t fusion_cos_topk_warpsort_large<float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*, bool, cudaStream_t);
template cudaError_t fusion_cos_topk_warpsort_large<float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

template cudaError_t fusion_l2_topk_warpsort_large<float, int>(
    const float*, const float*, const float*, int, int, int, float*, int*, bool, cudaStream_t);
template cudaError_t fusion_l2_topk_warpsort_large<float, uint32_t>(
    const float*, const float*, const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace fusion_dist_topk_warpsort
} // namespace paraselect
