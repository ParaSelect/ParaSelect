#include <cfloat>
#include <cstdint>
#include <optional>

#include "fusion_dist_topk.cuh"
#include "utils/pch.h"
#include "utils/utils.cuh"

#include "l2norm/l2norm.cuh"

#include <cuvs/selection/select_k.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/device_mdspan.hpp>

#define ENABLE_CUDA_TIMING 0

namespace {

template <typename T>
__global__ void cosine_distance_from_inner_product_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    T* __restrict__ d_dist,
    int batch_size,
    int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * len;
    if (tid >= total) return;

    int row = tid / len;
    int col = tid % len;

    T qn = d_query_norm[row];
    T dn = d_data_norm[col];
    T ip = d_inner_product[tid];

    T denom = qn * dn;
    if (denom < (T)1e-12) denom = (T)1e-12;

    //
    d_dist[tid] = (T)1.0 - ip / denom;
}

template <typename T>
__global__ void l2_distance_from_inner_product_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    T* __restrict__ d_dist,
    int batch_size,
    int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * len;
    if (tid >= total) return;

    int row = tid / len;
    int col = tid % len;

    T qn = d_query_norm[row];
    T dn = d_data_norm[col];
    T ip = d_inner_product[tid];

    // L2^2 = ||q||^2 + ||v||^2 - 2*<q,v>
    d_dist[tid] = qn * qn + dn * dn - (T)2.0 * ip;
}

inline void start_timer(cudaEvent_t& ev_start, cudaEvent_t& ev_stop) {
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, 0);
}

inline float stop_timer(cudaEvent_t ev_start, cudaEvent_t ev_stop) {
    cudaEventRecord(ev_stop, 0);
    cudaEventSynchronize(ev_stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return ms;
}

template <typename IdxT>
__global__ void init_rowwise_index_kernel_generic(IdxT* d_index, int batch_size, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * len;
    if (tid < total) {
        d_index[tid] = static_cast<IdxT>(tid % len);
    }
}

} // anonymous namespace

namespace paraselect {
namespace fusion_dist_topk_cuvs {

template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_cuvs(
    const T* d_query_norm,
    const T* d_data_norm,
    const T* d_inner_product,
    const IdxT* d_index,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream,
    raft::resources* external_res)
{
    if (!d_query_norm || !d_data_norm || !d_inner_product || !d_index ||
        !output_vals || !output_idx || batch_size <= 0 || len <= 0 || k <= 0) {
        return cudaErrorInvalidValue;
    }

    if (k > len) {
        return cudaErrorInvalidValue;
    }

    T* d_dist = nullptr;
    cudaError_t err = cudaMalloc(&d_dist, static_cast<size_t>(batch_size) * len * sizeof(T));
    if (err != cudaSuccess) {
        return err;
    }

    int threads = 256;
    int blocks = (batch_size * len + threads - 1) / threads;
    cosine_distance_from_inner_product_kernel<<<blocks, threads, 0, stream>>>(
        d_query_norm, d_data_norm, d_inner_product, d_dist, batch_size, len);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_dist);
        return err;
    }

    bool need_sync_stream = false;
    if (external_res) {
        raft::resource::set_cuda_stream(*external_res, rmm::cuda_stream_view(stream));
    } else {
        external_res = new raft::resources();
        raft::resource::set_cuda_stream(*external_res, rmm::cuda_stream_view(stream));
        need_sync_stream = true;
    }

    auto in_vals = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
        d_dist, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));

    auto in_idx_view = raft::make_device_matrix_view<const IdxT, int64_t, raft::row_major>(
        d_index, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));

    auto out_vals = raft::make_device_matrix_view<T, int64_t, raft::row_major>(
        output_vals, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));

    auto out_idx = raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
        output_idx, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));

    std::optional<decltype(in_idx_view)> in_idx_opt = in_idx_view;

    cuvs::selection::select_k(
        *external_res,
        in_vals,
        in_idx_opt,
        out_vals,
        out_idx,
        select_min,
        true
    );

    if (need_sync_stream) {
        raft::resource::sync_stream(*external_res);
        delete external_res;
    }
    err = cudaGetLastError();
    cudaFree(d_dist);
    return err;
}

template<typename T, typename IdxT>
cudaError_t select_k_cuvs(
    const T* input,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream,
    raft::resources* external_res)
{
    if (!input || !output_vals || !output_idx) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0 || len <= 0 || k <= 0 || k > len) {
        return cudaErrorInvalidValue;
    }

    IdxT* d_index = nullptr;
    cudaError_t err = cudaMalloc(&d_index, static_cast<size_t>(batch_size) * len * sizeof(IdxT));
    if (err != cudaSuccess) {
        return err;
    }

    int threads = 256;
    int blocks = (batch_size * len + threads - 1) / threads;
    init_rowwise_index_kernel_generic<IdxT><<<blocks, threads, 0, stream>>>(
        d_index, batch_size, len);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_index);
        return err;
    }

    bool need_sync_stream = false;
    if (external_res) {
        raft::resource::set_cuda_stream(*external_res, rmm::cuda_stream_view(stream));
    } else {
        external_res = new raft::resources();
        raft::resource::set_cuda_stream(*external_res, rmm::cuda_stream_view(stream));
        need_sync_stream = true;
    }

    auto in_vals = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
        input, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));

    auto in_idx_view = raft::make_device_matrix_view<const IdxT, int64_t, raft::row_major>(
        d_index, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));

    auto out_vals = raft::make_device_matrix_view<T, int64_t, raft::row_major>(
        output_vals, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));

    auto out_idx = raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
        output_idx, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));

    std::optional<decltype(in_idx_view)> in_idx_opt = in_idx_view;

    cuvs::selection::select_k(
        *external_res,
        in_vals,
        in_idx_opt,
        out_vals,
        out_idx,
        select_min,
        true
    );

    if (need_sync_stream) {
        raft::resource::sync_stream(*external_res);
        delete external_res;
    }
    err = cudaGetLastError();
    cudaFree(d_index);
    return err;
}

template cudaError_t fusion_cos_topk_cuvs<float, int>(
    const float*, const float*, const float*, const int*,
    int, int, int, float*, int*, bool, cudaStream_t, raft::resources*);

template cudaError_t fusion_cos_topk_cuvs<float, uint32_t>(
    const float*, const float*, const float*, const uint32_t*,
    int, int, int, float*, uint32_t*, bool, cudaStream_t, raft::resources*);

template cudaError_t fusion_cos_topk_cuvs<float, idx_t>(
    const float*, const float*, const float*, const idx_t*,
    int, int, int, float*, idx_t*, bool, cudaStream_t, raft::resources*);

template cudaError_t select_k_cuvs<float, int>(
    const float*, int, int, int, float*, int*, bool, cudaStream_t, raft::resources*);

template cudaError_t select_k_cuvs<float, uint32_t>(
    const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t, raft::resources*);

template cudaError_t select_k_cuvs<float, idx_t>(
    const float*, int, int, int, float*, idx_t*, bool, cudaStream_t, raft::resources*);

/**
 * L2 : L2  cuVS select_k
 */
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_cuvs(
    const T* d_query_norm,
    const T* d_data_norm,
    const T* d_inner_product,
    const IdxT* d_index,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream,
    raft::resources* external_res)
{
    if (!d_query_norm || !d_data_norm || !d_inner_product || !d_index ||
        !output_vals || !output_idx || batch_size <= 0 || len <= 0 || k <= 0) {
        return cudaErrorInvalidValue;
    }

    if (k > len) {
        return cudaErrorInvalidValue;
    }

    T* d_dist = nullptr;
    cudaError_t err = cudaMalloc(&d_dist, static_cast<size_t>(batch_size) * len * sizeof(T));
    if (err != cudaSuccess) {
        return err;
    }

    int threads = 256;
    int blocks = (batch_size * len + threads - 1) / threads;
    l2_distance_from_inner_product_kernel<<<blocks, threads, 0, stream>>>(
        d_query_norm, d_data_norm, d_inner_product, d_dist, batch_size, len);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_dist);
        return err;
    }

    bool need_sync_stream = false;
    if (external_res) {
        raft::resource::set_cuda_stream(*external_res, rmm::cuda_stream_view(stream));
    } else {
        external_res = new raft::resources();
        raft::resource::set_cuda_stream(*external_res, rmm::cuda_stream_view(stream));
        need_sync_stream = true;
    }

    auto in_vals = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
        d_dist, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));

    auto in_idx_view = raft::make_device_matrix_view<const IdxT, int64_t, raft::row_major>(
        d_index, static_cast<int64_t>(batch_size), static_cast<int64_t>(len));

    auto out_vals = raft::make_device_matrix_view<T, int64_t, raft::row_major>(
        output_vals, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));

    auto out_idx = raft::make_device_matrix_view<IdxT, int64_t, raft::row_major>(
        output_idx, static_cast<int64_t>(batch_size), static_cast<int64_t>(k));

    std::optional<decltype(in_idx_view)> in_idx_opt = in_idx_view;

    cuvs::selection::select_k(
        *external_res,
        in_vals,
        in_idx_opt,
        out_vals,
        out_idx,
        select_min,
        true
    );

    if (need_sync_stream) {
        raft::resource::sync_stream(*external_res);
        delete external_res;
    }
    err = cudaGetLastError();
    cudaFree(d_dist);
    return err;
}

template cudaError_t fusion_l2_topk_cuvs<float, int>(
    const float*, const float*, const float*, const int*,
    int, int, int, float*, int*, bool, cudaStream_t, raft::resources*);

template cudaError_t fusion_l2_topk_cuvs<float, uint32_t>(
    const float*, const float*, const float*, const uint32_t*,
    int, int, int, float*, uint32_t*, bool, cudaStream_t, raft::resources*);

template cudaError_t fusion_l2_topk_cuvs<float, idx_t>(
    const float*, const float*, const float*, const idx_t*,
    int, int, int, float*, idx_t*, bool, cudaStream_t, raft::resources*);

} // namespace fusion_dist_topk_cuvs
} // namespace paraselect
