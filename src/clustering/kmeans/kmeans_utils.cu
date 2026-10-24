#include "kmeans.cuh"
#include "utils/utils.cuh"
#include "clustering/utils.cuh"
#include "l2norm/l2norm.cuh"
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// ============================================================
// StreamEnv Implementation
// ============================================================

void StreamEnv::allocate(int B, int Ktile, int dot_B) {
    if (dot_B <= 0) {
        dot_B = B;
    }
    cudaMalloc(&d_xnorm2,    sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_dist2, sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_idx,   sizeof(idx_t)   * (size_t)B);
    cudaMalloc(&d_dot,        sizeof(float) * (size_t)dot_B * (size_t)Ktile);
}

void StreamEnv::allocate_accum_sort(int B, int k) {
    cudaMalloc(&d_sorted_keys, sizeof(idx_t) * (size_t)B);
    cudaMalloc(&d_sorted_vals, sizeof(idx_t) * (size_t)B);
    cudaMalloc(&d_sample_idx,  sizeof(idx_t) * (size_t)B);
    cudaMalloc(&d_unique_ids,  sizeof(int) * (size_t)B);
    cudaMalloc(&d_run_counts,  sizeof(int) * (size_t)B);
    cudaMalloc(&d_run_offsets, sizeof(int) * ((size_t)B + 1));
    cudaMalloc(&d_num_runs,    sizeof(int));
    cudaMallocHost(&h_num_runs, sizeof(int));

    //  CUB ()
    size_t bytes_sort = 0, bytes_rle = 0, bytes_scan = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, bytes_sort,
        d_best_idx, d_sorted_keys,   // key: assign
        d_sample_idx, d_sorted_vals, // val: sample index
        B, 0, static_cast<int>(std::ceil(std::log2(static_cast<double>(k) + 1))));
    cub::DeviceRunLengthEncode::Encode(
        nullptr, bytes_rle,
        d_sorted_keys, d_unique_ids, d_run_counts, d_num_runs, B);
    cub::DeviceScan::ExclusiveSum(
        nullptr, bytes_scan,
        d_run_counts, d_run_offsets, B);

    cub_tmp_bytes = std::max({bytes_sort, bytes_rle, bytes_scan});
    cudaMalloc(&d_cub_tmp, cub_tmp_bytes);
}

void StreamEnv::free() {
    if (d_xnorm2)    { cudaFree(d_xnorm2);    d_xnorm2 = nullptr; }
    if (d_best_dist2) { cudaFree(d_best_dist2); d_best_dist2 = nullptr; }
    if (d_best_idx)   { cudaFree(d_best_idx);   d_best_idx = nullptr; }
    if (d_dot)        { cudaFree(d_dot);        d_dot = nullptr; }
    if (d_sorted_keys) { cudaFree(d_sorted_keys); d_sorted_keys = nullptr; }
    if (d_sorted_vals) { cudaFree(d_sorted_vals); d_sorted_vals = nullptr; }
    if (d_sample_idx)  { cudaFree(d_sample_idx);  d_sample_idx = nullptr; }
    if (d_unique_ids)  { cudaFree(d_unique_ids);  d_unique_ids = nullptr; }
    if (d_run_counts)  { cudaFree(d_run_counts);  d_run_counts = nullptr; }
    if (d_run_offsets) { cudaFree(d_run_offsets); d_run_offsets = nullptr; }
    if (d_num_runs)    { cudaFree(d_num_runs);    d_num_runs = nullptr; }
    if (h_num_runs)    { cudaFreeHost(h_num_runs); h_num_runs = nullptr; }
    if (d_cub_tmp)     { cudaFree(d_cub_tmp);     d_cub_tmp = nullptr; }
    cub_tmp_bytes = 0;
}

// ============================================================
// perform_assignment_only Implementation
// ============================================================

template<bool Timed>
void perform_assignment_only_impl(
    const KMeansCase& cfg,
    const float* h_data,
    idx_t* d_assign,
    const float* d_centroids,
    float* d_cnorm2,
    StreamEnv& env,
    float* d_data_buffer,
    cudaStream_t stream,
    cublasHandle_t handle,
    int B,
    int Ktile,
    int n,
    int dim,
    int k,
    float* d_objective_sum,
    const float* d_precomputed_cnorm2,
    bool precomputed_cnorm_is_l2,
    KMeansAssignmentTiming* timing
) {
    struct EventList {
        std::vector<cudaEvent_t> events;
        ~EventList() { for (auto e : events) cudaEventDestroy(e); }
    } recorded;
    if constexpr (Timed) {
        recorded.events.resize(3 * ((k + Ktile - 1) / Ktile));
        for (auto& event : recorded.events) CUDA_CHECK(cudaEventCreate(&event));
    }
    const float alpha = 1.f;
    const float beta = 0.f;

    const float* centroid_norms = d_precomputed_cnorm2 ? d_precomputed_cnorm2 : d_cnorm2;
    if (!d_precomputed_cnorm2)
        compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream);
    if (d_objective_sum) {
        cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream);
    }

    //
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);

        // (,/)
        float* d_data_cur = d_data_buffer;

        // H2D  batch
        cudaMemcpyAsync(d_data_cur, h_data + (size_t)base * dim,
                       sizeof(float) * (size_t)curB * dim,
                       cudaMemcpyHostToDevice, stream);

        // xnorm2 for this batch
        compute_l2_squared_gpu(d_data_cur, env.d_xnorm2, curB, dim, L2NORM_AUTO, stream);

        // init best_dist2 = +inf, best_idx = 0
        {
            int threads = 256;
            int blocks = (curB + threads - 1) / threads;
            kernel_init_best<<<blocks, threads, 0, stream>>>(env.d_best_dist2, env.d_best_idx, curB);
        }

        // centroid  Ktile
        for (int cbase = 0; cbase < k; cbase += Ktile) {
            int curK = std::min(Ktile, k - cbase);

            const float* A = d_centroids + (size_t)cbase * dim;
            const float* Bm = d_data_cur;
            float* Cc = env.d_dot;

            // cuBLAS
            cublas_check(cublasSetStream(handle, stream), "cublasSetStream");

            if constexpr (Timed) CUDA_CHECK(cudaEventRecord(recorded.events[3*(cbase/Ktile)], stream));
            // GEMM: dotT[curB,curK] (col-major) = Xcm^T[curB,dim] * Ccm[dim,curK]
            cublas_check(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    curB, curK, dim,
                    &alpha,
                    Bm, dim,
                    A, dim,
                    &beta,
                    Cc, curB
                ),
                "cublasSgemm(Xcm^T * Ccm)"
            );

            if constexpr (Timed) CUDA_CHECK(cudaEventRecord(recorded.events[3*(cbase/Ktile)+1], stream));
            // best
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_update_best_from_dotT<<<blocks, threads, 0, stream>>>(
                    env.d_dot, env.d_xnorm2, centroid_norms,
                    curB, curK, cbase, precomputed_cnorm_is_l2,
                    env.d_best_idx, env.d_best_dist2
                );
            }
            if constexpr (Timed) CUDA_CHECK(cudaEventRecord(recorded.events[3*(cbase/Ktile)+2], stream));
        }
        if constexpr (Timed) {
            CUDA_CHECK(cudaEventSynchronize(recorded.events.back()));
            for (size_t i = 0; i < recorded.events.size(); i += 3) {
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, recorded.events[i], recorded.events[i+1]));
                timing->gemm_seconds += ms / 1000.;
                CUDA_CHECK(cudaEventElapsedTime(&ms, recorded.events[i+1], recorded.events[i+2]));
                timing->select_seconds += ms / 1000.;
            }
        }

        // assign
        if (d_assign != env.d_best_idx)
            cudaMemcpyAsync(d_assign + base, env.d_best_idx, sizeof(idx_t) * (size_t)curB,
                       cudaMemcpyDeviceToDevice, stream);

        if (d_objective_sum) {
            const int reduce_threads = 256;
            const int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
            const int reduce_shmem = reduce_threads * sizeof(float);
            kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream>>>(
                env.d_best_dist2,
                d_objective_sum,
                curB);
        }
    }

    cudaStreamSynchronize(stream);
}

// Dispatch once; production calls compile out all event instrumentation.
void perform_assignment_only(
    const KMeansCase& cfg, const float* h_data, idx_t* d_assign,
    const float* d_centroids, float* d_cnorm2, StreamEnv& env,
    float* d_data_buffer, cudaStream_t stream, cublasHandle_t handle,
    int B, int Ktile, int n, int dim, int k, float* d_objective_sum,
    const float* d_precomputed_cnorm2, bool precomputed_cnorm_is_l2,
    KMeansAssignmentTiming* timing) {
    if (timing) {
        perform_assignment_only_impl<true>(cfg,h_data,d_assign,d_centroids,d_cnorm2,
            env,d_data_buffer,stream,handle,B,Ktile,n,dim,k,d_objective_sum,
            d_precomputed_cnorm2,precomputed_cnorm_is_l2,timing);
    } else {
        perform_assignment_only_impl<false>(cfg,h_data,d_assign,d_centroids,d_cnorm2,
            env,d_data_buffer,stream,handle,B,Ktile,n,dim,k,d_objective_sum,
            d_precomputed_cnorm2,precomputed_cnorm_is_l2,nullptr);
    }
}

// ============================================================
// GPU Kernels Implementation
// ============================================================

/**
 * Kernel:  best_dist2 (block reduce)
 *  block ,
 */
 __global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  //  float
    int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //
    float val = (i < n) ? data[i] : 0.0f;

    // Block
    sdata[tid] = val;
    __syncthreads();

    // : warp shuffle + shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block 0
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int cnt = counts[c];
    if (cnt <= 0) return;  // keep old centroid
    float inv = 1.0f / (float)cnt;

    for (int col = j; col < dim; col += blockDim.x) {
        centroids[(size_t)c * dim + col] = accum[(size_t)c * dim + col] * inv;
    }
}

/**
 * Kernel: Minibatch()
 * :lr = 1 / (total_count + 1)
 * :centroid_new = (1 - lr) * centroid_old + lr * (accum / count)
 */
__global__ void kernel_update_centroids_minibatch(
    float* __restrict__ centroids,      // [k, dim] (in/out)
    const float* __restrict__ accum,     // [k, dim]
    const int* __restrict__ counts,      // [k]
    int* __restrict__ total_counts,      // [k] (in/out)
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int count = counts[c];
    if (count <= 0) return;  // minibatch,

    // :total_count()
    // :blockcentroid,,
    // ,,
    __shared__ int old_total;
    __shared__ float lr_shared;
    if (j == 0) {
        old_total = total_counts[c];
        // :lr = 1 / (old_total + 1)
        // :old_total
        // ,"centroid"
        // ,
        lr_shared = 1.0f / ((float)old_total + 1.0f);
        // total_counts()
        total_counts[c] = old_total + count;
    }
    __syncthreads();

    float lr = lr_shared;
    float inv_count = 1.0f / (float)count;

    // dim
    for (int col = j; col < dim; col += blockDim.x) {
        size_t idx = (size_t)c * dim + col;
        float avg = accum[idx] * inv_count;  // minibatchcentroid
        float old_val = centroids[idx];
        centroids[idx] = (1.0f - lr) * old_val + lr * avg;
    }
}

// ============================================================
// GEMM-based KMeans Kernels
// ============================================================

/**
 * Kernel: (best_dist2 = INF, best_idx = 0)
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    idx_t* __restrict__ best_idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_dist2[i] = 3.402823e38f;  // FLT_MAX
        best_idx[i] = 0;
    }
}

/**
 * Kernel:  GEMM (col-major dotT [curB, curK])
 *
 * dotT: [curB, curK] col-major, dotT(i, t) = dotT[i + t*curB] = dot(data[i], centroid[cbase+t])
 *  i, i+1  dotT[i + t*curB]  dotT[i+1 + t*curB], → coalesced
 */
__global__ void kernel_update_best_from_dotT(
    const float* __restrict__ dotT,      // [curB, curK] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] centroid
    int curB,
    int curK,
    int cbase,                            // centroid
    bool cnorm_is_l2,
    idx_t* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= curB) return;

    const float xn = xnorm2[i];
    float bestd = best_dist2[i];
    idx_t bestc = best_idx[i];

    const int unroll_factor = 4;
    int t = 0;

    for (; t + unroll_factor <= curK; t += unroll_factor) {
        #pragma unroll
        for (int u = 0; u < unroll_factor; ++u) {
            int tidx = t + u;
            float dot = dotT[i + (size_t)tidx * curB];
            float cn = cnorm2_global[cbase + tidx];
            if (cnorm_is_l2) cn *= cn;
            float d2 = xn + cn - 2.f * dot;
            if (d2 < bestd) {
                bestd = d2;
                bestc = cbase + tidx;
            }
        }
    }

    for (; t < curK; ++t) {
        float dot = dotT[i + (size_t)t * curB];
        float cn = cnorm2_global[cbase + t];
        if (cnorm_is_l2) cn *= cn;
        float d2 = xn + cn - 2.f * dot;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cbase + t;
        }
    }

    best_dist2[i] = bestd;
    best_idx[i] = bestc;
}

/**
 * Kernel:  GEMM (col-major dotT [curB, curK])(Warp)
 *
 *  warp , warp  min
 */
__global__ void kernel_update_best_from_dotT_warp_reduce(
    const float* __restrict__ dotT,      // [curB, curK] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] centroid
    int curB,
    int curK,
    int cbase,                            // centroid
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_count = (blockDim.x + 31) / 32;
    const int data_idx = blockIdx.x * warp_count + warp_id;

    if (data_idx >= curB) return;

    const float xn = xnorm2[data_idx];
    float bestd = (lane_id == 0) ? best_dist2[data_idx] : 3.402823e38f;
    int bestc = (lane_id == 0) ? best_idx[data_idx] : -1;

    //  lane  stride-32  centroid, lane  → coalesced
    for (int t = lane_id; t < curK; t += 32) {
        float dot = dotT[data_idx + (size_t)t * curB];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cbase + t;
        }
    }

    // Warp shuffle : warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_d = __shfl_down_sync(0xffffffff, bestd, offset);
        int other_c = __shfl_down_sync(0xffffffff, bestc, offset);
        if (other_d < bestd) {
            bestd = other_d;
            bestc = other_c;
        }
    }

    // Lane 0
    if (lane_id == 0) {
        best_dist2[data_idx] = bestd;
        best_idx[data_idx] = bestc;
    }
}

/**
 * Kernel: ( GEMM )-
 * :
 * 1.
 * 2.
 * 3.
 */
__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const idx_t* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    idx_t c = assign[i];
    int base = i * dim;

    //  counts
    atomicAdd(&counts[c], 1);

    //  accum,
    float* acc = accum + (size_t)c * dim;

    //
    if (dim <= 4) {
        // :
        if (dim >= 1) atomicAdd(&acc[0], data[base + 0]);
        if (dim >= 2) atomicAdd(&acc[1], data[base + 1]);
        if (dim >= 3) atomicAdd(&acc[2], data[base + 2]);
        if (dim >= 4) atomicAdd(&acc[3], data[base + 3]);
    } else if (dim <= 16) {
        // :(4)
        int j = 0;
        #pragma unroll 4
        for (; j + 4 <= dim; j += 4) {
            atomicAdd(&acc[j], data[base + j]);
            atomicAdd(&acc[j+1], data[base + j+1]);
            atomicAdd(&acc[j+2], data[base + j+2]);
            atomicAdd(&acc[j+3], data[base + j+3]);
        }
        //
        for (; j < dim; ++j) {
            atomicAdd(&acc[j], data[base + j]);
        }
    } else {
        // :
        int j = 0;
        #pragma unroll 8
        for (; j + 8 <= dim; j += 8) {
            atomicAdd(&acc[j], data[base + j]);
            atomicAdd(&acc[j+1], data[base + j+1]);
            atomicAdd(&acc[j+2], data[base + j+2]);
            atomicAdd(&acc[j+3], data[base + j+3]);
            atomicAdd(&acc[j+4], data[base + j+4]);
            atomicAdd(&acc[j+5], data[base + j+5]);
            atomicAdd(&acc[j+6], data[base + j+6]);
            atomicAdd(&acc[j+7], data[base + j+7]);
        }
        //
        for (; j < dim; ++j) {
            atomicAdd(&acc[j], data[base + j]);
        }
    }
}


/**
 * Kernel: Sort-by-cluster ( atomic )
 *
 *  Block  cluster .
 *  stride=blockDim.x  dim_v4  float4 , cluster ,
 *  atomicAdd  accum( Block ,).
 */
__global__ void kernel_segmented_accum_v2(
    const float4* __restrict__ data4,
    const idx_t*   __restrict__ sorted_indices,
    const int*   __restrict__ cluster_offsets,
    const int*   __restrict__ active_clusters,
    float*       __restrict__ accum,
    int*         __restrict__ counts,
    int num_active, int dim_v4
) {
    const int idx = blockIdx.x;
    if (idx >= num_active) return;

    const int cluster_id = active_clusters[idx];
    const int seg_start  = cluster_offsets[idx];
    const int seg_end    = cluster_offsets[idx + 1];
    const int seg_len    = seg_end - seg_start;

    //  thread 0  count( atomicAdd,)
    if (threadIdx.x == 0)
        atomicAdd(&counts[cluster_id], seg_len);

    //  dim_v4  float4
    for (int d = threadIdx.x; d < dim_v4; d += blockDim.x) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int i = seg_start; i < seg_end; ++i) {
            const float4 v = data4[(size_t)sorted_indices[i] * dim_v4 + d];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        //  Block , Block  cluster →
        float* out = accum + (size_t)cluster_id * (dim_v4 * 4) + (size_t)d * 4;
        atomicAdd(out + 0, sum.x);
        atomicAdd(out + 1, sum.y);
        atomicAdd(out + 2, sum.z);
        atomicAdd(out + 3, sum.w);
    }
}

// ============================================================
// Vector Reordering Kernels
// ============================================================

/**
 * Kernel: cluster
 * ,cluster
 */
__global__ void kernel_compute_cluster_indices(
    const int* __restrict__ assign,      // [n]
    int* __restrict__ cluster_indices,   // [n] :cluster
    int* __restrict__ cluster_counts,    // [k] :cluster()
    int n, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int cluster_id = assign[idx];
    if (cluster_id >= 0 && cluster_id < k) {
        // ,cluster
        int pos = atomicAdd(&cluster_counts[cluster_id], 1);
        cluster_indices[idx] = pos;
    } else {
        cluster_indices[idx] = -1;  // cluster ID
    }
}

/**
 * Kernel: Exclusive scan ()
 *  offsets[i] = sum(counts[0..i-1])
 * :counts = [2, 3, 1],offsets = [0, 2, 5]
 *
 * :kernel,
 */
__global__ void kernel_exclusive_scan(
    const int* __restrict__ counts,   // [k]
    int* __restrict__ offsets,        // [k]
    int k
) {
    // ,
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            offsets[i] = sum;
            sum += counts[i];
        }
    }
}

/**
 * Kernel: clustercluster
 */
__global__ void kernel_reorder_vectors_by_cluster(
    const float* __restrict__ data_in,   // [n, dim]
    const int* __restrict__ assign,      // [n]
    const int* __restrict__ cluster_offsets,  // [k]
    const int* __restrict__ cluster_indices,   // [n]
    float* __restrict__ data_out,        // [n, dim]
    int n, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int cluster_id = assign[idx];
    int cluster_idx = cluster_indices[idx];

    if (cluster_id >= 0 && cluster_idx >= 0) {
        //
        int out_pos = cluster_offsets[cluster_id] + cluster_idx;

        //
        for (int d = 0; d < dim; ++d) {
            data_out[(size_t)out_pos * dim + d] = data_in[(size_t)idx * dim + d];
        }
    }
}
