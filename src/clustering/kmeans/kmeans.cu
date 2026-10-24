#include "kmeans.cuh"
#include "clustering/flash_kmeans/flash_kmeans_assign.cuh"
#include "utils/utils.cuh"
#include "l2norm/l2norm.cuh"
#include "utils/cudatimer.h"
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdint>
#include <vector>

#define ENABLE_CUDA_TIMING 0 /*CUDATimer*/

namespace {

static inline double kmeans_now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct ScopedKMeansTimer {
    double* dst = nullptr;
    double start = 0.0;

    explicit ScopedKMeansTimer(double* value) : dst(value), start(kmeans_now_ms()) {}

    ~ScopedKMeansTimer() {
        if (dst) {
            *dst += kmeans_now_ms() - start;
        }
    }
};

}  // namespace

static inline void kmeans_cuda_log_check(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[kmeans.cu] CUDA error at %s: %s\n", expr, cudaGetErrorString(err));
        fflush(stderr);
    }
}

// static inline void cublas_check(cublasStatus_t st, const char* msg) {
//     if (st != CUBLAS_STATUS_SUCCESS) {
//         fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
//         std::abort();
//     }
// }

/**  [0, 1, ..., n-1] , SortPairs  value  */
__global__ static void kernel_fill_seq(idx_t* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = static_cast<idx_t>(i);
}

__global__ static void kernel_recompute_selected_l2(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    const idx_t* __restrict__ best_idx,
    float* __restrict__ best_dist2,
    int n,
    int dim,
    int k)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= n) return;

    const idx_t c = best_idx[row];
    if (c < 0 || c >= k) {
        if (tid == 0) {
            best_dist2[row] = 3.4028234663852886e38f;
        }
        return;
    }

    __shared__ float smem[256];
    float sum = 0.0f;
    const float* x = data + (size_t)row * dim;
    const float* centroid = centroids + (size_t)c * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        const float diff = x[d] - centroid[d];
        sum += diff * diff;
    }
    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_dist2[row] = smem[0];
    }
}

static void perform_flash_assignment_only(
    const float* h_data,
    idx_t* d_assign,
    const float* d_centroids,
    float* d_cnorm2,
    StreamEnv& env,
    paraselect::clustering::kmeans::FlashKMeansAssignWorkspace& flash_workspace,
    float* d_data_buffer,
    cudaStream_t stream,
    int B,
    int n,
    int dim,
    int k,
    const paraselect::clustering::kmeans::FlashKMeansAssignConfig& flash_config,
    float* d_objective_sum)
{
    compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream);
    kmeans_cuda_log_check(cudaGetLastError(), "perform_flash_assignment_only compute centroid norms");
    if (d_objective_sum) {
        kmeans_cuda_log_check(cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream),
                              "perform_flash_assignment_only memset objective");
    }

    for (int base = 0; base < n; base += B) {
        const int curB = std::min(B, n - base);
        float* d_data_cur = d_data_buffer;

        kmeans_cuda_log_check(cudaMemcpyAsync(
            d_data_cur,
            h_data + (size_t)base * dim,
            sizeof(float) * (size_t)curB * dim,
            cudaMemcpyHostToDevice,
            stream), "perform_flash_assignment_only H2D");

        compute_l2_squared_gpu(d_data_cur, env.d_xnorm2, curB, dim, L2NORM_AUTO, stream);
        kmeans_cuda_log_check(cudaGetLastError(), "perform_flash_assignment_only compute batch norms");

        paraselect::clustering::kmeans::flash_kmeans_assign_l2_precomputed_norms(
            d_data_cur,
            d_centroids,
            env.d_xnorm2,
            d_cnorm2,
            curB,
            k,
            dim,
            env.d_best_idx,
            env.d_best_dist2,
            flash_config,
            flash_workspace,
            stream);
        kmeans_cuda_log_check(cudaGetLastError(), "perform_flash_assignment_only flash assign");

        kernel_recompute_selected_l2<<<curB, 256, 0, stream>>>(
            d_data_cur,
            d_centroids,
            env.d_best_idx,
            env.d_best_dist2,
            curB,
            dim,
            k);
        kmeans_cuda_log_check(cudaGetLastError(), "perform_flash_assignment_only recompute selected l2");

        kmeans_cuda_log_check(cudaMemcpyAsync(
            d_assign + base,
            env.d_best_idx,
            sizeof(idx_t) * (size_t)curB,
            cudaMemcpyDeviceToDevice,
            stream), "perform_flash_assignment_only best_idx to d_assign");

        if (d_objective_sum) {
            const int reduce_threads = 256;
            const int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
            const int reduce_shmem = reduce_threads * sizeof(float);
            kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream>>>(
                env.d_best_dist2,
                d_objective_sum,
                curB);
            kmeans_cuda_log_check(cudaGetLastError(), "perform_flash_assignment_only objective reduce");
        }
    }

    kmeans_cuda_log_check(cudaStreamSynchronize(stream), "perform_flash_assignment_only synchronize");
}

// ============================================================
// Lloyd workspace: K-means ( cublasCreate /  cudaMalloc)
// ============================================================

struct KMeansLloydWorkspace {
    int max_n = 0;
    int max_k = 0;
    int dim = 0;
    cublasHandle_t handle{};
    float* d_accum = nullptr;
    int* d_counts = nullptr;
    float* d_cnorm2 = nullptr;
    float* d_data_buf[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {};
    cudaEvent_t event[2] = {};
    cudaEvent_t init_event = nullptr;
    StreamEnv stream_env[2];
    paraselect::clustering::kmeans::FlashKMeansAssignWorkspace flash_env[2];
    float* d_objective_sum = nullptr;
    bool initialized = false;
};

namespace {
thread_local KMeansLloydStats g_last_lloyd_stats;
}

static inline paraselect::clustering::kmeans::FlashKMeansAssignConfig
make_kmeans_flash_assign_config(int dim) {
    auto config = paraselect::clustering::kmeans::default_flash_kmeans_assign_config(dim);
    config.variant = 8;
    config.point_tile = 64;
    config.centroid_tile = 256;
    config.cutlass_k_tile = 16;
    config.cutlass_stages = 2;
    config.collect_timing = false;
    return config;
}

KMeansLloydWorkspace* kmeans_lloyd_workspace_create(int max_n, int max_k, int dim) {
    const int B = 1 << 15;
    const int FlashB = 1 << 17;
    const int Ktile = 4096;
    auto* ws = new KMeansLloydWorkspace();
    ws->max_n = max_n;
    ws->max_k = max_k;
    ws->dim = dim;
    kmeans_cuda_log_check(cudaMalloc(&ws->d_accum, sizeof(float) * (size_t)max_k * dim), "cudaMalloc d_accum");
    kmeans_cuda_log_check(cudaMalloc(&ws->d_counts, sizeof(int) * (size_t)max_k), "cudaMalloc d_counts");
    kmeans_cuda_log_check(cudaMalloc(&ws->d_cnorm2, sizeof(float) * (size_t)max_k), "cudaMalloc d_cnorm2");
    ws->stream_env[0].allocate(FlashB, Ktile, B);
    ws->stream_env[1].allocate(FlashB, Ktile, B);
    ws->stream_env[0].allocate_accum_sort(FlashB, max_k);
    ws->stream_env[1].allocate_accum_sort(FlashB, max_k);
    const auto flash_config = make_kmeans_flash_assign_config(dim);
    paraselect::clustering::kmeans::reserve_flash_kmeans_assign_scratch_workspace(
        ws->flash_env[0], FlashB, max_k, flash_config);
    paraselect::clustering::kmeans::reserve_flash_kmeans_assign_scratch_workspace(
        ws->flash_env[1], FlashB, max_k, flash_config);
    kmeans_cuda_log_check(cudaMalloc(&ws->d_data_buf[0], sizeof(float) * (size_t)FlashB * dim), "cudaMalloc d_data_buf[0]");
    kmeans_cuda_log_check(cudaMalloc(&ws->d_data_buf[1], sizeof(float) * (size_t)FlashB * dim), "cudaMalloc d_data_buf[1]");
    kmeans_cuda_log_check(cudaStreamCreate(&ws->stream[0]), "cudaStreamCreate stream[0]");
    kmeans_cuda_log_check(cudaStreamCreate(&ws->stream[1]), "cudaStreamCreate stream[1]");
    kmeans_cuda_log_check(cudaEventCreate(&ws->event[0]), "cudaEventCreate event[0]");
    kmeans_cuda_log_check(cudaEventCreate(&ws->event[1]), "cudaEventCreate event[1]");
    kmeans_cuda_log_check(cudaEventCreate(&ws->init_event), "cudaEventCreate init_event");
    kmeans_cuda_log_check(cudaMalloc(&ws->d_objective_sum, sizeof(float)), "cudaMalloc d_objective_sum");
    cublas_check(cublasCreate(&ws->handle), "cublasCreate");
    cublas_check(cublasSetMathMode(ws->handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");
    ws->initialized = true;
    return ws;
}

void kmeans_lloyd_workspace_destroy(KMeansLloydWorkspace* ws) {
    if (!ws) {
        return;
    }
    if (ws->handle) {
        cublasDestroy(ws->handle);
    }
    if (ws->d_data_buf[0]) {
        cudaFree(ws->d_data_buf[0]);
    }
    if (ws->d_data_buf[1]) {
        cudaFree(ws->d_data_buf[1]);
    }
    cudaStreamDestroy(ws->stream[0]);
    cudaStreamDestroy(ws->stream[1]);
    cudaEventDestroy(ws->event[0]);
    cudaEventDestroy(ws->event[1]);
    if (ws->init_event) {
        cudaEventDestroy(ws->init_event);
    }
    ws->stream_env[0].free();
    ws->stream_env[1].free();
    paraselect::clustering::kmeans::release_flash_kmeans_assign_workspace(ws->flash_env[0]);
    paraselect::clustering::kmeans::release_flash_kmeans_assign_workspace(ws->flash_env[1]);
    if (ws->d_objective_sum) {
        cudaFree(ws->d_objective_sum);
    }
    cudaFree(ws->d_accum);
    cudaFree(ws->d_counts);
    cudaFree(ws->d_cnorm2);
    delete ws;
}

// ============================================================
// GPU KMeans Runner Implementation (GEMM-optimized)
// ============================================================

void gpu_kmeans_lloyd_with_workspace(
    KMeansLloydWorkspace* ws,
    const KMeansCase& cfg,
    const float* h_data,             // [n, dim] row-major,
    idx_t* d_assign,                   // [n]
    float* d_centroids,              // [k, dim] (in/out),
    float* h_objective                // sum dist2
) {
    if (!ws || !ws->initialized || !ws->handle) {
        fprintf(stderr, "gpu_kmeans_lloyd_with_workspace: invalid workspace\n");
        return;
    }
    if (cfg.n > ws->max_n || cfg.k > ws->max_k || cfg.dim != ws->dim) {
        fprintf(
            stderr,
            "gpu_kmeans_lloyd_with_workspace: cfg (n=%d,k=%d,dim=%d) exceeds workspace "
            "(max_n=%d,max_k=%d,dim=%d)\n",
            cfg.n,
            cfg.k,
            cfg.dim,
            ws->max_n,
            ws->max_k,
            ws->dim);
        return;
    }

    CUDATimer timer_total("gpu_kmeans_lloyd: Total Time", ENABLE_CUDA_TIMING);
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== :batch & ktile( kmeans_lloyd_workspace_create )=====
    const int Ktile = 4096;
    const int baseline_B = 1 << 15;
    const int B = cfg.use_flash_assign ? (1 << 17) : baseline_B;
    const int total_batches = (n + B - 1) / B;

    float* d_accum = ws->d_accum;
    int* d_counts = ws->d_counts;
    float* d_cnorm2 = ws->d_cnorm2;
    float* d_data_buf[2] = {ws->d_data_buf[0], ws->d_data_buf[1]};
    cudaStream_t stream[2] = {ws->stream[0], ws->stream[1]};
    cudaEvent_t event[2] = {ws->event[0], ws->event[1]};
    StreamEnv* stream_env = ws->stream_env;
    cublasHandle_t handle = ws->handle;
    float* d_objective_sum = ws->d_objective_sum;
    const bool use_flash_assign = cfg.use_flash_assign;
    const auto flash_config = make_kmeans_flash_assign_config(dim);
    KMeansLloydStats stats;
    stats.n = n;
    stats.dim = dim;
    stats.k = k;
    stats.iters = cfg.iters;
    stats.batch_size = B;
    stats.total_batches = total_batches;
    stats.ktile = Ktile;
    stats.use_flash_assign = use_flash_assign;
    stats.timing_valid = cfg.collect_timing;
    const double total_start_ms = cfg.collect_timing ? kmeans_now_ms() : 0.0;
    //  sample_idx  [0, 1, ..., len-1](SortPairs  value )
    auto init_sample_idx = [&](idx_t* d_idx, int len, cudaStream_t s) {
        int threads = 256;
        int blocks = (len + threads - 1) / threads;
        kernel_fill_seq<<<blocks, threads, 0, s>>>(d_idx, len);
    };

    // ======  ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // (01)
    int cur_buf = 0;

    // ====== iteration ======
    for (int it = 0; it < cfg.iters; ++it) {
        // CUDATimer timer_iter("gpu_kmeans_lloyd: Iteration " + std::to_string(it), ENABLE_CUDA_TIMING);
        // clear accum & counts
        // batch
        {
            CUDATimer timer_init("  Iter " + std::to_string(it) + ": Initialize (clear accum/counts, compute centroid norms)", ENABLE_CUDA_TIMING);
            ScopedKMeansTimer timing(cfg.collect_timing ? &stats.init_ms : nullptr);
            kmeans_cuda_log_check(cudaMemsetAsync(d_accum, 0, sizeof(float) * (size_t)k * dim, stream[0]), "cudaMemsetAsync d_accum");
            kmeans_cuda_log_check(cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream[0]), "cudaMemsetAsync d_counts");
            //  stream[0],batch
            compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);
            kmeans_cuda_log_check(cudaGetLastError(), "compute_l2_squared_gpu centroids launch");
        }

        // ====== () ======
        cur_buf = 0;

        //  objective (,)
        if (h_objective && d_objective_sum) {
            kmeans_cuda_log_check(cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream[0]), "cudaMemsetAsync d_objective_sum");
        }

        //  stream[0] (accum,centroid)
        // batch d_cnorm2  d_accum
        cudaEvent_t init_event = ws->init_event;
        kmeans_cuda_log_check(cudaEventRecord(init_event, stream[0]), "cudaEventRecord init_event");
        kmeans_cuda_log_check(cudaStreamWaitEvent(stream[0], init_event, 0), "cudaStreamWaitEvent stream[0] init_event");
        kmeans_cuda_log_check(cudaStreamWaitEvent(stream[1], init_event, 0), "cudaStreamWaitEvent stream[1] init_event");

        // batch
        int first_base = 0;
        int first_curB = std::min(B, n - first_base);
        // , d_cnorm2  d_accum
        kmeans_cuda_log_check(cudaMemcpyAsync(d_data_buf[cur_buf], h_data + (size_t)first_base * dim,
                       sizeof(float) * (size_t)first_curB * dim,
                       cudaMemcpyHostToDevice, stream[cur_buf]), "cudaMemcpyAsync first H2D");
        // ,
        kmeans_cuda_log_check(cudaEventRecord(event[cur_buf], stream[cur_buf]), "cudaEventRecord first preload");

        {
            CUDATimer timer_batch("  Iter " + std::to_string(it) + ": Process All Batches", ENABLE_CUDA_TIMING);
            ScopedKMeansTimer timing(cfg.collect_timing ? &stats.batch_ms : nullptr);
        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);
            float* d_data_cur = d_data_buf[cur_buf];

            // batch()
            // :;:batch
            kmeans_cuda_log_check(cudaStreamWaitEvent(stream[cur_buf], event[cur_buf], 0), "cudaStreamWaitEvent current batch");

            // batch()
            // :,()
            int next_base = base + B;
            int next_buf = 1 - cur_buf;

            //  StreamEnv buffer,
            StreamEnv& cur_env = stream_env[cur_buf];

            // xnorm2 for this batch()
            compute_l2_squared_gpu(d_data_cur, cur_env.d_xnorm2, curB, dim, L2NORM_AUTO, stream[cur_buf]);
            kmeans_cuda_log_check(cudaGetLastError(), "compute_l2_squared_gpu batch launch");

            // init best_dist2 = +inf, best_idx = 0()
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[cur_buf]>>>(cur_env.d_best_dist2, cur_env.d_best_idx, curB);
                kmeans_cuda_log_check(cudaGetLastError(), "kernel_init_best launch");
            }

            if (use_flash_assign) {
                paraselect::clustering::kmeans::flash_kmeans_assign_l2_precomputed_norms(
                    d_data_cur,
                    d_centroids,
                    cur_env.d_xnorm2,
                    d_cnorm2,
                    curB,
                    k,
                    dim,
                    cur_env.d_best_idx,
                    cur_env.d_best_dist2,
                    flash_config,
                    ws->flash_env[cur_buf],
                    stream[cur_buf]);
                kmeans_cuda_log_check(cudaGetLastError(), "flash_kmeans_assign_l2_precomputed_norms launch");
                kernel_recompute_selected_l2<<<curB, 256, 0, stream[cur_buf]>>>(
                    d_data_cur,
                    d_centroids,
                    cur_env.d_best_idx,
                    cur_env.d_best_dist2,
                    curB,
                    dim,
                    k);
                kmeans_cuda_log_check(cudaGetLastError(), "kernel_recompute_selected_l2 launch");
            } else {
                // ====== centroid  Ktile ======
                for (int cbase = 0; cbase < k; cbase += Ktile) {
                    int curK = std::min(Ktile, k - cbase);

                    const float* A = d_centroids + (size_t)cbase * dim;
                    const float* Bm = d_data_cur;
                    float* Cc = cur_env.d_dot;

                    // cuBLAS
                    cublas_check(cublasSetStream(handle, stream[cur_buf]), "cublasSetStream");

                    // Treat row-major C[curK,dim] as col-major Ccm[dim,curK]
                    // Treat row-major X[curB,dim] as col-major Xcm[dim,curB]
                    // Compute dotT[curB,curK] (col-major) = Xcm^T[curB,dim] * Ccm[dim,curK]
                    //  dotT(i,t) = dotT[i + t*curB], i  → coalesced
                    cublas_check(
                        cublasSgemm(
                            handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,     // A^T * B
                            curB, curK, dim,              // m=curB, n=curK, k=dim
                            &alpha,
                            Bm, dim,                      // A is Xcm with shape [dim,curB], lda=dim
                            A, dim,                       // B is Ccm with shape [dim,curK], ldb=dim
                            &beta,
                            Cc, curB                      // C is [curB,curK] col-major, ldc=curB
                        ),
                        "cublasSgemm(Xcm^T * Ccm)"
                    );


                    // best()
                    {
                        int threads = 256;
                        int blocks = (curB + threads - 1) / threads;
                        kernel_update_best_from_dotT<<<blocks, threads, 0, stream[cur_buf]>>>(
                            cur_env.d_dot, cur_env.d_xnorm2, d_cnorm2,
                            curB, curK, cbase, false,
                            cur_env.d_best_idx, cur_env.d_best_dist2
                        );
                        kmeans_cuda_log_check(cudaGetLastError(), "kernel_update_best_from_dotT launch");
                    }
                }
            }

            // assign(,)
            kmeans_cuda_log_check(cudaMemcpyAsync(d_assign + base, cur_env.d_best_idx, sizeof(idx_t) * (size_t)curB,
                           cudaMemcpyDeviceToDevice, stream[cur_buf]), "cudaMemcpyAsync best_idx to d_assign");

            // ====== Sort-by-cluster ( kernel_accum_from_assign)======
            // Step 0:  sample_idx = [0, 1, ..., curB-1]( batch )
            init_sample_idx(cur_env.d_sample_idx, curB, stream[cur_buf]);

            // Step 1: SortPairs(assign → sorted_keys, sample_idx → sorted_vals)
            {
                int end_bit = static_cast<int>(
                    std::ceil(std::log2(static_cast<double>(k) + 1)));
                cub::DeviceRadixSort::SortPairs(
                    cur_env.d_cub_tmp, cur_env.cub_tmp_bytes,
                    cur_env.d_best_idx,   cur_env.d_sorted_keys,
                    cur_env.d_sample_idx, cur_env.d_sorted_vals,
                    curB, 0, end_bit, stream[cur_buf]);
            }

            // Step 2: RunLengthEncode → unique cluster ids + run lengths + num_runs
            cub::DeviceRunLengthEncode::Encode(
                cur_env.d_cub_tmp, cur_env.cub_tmp_bytes,
                cur_env.d_sorted_keys,
                cur_env.d_unique_ids,
                cur_env.d_run_counts,
                cur_env.d_num_runs,
                curB, stream[cur_buf]);

            // Step 3: ExclusiveSum(run_counts → run_offsets),
            cub::DeviceScan::ExclusiveSum(
                cur_env.d_cub_tmp, cur_env.cub_tmp_bytes,
                cur_env.d_run_counts, cur_env.d_run_offsets,
                curB, stream[cur_buf]);
            //  num_runs  pinned host,()
            cudaMemcpyAsync(cur_env.h_num_runs, cur_env.d_num_runs, sizeof(int),
                            cudaMemcpyDeviceToHost, stream[cur_buf]);

            //  num_runs , next batch  H2D( D2H )
            if (next_base < n) {
                int next_curB = std::min(B, n - next_base);
                // next_buf  GPU buffer
                kmeans_cuda_log_check(cudaStreamWaitEvent(stream[next_buf], event[next_buf], 0), "cudaStreamWaitEvent next buffer");
                kmeans_cuda_log_check(cudaMemcpyAsync(d_data_buf[next_buf], h_data + (size_t)next_base * dim,
                               sizeof(float) * (size_t)next_curB * dim,
                               cudaMemcpyHostToDevice, stream[next_buf]), "cudaMemcpyAsync next H2D");
                //  H2D ,
                kmeans_cuda_log_check(cudaEventRecord(event[next_buf], stream[next_buf]), "cudaEventRecord next preload");
            }

            // , h_num_runs( next batch H2D )
            kmeans_cuda_log_check(cudaStreamSynchronize(stream[cur_buf]), "cudaStreamSynchronize current batch before h_num_runs");
            int h_num_runs = *cur_env.h_num_runs;

            // Step 4:  kernel( Block  cluster)
            if (h_num_runs > 0) {
                //  run_offsets  curB(kernel  offsets[num_runs] = curB)
                kmeans_cuda_log_check(cudaMemcpyAsync(cur_env.d_run_offsets + h_num_runs, &curB,
                                sizeof(int), cudaMemcpyHostToDevice, stream[cur_buf]), "cudaMemcpyAsync run_offsets tail");
                const int dim_v4 = dim / 4;
                const int block_threads = std::min(256, dim_v4);
                kernel_segmented_accum_v2<<<h_num_runs, block_threads, 0, stream[cur_buf]>>>(
                    reinterpret_cast<const float4*>(d_data_cur),
                    cur_env.d_sorted_vals,
                    cur_env.d_run_offsets,
                    cur_env.d_unique_ids,
                    d_accum, d_counts,
                    h_num_runs, dim_v4);
                kmeans_cuda_log_check(cudaGetLastError(), "kernel_segmented_accum_v2 launch");
            }

            // objective: GPU  reduce, host
            if (h_objective && d_objective_sum) {
                int reduce_threads = 256;
                int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
                int reduce_shmem = reduce_threads * sizeof(float);
                kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream[cur_buf]>>>(
                    cur_env.d_best_dist2, d_objective_sum, curB);
                kmeans_cuda_log_check(cudaGetLastError(), "kernel_reduce_sum launch");
            }

            //  buf ( d_data_buf[cur_buf])
            kmeans_cuda_log_check(cudaEventRecord(event[cur_buf], stream[cur_buf]), "cudaEventRecord batch done");

            //
            cur_buf = next_buf;
        }
        }

        // batch()
        // ,batchaccum
        kmeans_cuda_log_check(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize stream[0] after batches");
        kmeans_cuda_log_check(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize stream[1] after batches");

        // update centroids: one block per centroid( stream[0])
        {
            CUDATimer timer_update("  Iter " + std::to_string(it) + ": Update Centroids", ENABLE_CUDA_TIMING);
        ScopedKMeansTimer timing(cfg.collect_timing ? &stats.update_ms : nullptr);
        kernel_update_centroids<<<k, 256, 0, stream[0]>>>(d_centroids, d_accum, d_counts, k, dim);
        kmeans_cuda_log_check(cudaGetLastError(), "kernel_update_centroids launch");

        //  centroid ( stream[0])
        kmeans_cuda_log_check(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize update centroids");
        }

        //  GPU  objective ()
        if (h_objective && d_objective_sum) {
            ScopedKMeansTimer timing(cfg.collect_timing ? &stats.objective_ms : nullptr);
            float obj_val = 0.0f;
            kmeans_cuda_log_check(cudaMemcpy(&obj_val, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy objective D2H");
            *h_objective = obj_val;
            stats.last_iter_objective = obj_val;
        }
    }

    // ====== Final Assignment Pass ======
    // 2:
    // ,d_centroids , d_assign
    //  assignment, d_assign  d_centroids
    {
        CUDATimer timer_final("gpu_kmeans_lloyd: Final Assignment Pass", ENABLE_CUDA_TIMING);
        ScopedKMeansTimer timing(cfg.collect_timing ? &stats.final_assign_ms : nullptr);
        //  stream[0]  stream_env[0]  assignment.
        // Flash  workspace, GEMM baseline.
        if (use_flash_assign) {
            perform_flash_assignment_only(
                h_data,
                d_assign,
                d_centroids,
                d_cnorm2,
                *stream_env,
                ws->flash_env[0],
                d_data_buf[0],
                stream[0],
                B,
                n,
                dim,
                k,
                flash_config,
                (h_objective && d_objective_sum) ? d_objective_sum : nullptr);
        } else {
            perform_assignment_only(
                cfg, h_data, d_assign, d_centroids, d_cnorm2,
                *stream_env, d_data_buf[0], stream[0], handle, B, Ktile, n, dim, k,
                (h_objective && d_objective_sum) ? d_objective_sum : nullptr
            );
        }
    }
    if (h_objective && d_objective_sum) {
        ScopedKMeansTimer timing(cfg.collect_timing ? &stats.objective_ms : nullptr);
        float final_obj = 0.0f;
        kmeans_cuda_log_check(cudaMemcpy(&final_obj, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy final objective D2H");
        *h_objective = final_obj;
        stats.final_objective = final_obj;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_kmeans_lloyd_with_workspace: %s\n", cudaGetErrorString(err));
    }
    if (cfg.collect_timing) {
        stats.total_ms = kmeans_now_ms() - total_start_ms;
        fprintf(stderr,
                "[paraselect:kmeans_lloyd:summary] n=%d k=%d dim=%d iters=%d flash=%d "
                "total_ms=%.6f init_ms=%.6f batch_ms=%.6f update_ms=%.6f "
                "final_assign_ms=%.6f objective_ms=%.6f last_iter_obj=%.9g final_obj=%.9g\n",
                stats.n,
                stats.k,
                stats.dim,
                stats.iters,
                stats.use_flash_assign ? 1 : 0,
                stats.total_ms,
                stats.init_ms,
                stats.batch_ms,
                stats.update_ms,
                stats.final_assign_ms,
                stats.objective_ms,
                stats.last_iter_objective,
                stats.final_objective);
        fflush(stderr);
    }
    g_last_lloyd_stats = stats;
}

void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* h_data,
    idx_t* d_assign,
    float* d_centroids,
    float* h_objective
) {
    KMeansLloydWorkspace* ws = kmeans_lloyd_workspace_create(cfg.n, cfg.k, cfg.dim);
    gpu_kmeans_lloyd_with_workspace(ws, cfg, h_data, d_assign, d_centroids, h_objective);
    kmeans_lloyd_workspace_destroy(ws);
}


// ============================================================
// CPU Initialization Functions Implementation
// ============================================================

namespace {
thread_local KMeansInitStats g_last_init_stats = {
    1,    // default mode: random_fallback
    0.0,  // init_ms
    0,    // n
    0     // k_eff
};
} // namespace

__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int n = cfg.n;
    const int dim = cfg.dim;
    const int k = cfg.k;
    if (n <= 0 || dim <= 0 || k <= 0) {
        g_last_init_stats.mode = 1;
        g_last_init_stats.init_ms = 0.0;
        g_last_init_stats.n = n;
        g_last_init_stats.k_eff = 0;
        return;
    }

    // kmeans++  CPU  O(n * k_eff * dim); n  GPU  0.
    // KMEANS_INIT_AUTO_THRESHOLD: n*k_eff .
    // KMEANS_INIT_FORCE_KMEANSPP / FORCE_RANDOM:.
    //
    // : GPU assignment  squared L2(d2 = ||x||^2 + ||c||^2 - 2 x·c),
    //  COSINE_DISTANCE ; squared L2 .
    const int k_eff = std::min(k, n);

    std::mt19937 rng(cfg.seed);

    auto copy_centroid = [&](int c, int idx) {
        std::memcpy(out_centroids + (size_t)c * dim,
                    data + (size_t)idx * dim,
                    sizeof(float) * dim);
    };

    auto dist2_l2 = [&](int i, const float* centroid) -> float {
        const float* xi = data + (size_t)i * dim;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = xi[d] - centroid[d];
            acc += diff * diff;
        }
        return acc;
    };

    bool use_kmeanspp;
    if (cfg.init_policy == KMEANS_INIT_FORCE_RANDOM) {
        use_kmeanspp = false;
    } else if (cfg.init_policy == KMEANS_INIT_FORCE_KMEANSPP) {
        use_kmeanspp = true;
    } else {
        // KMEANS_INIT_AUTO_THRESHOLD()
        const int64_t n_times_k = (int64_t)n * (int64_t)k_eff;
        constexpr int64_t kKmeansPpBudget = 20000000LL;
        use_kmeanspp = (n_times_k <= kKmeansPpBudget);
    }

    if (!use_kmeanspp) {
        // ----------------------------
        // Random no-replacement sampling()
        // ----------------------------
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;

        for (int i = 0; i < k_eff; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            int j = dist(rng);
            std::swap(indices[i], indices[j]);
        }

        for (int c = 0; c < k_eff; ++c) {
            copy_centroid(c, indices[c]);
        }

        // k > n ,( k<=n),
        for (int c = k_eff; c < k; ++c) {
            copy_centroid(c, indices[0]);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        g_last_init_stats.mode = 1;
        g_last_init_stats.init_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_last_init_stats.n = n;
        g_last_init_stats.k_eff = k_eff;
        return;
    }

    // ----------------------------
    // kmeans++ seeding
    // ----------------------------
    std::uniform_int_distribution<int> uni(0, n - 1);
    int first_idx = uni(rng);
    copy_centroid(0, first_idx);

    std::vector<float> best_dist2((size_t)n, 0.0f);
    {
        const float* c0 = out_centroids;
        for (int i = 0; i < n; ++i) {
            best_dist2[(size_t)i] = dist2_l2(i, c0);
        }
    }

    std::vector<int> chosen_indices;
    chosen_indices.reserve((size_t)k_eff);
    chosen_indices.push_back(first_idx);

    for (int c = 1; c < k_eff; ++c) {
        double total = 0.0;
        for (int i = 0; i < n; ++i) total += (double)best_dist2[(size_t)i];

        int next_idx = -1;
        if (total <= 0.0) {
            //  0(),
            next_idx = uni(rng);
        } else {
            std::uniform_real_distribution<double> urd(0.0, total);
            double r = urd(rng);
            double cumsum = 0.0;
            for (int i = 0; i < n; ++i) {
                cumsum += (double)best_dist2[(size_t)i];
                if (cumsum >= r) {
                    next_idx = i;
                    break;
                }
            }
            if (next_idx < 0) next_idx = n - 1;
        }

        copy_centroid(c, next_idx);
        chosen_indices.push_back(next_idx);

        //  best_dist2:min(old, dist(x, newly_added_centroid))
        const float* cc = out_centroids + (size_t)c * dim;
        for (int i = 0; i < n; ++i) {
            float d2 = dist2_l2(i, cc);
            float& bd2 = best_dist2[(size_t)i];
            if (d2 < bd2) bd2 = d2;
        }
    }

    // k > n
    for (int c = k_eff; c < k; ++c) {
        copy_centroid(c, chosen_indices.empty() ? 0 : chosen_indices[0]);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    g_last_init_stats.mode = 0;
    g_last_init_stats.init_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    g_last_init_stats.n = n;
    g_last_init_stats.k_eff = k_eff;
}

__host__ KMeansInitStats get_last_kmeans_init_stats() {
    return g_last_init_stats;
}

__host__ KMeansLloydStats get_last_kmeans_lloyd_stats() {
    return g_last_lloyd_stats;
}
