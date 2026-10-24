// ============================================================
//  - ()
// :,Pinned Memory,GPU topk,
// ============================================================
#include "search/kernels/brute_force/brute_force_scan.h"
#include "l2norm/l2norm.cuh"
#include "utils/utils.cuh"
#include "search/kernels/topk/reducer/topk_reducer.cuh"
#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

const char* gemm_math_mode_name(BruteForceGemmMathMode mode) {
    switch (mode) {
        case BruteForceGemmMathMode::kCudaCore:
            return "cuda_core";
        case BruteForceGemmMathMode::kTensorCore:
            return "tensor_core";
    }
    return "unknown";
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

size_t env_size_t_or_default(const char* name, size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return (end != value && parsed > 0) ? static_cast<size_t>(parsed) : fallback;
}

}  // namespace

// (MB)
static size_t get_available_memory_mb() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem / (1024 * 1024);
}

// ============================================================
// CUDA Kernel:  GEMM  L2
// : ||q - d||^2 = ||q||^2 + ||d||^2 - 2<q, d>
// ============================================================
__global__ void convert_inner_to_l2_sq(float* dist, const float* q_norm, const float* d_norm,
                                       int n_query, int chunk_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // chunk
    int q = blockIdx.y;                            // query

    if (i < chunk_size && q < n_query) {
        float inner = dist[q * chunk_size + i];
        float qn = q_norm[q];
        float dn = d_norm[i];

        //  L2 , fmaxf
        dist[q * chunk_size + i] = fmaxf(0.0f, qn * qn + dn * dn - 2.0f * inner);
    }
}

// : CPU  TopK
static void merge_into_global_topk(
    int n_query, int k, idx_t chunk_start_idx, int actual_k,
    const float* chunk_topk_dist, const idx_t* chunk_topk_idx,
    float* global_topk_dist, idx_t* global_topk_idx)
{
    for (int q = 0; q < n_query; q++) {
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(2 * k);

        // 1.  TopK
        for (int i = 0; i < k; i++) {
            candidates.push_back({global_topk_dist[q * k + i], global_topk_idx[q * k + i]});
        }
        // 2.  Chunk  TopK ()
        for (int i = 0; i < actual_k; i++) {
            float dist = chunk_topk_dist[q * actual_k + i];
            idx_t local_idx = chunk_topk_idx[q * actual_k + i];
            //  idx_t  size_t (),
            if (local_idx != static_cast<idx_t>(-1)) {
                candidates.push_back({dist, chunk_start_idx + local_idx});
            }
        }

        // 3.  TopK ()
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());

        // 4.
        for (int i = 0; i < k; i++) {
            global_topk_dist[q * k + i] = candidates[i].first;
            global_topk_idx[q * k + i] = candidates[i].second;
        }
    }
}

// ============================================================
//  API
// ============================================================
void brute_force_scan_chunked(
    const float* h_query, const float* h_data,
    int n_query, int n_data, int n_dim, int k,
    int distance_mode,
    float* h_topk_dist, idx_t* h_topk_index,
    size_t max_chunk_size_mb, bool enable_timing,
    BruteForceGemmMathMode gemm_math_mode)
{
    if (distance_mode != L2_DISTANCE) {
        throw std::runtime_error("brute_force_scan_chunked currently supports L2_DISTANCE only");
    }
    if (!h_query || !h_data || !h_topk_dist || !h_topk_index || n_query <= 0 || n_data <= 0 || n_dim <= 0 || k <= 0 || k > n_data) {
        throw std::runtime_error("brute_force_scan_chunked invalid arguments");
    }

    const size_t query_block_size = std::min<size_t>(
        env_size_t_or_default("PARASELECT_BF_QUERY_BLOCK", 512), static_cast<size_t>(n_query));

    printf("[Debug] FUNCTION START: n_query=%d, n_data=%d, n_dim=%d, k=%d, max_chunk_size_mb=%zu, query_block=%zu, gemm_math_mode=%s\n",
           n_query, n_data, n_dim, k, max_chunk_size_mb, query_block_size, gemm_math_mode_name(gemm_math_mode));
    fflush(stdout);

    size_t available_mem_mb = get_available_memory_mb();
    size_t chunk_size_mb = max_chunk_size_mb > 0 ? max_chunk_size_mb : (available_mem_mb * 0.7);
    chunk_size_mb = std::min(chunk_size_mb, (size_t)4000);

    const size_t bytes_per_vec = (size_t)n_dim * sizeof(float);
    const size_t chunk_budget_bytes = chunk_size_mb * 1024ULL * 1024ULL;
    const size_t bytes_per_data_candidate = bytes_per_vec + query_block_size * sizeof(float) + sizeof(float);
    size_t max_chunk_data = chunk_budget_bytes / std::max<size_t>(bytes_per_data_candidate, 1);
    max_chunk_data = std::max(max_chunk_data, static_cast<size_t>(k));
    max_chunk_data = std::min(max_chunk_data, static_cast<size_t>(n_data));
    if (max_chunk_data == 0) {
        throw std::runtime_error("brute_force_scan_chunked computed an empty data block; increase max_chunk_size_mb");
    }

    const int num_query_blocks = (n_query + (int)query_block_size - 1) / (int)query_block_size;
    const int num_data_chunks = (n_data + (int)max_chunk_data - 1) / (int)max_chunk_data;

    if (enable_timing) {
        printf("[brute_force_scan_chunked] Memory: %zu MB available, %zu MB per data block budget\n",
               available_mem_mb, chunk_size_mb);
        printf("[brute_force_scan_chunked] 2D blocks: query_blocks=%d query_block_size=%zu data_chunks=%d max_data_chunk=%zu dist_matrix=%.2f MB\n",
               num_query_blocks, query_block_size, num_data_chunks, max_chunk_data,
               (double)query_block_size * (double)max_chunk_data * (double)sizeof(float) / (1024.0 * 1024.0));
    }

    float* h_data_pinned[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; i++) {
        check_cuda(cudaMallocHost(&h_data_pinned[i], max_chunk_data * n_dim * sizeof(float)), "cudaMallocHost h_data_pinned");
    }

    float* d_query = nullptr;
    float* d_query_norm = nullptr;
    float* d_data[2] = {nullptr, nullptr};
    float* d_dist[2] = {nullptr, nullptr};
    float* d_data_norm[2] = {nullptr, nullptr};
    float* d_topk_dist[2] = {nullptr, nullptr};
    idx_t* d_topk_idx[2] = {nullptr, nullptr};
    void* d_topk_workspace[2] = {nullptr, nullptr};
    size_t d_topk_workspace_bytes[2] = {0, 0};

    check_cuda(cudaMalloc(&d_query, query_block_size * n_dim * sizeof(float)), "cudaMalloc d_query");
    check_cuda(cudaMalloc(&d_query_norm, query_block_size * sizeof(float)), "cudaMalloc d_query_norm");

    for (int i = 0; i < 2; i++) {
        check_cuda(cudaMalloc(&d_data[i], max_chunk_data * n_dim * sizeof(float)), "cudaMalloc d_data");
        check_cuda(cudaMalloc(&d_dist[i], query_block_size * max_chunk_data * sizeof(float)), "cudaMalloc d_dist");
        check_cuda(cudaMalloc(&d_data_norm[i], max_chunk_data * sizeof(float)), "cudaMalloc d_data_norm");
        check_cuda(cudaMalloc(&d_topk_dist[i], query_block_size * k * sizeof(float)), "cudaMalloc d_topk_dist");
        check_cuda(cudaMalloc(&d_topk_idx[i], query_block_size * k * sizeof(idx_t)), "cudaMalloc d_topk_idx");

        paraselect::topk::RowWiseTopKArgs topk_args;
        topk_args.n_rows = static_cast<int>(query_block_size);
        topk_args.n_cols = static_cast<int>(max_chunk_data);
        topk_args.k = k;
        d_topk_workspace_bytes[i] = paraselect::topk::rowwise_topk_workspace_bytes(
            topk_args, paraselect::topk::TopKReducerBackend::kCubBlock);
        if (d_topk_workspace_bytes[i] > 0) {
            check_cuda(cudaMalloc(&d_topk_workspace[i], d_topk_workspace_bytes[i]), "cudaMalloc d_topk_workspace");
        }
    }

    std::vector<float> h_chunk_topk_dist(query_block_size * k);
    std::vector<idx_t> h_chunk_topk_idx(query_block_size * k);
    std::vector<float> h_global_topk_dist(query_block_size * k);
    std::vector<idx_t> h_global_topk_idx(query_block_size * k);

    cudaStream_t streams[2];
    cudaEvent_t events[2];
    for (int i = 0; i < 2; i++) {
        check_cuda(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        check_cuda(cudaEventCreate(&events[i]), "cudaEventCreate");
    }
    cublasHandle_t handles[2];
    for (int i = 0; i < 2; i++) {
        const cublasStatus_t create_status = cublasCreate(&handles[i]);
        if (create_status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        if (gemm_math_mode == BruteForceGemmMathMode::kTensorCore) {
            cublasSetMathMode(handles[i], CUBLAS_TF32_TENSOR_OP_MATH);
        } else {
            cublasSetMathMode(handles[i], CUBLAS_PEDANTIC_MATH);
        }
    }

    const cublasComputeType_t compute_type =
        (gemm_math_mode == BruteForceGemmMathMode::kTensorCore)
            ? CUBLAS_COMPUTE_32F_FAST_TF32
            : CUBLAS_COMPUTE_32F_PEDANTIC;
    const cublasGemmAlgo_t gemm_algo =
        (gemm_math_mode == BruteForceGemmMathMode::kTensorCore)
            ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
            : CUBLAS_GEMM_DEFAULT;

    printf("[Debug] ==== START 2D MAIN LOOP ====\n");
    fflush(stdout);

    for (int q_start = 0; q_start < n_query; q_start += (int)query_block_size) {
        const int q_count = std::min<int>((int)query_block_size, n_query - q_start);
        cudaStream_t query_stream = streams[0];
        check_cuda(cudaMemcpyAsync(d_query, h_query + static_cast<size_t>(q_start) * n_dim,
                                   static_cast<size_t>(q_count) * n_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, query_stream), "cudaMemcpyAsync d_query");
        check_cuda(cudaStreamSynchronize(query_stream), "cudaStreamSynchronize d_query");
        compute_l2_norm_gpu(d_query, d_query_norm, q_count, n_dim, L2NORM_AUTO, query_stream);
        check_cuda(cudaStreamSynchronize(query_stream), "cudaStreamSynchronize query norm");

        std::fill(h_global_topk_dist.begin(), h_global_topk_dist.begin() + static_cast<size_t>(q_count) * k,
                  std::numeric_limits<float>::max());
        std::fill(h_global_topk_idx.begin(), h_global_topk_idx.begin() + static_cast<size_t>(q_count) * k,
                  static_cast<idx_t>(-1));

        for (int chunk_id = 0; chunk_id < num_data_chunks; chunk_id++) {
            int curr_buf = chunk_id % 2;
            idx_t chunk_start = static_cast<idx_t>(chunk_id) * max_chunk_data;
            int chunk_size = std::min(static_cast<int>(max_chunk_data), static_cast<int>(n_data - chunk_start));
            int actual_k = std::min(k, chunk_size);
            cudaStream_t stream = streams[curr_buf];

            std::memcpy(h_data_pinned[curr_buf], h_data + static_cast<size_t>(chunk_start) * n_dim,
                        static_cast<size_t>(chunk_size) * n_dim * sizeof(float));
            check_cuda(cudaMemcpyAsync(d_data[curr_buf], h_data_pinned[curr_buf],
                                       static_cast<size_t>(chunk_size) * n_dim * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync d_data");
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize d_data");

            compute_l2_norm_gpu(d_data[curr_buf], d_data_norm[curr_buf], chunk_size, n_dim, L2NORM_AUTO, stream);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize data norm");

            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(handles[curr_buf], stream);
            const cublasStatus_t gemm_status = cublasGemmEx(
                handles[curr_buf],
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                chunk_size,
                q_count,
                n_dim,
                &alpha,
                d_data[curr_buf],
                CUDA_R_32F,
                n_dim,
                d_query,
                CUDA_R_32F,
                n_dim,
                &beta,
                d_dist[curr_buf],
                CUDA_R_32F,
                chunk_size,
                compute_type,
                gemm_algo);
            if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("brute_force_scan_chunked cublasGemmEx failed");
            }
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize gemm");

            paraselect::topk::RowWiseTopKArgs topk_args;
            topk_args.query_norm = d_query_norm;
            topk_args.data_norm = d_data_norm[curr_buf];
            topk_args.inner_product = d_dist[curr_buf];
            topk_args.n_rows = q_count;
            topk_args.n_cols = chunk_size;
            topk_args.k = actual_k;
            topk_args.distance_op = paraselect::topk::TopKDistanceOp::kL2;
            topk_args.select_min = true;
            topk_args.out_dist = d_topk_dist[curr_buf];
            topk_args.out_idx = d_topk_idx[curr_buf];
            topk_args.workspace = d_topk_workspace[curr_buf];
            topk_args.workspace_bytes = d_topk_workspace_bytes[curr_buf];
            topk_args.stream = stream;
            const cudaError_t topk_err = paraselect::topk::rowwise_topk_reduce(
                topk_args, paraselect::topk::TopKReducerBackend::kCubBlock);
            if (topk_err != cudaSuccess) {
                throw std::runtime_error(std::string("rowwise_topk_reduce failed: ") + cudaGetErrorString(topk_err));
            }
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize topk");

            check_cuda(cudaMemcpy(h_chunk_topk_dist.data(), d_topk_dist[curr_buf],
                                  static_cast<size_t>(q_count) * actual_k * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy h_chunk_topk_dist");
            check_cuda(cudaMemcpy(h_chunk_topk_idx.data(), d_topk_idx[curr_buf],
                                  static_cast<size_t>(q_count) * actual_k * sizeof(idx_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy h_chunk_topk_idx");

            merge_into_global_topk(q_count, k, chunk_start, actual_k,
                                   h_chunk_topk_dist.data(), h_chunk_topk_idx.data(),
                                   h_global_topk_dist.data(), h_global_topk_idx.data());
        }

        for (int q = 0; q < q_count; ++q) {
            std::memcpy(h_topk_dist + static_cast<size_t>(q_start + q) * k,
                        h_global_topk_dist.data() + static_cast<size_t>(q) * k,
                        static_cast<size_t>(k) * sizeof(float));
            std::memcpy(h_topk_index + static_cast<size_t>(q_start + q) * k,
                        h_global_topk_idx.data() + static_cast<size_t>(q) * k,
                        static_cast<size_t>(k) * sizeof(idx_t));
        }
    }

    for (int i = 0; i < 2; i++) {
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
        cudaFree(d_data[i]);
        cudaFree(d_dist[i]);
        cudaFree(d_data_norm[i]);
        cudaFree(d_topk_dist[i]);
        cudaFree(d_topk_idx[i]);
        if (d_topk_workspace[i]) cudaFree(d_topk_workspace[i]);
        cudaFreeHost(h_data_pinned[i]);
    }
    cudaFree(d_query);
    cudaFree(d_query_norm);
}
