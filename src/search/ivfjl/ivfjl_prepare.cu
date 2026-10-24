#include "search/kernels/coarse/coarse_prepare.h"
#include "search/ivfjl/ivfjl_prepare.cuh"

#include "cpu/utils/cpu_affinity.h"
#include "cpu/utils/numa_cluster_owner.h"
#include "l2norm/l2norm.cuh"
#include "search/ivfflat/pipeline/ivf_cpu_compact_metadata.h"
#include "search/kernels/fine/cpu_fine/dispatch/cpu_ivf_kernel_combos.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_query_prepare.h"
#include "utils/check_cuda_error.cuh"
#include "utils/utils.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#if PARASELECT_ENABLE_CUVS
#include <raft/core/resources.hpp>
#endif

namespace {

__global__ void ivfjl_round_up_to_group32_kernel(const int* __restrict__ in, int* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int v = in[i];
        out[i] = ((v + 31) / 32) * 32;
    }
}

int choose_ivfjl_coarse_sub_batch_size(int requested_batch_size, int nlist, int nprobe, int dim) {
    if (requested_batch_size <= 0) return requested_batch_size;
    constexpr long long kGiB = 1024LL * 1024 * 1024;
    constexpr long long kMaxGemmElems = 1900000000LL;
    constexpr int kAlign = 32;

    size_t gpu_free = 0;
    size_t gpu_total = 0;
    if (cudaMemGetInfo(&gpu_free, &gpu_total) != cudaSuccess) return requested_batch_size;

    const long long headroom = std::max(4LL * kGiB, (long long)gpu_total / 8);
    long long usable = (long long)gpu_free - headroom;
    if (usable < (long long)gpu_free / 5) usable = (long long)gpu_free / 5;
    usable = usable * 9 / 10;

    const long long bytes_per_query =
        (long long)nlist * sizeof(float) +
        (long long)nprobe * (sizeof(float) + sizeof(idx_t)) +
        (long long)dim * sizeof(float) + 4096;
    int max_sub_mem = (int)std::min((long long)requested_batch_size,
                                    usable / std::max(bytes_per_query, 1LL));
    int max_sub_int = (int)(kMaxGemmElems / std::max(nlist, 1));
    int max_sub = std::min(max_sub_mem, max_sub_int);
    if (max_sub < 1) max_sub = 1;
    if (max_sub >= kAlign) max_sub = (max_sub / kAlign) * kAlign;
    if (max_sub < 1) max_sub = 1;
    if (max_sub < requested_batch_size) {
        std::fprintf(stderr,
                     "[ivfjl_prepare] coarse auto-tiling enabled: coarse_sub_batch_size=%d "
                     "(query_batch_size=%d, nlist=%d, nprobe=%d)\n",
                     max_sub, requested_batch_size, nlist, nprobe);
    }
    return max_sub;
}

int choose_ivfjl_coarse_cluster_tile_size(int query_tile, int nlist, int nprobe) {
    if (query_tile <= 0 || nlist <= 0) return nlist;
    constexpr long long kGiB = 1024LL * 1024 * 1024;
    constexpr long long kMaxGemmElems = 1900000000LL;
    constexpr int kAlign = 256;

    size_t gpu_free = 0;
    size_t gpu_total = 0;
    if (cudaMemGetInfo(&gpu_free, &gpu_total) != cudaSuccess) return nlist;

    const long long headroom = std::max(4LL * kGiB, (long long)gpu_total / 8);
    long long usable = (long long)gpu_free - headroom;
    if (usable < (long long)gpu_free / 5) usable = (long long)gpu_free / 5;
    usable = usable * 9 / 10;

    const long long bytes_per_cluster = (long long)query_tile * (sizeof(float) + sizeof(idx_t));
    int max_tile_mem = (int)std::min((long long)nlist, usable / std::max(bytes_per_cluster, 1LL));
    int max_tile_int = (int)(kMaxGemmElems / std::max(query_tile, 1));
    int max_tile = std::min(max_tile_mem, max_tile_int);
    max_tile = std::max(max_tile, std::min(nlist, nprobe));
    if (max_tile >= kAlign) max_tile = (max_tile / kAlign) * kAlign;
    if (max_tile < nprobe) max_tile = nprobe;
    if (max_tile > nlist) max_tile = nlist;
    if (max_tile < nlist) {
        std::fprintf(stderr,
                     "[ivfjl_prepare] coarse cluster tiling enabled: coarse_cluster_tile_size=%d "
                     "(nlist=%d, nprobe=%d)\n",
                     max_tile, nlist, nprobe);
    }
    return max_tile;
}

paraselect::cpu_fine::CpuIvfStorageView make_ivfjl_fp32_storage_view(
    const IVFSearchArg* arg,
    const IVFFlatFineCtx* fine) {
    if (!fine->h_cluster_vectors || !fine->h_cluster_offsets_ll) {
        throw std::runtime_error("ivfjl_prepare: missing fp32 cluster storage");
    }
    return paraselect::cpu_fine::make_fp32_row_major_storage(
        fine->h_cluster_vectors,
        fine->h_cluster_offsets_ll,
        fine->h_cluster_sizes,
        arg->n_dim,
        arg->n_total_clusters,
        arg->n_total_vectors);
}

}  // namespace

void ivfjl_prepare(IVFSearchCtx* ctx) {
    if (!ctx) throw std::invalid_argument("ivfjl_prepare: ctx is null");
    IVFFlatCoarseCtx* coarse = &ctx->flat_coarse;
    IVFFlatFineCtx* fine = &ctx->flat_fine;
    IVFSearchArg* arg = &ctx->arg;

    if (arg->distance_mode != (int)L2_DISTANCE) {
        throw std::invalid_argument("ivfjl_prepare: only L2 distance is supported");
    }
    if (arg->fp16_coarse) {
        throw std::invalid_argument("ivfjl_prepare: fp16 coarse is not implemented");
    }
    if (arg->n_query <= 0 || arg->query_batch_size <= 0 || arg->n_dim <= 0 ||
        arg->n_total_clusters <= 0 || arg->n_total_vectors <= 0 || arg->n_probes <= 0 || arg->k <= 0) {
        throw std::invalid_argument("ivfjl_prepare: invalid shape");
    }
    if (arg->n_probes > arg->n_total_clusters) {
        throw std::invalid_argument("ivfjl_prepare: nprobe exceeds nlist");
    }
    if (!coarse->h_cluster_sizes || !coarse->h_cluster_centers || !fine->h_cluster_vectors ||
        !fine->h_reordered_indices || !fine->h_cluster_norm) {
        throw std::invalid_argument("ivfjl_prepare: missing host inputs");
    }

    const int max_batch = arg->query_batch_size;
    const int nlist = arg->n_total_clusters;
    const int nprobe = arg->n_probes;
    const int dim = arg->n_dim;
    const int k = arg->k;
    const int max_cpu_threads = paraselect::resolve_cpu_fine_thread_count();

    const auto* combo = paraselect::cpu_fine::resolve_cpu_fine_combo_ptr_or_throw(
        arg->cpu_fine_combo_name);
    if (combo->layout != CPU_FINE_LAYOUT_FP32_ROW_MAJOR) {
        throw std::invalid_argument("ivfjl_prepare: IVFJL requires fp32 row-major CPU fine combo");
    }
    if (!paraselect::cpu_fine::cpu_fine_combo_supported_by_host(*combo)) {
        throw std::runtime_error("ivfjl_prepare: selected CPU fine combo is not supported by this host");
    }
    fine->cached_combo = combo;
    fine->cached_cluster_runner = combo->cluster_metadata;
    fine->cached_cluster_metadata_mode = CPU_FINE_CLUSTER_METADATA_FULL;
    fine->h_use_compact_metadata = 0;
    if (!fine->cached_cluster_runner) {
        throw std::runtime_error("ivfjl_prepare: selected CPU fine combo has no full-metadata runner");
    }
    fine->cached_query_prepare = paraselect::cpu_fine::query_prepare_for_layout(combo->layout);
    if (!fine->cached_query_prepare) {
        throw std::runtime_error("ivfjl_prepare: selected CPU fine combo has no query prepare function");
    }

    coarse->coarse_sub_batch_size = choose_ivfjl_coarse_sub_batch_size(max_batch, nlist, nprobe, dim);
    const int query_tile = std::min(coarse->coarse_sub_batch_size, max_batch);
    coarse->coarse_cluster_tile_size = choose_ivfjl_coarse_cluster_tile_size(query_tile, nlist, nprobe);

    cudaStream_t stream = coarse->stream_0 ? coarse->stream_0 : 0;

    CUDA_CHECK(cudaMalloc(&coarse->d_cluster_size, (size_t)nlist * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(coarse->d_cluster_size, coarse->h_cluster_sizes,
                               (size_t)nlist * sizeof(int), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMalloc(&coarse->d_probe_vector_offset, (size_t)(nlist + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&coarse->d_probe_vector_offset_compact, (size_t)(nlist + 1) * sizeof(int)));
    int* d_cluster_size_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cluster_size_padded, (size_t)(nlist + 1) * sizeof(int)));
    const dim3 block(256);
    const dim3 grid((nlist + block.x - 1) / block.x);
    ivfjl_round_up_to_group32_kernel<<<grid, block, 0, stream>>>(coarse->d_cluster_size, d_cluster_size_padded, nlist);
    CUDA_LAUNCH_CHECK("ivfjl_round_up_to_group32_kernel");
    compute_prefix_sum(d_cluster_size_padded, coarse->d_probe_vector_offset, nlist, stream);
    compute_prefix_sum(coarse->d_cluster_size, coarse->d_probe_vector_offset_compact, nlist, stream);
    CUDA_SYNC_CHECK(stream, "ivfjl_prepare after prefix sums");
    CUDA_CHECK(cudaFree(d_cluster_size_padded));

    int n_padded = 0;
    CUDA_CHECK(cudaMemcpy(&n_padded, coarse->d_probe_vector_offset + nlist, sizeof(int), cudaMemcpyDeviceToHost));
    coarse->n_padded_cache = n_padded;

    CUDA_CHECK(cudaMallocHost(&fine->h_cluster_offsets, (size_t)(nlist + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(fine->h_cluster_offsets, coarse->d_probe_vector_offset_compact,
                          (size_t)(nlist + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    fine->h_cluster_offsets_ll = static_cast<long long*>(std::malloc((size_t)(nlist + 1) * sizeof(long long)));
    if (!fine->h_cluster_offsets_ll) throw std::bad_alloc();
    for (int i = 0; i <= nlist; ++i) fine->h_cluster_offsets_ll[i] = (long long)fine->h_cluster_offsets[i];

    CUDA_CHECK(cudaMallocHost(&fine->h_top_nprobe_batch, (size_t)max_batch * nprobe * sizeof(idx_t)));
    CUDA_CHECK(cudaMallocHost(&fine->h_cluster_query_offset_batch, (size_t)(nlist + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&fine->h_cluster_query_data_batch, (size_t)max_batch * nprobe * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&fine->h_active_cluster_ids_batch, (size_t)max_batch * nprobe * sizeof(int)));
    fine->h_topk_local_idx_batch = static_cast<int*>(std::malloc((size_t)max_batch * k * sizeof(int)));
    fine->h_query_u8_tmp = nullptr;
    fine->h_thread_topk_capacity = (size_t)max_cpu_threads * (size_t)max_batch * (size_t)k;
    fine->h_thread_query_capacity = (size_t)max_cpu_threads * (size_t)max_batch;
    fine->h_thread_dists = static_cast<float*>(std::malloc(fine->h_thread_topk_capacity * sizeof(float)));
    fine->h_thread_idxs = static_cast<int*>(std::malloc(fine->h_thread_topk_capacity * sizeof(int)));
    fine->h_thread_sizes = static_cast<int*>(std::malloc(fine->h_thread_query_capacity * sizeof(int)));
    fine->h_thread_touched = static_cast<uint8_t*>(std::malloc(fine->h_thread_query_capacity * sizeof(uint8_t)));

    const int numa_nodes = std::max(1, std::min(paraselect::get_numa_node_count(), max_cpu_threads));
    const std::vector<int> boundaries = paraselect::numa::build_cluster_owner_boundaries_by_counts(
        coarse->h_cluster_sizes, nlist, numa_nodes);
    fine->h_numa_cluster_boundaries = static_cast<int*>(std::malloc(boundaries.size() * sizeof(int)));
    fine->h_numa_cluster_boundary_count = (int)boundaries.size();
    if (fine->h_numa_cluster_boundaries) {
        std::memcpy(fine->h_numa_cluster_boundaries, boundaries.data(), boundaries.size() * sizeof(int));
    }
    if (!fine->h_topk_local_idx_batch || !fine->h_thread_dists || !fine->h_thread_idxs ||
        !fine->h_thread_sizes || !fine->h_thread_touched || !fine->h_numa_cluster_boundaries) {
        throw std::bad_alloc();
    }
    fine->cached_storage_view = make_ivfjl_fp32_storage_view(arg, fine);

    const int max_entries = max_batch * nprobe;
    const int max_topk = max_batch * k;
    for (int s = 0; s < 1; ++s) {
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_query_count[s], (size_t)(nlist + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_write_pos[s], (size_t)(nlist + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_cluster_offset[s], (size_t)(nlist + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_cluster_query_data[s], (size_t)max_entries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_topk_tmp[s], (size_t)max_topk * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_topk_dist[s], (size_t)max_topk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_topk_index[s], (size_t)max_topk * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_top_nprobe_dist[s], (size_t)query_tile * nprobe * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&coarse->d_top_nprobe_index[s], (size_t)query_tile * nprobe * sizeof(idx_t)));
        coarse->coarse_workspace[s]=paraselect::coarse::prepare_coarse(
            false,arg->distance_mode,query_tile,dim,nlist,nprobe).release();
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_query_batch[s], (size_t)max_batch * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&coarse->d_pool_query_norm[s], (size_t)query_tile * sizeof(float)));

    }

    cublasStatus_t cublas_status = cublasCreate(&coarse->cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("ivfjl_prepare: cublasCreate failed");
    }
    cublasSetStream(coarse->cublas_handle, stream);

    CUDA_CHECK(cudaMalloc(&coarse->d_cluster_centers, (size_t)nlist * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(coarse->d_cluster_centers, coarse->h_cluster_centers,
                               (size_t)nlist * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMalloc(&coarse->d_cluster_centers_norm, (size_t)nlist * sizeof(float)));
    compute_l2_norm_gpu(coarse->d_cluster_centers, coarse->d_cluster_centers_norm,
                        nlist, dim, L2NORM_AUTO, stream);
    CUDA_SYNC_CHECK(stream, "ivfjl_prepare after centers upload");

#if PARASELECT_ENABLE_CUVS
    coarse->cuvs_res = new raft::resources();
#endif
}
