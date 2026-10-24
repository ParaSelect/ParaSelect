#include "ivfflat_cpu_overlap.cuh"
#include "ivfflat_probe_metadata.cuh"

#include "search/ivfflat/ivfflat_search.cuh"
#include "search/ivfflat/ivfflat_search_ctx.h"
#include "utils/utils.cuh"
#include "search/ivfflat/stage/ivfflat_stages.cuh"
#include "search/kernels/fine/cpu_fine/schedulers/thread_stats.h"
#include "search/kernels/fine/cpu_fine/executor/cpu_fine_executor.h"
#include "l2norm/l2norm.cuh"
#include "search/ivf_common/ivf_search_stats_collector.h"

#include <algorithm>
#include <chrono>
#include <new>
#include <stdexcept>
#include <vector>

using paraselect::search::IVFSearchStats;


namespace {

void ivfflat_cpu_fine_job(void* owner, const paraselect::cpu_fine::CpuFineBatchView& view) {
    ivfflat_fine_cpu(static_cast<IVFSearchCtx*>(owner), view);
}

// Device-side fp32 -> fp16 query conversion (see ivfflat_pipeline.cu).
__global__ void ivfflat_fp32_to_fp16_query_kernel(const float* __restrict__ src,
                                                  __half* __restrict__ dst,
                                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half_rn(src[i]);
    }
}

struct CpuFineScanStats {
    long long candidate_vectors = 0;
    long long max_candidates_per_query = 0;
    int max_probe_list_size = 0;
};

struct CpuBatchTiming {
    double query_preprocess_ms = 0.0;
    double coarse_search_ms = 0.0;
    double coarse_stage1_ms = 0.0;
    double coarse_d2h_ms = 0.0;
    double coarse_gemm_ms = 0.0;
    double coarse_topk_ms = 0.0;
    double probe_metadata_ms = 0.0;
    double probe_metadata_d2h_ms = 0.0;
    double cpu_fine_ms = 0.0;
    double cpu_fine_submit_wait_ms = 0.0;
    double cpu_fine_exposed_wait_ms = 0.0;
    paraselect::cpu_fine::thread_stats_detail::FineTimingSnapshot fine_timing;
};

struct CpuPipelineSlot {
    CpuPipelineSlot() = default;
    CpuPipelineSlot(const CpuPipelineSlot&) = delete;
    CpuPipelineSlot& operator=(const CpuPipelineSlot&) = delete;
    CpuPipelineSlot(CpuPipelineSlot&&) = delete;
    CpuPipelineSlot& operator=(CpuPipelineSlot&&) = delete;

    int slot_id = 0;
    int batch_id = -1;
    int batch_start = 0;
    int n_query = 0;
    const float* h_queries = nullptr;
    const uint8_t* h_queries_u8 = nullptr;
    float* h_topk_dist_out = nullptr;
    idx_t* h_topk_index_out = nullptr;
    idx_t* h_top_nprobe_batch = nullptr;
    int* h_cluster_query_offset_batch = nullptr;
    int* h_cluster_query_data_batch = nullptr;
    int* h_active_cluster_ids_batch = nullptr;
    int h_active_cluster_count_batch = 0;
    int* h_topk_local_idx_batch = nullptr;
    uint8_t* h_query_u8_tmp = nullptr;
    float* h_thread_dists = nullptr;
    int* h_thread_idxs = nullptr;
    int* h_thread_sizes = nullptr;
    uint8_t* h_thread_touched = nullptr;
    size_t h_thread_topk_capacity = 0;
    size_t h_thread_query_capacity = 0;
    CpuBatchTiming timing;
    CpuFineScanStats scan_stats;

    void allocate(const IVFSearchArg& arg, const IVFFlatFineCtx& fine_template) {
        CUDA_CHECK(cudaMallocHost(&h_top_nprobe_batch,
                                  (size_t)arg.query_batch_size * arg.n_probes * sizeof(idx_t)));
        CUDA_CHECK(cudaMallocHost(&h_cluster_query_offset_batch,
                                  (size_t)(arg.n_total_clusters + 1) * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_cluster_query_data_batch,
                                  (size_t)arg.query_batch_size * arg.n_probes * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_active_cluster_ids_batch,
                                  (size_t)arg.query_batch_size * arg.n_probes * sizeof(int)));
        h_topk_local_idx_batch = static_cast<int*>(
            std::malloc((size_t)arg.query_batch_size * arg.k * sizeof(int)));
        h_query_u8_tmp = static_cast<uint8_t*>(
            std::malloc((size_t)arg.query_batch_size * arg.n_dim * sizeof(uint8_t)));
        h_thread_topk_capacity = fine_template.h_thread_topk_capacity;
        h_thread_query_capacity = fine_template.h_thread_query_capacity;
        h_thread_dists = static_cast<float*>(std::malloc(h_thread_topk_capacity * sizeof(float)));
        h_thread_idxs = static_cast<int*>(std::malloc(h_thread_topk_capacity * sizeof(int)));
        h_thread_sizes = static_cast<int*>(std::malloc(h_thread_query_capacity * sizeof(int)));
        h_thread_touched = static_cast<uint8_t*>(std::malloc(h_thread_query_capacity * sizeof(uint8_t)));
        if (!h_topk_local_idx_batch || !h_query_u8_tmp || !h_thread_dists ||
            !h_thread_idxs || !h_thread_sizes || !h_thread_touched) {
            throw std::bad_alloc();
        }
    }

    void release() {
        if (h_top_nprobe_batch) cudaFreeHost(h_top_nprobe_batch);
        if (h_cluster_query_offset_batch) cudaFreeHost(h_cluster_query_offset_batch);
        if (h_cluster_query_data_batch) cudaFreeHost(h_cluster_query_data_batch);
        if (h_active_cluster_ids_batch) cudaFreeHost(h_active_cluster_ids_batch);
        std::free(h_topk_local_idx_batch);
        std::free(h_query_u8_tmp);
        std::free(h_thread_dists);
        std::free(h_thread_idxs);
        std::free(h_thread_sizes);
        std::free(h_thread_touched);
        h_top_nprobe_batch = nullptr;
        h_cluster_query_offset_batch = nullptr;
        h_cluster_query_data_batch = nullptr;
        h_active_cluster_ids_batch = nullptr;
        h_topk_local_idx_batch = nullptr;
        h_query_u8_tmp = nullptr;
        h_thread_dists = nullptr;
        h_thread_idxs = nullptr;
        h_thread_sizes = nullptr;
        h_thread_touched = nullptr;
    }

    ~CpuPipelineSlot() { release(); }
};

class CpuOverlapRunner {
public:
    CpuOverlapRunner(
        IVFSearchCtx* ctx,
        IVFSearchStats* stats,
        bool async_fine_worker)
        : ctx_(ctx),
          collector_storage_(stats),
          collector_(stats ? &collector_storage_ : nullptr),
          coarse_(&ctx->flat_coarse),
          fine_(&ctx->flat_fine),
          arg_(&ctx->arg),
          dim_(arg_->n_dim),
          k_(arg_->k),
          n_probes_(arg_->n_probes),
          n_query_(arg_->n_query),
          batch_size_(arg_->query_batch_size),
          h_queries_(coarse_->h_queries),
          h_queries_u8_(coarse_->h_queries_u8),
          h_dist_out_(coarse_->h_topk_dist_out),
          h_idx_out_(coarse_->h_topk_index_out),
          n_batches_((n_query_ + batch_size_ - 1) / batch_size_),
          stream_(coarse_->stream_0),
          fine_executor_(async_fine_worker) {}

    ~CpuOverlapRunner() noexcept {
        try {
            fine_executor_.wait();
        } catch (...) {
        }
        restore_owner_fine();
    }

    void run() {
        ivfflat_prepare(ctx_);
        owner_fine_ = *fine_;
        owner_fine_valid_ = true;
        coarse_sub_batch_size_ = (coarse_->coarse_sub_batch_size > 0)
            ? std::min(coarse_->coarse_sub_batch_size, batch_size_)
            : batch_size_;
        if (coarse_sub_batch_size_ < batch_size_) {
            throw std::runtime_error("IVF CPU pipeline requires coarse_sub_batch_size >= query_batch_size");
        }

        for (int sid = 0; sid < 2; ++sid) {
            slots_[sid].slot_id = sid;
            slots_[sid].allocate(*arg_, *fine_);
        }
        if (collector_) {
            collector_->startTotal();
        }

        const auto pipeline_t0 = std::chrono::high_resolution_clock::now();
        for (int bi = 0; bi < n_batches_; ++bi) {
            CpuPipelineSlot* slot = &slots_[bi & 1];
            run_query_tile(bi, slot);
            wait_fine_worker();
            run_metadata_batch(slot);
            submit_fine_job(slot);
        }
        wait_fine_worker();
        cudaStreamSynchronize(stream_);
        CHECK_CUDA_ERRORS;
        const auto pipeline_t1 = std::chrono::high_resolution_clock::now();
        pipeline_wall_ms_ = std::chrono::duration<double, std::milli>(pipeline_t1 - pipeline_t0).count();
        finish_collector();
    }

private:
    template <typename Fn>
    double time_gpu_stage(Fn&& fn) {
        if (!collector_) {
            fn();
            cudaStreamSynchronize(stream_);
            CHECK_CUDA_ERRORS;
            return 0.0;
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        cudaStreamSynchronize(stream_);
        CHECK_CUDA_ERRORS;
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    void run_query_tile(int bi, CpuPipelineSlot* slot) {
        bind_batch_to_slot(bi, slot);
        bind_ctx_to_slot(slot);
        run_query_preprocess(slot);
        run_coarse_search(slot);
    }

    void run_metadata_batch(CpuPipelineSlot* slot) {
        bind_ctx_to_slot(slot);
        run_probe_metadata(slot);
        slot->scan_stats = collect_scan_stats(slot);
    }

    void bind_batch_to_slot(int bi, CpuPipelineSlot* slot) {
        slot->batch_id = bi;
        slot->batch_start = bi * batch_size_;
        slot->n_query = std::min(batch_size_, n_query_ - slot->batch_start);
        slot->h_queries = h_queries_ + (size_t)slot->batch_start * dim_;
        slot->h_queries_u8 = h_queries_u8_ ? (h_queries_u8_ + (size_t)slot->batch_start * dim_) : nullptr;
        slot->h_topk_dist_out = h_dist_out_ + (size_t)slot->batch_start * k_;
        slot->h_topk_index_out = h_idx_out_ + (size_t)slot->batch_start * k_;
        slot->h_active_cluster_count_batch = 0;
        slot->timing = CpuBatchTiming{};
        slot->scan_stats = CpuFineScanStats{};
    }

    void restore_owner_fine() {
        if (!owner_fine_valid_) return;
        *fine_ = owner_fine_;
        owner_fine_valid_ = false;
    }

    void bind_ctx_to_slot(CpuPipelineSlot* slot) {
        coarse_->h_queries = slot->h_queries;
        coarse_->h_queries_u8 = slot->h_queries_u8;
        coarse_->h_topk_dist_out = slot->h_topk_dist_out;
        coarse_->h_topk_index_out = slot->h_topk_index_out;
        coarse_->current_n_querys = slot->n_query;
    }

    void run_query_preprocess(CpuPipelineSlot* slot) {
        const int sid = slot->slot_id;
        slot->timing.query_preprocess_ms += time_gpu_stage([&]() {
            NVTX_PUSH("Query_Preprocess_H2D_Norm");
            CUDA_CHECK(cudaMemcpyAsync(coarse_->d_pool_query_batch[sid],
                                       slot->h_queries,
                                       (size_t)slot->n_query * dim_ * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_));
            if (coarse_->d_pool_query_batch_u8[sid]) {
                if (!slot->h_queries_u8) throw std::runtime_error("u8 resident query buffer missing");
                CUDA_CHECK(cudaMemcpyAsync(coarse_->d_pool_query_batch_u8[sid],
                                           slot->h_queries_u8,
                                           (size_t)slot->n_query * dim_ * sizeof(uint8_t),
                                           cudaMemcpyHostToDevice, stream_));
            }
            compute_l2_norm_gpu(
                coarse_->d_pool_query_batch[sid],
                coarse_->d_pool_query_norm[sid],
                slot->n_query,
                arg_->n_dim,
                L2NORM_AUTO, stream_);
            if (arg_->fp16_coarse) {
                const int n_elems = slot->n_query * dim_;
                if (n_elems > 0 && coarse_->d_pool_query_batch[sid] &&
                    coarse_->d_pool_query_fp16[sid]) {
                    constexpr int kThreads = 256;
                    const int n_blocks = (n_elems + kThreads - 1) / kThreads;
                    ivfflat_fp32_to_fp16_query_kernel<<<n_blocks, kThreads, 0, stream_>>>(
                        coarse_->d_pool_query_batch[sid], coarse_->d_pool_query_fp16[sid], n_elems);
                }
            }
            NVTX_POP();
        });
    }

    void run_coarse_search(CpuPipelineSlot* slot) {
        const int sid = slot->slot_id;
        slot->timing.coarse_search_ms += time_gpu_stage([&]() {
            coarse_->debug_coarse_gemm_ms = 0.0;
            coarse_->debug_coarse_topk_ms = 0.0;
            slot->timing.coarse_stage1_ms += time_gpu_stage([&]() {
                ivfflat_coarse(ctx_, sid);
                CHECK_CUDA_ERRORS;
            });
            slot->timing.coarse_gemm_ms += coarse_->debug_coarse_gemm_ms;
            slot->timing.coarse_topk_ms += coarse_->debug_coarse_topk_ms;
            slot->timing.coarse_d2h_ms += time_gpu_stage([&]() {
                CUDA_CHECK(cudaMemcpyAsync(slot->h_top_nprobe_batch,
                                           coarse_->d_top_nprobe_index[sid],
                                           (size_t)slot->n_query * n_probes_ * sizeof(idx_t),
                                           cudaMemcpyDeviceToHost, stream_));
                if (coarse_->h_coarse_dist != nullptr) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        coarse_->h_coarse_dist[0] + (size_t)slot->batch_start * n_probes_,
                        coarse_->d_pool_top_nprobe_dist[sid],
                        (size_t)slot->n_query * n_probes_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_));
                }
            });
        });
    }

    void run_probe_metadata(CpuPipelineSlot* slot) {
        const int sid = slot->slot_id;
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto metadata = paraselect::cpu_fine::transfer_cpu_probe_metadata(
            ctx_,
            sid,
            slot->n_query,
            stream_,
            {slot->h_cluster_query_offset_batch,
             slot->h_cluster_query_data_batch,
             slot->h_active_cluster_ids_batch},
            "pipeline overlap");
        slot->h_active_cluster_count_batch = metadata.active_cluster_count;
        slot->timing.probe_metadata_d2h_ms += metadata.d2h_ms;
        const auto t1 = std::chrono::high_resolution_clock::now();
        slot->timing.probe_metadata_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    CpuFineScanStats collect_scan_stats(const CpuPipelineSlot* slot) const {
        CpuFineScanStats out;
        if (!fine_->h_cluster_sizes || !slot->h_top_nprobe_batch) return out;
        for (int q = 0; q < slot->n_query; ++q) {
            long long query_candidates = 0;
            for (int p = 0; p < n_probes_; ++p) {
                const idx_t cluster = slot->h_top_nprobe_batch[(size_t)q * n_probes_ + p];
                if (cluster < 0 || cluster >= arg_->n_total_clusters) continue;
                const int list_size = fine_->h_cluster_sizes[(int)cluster];
                query_candidates += list_size;
                out.max_probe_list_size = std::max(out.max_probe_list_size, list_size);
            }
            out.candidate_vectors += query_candidates;
            out.max_candidates_per_query = std::max(out.max_candidates_per_query, query_candidates);
        }
        return out;
    }

    paraselect::cpu_fine::CpuFineBatchView make_fine_view(CpuPipelineSlot* slot) const {
        paraselect::cpu_fine::CpuFineBatchView view{};
        paraselect::search::ivfflat_stage::bind_cpu_fine_batch(
            &view,
            paraselect::search::ivfflat_stage::CpuFineBatchBindConfig{
                slot->n_query,
                arg_->k,
                slot->h_queries,
                slot->h_queries_u8,
                slot->h_topk_dist_out,
                slot->h_topk_index_out,
                slot->h_top_nprobe_batch,
                slot->h_cluster_query_offset_batch,
                slot->h_cluster_query_data_batch,
                slot->h_active_cluster_ids_batch,
                slot->h_active_cluster_count_batch,
                slot->h_topk_local_idx_batch,
                slot->h_query_u8_tmp,
                slot->h_thread_dists,
                slot->h_thread_idxs,
                slot->h_thread_sizes,
                slot->h_thread_touched,
                slot->h_thread_topk_capacity,
                slot->h_thread_query_capacity,
            });
        return view;
    }

    void submit_fine_job(CpuPipelineSlot* slot) {
        const auto view = make_fine_view(slot);
        running_slot_ = slot;
        slot->timing.fine_timing = {};
        slot->timing.cpu_fine_ms = 0.0;
        fine_t0_ = std::chrono::high_resolution_clock::now();
        fine_executor_.submit({ctx_, ivfflat_cpu_fine_job, view});
    }

    void wait_fine_worker() {
        CpuPipelineSlot* completed_slot = running_slot_;
        if (!completed_slot) return;
        const auto wait_t0 = std::chrono::high_resolution_clock::now();
        fine_executor_.wait();
        const auto fine_t1 = std::chrono::high_resolution_clock::now();
        running_slot_ = nullptr;
        completed_slot->timing.cpu_fine_ms = fine_executor_.latest_stage_ms();
        completed_slot->timing.cpu_fine_submit_wait_ms =
            std::chrono::duration<double, std::milli>(fine_t1 - fine_t0_).count();
        completed_slot->timing.cpu_fine_exposed_wait_ms =
            std::chrono::duration<double, std::milli>(fine_t1 - wait_t0).count();
        completed_slot->timing.fine_timing = fine_executor_.latest_timing();
        total_candidate_vectors_ += completed_slot->scan_stats.candidate_vectors;
        global_max_candidates_per_query_ = std::max(
            global_max_candidates_per_query_, completed_slot->scan_stats.max_candidates_per_query);
        global_max_probe_list_size_ = std::max(
            global_max_probe_list_size_, completed_slot->scan_stats.max_probe_list_size);
        record_batch_timing(completed_slot->timing);
    }

    void record_batch_timing(const CpuBatchTiming& timing) {
        if (!collector_) return;
        collector_->addStageTime("query_preprocess", timing.query_preprocess_ms);
        collector_->addStageTime("coarse_search", timing.coarse_search_ms);
        collector_->addStageTime("coarse_stage1", timing.coarse_stage1_ms);
        collector_->addStageTime("coarse_gemm", timing.coarse_gemm_ms);
        collector_->addStageTime("coarse_topk", timing.coarse_topk_ms);
        collector_->addStageTime("coarse_d2h", timing.coarse_d2h_ms);
        collector_->addStageTime("probe_metadata", timing.probe_metadata_ms);
        collector_->addStageTime("probe_metadata_d2h", timing.probe_metadata_d2h_ms);
        collector_->addStageTime("cpu_fine", timing.cpu_fine_ms);
        collector_->addStageTime("cpu_fine_submit_wait", timing.cpu_fine_submit_wait_ms);
        collector_->addStageTime("cpu_fine_exposed_wait", timing.cpu_fine_exposed_wait_ms);
        collector_->addStageTime("cpu_fine_query_prepare", timing.fine_timing.query_prepare_ms);
        collector_->addStageTime("cpu_fine_metadata_build", timing.fine_timing.metadata_build_ms);
        collector_->addStageTime("cpu_fine_scratch_clear", timing.fine_timing.scratch_clear_ms);
        collector_->addStageTime("cpu_fine_task_build", timing.fine_timing.task_build_ms);
        collector_->addStageTime("cpu_fine_scan", timing.fine_timing.scan_ms);
        collector_->addStageTime("cpu_fine_merge", timing.fine_timing.merge_ms);
        collector_->addStageTime("cpu_fine_runner", timing.fine_timing.runner_ms);
        collector_->addStageTime("cpu_fine_id_map", timing.fine_timing.id_map_ms);
    }

    void finish_collector() {
        if (!collector_) return;
        collector_->stopTotal();
        add_collector_fields();
        collector_->finalizeCoreFields();
    }

    void add_collector_fields() {
        collector_->addField("ivf_cpu_batch_pipeline", 1.0);
        collector_->addField("pipeline_wall_ms", pipeline_wall_ms_);
        collector_->addField("cpu_fine_submit_wait_ms", collector_->avgStageTimeMs("cpu_fine_submit_wait"));
        collector_->addField("cpu_fine_exposed_wait_ms", collector_->avgStageTimeMs("cpu_fine_exposed_wait"));
        collector_->addField("coarse_stage1_ms", collector_->avgStageTimeMs("coarse_stage1"));
        collector_->addField("coarse_gemm_ms", collector_->avgStageTimeMs("coarse_gemm"));
        collector_->addField("coarse_topk_ms", collector_->avgStageTimeMs("coarse_topk"));
        collector_->addField("coarse_d2h_ms", collector_->avgStageTimeMs("coarse_d2h"));
        collector_->addField("probe_metadata_compact", (double)(fine_->h_use_compact_metadata != 0));
        collector_->addField("probe_metadata_d2h_ms", collector_->avgStageTimeMs("probe_metadata_d2h"));
        collector_->addField("cpu_fine_query_prepare_ms", collector_->avgStageTimeMs("cpu_fine_query_prepare"));
        collector_->addField("cpu_fine_metadata_build_ms", collector_->avgStageTimeMs("cpu_fine_metadata_build"));
        collector_->addField("cpu_fine_scratch_clear_ms", collector_->avgStageTimeMs("cpu_fine_scratch_clear"));
        collector_->addField("cpu_fine_task_build_ms", collector_->avgStageTimeMs("cpu_fine_task_build"));
        collector_->addField("cpu_fine_scan_ms", collector_->avgStageTimeMs("cpu_fine_scan"));
        collector_->addField("cpu_fine_merge_ms", collector_->avgStageTimeMs("cpu_fine_merge"));
        collector_->addField("cpu_fine_runner_ms", collector_->avgStageTimeMs("cpu_fine_runner"));
        collector_->addField("cpu_fine_id_map_ms", collector_->avgStageTimeMs("cpu_fine_id_map"));
        collector_->addField("cpu_fine_candidate_vectors", (double)total_candidate_vectors_);
        collector_->addField("cpu_fine_avg_candidates_per_query",
                             n_query_ > 0 ? (double)total_candidate_vectors_ / (double)n_query_ : 0.0);
        collector_->addField("cpu_fine_max_candidates_per_query", (double)global_max_candidates_per_query_);
        collector_->addField("cpu_fine_max_probe_list_size", (double)global_max_probe_list_size_);
    }

    IVFSearchCtx* ctx_ = nullptr;
    paraselect::search::IVFSearchStatsCollector collector_storage_;
    IVFFlatCoarseCtx* coarse_ = nullptr;
    IVFFlatFineCtx* fine_ = nullptr;
    IVFSearchArg* arg_ = nullptr;
    int dim_ = 0;
    int k_ = 0;
    int n_probes_ = 0;
    int n_query_ = 0;
    int batch_size_ = 0;
    int n_batches_ = 0;
    int coarse_sub_batch_size_ = 0;
    const float* h_queries_ = nullptr;
    const uint8_t* h_queries_u8_ = nullptr;
    float* h_dist_out_ = nullptr;
    idx_t* h_idx_out_ = nullptr;
    cudaStream_t stream_ = nullptr;
    IVFFlatFineCtx owner_fine_{};
    bool owner_fine_valid_ = false;
    CpuPipelineSlot slots_[2];
    paraselect::search::IVFSearchStatsCollector* collector_ = nullptr;
    paraselect::cpu_fine::CpuFineExecutor fine_executor_;
    CpuPipelineSlot* running_slot_ = nullptr;
    std::chrono::high_resolution_clock::time_point fine_t0_{};
    double pipeline_wall_ms_ = 0.0;
    long long total_candidate_vectors_ = 0;
    long long global_max_candidates_per_query_ = 0;
    int global_max_probe_list_size_ = 0;
};

}  // namespace

void ivfflat_cpu_overlap(
    IVFSearchCtx* ctx,
    IVFSearchStats* stats)
{
    CpuOverlapRunner(ctx, stats, ctx->arg.cpu_batch_pipeline_overlap).run();
}
