#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_SEARCH_STATS_COLLECTOR_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_SEARCH_STATS_COLLECTOR_H

#include "search/ivf_common/ivf_search_types.h"

#include <chrono>
#include <string>

namespace paraselect::search {

// Search-local timing and diagnostics. Recall, RSS, CSV output, and experiment
// metadata belong to the benchmark layer and are intentionally excluded.
class IVFSearchStatsCollector {
public:
    explicit IVFSearchStatsCollector(IVFSearchStats* output) : output_(output) {
        if (output_) *output_ = {};
    }

    explicit operator bool() const { return output_ != nullptr; }

    void startTotal() {
        if (!output_) return;
        total_start_ = Clock::now();
    }

    double stopTotal() {
        if (!output_) return 0.0;
        output_->total_time_ms = elapsed_ms(total_start_);
        return output_->total_time_ms;
    }

    void setAverageBatchTime(double time_ms) {
        if (output_) output_->avg_batch_time_ms = time_ms;
    }

    double averageBatchTimeMs() const {
        return output_ ? output_->avg_batch_time_ms : 0.0;
    }

    void startBatch(int batch_id, int start_query, int n_queries) {
        if (!output_) return;
        current_batch_ = {batch_id, start_query, n_queries, 0.0};
        batch_start_ = Clock::now();
    }

    double stopBatch() {
        if (!output_) return 0.0;
        current_batch_.time_ms = elapsed_ms(batch_start_);
        output_->batch_timings.push_back(current_batch_);
        batch_total_ms_ += current_batch_.time_ms;
        output_->avg_batch_time_ms =
            batch_total_ms_ / static_cast<double>(output_->batch_timings.size());
        return current_batch_.time_ms;
    }

    // Overlapped pipelines measure each batch on the GPU (events spanning the
    // per-batch streams) instead of host wall time; feed those durations here.
    void addBatchTime(int batch_id, int start_query, int n_queries, double time_ms) {
        if (!output_) return;
        current_batch_ = {batch_id, start_query, n_queries, time_ms};
        output_->batch_timings.push_back(current_batch_);
        batch_total_ms_ += time_ms;
        output_->avg_batch_time_ms =
            batch_total_ms_ / static_cast<double>(output_->batch_timings.size());
    }

    void addStageTime(const std::string& name, double time_ms) {
        if (output_) output_->stage_timings.push_back({name, time_ms});
    }

    void addField(const std::string& key, double value) {
        if (output_) output_->numeric_fields[key] = value;
    }

    void addToField(const std::string& key, double value) {
        if (output_) output_->numeric_fields[key] += value;
    }

    void addField(const std::string& key, const std::string& value) {
        if (output_) output_->string_fields[key] = value;
    }

    void addField(const std::string& key, const char* value) {
        if (output_) output_->string_fields[key] = value ? value : "";
    }

    double avgStageTimeMs(const std::string& name) const {
        if (!output_) return 0.0;
        double total_ms = 0.0;
        int count = 0;
        for (const auto& stage : output_->stage_timings) {
            if (stage.name == name) {
                total_ms += stage.time_ms;
                ++count;
            }
        }
        return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
    }

    // A stage may be emitted more than once inside one outer query batch
    // (for example, M20 fine search processes a batch in several tiles).
    // Sum those invocations before averaging across outer batches.
    double avgStageTimePerBatchMs(const std::string& name) const {
        if (!output_ || output_->batch_timings.empty()) return 0.0;
        double total_ms = 0.0;
        for (const auto& stage : output_->stage_timings) {
            if (stage.name == name) total_ms += stage.time_ms;
        }
        return total_ms / static_cast<double>(output_->batch_timings.size());
    }

    void finalizeCoreFields() {
        if (!output_) return;
        output_->query_preprocess_ms = avgStageTimeMs("query_preprocess");
        output_->coarse_search_ms = avgStageTimeMs("coarse_search");
        output_->probe_metadata_ms = avgStageTimeMs("probe_metadata");
        output_->cpu_fine_ms = avgStageTimeMs("cpu_fine");
        output_->numeric_fields["coarse_stage1_ms"] = avgStageTimeMs("coarse_stage1");
        output_->numeric_fields["coarse_gemm_ms"] = avgStageTimeMs("coarse_gemm");
        output_->numeric_fields["coarse_topk_ms"] = avgStageTimeMs("coarse_topk");
        output_->numeric_fields["coarse_d2h_ms"] = avgStageTimeMs("coarse_d2h");

        const double stage_total_ms = output_->query_preprocess_ms +
            output_->coarse_search_ms + output_->probe_metadata_ms +
            output_->cpu_fine_ms;
        output_->numeric_fields["stage_total_ms"] = stage_total_ms;
        output_->numeric_fields["query_preprocess_pct"] =
            stage_total_ms > 0.0 ? output_->query_preprocess_ms / stage_total_ms : 0.0;
        output_->numeric_fields["coarse_search_pct"] =
            stage_total_ms > 0.0 ? output_->coarse_search_ms / stage_total_ms : 0.0;
        output_->numeric_fields["probe_metadata_pct"] =
            stage_total_ms > 0.0 ? output_->probe_metadata_ms / stage_total_ms : 0.0;
        output_->numeric_fields["cpu_fine_pct"] =
            stage_total_ms > 0.0 ? output_->cpu_fine_ms / stage_total_ms : 0.0;
    }

private:
    using Clock = std::chrono::high_resolution_clock;

    static double elapsed_ms(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    IVFSearchStats* output_ = nullptr;
    IVFSearchBatchTiming current_batch_;
    double batch_total_ms_ = 0.0;
    Clock::time_point total_start_;
    Clock::time_point batch_start_;
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_SEARCH_STATS_COLLECTOR_H
