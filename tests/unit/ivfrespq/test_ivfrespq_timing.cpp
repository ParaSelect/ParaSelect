#include "search/ivfrespq/ivfrespq_timing.h"
#include "search/ivf_common/ivf_rerank_policy.h"
#include "search/ivf_common/ivf_search_stats_collector.h"

#include <cmath>
#include <iostream>

namespace {

bool near(double actual, double expected, double tolerance = 1e-9) {
    if (std::abs(actual - expected) <= tolerance) return true;
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    return false;
}

}  // namespace

int main() {
    const paraselect::search::IVFResPQSerialTiming timing{
        1.02374, 0.415270, 0.607048, 0.00113859, 0.218799, 0.387645};

    if (!near(timing.fine_overhead_ms(), 0.000604)) return 1;
    if (!near(timing.stage_sum_ms(), 1.02345659)) return 1;
    if (!near(timing.closure_error_ms(), 0.00028341)) return 1;

    // CUDA event time is nested inside GPU wall time and deliberately absent
    // from this additive decomposition.
    const double nested_gpu_event_ms = 0.198799;
    if (!(nested_gpu_event_ms < timing.fine_gpu_wall_ms)) return 1;

    paraselect::search::IVFSearchStats stats;
    paraselect::search::IVFSearchStatsCollector collector(&stats);
    collector.startBatch(0, 0, 8);
    collector.addStageTime("tiled_fine", 1.0);
    collector.addStageTime("tiled_fine", 2.0);
    collector.stopBatch();
    collector.startBatch(1, 8, 8);
    collector.addStageTime("tiled_fine", 3.0);
    collector.addStageTime("tiled_fine", 4.0);
    collector.stopBatch();
    if (!near(collector.avgStageTimeMs("tiled_fine"), 2.5)) return 1;
    if (!near(collector.avgStageTimePerBatchMs("tiled_fine"), 5.0)) return 1;

    if (paraselect::search::ivf_rerank_thread_count(16, 1) != 1) return 1;
    if (paraselect::search::ivf_rerank_thread_count(16, 8) != 8) return 1;
    if (paraselect::search::ivf_rerank_thread_count(16, 32) != 16) return 1;
    if (paraselect::search::ivf_rerank_thread_count(16, 256) != 16) return 1;
    if (paraselect::search::ivf_rerank_thread_count(4, 256) != 4) return 1;
    if (paraselect::search::ivf_rerank_input_count(512, 128, 10) != 128) return 1;
    if (paraselect::search::ivf_rerank_input_count(512, 8, 10) != 10) return 1;
    if (paraselect::search::ivf_rerank_input_count(512, 1024, 10) != 512) return 1;
    if (paraselect::search::ivf_rerank_input_count(512, 0, 10) != 512) return 1;
    return 0;
}
