#pragma once

#include "search/ivfrespq/ivfrespq_search.cuh"

namespace paraselect::search {

// Exhaustive search of an existing IVFResPQ index. IVF assets define the
// residual encoding only. Neither the seed nor scan stage performs coarse search.
struct BruteForceResPQSearchRequest {
    IVFResPQIndexView index;
    IVFQueryBatchView queries;
    IVFSearchOutputView output;
    int batch_size = 10000;
    int scan_batch = 256; // query cap, further bounded by scan workspace
    int topk = 10;
    int n_rerank = 512;
    // Zero keeps the full-sort reference; positive values enable sampled-row seed histogram.
    int seed_histogram_bins = 0;
    int seed_rows = 65536;
    bool verify_candidates = false; // diagnostic ADC top-k comparison, outside reported timing
    int rerank_threads = 16;
    int rerank_prefetch = 8;
};

void bruteforce_respq_search(const BruteForceResPQSearchRequest& request,
                            IVFSearchStats* stats = nullptr);

} // namespace paraselect::search
