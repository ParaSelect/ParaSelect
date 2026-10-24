#pragma once
#include "search/kernels/coarse/coarse.h"

namespace paraselect::coarse { struct CoarseHistogramTopKWorkspace; }
namespace paraselect::coarse::detail {
// Internal binding only: callers never allocate or select this scratch.
struct Execution : Request {
    explicit Execution(const Request& request) : Request(request) {}
    bool fused_gemm = false;
    int l2_topk_policy = -1;
    float* d_inner_product = nullptr;
    float* d_tile_top_nprobe_dist = nullptr;
    idx_t* d_tile_top_nprobe_index = nullptr;
    CoarseHistogramTopKWorkspace* histogram_workspace = nullptr;
};
void run_gemm_score_tiles(const Execution&, Timing* = nullptr);
}
