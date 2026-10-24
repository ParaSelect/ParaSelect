#ifndef PARASELECT_TOPK_REDUCER_CUH
#define PARASELECT_TOPK_REDUCER_CUH

#include "utils/pch.h"

#include <cuda_runtime.h>

namespace raft { class resources; }

namespace paraselect {
namespace topk {

enum class TopKReducerBackend {
    kWarpSort = 0,
    kCubBlock = 1,
    kTiered = 2,
    kTieredFusion = 3,
};

enum class TopKDistanceOp {
    kL2 = 0,
    kCosine = 1,
};

enum class TopKInputKind {
    kDenseDistanceMatrix = 0,
    kCandidateMatrix = 1,
};

struct RowWiseTopKArgs {
    const float* query_norm = nullptr;
    const float* data_norm = nullptr;
    const float* inner_product = nullptr;
    int n_rows = 0;
    int n_cols = 0;
    int k = 0;
    TopKDistanceOp distance_op = TopKDistanceOp::kL2;
    bool select_min = true;
    float* out_dist = nullptr;
    idx_t* out_idx = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaStream_t stream = 0;
    raft::resources* cuvs_res = nullptr;
};

struct CandidateTopKArgs {
    const int* candidate_idx = nullptr;
    const float* candidate_dist = nullptr;
    int n_rows = 0;
    int n_sources = 0;
    int local_k = 0;
    int global_k = 0;
    int thread_topk = 8;
    int* out_idx = nullptr;
    float* out_dist = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaStream_t stream = 0;
};

const char* topk_reducer_backend_name(TopKReducerBackend backend);
TopKReducerBackend parse_topk_reducer_backend(const char* name, TopKReducerBackend fallback);
size_t rowwise_topk_workspace_bytes(const RowWiseTopKArgs& args, TopKReducerBackend backend);
size_t candidate_topk_workspace_bytes(const CandidateTopKArgs& args, TopKReducerBackend backend);

cudaError_t rowwise_topk_reduce(
    const RowWiseTopKArgs& args,
    TopKReducerBackend backend);

cudaError_t candidate_topk_reduce(
    const CandidateTopKArgs& args,
    TopKReducerBackend backend);

}  // namespace topk
}  // namespace paraselect

#endif  // PARASELECT_TOPK_REDUCER_CUH
