#ifndef PARASELECT_SEARCH_TOPK_FIXED_TOPK_CUH
#define PARASELECT_SEARCH_TOPK_FIXED_TOPK_CUH

#include <cfloat>

namespace paraselect::search::topk_detail {

__device__ inline void insert_topk(float d, int idx, int k, float* dists, int* idxs) {
    if (idx < 0 || d >= dists[0]) return;
    dists[0] = d;
    idxs[0] = idx;
    int pos = 0;
    while (true) {
        const int l = pos * 2 + 1;
        const int r = l + 1;
        int largest = pos;
        if (l < k && dists[l] > dists[largest]) largest = l;
        if (r < k && dists[r] > dists[largest]) largest = r;
        if (largest == pos) break;
        float td = dists[pos];
        dists[pos] = dists[largest];
        dists[largest] = td;
        int ti = idxs[pos];
        idxs[pos] = idxs[largest];
        idxs[largest] = ti;
        pos = largest;
    }
}

template <int K>
__device__ inline void init_topk_fixed(float* dists, int* idxs) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
        dists[i] = FLT_MAX;
        idxs[i] = -1;
    }
}

}  // namespace paraselect::search::topk_detail

#endif  // PARASELECT_SEARCH_TOPK_FIXED_TOPK_CUH
