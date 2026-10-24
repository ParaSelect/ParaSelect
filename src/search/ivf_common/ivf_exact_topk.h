#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_EXACT_TOPK_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_EXACT_TOPK_H

#include <algorithm>
#include <limits>

namespace paraselect::search {

// The heap stores the worst retained item at its root. A deterministic
// distance/ID order keeps hash-dedup and sort-dedup paths equivalent when
// quantized or integer data produces equal distances.
inline bool ivf_exact_item_worse(float lhs_distance, int lhs_id,
                                 float rhs_distance, int rhs_id) {
    return lhs_distance > rhs_distance ||
        (lhs_distance == rhs_distance && lhs_id > rhs_id);
}

inline bool ivf_exact_item_better(float lhs_distance, int lhs_id,
                                  float rhs_distance, int rhs_id) {
    return lhs_distance < rhs_distance ||
        (lhs_distance == rhs_distance && lhs_id < rhs_id);
}

inline void ivf_insert_exact_topk(
    float distance,
    int id,
    int k,
    float* distances,
    int* ids,
    int& size) {
    if (size < k) {
        distances[size] = distance;
        ids[size] = id;
        int pos = size++;
        while (pos > 0) {
            const int parent = (pos - 1) >> 1;
            if (!ivf_exact_item_worse(
                    distances[pos], ids[pos], distances[parent], ids[parent])) break;
            std::swap(distances[parent], distances[pos]);
            std::swap(ids[parent], ids[pos]);
            pos = parent;
        }
        return;
    }
    if (!ivf_exact_item_better(distance, id, distances[0], ids[0])) return;
    distances[0] = distance;
    ids[0] = id;
    int pos = 0;
    for (;;) {
        const int left = pos * 2 + 1;
        const int right = left + 1;
        int largest = pos;
        if (left < size && ivf_exact_item_worse(
                distances[left], ids[left], distances[largest], ids[largest])) largest = left;
        if (right < size && ivf_exact_item_worse(
                distances[right], ids[right], distances[largest], ids[largest])) largest = right;
        if (largest == pos) break;
        std::swap(distances[pos], distances[largest]);
        std::swap(ids[pos], ids[largest]);
        pos = largest;
    }
}

inline void ivf_finalize_exact_topk(
    float* distances,
    int* ids,
    int size,
    int k) {
    for (int end = size - 1; end > 0; --end) {
        std::swap(distances[0], distances[end]);
        std::swap(ids[0], ids[end]);
        int pos = 0;
        for (;;) {
            const int left = pos * 2 + 1;
            const int right = left + 1;
            int largest = pos;
            if (left < end && ivf_exact_item_worse(
                    distances[left], ids[left], distances[largest], ids[largest])) largest = left;
            if (right < end && ivf_exact_item_worse(
                    distances[right], ids[right], distances[largest], ids[largest])) largest = right;
            if (largest == pos) break;
            std::swap(distances[pos], distances[largest]);
            std::swap(ids[pos], ids[largest]);
            pos = largest;
        }
    }
    for (int i = std::min(size, k); i < k; ++i) {
        distances[i] = std::numeric_limits<float>::max();
        ids[i] = -1;
    }
}

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_EXACT_TOPK_H
