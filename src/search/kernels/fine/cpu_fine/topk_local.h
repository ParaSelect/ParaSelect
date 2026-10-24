#ifndef PARASELECT_CPU_FINE_TOPK_LOCAL_H
#define PARASELECT_CPU_FINE_TOPK_LOCAL_H

namespace paraselect {
namespace cpu_fine {

#if defined(__GNUC__) || defined(__clang__)
#define PARASELECT_TOPK_LOCAL_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define PARASELECT_TOPK_LOCAL_ALWAYS_INLINE inline
#endif

struct LocalTopK {
    int size;
    float* dists;
    int* idxs;

    static PARASELECT_TOPK_LOCAL_ALWAYS_INLINE void sift_up(float* d, int* i, int pos) {
        while (pos > 0) {
            const int parent = (pos - 1) >> 1;
            if (d[parent] >= d[pos]) break;
            const float td = d[pos];
            d[pos] = d[parent];
            d[parent] = td;
            const int ti = i[pos];
            i[pos] = i[parent];
            i[parent] = ti;
            pos = parent;
        }
    }

    static PARASELECT_TOPK_LOCAL_ALWAYS_INLINE void sift_down(float* d, int* i, int n, int pos) {
        while (true) {
            const int l = 2 * pos + 1;
            const int r = 2 * pos + 2;
            int largest = pos;
            if (l < n && d[l] > d[largest]) largest = l;
            if (r < n && d[r] > d[largest]) largest = r;
            if (largest == pos) break;
            const float td = d[pos];
            d[pos] = d[largest];
            d[largest] = td;
            const int ti = i[pos];
            i[pos] = i[largest];
            i[largest] = ti;
            pos = largest;
        }
    }

    PARASELECT_TOPK_LOCAL_ALWAYS_INLINE void push(float dist, int idx, int cap) {
        if (size < cap) {
            dists[size] = dist;
            idxs[size] = idx;
            ++size;
            sift_up(dists, idxs, size - 1);
        } else if (dist < dists[0]) {
            dists[0] = dist;
            idxs[0] = idx;
            sift_down(dists, idxs, size, 0);
        }
    }
};

#undef PARASELECT_TOPK_LOCAL_ALWAYS_INLINE

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_FINE_TOPK_LOCAL_H
