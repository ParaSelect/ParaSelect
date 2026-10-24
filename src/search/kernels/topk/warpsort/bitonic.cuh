/**
 * Bitonic Sort Implementation for paraselect (Optimized for Compilation Speed)
 */

#ifndef BITONIC_CUH
#define BITONIC_CUH

#include "warpsort_utils.cuh"

namespace paraselect {
namespace bitonic {

using namespace warpsort_utils;

// ============================================================================
// Helper to prevent inlining for large sizes to speed up compilation
// ============================================================================
template<int Size, typename KeyT, typename IdxT>
__device__ __noinline__ void merge_step_noinline(bool ascending, int warp_width, KeyT* keys, IdxT* indices);

/**
 * Bitonic Sort Implementation
 * @tparam Size Number of elements per thread (must be power of 2)
 */
template<int Size = 1>
class Bitonic {
    static_assert(isPowerOf2(Size), "Size must be power of 2");

public:
    __device__ __forceinline__ Bitonic(bool ascending, int warp_width = kWarpSize)
        : ascending_(ascending), warp_width_(warp_width) {}

    template<typename KeyT, typename IdxT>
    __device__ __forceinline__ void merge(KeyT* keys, IdxT* indices) const {
        merge_impl(ascending_, warp_width_, keys, indices);
    }

    template<typename KeyT, typename IdxT>
    __device__ __forceinline__ void sort(KeyT* keys, IdxT* indices) const {
        sort_impl(ascending_, warp_width_, keys, indices);
    }

    //
    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void merge_impl(bool ascending, int warp_width, KeyT* keys, IdxT* indices) {
        const int lane = laneId();

        // 1. Thread-local sorting
        #pragma unroll
        for (int s = Size; s > 1; s >>= 1) {
            const int stride = s >> 1;
            #pragma unroll
            for (int offset = 0; offset < Size; offset += s) {
                #pragma unroll
                for (int i = 0; i < stride; i++) {
                    compare_and_swap(ascending, keys[offset + i], keys[offset + i + stride],
                                     indices[offset + i], indices[offset + i + stride]);
                }
            }
        }

        // 2. Cross-thread shuffle merge
        // :,
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            if (stride < warp_width) {
                #pragma unroll
                for (int offset = 0; offset < Size; offset++) {
                    KeyT other_key = shfl(keys[offset], lane ^ stride, warp_width);
                    IdxT other_idx = shfl(indices[offset], lane ^ stride, warp_width);

                    const bool is_second = bool(lane & stride);
                    const bool do_swap = (ascending != is_second)
                        ? (keys[offset] > other_key)
                        : (keys[offset] < other_key);

                    if (do_swap) {
                        keys[offset] = other_key;
                        indices[offset] = other_idx;
                    }
                }
            }
        }
    }

private:
    const bool ascending_;
    const int warp_width_;

    template<int S, typename KeyT, typename IdxT>
    static __device__ __forceinline__ void sort_recursive(bool ascending, int warp_width, KeyT* keys, IdxT* indices) {
        if constexpr (S == 1) {
            const int lane = laneId();
            #pragma unroll
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, keys, indices);
            }
        } else {
            constexpr int HalfSize = S / 2;
            sort_recursive<HalfSize>(false, warp_width, keys, indices);
            sort_recursive<HalfSize>(true, warp_width, keys + HalfSize, indices + HalfSize);
        }

        // For larger per-lane arrays, keep the merge out-of-line to avoid compile-time blowups.
        if constexpr (S > 4) {
            merge_step_noinline<S, KeyT, IdxT>(ascending, warp_width, keys, indices);
        } else {
            Bitonic<S>::merge_impl(ascending, warp_width, keys, indices);
        }
    }

    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void sort_impl(bool ascending, int warp_width, KeyT* keys, IdxT* indices) {
        sort_recursive<Size>(ascending, warp_width, keys, indices);
    }

    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void compare_and_swap(bool ascending, KeyT& a, KeyT& b, IdxT& ia, IdxT& ib) {
        bool cond = ascending ? (a > b) : (a < b);
        if (cond) {
            KeyT tk = a; a = b; b = tk;
            IdxT ti = ia; ia = ib; ib = ti;
        }
    }
};


// ============================================================================
// Non-inlined helper implementation
// ============================================================================
template<int Size, typename KeyT, typename IdxT>
__device__ __noinline__ void merge_step_noinline(bool ascending, int warp_width, KeyT* keys, IdxT* indices) {
    Bitonic<Size>::merge_impl(ascending, warp_width, keys, indices);
}

} // namespace bitonic
} // namespace paraselect

#endif // BITONIC_CUH
