/**
 * Warp-Sort Top-K Implementation for paraselect
 * Optimized for Compilation Speed and Peak Performance
 */

#ifndef WARPSORT_CUH
#define WARPSORT_CUH

#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "l2norm/l2norm.cuh"
#include "utils/pch.h"
#include "warpsort_utils.cuh"
#include "bitonic.cuh"


namespace paraselect {
namespace warpsort {

using namespace warpsort_utils;
using namespace bitonic;

// ============================================================================
// Warp Sort Base Class
// ============================================================================

template<int Capacity, bool Ascending, typename T, typename IdxT>
class WarpSort {
    static_assert(isPowerOf2(Capacity), "Capacity must be power of 2");
    static_assert(Capacity/2 <= kMaxCapacity, "Capacity exceeds maximum");

public:
    static constexpr int kWarpWidth = (Capacity < kWarpSize) ? Capacity : kWarpSize;
    static constexpr int kMaxArrLen = Capacity / kWarpWidth;

    static __device__ __forceinline__ T kDummy()
    {
        return Ascending ? upper_bound<T>() : lower_bound<T>();
    }

    const int k;

    __device__ WarpSort(int k_val) : k(k_val)
    {
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            val_arr_[i] = kDummy();
            idx_arr_[i] = IdxT{};
        }
    }

    __device__ __forceinline__ void store(T* out_val, IdxT* out_idx) const
    {
        int idx = Pow2<kWarpWidth>::mod(laneId());

        #pragma unroll
        for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
            out_val[idx] = val_arr_[i];
            out_idx[idx] = idx_arr_[i];
        }
    }

    __device__ __forceinline__ void store_values(T* out_val) const
    {
        int idx = Pow2<kWarpWidth>::mod(laneId());

        #pragma unroll
        for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
            out_val[idx] = val_arr_[i];
        }
    }

protected:
    T val_arr_[kMaxArrLen];
    IdxT idx_arr_[kMaxArrLen];
};

// ============================================================================
// Warp Sort Filtered (Optimized for Compilation and Speed)
// ============================================================================

template<int Capacity, bool Ascending, typename T, typename IdxT>
class WarpSortFiltered : public WarpSort<Capacity, Ascending, T, IdxT> {
    using Base = WarpSort<Capacity, Ascending, T, IdxT>;

public:
    using Base::kWarpWidth;
    using Base::kMaxArrLen;
    using Base::k;

    static_assert(kMaxArrLen >= 2, "Capacity must be >= 64 for WarpSortFiltered");

    __device__ WarpSortFiltered(int k_val, T limit = Base::kDummy())
        : Base(k_val), buf_len_(0), k_th_(limit)
    {
        k_arr_len_ = (k + kWarpWidth - 1) / kWarpWidth;
    }

    __device__ __forceinline__ void add(T val, IdxT idx)
    {
        int buf_max_len = kMaxArrLen - k_arr_len_;

        if (any(buf_len_ >= buf_max_len)) {
            merge_buf_();
        }

        if (is_ordered<Ascending>(val, k_th_)) {
            add_to_buf_(val, idx);
        }
    }

    __device__ __forceinline__ void done()
    {
        if (any(buf_len_ != 0)) {
            merge_buf_();
        }
    }

    __device__ __forceinline__ T threshold() const
    {
        return k_th_;
    }

private:
    using Base::val_arr_;
    using Base::idx_arr_;

    int buf_len_;
    int k_arr_len_;
    T k_th_;

    __device__ __forceinline__ void add_to_buf_(T val, IdxT idx)
    {
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            if (i == k_arr_len_ + buf_len_) {
                val_arr_[i] = val;
                idx_arr_[i] = idx;
            }
        }
        buf_len_++;
    }

    __device__ __forceinline__ void merge_buf_()
    {
        //  Bitonic
        Bitonic<kMaxArrLen>(Ascending, kWarpWidth).sort(val_arr_, idx_arr_);

        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            if (i >= k_arr_len_) {
                val_arr_[i] = Base::kDummy();
                idx_arr_[i] = IdxT{};
            }
        }

        buf_len_ = 0;
        set_k_th_();
    }

    __device__ __forceinline__ void set_k_th_()
    {
        int k_thread = (k - 1) % kWarpWidth;
        int k_idx = (k - 1) / kWarpWidth;
        k_th_ = shfl(val_arr_[k_idx], k_thread, kWarpWidth);
    }
};

} // namespace warpsort
} // namespace paraselect

#endif // WARPSORT_CUH
