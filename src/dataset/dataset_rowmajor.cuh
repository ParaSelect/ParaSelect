#ifndef PARASELECT_DATASET_ROWMAJOR_CUH
#define PARASELECT_DATASET_ROWMAJOR_CUH

#include "dataset_reordered_storage.cuh"
#include "dataset/ownership_helper.h"
#include "utils/pch.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <thread>
#include <vector>

namespace paraselect {
namespace dataset {

inline void release_rowmajor_storage(ReorderedRowMajorStorage* storage) {
    if (!storage) return;
    if (storage->fp32) {
        paraselect::ownership::free_raw(storage->fp32, storage->fp32_owner);
        storage->fp32 = nullptr;
    }
    if (storage->u8) {
        paraselect::ownership::free_raw(storage->u8, storage->u8_owner);
        storage->u8 = nullptr;
    }
    storage->fp32_owner = paraselect::ownership::BufferOwner::None;
    storage->u8_owner = paraselect::ownership::BufferOwner::None;
}

inline void prepare_rowmajor_from_existing(
    ReorderedRowMajorStorage* storage,
    const float* h_raw_data,
    const idx_t* h_reordered_indices,
    int n_total_vectors,
    int vector_dim,
    bool data_already_reordered,
    bool own_data,
    paraselect::ownership::BufferOwner own_data_owner)
{
    // ():
    // 1) own_data=true:
    //     h_raw_data (),/.
    // 2) own_data=false && data_already_reordered=true:
    //     cluster  row-major,.
    // 3) own_data=false && data_already_reordered=false:
    //    , h_reordered_indices .
    //
    // :
    // - Python ;
    // -  row-major owner (//).

    if (!storage || !h_raw_data || !h_reordered_indices ||
        n_total_vectors <= 0 || vector_dim <= 0) {
        throw std::invalid_argument("prepare_rowmajor_from_existing: invalid arguments");
    }

    using Owner = paraselect::ownership::BufferOwner;
    using Handle = paraselect::ownership::BufferHandle;

    const size_t data_size = (size_t)n_total_vectors * (size_t)vector_dim;
    if (own_data) {
        // 1:. own_data_owner .
        if (storage->fp32) {
            paraselect::ownership::free_raw(storage->fp32, storage->fp32_owner);
            storage->fp32 = nullptr;
        }
        storage->fp32 = const_cast<float*>(h_raw_data);
        storage->fp32_owner = own_data_owner;
        return;
    }

    const size_t bytes = data_size * sizeof(float);
    Handle new_fp32(memalign(64, bytes), bytes, Owner::StdMalloc);
    if (!new_fp32.data) throw std::bad_alloc();

    if (data_already_reordered) {
        // 2:, copy .
        std::memcpy(new_fp32.data, h_raw_data, bytes);
        if (storage->fp32) {
            paraselect::ownership::free_raw(storage->fp32, storage->fp32_owner);
        }
        paraselect::ownership::take_to(new_fp32, storage->fp32, storage->fp32_owner);
        return;
    }

    const int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    workers.reserve((size_t)num_threads);
    // 3:,.
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            for (int i = t; i < n_total_vectors; i += num_threads) {
                const int src = h_reordered_indices[i];
                std::memcpy(
                    static_cast<float*>(new_fp32.data) + (size_t)i * (size_t)vector_dim,
                    h_raw_data + (size_t)src * (size_t)vector_dim,
                    (size_t)vector_dim * sizeof(float));
            }
        });
    }
    for (auto& w : workers) w.join();
    if (storage->fp32) {
        paraselect::ownership::free_raw(storage->fp32, storage->fp32_owner);
    }
    paraselect::ownership::take_to(new_fp32, storage->fp32, storage->fp32_owner);
}

}  // namespace dataset
}  // namespace paraselect

#endif  // PARASELECT_DATASET_ROWMAJOR_CUH
