/**
 * Interleaved ( Rummy/Faiss IVFInterleaved)
 */

#include "layout/ivf_interleaved_layout.cuh"
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace paraselect {

namespace {

/**  [vec_off, vec_off + n_loc)  [block_begin, block_end) */
void pack_blocks_range_at_offset(
    const float* row_major,
    float* interleaved,
    int n_dim,
    int vec_off,
    int n_loc,
    int block_begin,
    int block_end) {
    const int kG = kInterleavedGroupSize;
    const size_t wpb = (size_t)kG * n_dim;
    const int v_end = vec_off + n_loc;
    for (int b = block_begin; b < block_end; ++b) {
        for (int lane = 0; lane < kG; ++lane) {
            const int global_v = b * kG + lane;
            if (global_v < vec_off || global_v >= v_end)
                continue;
            const int local_i = global_v - vec_off;
            for (int j = 0; j < n_dim; ++j) {
                const size_t dst =
                    (size_t)b * wpb + (size_t)j * kG + lane;
                interleaved[dst] = row_major[(size_t)local_i * n_dim + j];
            }
        }
    }
}

} // namespace

void pack_rowmajor_to_interleaved(
    const float* row_major,
    float* interleaved,
    int n_vec,
    int n_dim) {
    if (!row_major || !interleaved || n_vec <= 0 || n_dim <= 0) return;

    const int num_blocks = (n_vec + kInterleavedGroupSize - 1) / kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kInterleavedGroupSize * n_dim;

    //  padding ( 32 )
    std::memset(interleaved, 0, interleaved_storage_bytes(n_vec, n_dim));

    for (int i = 0; i < n_vec; ++i) {
        const int block = i / kInterleavedGroupSize;
        const int lane = i % kInterleavedGroupSize;

        for (int j = 0; j < n_dim; ++j) {
            const size_t dst_offset = (size_t)block * words_per_block + (size_t)j * kInterleavedGroupSize + lane;
            interleaved[dst_offset] = row_major[(size_t)i * n_dim + j];
        }
    }
}

void pack_rowmajor_to_interleaved_at_vec_offset(
    const float* row_major,
    float* interleaved,
    int n_vec,
    int n_dim,
    int vec_offset_global) {
    if (!row_major || !interleaved || n_vec <= 0 || n_dim <= 0 || vec_offset_global < 0)
        return;

    const int kG = kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kG * n_dim;

    for (int i = 0; i < n_vec; ++i) {
        const int global_v = vec_offset_global + i;
        const int block = global_v / kG;
        const int lane = global_v % kG;
        for (int j = 0; j < n_dim; ++j) {
            const size_t dst_offset =
                (size_t)block * words_per_block + (size_t)j * kG + lane;
            interleaved[dst_offset] = row_major[(size_t)i * n_dim + j];
        }
    }
}

/** float row-major → interleaved __half; `pack_rowmajor_to_interleaved_at_vec_offset` . */
void pack_rowmajor_to_interleaved_at_vec_offset_fp16(
    const float* row_major,
    __half* interleaved,
    int n_vec,
    int n_dim,
    int vec_offset_global) {
    if (!row_major || !interleaved || n_vec <= 0 || n_dim <= 0 || vec_offset_global < 0)
        return;
    const int kG = paraselect::kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kG * n_dim;
    for (int i = 0; i < n_vec; ++i) {
        const int global_v = vec_offset_global + i;
        const int block = global_v / kG;
        const int lane = global_v % kG;
        for (int j = 0; j < n_dim; ++j) {
            const size_t dst_off =
                (size_t)block * words_per_block + (size_t)j * kG + lane;
            interleaved[dst_off] = __float2half_rn(row_major[(size_t)i * n_dim + j]);
        }
    }
}


void unpack_interleaved_to_rowmajor(
    const float* interleaved,
    float* row_major,
    int n_vec,
    int n_dim) {
    if (!interleaved || !row_major || n_vec <= 0 || n_dim <= 0) return;

    const int num_blocks = (n_vec + kInterleavedGroupSize - 1) / kInterleavedGroupSize;
    const size_t words_per_block = (size_t)kInterleavedGroupSize * n_dim;

    for (int i = 0; i < n_vec; ++i) {
        const int block = i / kInterleavedGroupSize;
        const int lane = i % kInterleavedGroupSize;

        for (int j = 0; j < n_dim; ++j) {
            const size_t src_offset = (size_t)block * words_per_block + (size_t)j * kInterleavedGroupSize + lane;
            row_major[(size_t)i * n_dim + j] = interleaved[src_offset];
        }
    }
}

}  // namespace paraselect
