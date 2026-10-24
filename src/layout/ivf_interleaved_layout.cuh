/**
 * Interleaved (32 )
 *
 *  Rummy/Faiss GPU  IVFBase.cuh  IVFAppend.cu:
 * - :[vector/32][dim][vector%32], 32
 * - :(v0 d0)(v1 d0)...(v31 d0)(v0 d1)(v1 d1)...(v31 dD-1)(v32 d0)...
 * - :warp  32  32 , coalesced
 *
 * Host :pack_rowmajor_to_interleaved / pack_rowmajor_to_interleaved_at_vec_offset;
 * :pack_rowmajor_to_interleaved_mt / pack_rowmajor_to_interleaved_at_vec_offset_mt( .cpp).
 */

#ifndef PARASELECT_IVF_INTERLEAVED_LAYOUT_CUH
#define PARASELECT_IVF_INTERLEAVED_LAYOUT_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
namespace paraselect {

/// ( warp size )
constexpr int kInterleavedGroupSize = 32;

/**
 *  interleaved  n_vec  d  float .
 *  32  padding .
 */
inline size_t interleaved_storage_bytes(int n_vec, int n_dim) {
    int num_blocks = (n_vec + kInterleavedGroupSize - 1) / kInterleavedGroupSize;
    // :32  × n_dim  × sizeof(float)
    return (size_t)num_blocks * kInterleavedGroupSize * n_dim * sizeof(float);
}

/**
 *  interleaved  n_vec  float ( padding).
 */
inline size_t interleaved_storage_floats(int n_vec, int n_dim) {
    int num_blocks = (n_vec + kInterleavedGroupSize - 1) / kInterleavedGroupSize;
    return (size_t)num_blocks * kInterleavedGroupSize * n_dim;
}

/**
 *  (vec_id, dim_id), interleaved (float ).
 *
 * :[block][dim][lane]
 *   block = vec_id / 32
 *   lane  = vec_id % 32
 *   offset = block * (32 * n_dim) + dim_id * 32 + lane
 */
inline size_t interleaved_offset(int vec_id, int dim_id, int n_dim) {
    int block = vec_id / kInterleavedGroupSize;
    int lane = vec_id % kInterleavedGroupSize;
    return (size_t)block * kInterleavedGroupSize * n_dim + (size_t)dim_id * kInterleavedGroupSize + lane;
}

/**
 *  block  dim, (block,dim)  float .
 *  lane 0..31  32 .
 */
inline size_t interleaved_block_dim_offset(int block, int dim_id, int n_dim) {
    return (size_t)block * kInterleavedGroupSize * n_dim + (size_t)dim_id * kInterleavedGroupSize;
}

/**
 *  row-major [n_vec][n_dim]  interleaved .
 * :row_major[i*n_dim + j] =  i  j
 * :interleaved  [block][dim][lane]
 *
 * @param row_major   , [n_vec * n_dim] float
 * @param interleaved , interleaved_storage_floats(n_vec, n_dim)
 */
void pack_rowmajor_to_interleaved(
    const float* row_major,
    float* interleaved,
    int n_vec,
    int n_dim);

/**
 *  row-major  n_vec  interleaved  ****
 * [vec_offset_global, vec_offset_global + n_vec) .
 *
 *  reorder: compact  row-major , interleaved
 * per-cluster 32 ( _build_interleaved_aligned_from_compact ).
 *
 * : memset; padding lane  0, interleaved  memset,
 * .
 */
void pack_rowmajor_to_interleaved_at_vec_offset(
    const float* row_major,
    float* interleaved,
    int n_vec,
    int n_dim,
    int vec_offset_global);

/**
 *  float row-major → interleaved __half; `pack_rowmajor_to_interleaved_at_vec_offset` .
 */
void pack_rowmajor_to_interleaved_at_vec_offset_fp16(
    const float* row_major,
    __half* interleaved,
    int n_vec,
    int n_dim,
    int vec_offset_global);

/**
 *  interleaved  row-major.
 *
 * @param interleaved ,interleaved
 * @param row_major   , [n_vec * n_dim] float
 */
void unpack_interleaved_to_rowmajor(
    const float* interleaved,
    float* row_major,
    int n_vec,
    int n_dim);

}  // namespace paraselect

#endif /* PARASELECT_IVF_INTERLEAVED_LAYOUT_CUH */
