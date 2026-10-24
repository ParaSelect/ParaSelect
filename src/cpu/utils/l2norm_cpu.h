/**
 * @file l2norm_cpu.h
 *
 * Host(CPU) L2 , `l2norm.cu` / `l2norm_interleaved.cu`  GPU ,.
 */

#ifndef PARASELECT_L2NORM_CPU_H
#define PARASELECT_L2NORM_CPU_H

namespace paraselect {

/**
 *  L2 :sqrt(sum_d x[d]^2)
 */
float l2norm_cpu_vector(const float* x, int n_dim);

/**
 * row-major [n_batch, n_dim]: L2  norms[n_batch]
 */
void compute_l2_norm_cpu_rowmajor(
    const float* row_major,
    float* norms,
    int n_batch,
    int n_dim);

/**
 * interleaved  [block][dim][lane]( ivf_interleaved_layout / GPU interleaved l2norm ):
 *  n_batch  L2  norms[n_batch]
 */
void compute_l2_norm_cpu_interleaved(
    const float* interleaved,
    float* norms,
    int n_batch,
    int n_dim);

}  // namespace paraselect

#endif  // PARASELECT_L2NORM_CPU_H
