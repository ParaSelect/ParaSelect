/**
 * @file l2norm_cpu.cpp
 *
 * CPU :row-major  interleaved  L2 .
 */

#include "utils/l2norm_cpu.h"
#include "src/layout/ivf_interleaved_layout.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace paraselect {

float l2norm_cpu_vector(const float* x, int n_dim) {
    if (!x || n_dim <= 0)
        return 0.f;
    float s = 0.f;
    for (int d = 0; d < n_dim; ++d) {
        float v = x[d];
        s += v * v;
    }
    return std::sqrt(std::max(0.f, s));
}

void compute_l2_norm_cpu_rowmajor(
    const float* row_major,
    float* norms,
    int n_batch,
    int n_dim) {
    if (!row_major || !norms || n_batch <= 0 || n_dim <= 0)
        return;
    for (int i = 0; i < n_batch; ++i) {
        norms[i] = l2norm_cpu_vector(row_major + static_cast<size_t>(i) * n_dim, n_dim);
    }
}

void compute_l2_norm_cpu_interleaved(
    const float* interleaved,
    float* norms,
    int n_batch,
    int n_dim) {
    if (!interleaved || !norms || n_batch <= 0 || n_dim <= 0)
        return;
    for (int v = 0; v < n_batch; ++v) {
        float s = 0.f;
        for (int d = 0; d < n_dim; ++d) {
            float x = interleaved[interleaved_offset(v, d, n_dim)];
            s += x * x;
        }
        norms[v] = std::sqrt(std::max(0.f, s));
    }
}

}  // namespace paraselect
