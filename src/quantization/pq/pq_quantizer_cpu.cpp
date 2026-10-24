/**
 * PQQuantizerCPU: KMeans  + /,host , linear_fp16.
 */
#include "quantization/pq/pq_quantizer.cuh"
#include "quantization/pq/pq_kmeans.h"
#include <cstring>
#include <vector>

namespace ivf {

namespace {

/** L2  */
inline float l2_sq(const float* a, const float* b, int len) {
    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

} // namespace

PQQuantizerCPU::PQQuantizerCPU(int n_subspace, int n_codes)
    : n_subspace_(n_subspace), n_codes_(n_codes), sub_dim_(0) {}

void PQQuantizerCPU::train(const float* h_data, int n, int dim) {
    if (!h_data || n <= 0 || dim <= 0 || n_subspace_ <= 0 || n_codes_ <= 0) return;
    if (dim % n_subspace_ != 0) return;
    if (n_codes_ > 256) return;

    sub_dim_ = dim / n_subspace_;
    const size_t codebook_size = (size_t)n_subspace_ * n_codes_ * sub_dim_;
    codebooks_.resize(codebook_size);

    std::vector<float> sub_data((size_t)n * sub_dim_);
    for (int m = 0; m < n_subspace_; ++m) {
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < sub_dim_; ++d)
                sub_data[(size_t)i * sub_dim_ + d] = h_data[(size_t)i * dim + (size_t)m * sub_dim_ + d];

        float* out = codebooks_.data() + (size_t)m * n_codes_ * sub_dim_;
        pq_kmeans_subspace(sub_data.data(), n, sub_dim_, n_codes_, 25, 42u + (unsigned)m, out);
    }
}

void PQQuantizerCPU::encode(const float* h_data, int n, int dim, void* h_codes) {
    if (!h_data || !h_codes || sub_dim_ == 0 || (dim / n_subspace_) != sub_dim_) return;
    uint8_t* codes = static_cast<uint8_t*>(h_codes);

    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < n_subspace_; ++m) {
            const float* sub = h_data + (size_t)i * dim + (size_t)m * sub_dim_;
            const float* cb = codebooks_.data() + (size_t)m * n_codes_ * sub_dim_;
            int best = 0;
            float best_sq = l2_sq(sub, cb, sub_dim_);
            for (int k = 1; k < n_codes_; ++k) {
                float sq = l2_sq(sub, cb + (size_t)k * sub_dim_, sub_dim_);
                if (sq < best_sq) {
                    best_sq = sq;
                    best = k;
                }
            }
            codes[(size_t)i * n_subspace_ + m] = (uint8_t)best;
        }
    }
}

void PQQuantizerCPU::decode(const void* h_codes, int n, int dim, float* h_data) {
    if (!h_codes || !h_data || sub_dim_ == 0 || (dim / n_subspace_) != sub_dim_) return;
    const uint8_t* codes = static_cast<const uint8_t*>(h_codes);

    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < n_subspace_; ++m) {
            int k = codes[(size_t)i * n_subspace_ + m];
            if (k >= n_codes_) k = 0;
            const float* cen = codebooks_.data() + (size_t)m * n_codes_ * sub_dim_ + (size_t)k * sub_dim_;
            float* out = h_data + (size_t)i * dim + (size_t)m * sub_dim_;
            std::memcpy(out, cen, (size_t)sub_dim_ * sizeof(float));
        }
    }
}

int PQQuantizerCPU::code_size_bytes(int n, int dim) const {
    (void)dim;
    return n * n_subspace_;
}

} // namespace ivf
