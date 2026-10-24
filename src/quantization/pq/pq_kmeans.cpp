/**
 *  KMeans, PQ CPU/GPU .
 */
#include "quantization/pq/pq_kmeans.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace ivf {

void pq_kmeans_subspace(
    const float* data,
    int n,
    int sub_dim,
    int n_codes,
    int max_iter,
    unsigned seed,
    float* centroids_out
) {
    if (!data || !centroids_out || n <= 0 || sub_dim <= 0 || n_codes <= 0) return;
    std::vector<float> centroids((size_t)n_codes * sub_dim);
    std::vector<int> assign(n);
    std::vector<int> count(n_codes, 0);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int k = 0; k < n_codes; ++k) {
        int idx = dist(gen);
        for (int d = 0; d < sub_dim; ++d)
            centroids[(size_t)k * sub_dim + d] = data[(size_t)idx * sub_dim + d];
    }

    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < n; ++i) {
            int best = 0;
            float best_sq = 0.f;
            for (int d = 0; d < sub_dim; ++d) {
                float v = data[(size_t)i * sub_dim + d] - centroids[d];
                best_sq += v * v;
            }
            for (int k = 1; k < n_codes; ++k) {
                float sq = 0.f;
                for (int d = 0; d < sub_dim; ++d) {
                    float v = data[(size_t)i * sub_dim + d] - centroids[(size_t)k * sub_dim + d];
                    sq += v * v;
                }
                if (sq < best_sq) {
                    best_sq = sq;
                    best = k;
                }
            }
            assign[i] = best;
        }

        std::fill(centroids.begin(), centroids.end(), 0.f);
        std::fill(count.begin(), count.end(), 0);
        for (int i = 0; i < n; ++i) {
            int k = assign[i];
            count[k]++;
            for (int d = 0; d < sub_dim; ++d)
                centroids[(size_t)k * sub_dim + d] += data[(size_t)i * sub_dim + d];
        }
        for (int k = 0; k < n_codes; ++k) {
            int c = count[k];
            if (c > 0) {
                for (int d = 0; d < sub_dim; ++d)
                    centroids[(size_t)k * sub_dim + d] /= (float)c;
            }
        }
    }

    std::memcpy(centroids_out, centroids.data(), (size_t)n_codes * sub_dim * sizeof(float));
}

} // namespace ivf
