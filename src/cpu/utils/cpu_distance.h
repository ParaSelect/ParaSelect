#ifndef CPU_DISTANCE_H
#define CPU_DISTANCE_H

#include <cmath>

/**
 * CPU
 *
 */

/**
 * (sum of squares)
 * @param vector
 * @param n_dim
 * @return
 */
inline float compute_squared_sum(const float* vector, int n_dim) {
    float sum = 0.0f;
    for (int d = 0; d < n_dim; d++) {
        sum += vector[d] * vector[d];
    }
    return sum;
}

/**
 *  L2 (L2 norm)
 * @param vector
 * @param n_dim
 * @return L2 (sqrt(sum of squares))
 */
inline float compute_l2_norm(const float* vector, int n_dim) {
    float sum = compute_squared_sum(vector, n_dim);
    return std::sqrt(sum);
}

/**
 * L2 ()
 * @param x ()
 * @param dim
 */
inline void l2_normalize_inplace(float* x, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
        s += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    double inv = (s > 1e-12) ? (1.0 / std::sqrt(s)) : 0.0;
    for (int i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(x[i] * inv);
    }
}

/**
 * (dot product)
 * @param a  a
 * @param b  b
 * @param dim
 * @return
 */
inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 *  L2 (squared L2 distance)
 * @param a  a
 * @param b  b
 * @param dim
 * @return L2
 */
inline float l2_distance_squared(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 *  L2 (L2 distance)
 * @param a  a
 * @param b  b
 * @param dim
 * @return L2
 */
inline float l2_distance(const float* a, const float* b, int dim) {
    return std::sqrt(l2_distance_squared(a, b, dim));
}

/**
 * (cosine distance)
 *  = 1 -  = 1 - (a·b) / (||a|| * ||b||)
 * @param a  a
 * @param b  b
 * @param dim
 * @return
 */
inline float cosine_distance(const float* a, const float* b, int dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = compute_l2_norm(a, dim);
    float norm_b = compute_l2_norm(b, dim);

    if (norm_a < 1e-12f || norm_b < 1e-12f) {
        return 1.0f;  // ,
    }

    float cosine_similarity = dot / (norm_a * norm_b);
    //  [-1, 1],
    cosine_similarity = std::max(-1.0f, std::min(1.0f, cosine_similarity));

    return 1.0f - cosine_similarity;
}

/**
 * (cosine similarity)
 *  = (a·b) / (||a|| * ||b||)
 * @param a  a
 * @param b  b
 * @param dim
 * @return
 */
inline float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = compute_l2_norm(a, dim);
    float norm_b = compute_l2_norm(b, dim);

    if (norm_a < 1e-12f || norm_b < 1e-12f) {
        return 0.0f;  // , 0
    }

    float similarity = dot / (norm_a * norm_b);
    //  [-1, 1],
    return std::max(-1.0f, std::min(1.0f, similarity));
}

/**
 *
 * @param vectors  [n_batch][n_dim]  [n_batch * n_dim]
 * @param squared_sums  [n_batch]
 * @param n_batch
 * @param n_dim
 * @param stride ( n_dim)
 */
inline void compute_squared_sums_batch(
    const float** vectors,
    float* squared_sums,
    int n_batch,
    int n_dim
) {
    for (int i = 0; i < n_batch; i++) {
        squared_sums[i] = compute_squared_sum(vectors[i], n_dim);
    }
}

/**
 *  L2
 * @param vectors  [n_batch][n_dim]  [n_batch * n_dim]
 * @param norms  L2  [n_batch]
 * @param n_batch
 * @param n_dim
 * @param stride ( n_dim)
 */
inline void compute_l2_norms_batch(
    const float** vectors,
    float* norms,
    int n_batch,
    int n_dim
) {
    for (int i = 0; i < n_batch; i++) {
        norms[i] = compute_l2_norm(vectors[i], n_dim);
    }
}

/**
 *  L2
 * @param vectors_a  a [n_batch][n_dim]  [n_batch * n_dim]
 * @param vectors_b  b [n_batch][n_dim]  [n_batch * n_dim]
 * @param distances  [n_batch]
 * @param n_batch
 * @param n_dim
 * @param stride_a  a ( n_dim)
 * @param stride_b  b ( n_dim)
 */
inline void compute_l2_distances_squared_batch(
    const float* vectors_a,
    const float* vectors_b,
    float* distances,
    int n_batch,
    int n_dim,
    int stride_a = 0,
    int stride_b = 0
) {
    if (stride_a == 0) stride_a = n_dim;
    if (stride_b == 0) stride_b = n_dim;
    for (int i = 0; i < n_batch; i++) {
        distances[i] = l2_distance_squared(
            vectors_a + i * stride_a,
            vectors_b + i * stride_b,
            n_dim
        );
    }
}

/**
 *
 * @param vectors_a  a [n_batch][n_dim]  [n_batch * n_dim]
 * @param vectors_b  b [n_batch][n_dim]  [n_batch * n_dim]
 * @param distances  [n_batch]
 * @param n_batch
 * @param n_dim
 * @param stride_a  a ( n_dim)
 * @param stride_b  b ( n_dim)
 */
inline void compute_cosine_distances_batch(
    const float* vectors_a,
    const float* vectors_b,
    float* distances,
    int n_batch,
    int n_dim,
    int stride_a = 0,
    int stride_b = 0
) {
    if (stride_a == 0) stride_a = n_dim;
    if (stride_b == 0) stride_b = n_dim;
    for (int i = 0; i < n_batch; i++) {
        distances[i] = cosine_distance(
            vectors_a + i * stride_a,
            vectors_b + i * stride_b,
            n_dim
        );
    }
}

#endif // CPU_DISTANCE_H

