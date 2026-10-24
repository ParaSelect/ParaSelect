#ifndef IVF_PQ_KMEANS_H
#define IVF_PQ_KMEANS_H

namespace ivf {

/**  KMeans:data (n, sub_dim), centroids (n_codes, sub_dim) */
void pq_kmeans_subspace(
    const float* data,
    int n,
    int sub_dim,
    int n_codes,
    int max_iter,
    unsigned seed,
    float* centroids_out
);

} // namespace ivf

#endif
