#ifndef PARASELECT_DATASET_CACHE_PLAIN_PQ_CACHE_H
#define PARASELECT_DATASET_CACHE_PLAIN_PQ_CACHE_H

#include "dataset/cache/residual_pq_cache.h"

#include <algorithm>
#include <string>

namespace paraselect::dataset {

inline PQCacheRequest make_plain_pq_cache_request(
    int n_vectors,
    int dim,
    int n_lists,
    int pq_m,
    int pq_k,
    int pq_iters,
    int pq_train_n) {
    PQCacheRequest spec;
    spec.n_vectors = n_vectors;
    spec.dim = dim;
    spec.n_lists = n_lists;
    spec.pq_m = pq_m;
    spec.pq_k = pq_k;
    spec.pq_iters = pq_iters;
    spec.trainer = "plain-pq-ooc-v2";
    spec.training_sampling = "sample";
    spec.training_rows = std::min(n_vectors, std::max(pq_k, pq_train_n));
    spec.pq_type = "plain";
    spec.codes_layout = "cluster-reordered-[N,M]";
    return spec;
}

}  // namespace paraselect::dataset

#endif  // PARASELECT_DATASET_CACHE_PLAIN_PQ_CACHE_H
