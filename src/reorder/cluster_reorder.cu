// cluster_reorder.cu
//  Cluster (American Flag / In-Place Bucket Permutation)
#include "reorder/reorder.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================
//  Cluster ()
// ============================================================

template<class T>
void cluster_reorder_vectors_inplace(
    T* h_data,
    idx_t* h_cluster_id,
    int n, int dim, int k
) {
    if (n <= 0 || dim <= 0 || k <= 0) {
        return;
    }

    // Step 1:  counts[c]
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
        int c = h_cluster_id[i];
        if (c >= 0 && c < k) {
            counts[c]++;
        }
    }

    // Step 2:  offsets[c]  next_free[c]
    std::vector<int> offsets(k, 0);
    std::vector<int> next_free(k, 0);
    for (int c = 1; c < k; ++c) {
        offsets[c] = offsets[c-1] + counts[c-1];
    }
    for (int c = 0; c < k; ++c) {
        next_free[c] = offsets[c];
    }

    // Step 3:
    for (int c = 0; c < k; ++c) {
        int start = offsets[c];
        int end = start + counts[c];

        for (int i = start; i < end; ++i) {
            int current_c = h_cluster_id[i];

            //  bucket ,
            if (current_c == c) {
                continue;
            }

            // , swap  bucket
            //  i  bucket
            while (true) {
                int dst = next_free[current_c];

                //  -  64-bit
                size_t i_base = (size_t)i * dim;
                size_t dst_base = (size_t)dst * dim;
                for (int d = 0; d < dim; ++d) {
                    T tmp = h_data[i_base + d];
                    h_data[i_base + d] = h_data[dst_base + d];
                    h_data[dst_base + d] = tmp;
                }

                //  cluster_id
                int tmp_c = h_cluster_id[i];
                h_cluster_id[i] = h_cluster_id[dst];
                h_cluster_id[dst] = tmp_c;

                //  next_free
                next_free[current_c]++;

                //  swap  cluster_id
                current_c = h_cluster_id[i];

                //  bucket, i ,
                if (current_c == c) {
                    break;
                }
            }
        }
    }
}

void cpu_cluster_reorder_vectors_inplace(float* data,idx_t* assignments,int n,int dim,int k) {
    cluster_reorder_vectors_inplace(data,assignments,n,dim,k);
}
void cpu_cluster_reorder_vectors_inplace(uint8_t* data,idx_t* assignments,int n,int dim,int k) {
    cluster_reorder_vectors_inplace(data,assignments,n,dim,k);
}

// ============================================================
//  Cluster ()
// ============================================================

void cpu_cluster_reorder_indices_inplace(
    idx_t* h_indices,
    idx_t* h_cluster_id,
    int n, int k
) {
    if (n <= 0 || k <= 0) {
        return;
    }

    // Step 1:  counts[c]
    std::vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
        int c = h_cluster_id[i];
        if (c >= 0 && c < k) {
            counts[c]++;
        }
    }

    // Step 2:  offsets[c]  next_free[c]
    std::vector<int> offsets(k, 0);
    std::vector<int> next_free(k, 0);
    for (int c = 1; c < k; ++c) {
        offsets[c] = offsets[c-1] + counts[c-1];
    }
    for (int c = 0; c < k; ++c) {
        next_free[c] = offsets[c];
    }

    // Step 3:
    for (int c = 0; c < k; ++c) {
        int start = offsets[c];
        int end = start + counts[c];

        for (int i = start; i < end; ++i) {
            int current_c = h_cluster_id[i];

            //  bucket ,
            if (current_c == c) {
                continue;
            }

            // , swap  bucket
            //  i  bucket
            while (true) {
                int dst = next_free[current_c];

                //
                idx_t tmp = h_indices[i];
                h_indices[i] = h_indices[dst];
                h_indices[dst] = tmp;

                //  cluster_id
                int tmp_c = h_cluster_id[i];
                h_cluster_id[i] = h_cluster_id[dst];
                h_cluster_id[dst] = tmp_c;

                //  next_free
                next_free[current_c]++;

                //  swap  cluster_id
                current_c = h_cluster_id[i];

                //  bucket, i ,
                if (current_c == c) {
                    break;
                }
            }
        }
    }
}
