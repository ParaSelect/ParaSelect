#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "test_utils.cuh"
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include "cpu/utils/cpu_array_utils.h"

struct ClusterQueryData {
    int* cluster_map; //cluster_idx -> cluster_id
    int* cluster_query_data; //cluster_id -> query_ids
    int* cluster_query_offset; //cluster_id -> query_offset
    int* cluster_query_data_size; //cluster_id -> query_count
    int  cluster_size;
    int* cluster_vector_index; //cluster_id -> vector_index
    float** cluster_vector; //cluster_id -> vectors
    int* cluster_vector_num; //cluster_id -> vector_num
    int  tol_vector;
};

// /**
//  *
//  */
// void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size) {
//     //
//     void* data = malloc(n_batch * n_dim * elem_size);
//     if (data == NULL) return NULL;

//     //
//     void** vector_list = (void**)malloc(n_batch * sizeof(void*));
//     if (vector_list == NULL) {
//         free(data);
//         return NULL;
//     }

//     //
//     for (size_t i = 0; i < n_batch; i++) {
//         vector_list[i] = (char*)data + i * n_dim * elem_size;
//     }

//     return vector_list;
// }

//topkcluster-query
void* generate_cluster_query_data(int* query_cluster_group, int n_query, int k, int batch_size) {
    // :query-clustercluster-query
    std::map<int, std::set<int>> cluster_query_map;
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < k; j++) {
            cluster_query_map[query_cluster_group[i * k + j]].insert(i);
        }
    }

    // :batch(10cluster,batch_size=3,4batch:[3,3,3,1])
    int total_clusters = cluster_query_map.size();
    int num_batches = (total_clusters + batch_size - 1) / batch_size; //

    // :ClusterQueryData(batch)
    ClusterQueryData* cluster_query_data_array = (ClusterQueryData*)malloc(num_batches * sizeof(ClusterQueryData));

    // :batch
    int cluster_idx = 0;
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        // batchcluster
        int clusters_in_this_batch = std::min(batch_size, total_clusters - batch_idx * batch_size);

        ClusterQueryData& current_batch = cluster_query_data_array[batch_idx];

        // 1. cluster_map: batchclusterID
        current_batch.cluster_map = (int*)malloc(clusters_in_this_batch * sizeof(int));

        // 2. cluster_size: batchcluster
        current_batch.cluster_size = clusters_in_this_batch;

        // 3. cluster_query_offset: clusterquery_data
        current_batch.cluster_query_offset = (int*)malloc(clusters_in_this_batch * sizeof(int));

        // 4. cluster_query_data_size: clusterquery
        current_batch.cluster_query_data_size = (int*)malloc(clusters_in_this_batch * sizeof(int));

        // 5. cluster_vector_index: cluster
        current_batch.cluster_vector_index = (int*)malloc(clusters_in_this_batch * sizeof(int));

        // 6. cluster_vector_num: cluster
        current_batch.cluster_vector_num = (int*)malloc(clusters_in_this_batch * sizeof(int));

        // batchcluster
        auto it = cluster_query_map.begin();
        std::advance(it, batch_idx * batch_size);

        // :query
        int total_queries_in_batch = 0;
        int total_vectors_in_batch = 0;

        for (int i = 0; i < clusters_in_this_batch; i++) {
            int cluster_id = it->first;
            int query_count = it->second.size();

            // ()
            float* dummy_ptr = nullptr;
            int vector_num = 0;
            get_cluster_vector(cluster_id, &dummy_ptr, &vector_num);

            total_queries_in_batch += query_count;
            total_vectors_in_batch += vector_num;
            it++;
        }

        // query
        current_batch.cluster_query_data = (int*)malloc(total_queries_in_batch * sizeof(int));

        //
        current_batch.cluster_vector = (float**)malloc(clusters_in_this_batch * sizeof(float*));

        // :
        it = cluster_query_map.begin();
        std::advance(it, batch_idx * batch_size);

        int query_data_idx = 0;
        int vector_offset = 0;

        for (int i = 0; i < clusters_in_this_batch; i++) {
            int cluster_id = it->first;
            int query_count = it->second.size();

            // cluster
            current_batch.cluster_map[i] = cluster_id;
            current_batch.cluster_query_data_size[i] = query_count;
            current_batch.cluster_query_offset[i] = query_data_idx;

            // query
            for (int query_id : it->second) {
                current_batch.cluster_query_data[query_data_idx++] = query_id;
            }

            //
            float* cluster_vector_ptr = nullptr;
            int vector_num = 0;
            get_cluster_vector(cluster_id, &cluster_vector_ptr, &vector_num);

            //
            current_batch.cluster_vector_num[i] = vector_num;
            current_batch.cluster_vector_index[i] = vector_offset;
            current_batch.cluster_vector[i] = cluster_vector_ptr;

            vector_offset += vector_num;
            cluster_idx++;
            it++;
        }

        current_batch.tol_vector = total_vectors_in_batch;
    }

    return (void*)cluster_query_data_array;
}

void get_cluster_vector(int cluster_id, float** cluster_vector, int* vector_num) {
    // ,
    // :

    // cluster(50-200)
    *vector_num = 50 + (rand() % 151);

    // :
    int vector_dim = 128;
    *cluster_vector = (float*)malloc(*vector_num * vector_dim * sizeof(float));

    // , [-1, 1]
    //  cluster_id , cluster
    init_array_multithreaded(*cluster_vector, *vector_num * vector_dim, 5678 + cluster_id, -1.0f, 1.0f);
}

