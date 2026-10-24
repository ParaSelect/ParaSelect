#ifndef PARASELECT_CLUSTERING_KMEANS_CAGRA_TRAIN_CUH
#define PARASELECT_CLUSTERING_KMEANS_CAGRA_TRAIN_CUH

#include "clustering/kmeans/kmeans_policy.cuh"

#include <cstdint>
#include <stdexcept>
#include <vector>

struct KMeansCagraParams {
    int train_batch = 131072;
    int final_batch = 65536;
    int topk_centroids = 8;
    int itopk_size = 96;
    int graph_degree = 32;
    int intermediate_graph_degree = 64;
    int search_width = 1;
    int max_iterations = 0;
    int team_size = 0;
    int thread_block_size = 0;
    float index_scale = 1.0f / 255.0f;
    KMeansCagraBuildAlgo build_algo = KMeansCagraBuildAlgo::IvfPq;
};

inline KMeansCagraParams resolve_kmeans_cagra_profile(KMeansCagraProfile profile) {
    switch (profile) {
        case KMeansCagraProfile::FastV1: return {};
    }
    throw std::runtime_error("unknown KMeansCagraProfile");
}

struct KMeansCagraTrainConfig {
    int n = 0;
    int dim = 0;
    int k = 0;
    int iters = 0;
    int seed = 12345;
    int device_id = 0;
    KMeansCagraParams params{};
};

bool kmeans_train_centroids_cagra_u8(
    const KMeansCagraTrainConfig& cfg,
    const uint8_t* h_data,
    std::vector<float>* centroids);

bool kmeans_train_centroids_cagra_fp32(
    const KMeansCagraTrainConfig& cfg,
    const float* h_data,
    std::vector<float>* centroids);

#endif
