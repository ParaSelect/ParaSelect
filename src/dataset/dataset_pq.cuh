#ifndef PARASELECT_CUDA_DATASET_DATASET_PQ_CUH
#define PARASELECT_CUDA_DATASET_DATASET_PQ_CUH

#include <cstdint>
#include <vector>
#include "dataset/cache/cluster_cache.h"
#include "dataset/cache/residual_pq_cache.h"
#include "dataset/cache/turboquant_cache.h"
#include "dataset/cache/res_row_correction_assets.h"

namespace paraselect::dataset {

struct ClusterDatasetPQArtifacts {
    std::vector<uint8_t> pq_codes;
    std::vector<uint8_t> pq_codes_aosoa;
    std::vector<long long> pq_codes_aosoa_offsets;
    std::vector<float> pq_codebook;
    std::vector<float> residual_rotation;
    ClusterCache::MappedResult mapped_clusters;
    PQCacheMappedResult mapped_pq;
    TurboQuantCacheMappedResult mapped_tq;
    RowCorrectionAsset row_correction;
    int pq_m = 0;
    int pq_k = 0;
    int pq_d_sub = 0;
    int pq_codes_aosoa_tile = 0;
    double build_ms = 0.0;

    const uint8_t* codes_data() const {
        return mapped_pq.hit ? mapped_pq.codes.data() : pq_codes.data();
    }

    const float* codebook_data() const {
        return mapped_pq.hit ? mapped_pq.codebook.data() : pq_codebook.data();
    }

    const std::uint32_t* tq_codes_data() const {
        return mapped_tq.hit ? mapped_tq.codes.data() : nullptr;
    }

    const float* tq_scales_data() const {
        return mapped_tq.hit ? mapped_tq.scales.data() : nullptr;
    }

    void reset() {
        pq_codes.clear();
        pq_codes_aosoa.clear();
        pq_codes_aosoa_offsets.clear();
        pq_codebook.clear();
        residual_rotation.clear();
        mapped_clusters = {};
        mapped_pq = {};
        mapped_tq = {};
        row_correction = {};
        pq_m = 0;
        pq_k = 0;
        pq_d_sub = 0;
        pq_codes_aosoa_tile = 0;
        build_ms = 0.0;
    }
};

}  // namespace paraselect::dataset

#endif  // PARASELECT_CUDA_DATASET_DATASET_PQ_CUH
