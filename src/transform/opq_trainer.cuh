#ifndef PARASELECT_CUDA_TRANSFORM_OPQ_TRAINER_CUH
#define PARASELECT_CUDA_TRANSFORM_OPQ_TRAINER_CUH

#include "utils/pch.h"

#include <vector>

namespace paraselect::transform {

struct OPQTrainSpec {
    int n = 0;
    int dim = 0;
    int pq_m = 0;
    int pq_k = 256;
    int opq_iters = 8;
    int pq_iters = 25;
    unsigned seed = 12345;
};

struct OPQTrainResult {
    std::vector<float> rotation_rowmajor_out_in;
    std::vector<float> codebooks;
    std::vector<uint8_t> codes;
    std::vector<float> reconstructed_rotated;
    std::vector<float> objective_by_iter;
};

OPQTrainResult train_opq_transform(const float* base_rowmajor, const OPQTrainSpec& spec);

void opq_encode_plain_pq(
    const float* data,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    const float* codebooks,
    uint8_t* codes);

void opq_encode_plain_pq_gpu(
    const float* data,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    const float* codebooks,
    uint8_t* codes,
    size_t chunk_rows = 1 << 20);

void opq_decode_plain_pq(
    const uint8_t* codes,
    int n,
    int dim,
    int pq_m,
    int pq_k,
    const float* codebooks,
    float* reconstructed);

float opq_reconstruction_objective(const float* data, const float* reconstructed, int n, int dim);

}  // namespace paraselect::transform

#endif
