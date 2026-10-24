#ifndef IVF_PQ_TRAINER_CUH
#define IVF_PQ_TRAINER_CUH

#include <vector>
#include <cstdint>

namespace ivf {

struct PlainPQTrainConfig {
    int n = 0;
    int dim = 0;
    int n_subspace = 0;
    int n_codes = 256;
    int iters = 25;
    unsigned seed = 42;
};

void train_plain_pq_codebooks_gpu(
    const float* h_data,
    const PlainPQTrainConfig& cfg,
    std::vector<float>& codebooks);

void train_plain_pq_codebooks_gpu(
    const uint8_t* h_data,
    const PlainPQTrainConfig& cfg,
    std::vector<float>& codebooks);

}  // namespace ivf

#endif
