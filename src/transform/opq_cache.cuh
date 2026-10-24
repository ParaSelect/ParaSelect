#ifndef PARASELECT_CUDA_TRANSFORM_OPQ_CACHE_CUH
#define PARASELECT_CUDA_TRANSFORM_OPQ_CACHE_CUH

#include "transform/opq_trainer.cuh"

#include <string>
#include <vector>

namespace paraselect::transform {

std::string default_opq_cache_dir();

std::string make_opq_cache_key(
    const std::string& dataset_name,
    int n,
    int dim,
    const OPQTrainSpec& spec);

bool load_opq_rotation_cache(
    const std::string& cache_dir,
    const std::string& cache_key,
    int dim,
    std::vector<float>& rotation_out);

void save_opq_rotation_cache(
    const std::string& cache_dir,
    const std::string& cache_key,
    const OPQTrainSpec& spec,
    const std::vector<float>& rotation);

}  // namespace paraselect::transform

#endif
