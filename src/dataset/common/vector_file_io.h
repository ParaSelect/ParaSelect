#ifndef PARASELECT_CUDA_DATASET_COMMON_VECTOR_FILE_IO_H
#define PARASELECT_CUDA_DATASET_COMMON_VECTOR_FILE_IO_H

#include <cstddef>
#include <string>
#include <vector>

namespace paraselect {
namespace dataset {
namespace vector_file_io {

/** n_max_rows == 0 means load all rows (for heap loaders). */
inline constexpr size_t kLoadAllRows = 0;

int read_dim_from_file(const std::string& filepath);

bool load_bvecs_to_host(
    const std::string& filepath,
    float** data,
    int& n,
    int& dim,
    size_t n_max_rows = kLoadAllRows
);

bool load_fvecs_to_host(
    const std::string& filepath,
    float** data,
    int& n,
    int& dim,
    size_t n_max_rows = kLoadAllRows
);

bool load_ivecs_to_host(
    const std::string& filepath,
    int** data,
    int& n,
    int& dim,
    size_t n_max_rows = kLoadAllRows
);

bool load_fbin_to_host(
    const std::string& filepath,
    float** data,
    int& n,
    int& dim,
    size_t n_max_rows = kLoadAllRows
);

bool load_ibin_to_host(
    const std::string& filepath,
    int** data,
    int& n,
    int& dim,
    size_t n_max_rows = kLoadAllRows
);

void load_fbin_into_vector(
    const std::string& filepath,
    size_t n_max_rows,
    std::vector<float>& out_values,
    size_t& out_n,
    int& out_dim
);

void load_bvecs_into_vector(
    const std::string& filepath,
    size_t n_max_rows,
    std::vector<float>& out_values,
    size_t& out_n,
    int& out_dim
);

void load_u8bin_into_vector(
    const std::string& filepath,
    size_t n_max_rows,
    std::vector<float>& out_values,
    size_t& out_n,
    int& out_dim
);

void save_ibin(const std::string& path, const std::vector<int>& values, int n, int dim);
void save_fbin(const std::string& path, const std::vector<float>& values, int n, int dim);
void save_ivecs(const std::string& path, const std::vector<int>& values, int n, int dim);
void save_fvecs(const std::string& path, const std::vector<float>& values, int n, int dim);

} // namespace vector_file_io
} // namespace dataset
} // namespace paraselect

#endif
