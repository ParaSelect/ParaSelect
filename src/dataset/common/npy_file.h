#ifndef PARASELECT_CUDA_DATASET_COMMON_NPY_FILE_H
#define PARASELECT_CUDA_DATASET_COMMON_NPY_FILE_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <malloc.h>
#include <iostream>

namespace paraselect {
namespace dataset {

/**
 * NumPy .npy
 *
 *  NumPy .npy ( 1.0  2.0)
 * (C-order)(little-endian)
 */
class NpyFile {
public:
    enum class DataType {
        FLOAT32,
        FLOAT64,
        INT32,
        INT64,
        UNKNOWN
    };

    struct ArrayInfo {
        DataType dtype;
        std::vector<size_t> shape;
        size_t total_elements;
        size_t data_offset;  //
        size_t data_size;    //
    };

    /**
     *  NumPy ()
     */
    static bool parse_header(const std::string& filepath, ArrayInfo& info);

    /**
     *  float32
     * @param filepath
     * @param data :, free
     * @param n :
     * @param dim :(2D,dim=shape[1],0)
     * @return
     */
    static bool load_float32(const std::string& filepath, float** data, int& n, int& dim);

    /**
     *  int32
     * @param filepath
     * @param data :, free
     * @param n :
     * @param dim :
     * @return
     */
    static bool load_int32(const std::string& filepath, int** data, int& n, int& dim);

    /**
     *  int64
     * @param filepath
     * @param data :, free
     * @param n :
     * @param dim :
     * @return
     */
    static bool load_int64(const std::string& filepath, long long** data, int& n, int& dim);

    /**
     *  float32  .npy
     */
    static bool save_float32(const std::string& filepath, const float* data,
                             const std::vector<size_t>& shape);

    /**
     *  int32  .npy
     */
    static bool save_int32(const std::string& filepath, const int* data,
                           const std::vector<size_t>& shape);

    /**
     *  int64  .npy
     */
    static bool save_int64(const std::string& filepath, const long long* data,
                           const std::vector<size_t>& shape);

private:
    static DataType parse_dtype(const std::string& dtype_str);
    static std::string dtype_to_str(DataType dtype);
};

} // namespace dataset
} // namespace paraselect

#endif // PARASELECT_CUDA_DATASET_COMMON_NPY_FILE_H
