#ifndef PARASELECT_CUDA_DATASET_COMMON_BINARY_FORMAT_H
#define PARASELECT_CUDA_DATASET_COMMON_BINARY_FORMAT_H

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace paraselect {
namespace dataset {

/**
 *
 */
enum class DatasetFormat {
    UNKNOWN,
    BVECS,      // uint8 + 4-byte prefix per vector
    FVECS,      // float32 + 4-byte prefix per vector
    IVECS,      // int32 + 4-byte prefix per vector
    FBIN,       // float32 with header [n, dim]
    IBIN,       // int32 with header [n, dim]
};

/**
 *
 */
struct DatasetInfo {
    size_t n = 0;           //
    int dim = 0;            //
    size_t data_size = 0;   //

    DatasetInfo() = default;
    DatasetInfo(size_t n_, int dim_, size_t data_size_)
        : n(n_), dim(dim_), data_size(data_size_) {}
};

/**
 *
 *
 */
class BinaryFormat {
public:
    /**
     *
     */
    static DatasetFormat detect_format(const std::string& filepath);

    /**
     * ()
     */
    static std::string extract_name(const std::string& filepath);

    /**
     *  ( bvecs/fvecs/ivecs )
     * @param filepath
     * @return
     * @throws std::runtime_error
     */
    static int read_dim_from_file(const std::string& filepath);

    /**
     *  fbin/ibin  header [n, dim]
     */
    static DatasetInfo read_header_fbin(const std::string& filepath);
    static DatasetInfo read_header_ibin(const std::string& filepath);

    /**
     *  [uint32 n][uint32 dim] + uint8 payload .
     */
    static bool is_header_u8bin(const std::string& filepath);

    /**
     *  ( bvecs/fvecs/ivecs )
     * @param filepath
     * @param dim
     * @param prefix_bytes  (4)
     * @return
     */
    static size_t count_vectors(const std::string& filepath, int dim, int prefix_bytes = 4);

    /**
     *  ()
     */
    static DatasetInfo get_file_info(const std::string& filepath);

    /**
     *  CUDA Pinned Memory
     * @param size_bytes
     * @return
     * @throws std::runtime_error
     */
    static void* allocate_pinned(size_t size_bytes);

    /**
     *  Pinned Memory
     */
    static void free_pinned(void* ptr);

    /**
     *  bvecs  heap
     * @param filepath
     * @param data :, delete[]
     * @param n :
     * @param dim :
     * @return
     */
    static bool load_bvecs_to_host(
        const std::string& filepath,
        float** data,
        int& n,
        int& dim,
        size_t n_max = 0
    );

    /**
     *  fvecs  heap
     */
    static bool load_fvecs_to_host(
        const std::string& filepath,
        float** data,
        int& n,
        int& dim
    );

    /**
     *  ivecs  heap
     */
    static bool load_ivecs_to_host(
        const std::string& filepath,
        int** data,
        int& n,
        int& dim
    );

    /**
     *  fbin  heap
     */
    static bool load_fbin_to_host(
        const std::string& filepath,
        float** data,
        int& n,
        int& dim,
        size_t n_max = 0
    );

    /**
     *  ibin  heap
     */
    static bool load_ibin_to_host(
        const std::string& filepath,
        int** data,
        int& n,
        int& dim,
        size_t n_max = 0
    );

    /**
     * Load the standard ANN ground-truth container:
     * [uint32 n][uint32 k][int32 ids (n*k)][float32 distances (n*k)].
     */
    static bool load_groundtruth_bin_to_host(
        const std::string& filepath,
        int** ids,
        float** distances,
        int& n,
        int& k,
        size_t n_max = 0
    );

    /**
     *  bvecs  Pinned Memory
     * @param filepath
     * @param h_data :pinned memory , free_pinned
     * @param n_max  (0 = )
     * @return
     */
    static DatasetInfo load_bvecs_pinned(
        const std::string& filepath,
        float** h_data,
        size_t n_max = 0
    );

    /**
     *  bvecs  uint8 Pinned Memory(, float )
     */
    static DatasetInfo load_bvecs_pinned_u8(
        const std::string& filepath,
        uint8_t** h_data,
        size_t n_max = 0
    );

    /**
     *  fvecs  Pinned Memory
     */
    static DatasetInfo load_fvecs_pinned(
        const std::string& filepath,
        float** h_data,
        size_t n_max = 0
    );

    /**
     *  ivecs  Pinned Memory
     */
    static DatasetInfo load_ivecs_pinned(
        const std::string& filepath,
        int** h_data,
        size_t n_max = 0
    );

    /**
     *  fbin  Pinned Memory
     */
    static DatasetInfo load_fbin_pinned(
        const std::string& filepath,
        float** h_data,
        size_t n_max = 0
    );

    /**
     *  ibin  Pinned Memory
     */
    static DatasetInfo load_ibin_pinned(
        const std::string& filepath,
        int** h_data,
        size_t n_max = 0
    );

    /**
     *  bvecs
     * @param filepath
     * @param h_data
     * @param n_start
     * @param n_read
     * @return
     */
    static size_t load_bvecs_chunk(
        const std::string& filepath,
        float* h_data,
        size_t n_start,
        size_t n_read
    );

    /**
     *  bvecs  uint8 ()
     */
    static size_t load_bvecs_chunk_u8(
        const std::string& filepath,
        uint8_t* h_data,
        size_t n_start,
        size_t n_read
    );



    /**
     *  [uint32 n][uint32 dim] + uint8 payload , float.
     */
    static size_t load_header_u8bin_chunk(
        const std::string& filepath,
        float* h_data,
        size_t n_start,
        size_t n_read
    );

    static size_t load_header_u8bin_chunk_u8(
        const std::string& filepath,
        uint8_t* h_data,
        size_t n_start,
        size_t n_read
    );

    /**
     *  id , bvecs,header-u8bin  fbin.
     * out_vectors  ids ,out_ids  id.
     */
    static bool load_selected_vectors_as_float(
        const std::string& filepath,
        bool bvecs_format,
        const long long* ids,
        size_t n_ids,
        int n_base_limit,
        float** out_vectors,
        long long** out_ids,
        size_t* out_count,
        int* dim_out
    );

    /**
     *  fbin
     */
    static size_t load_fbin_chunk(
        const std::string& filepath,
        float* h_data,
        size_t n_start,
        size_t n_read
    );

private:
    // ,
    BinaryFormat() = delete;
};

} // namespace dataset
} // namespace paraselect

#endif // PARASELECT_CUDA_DATASET_COMMON_BINARY_FORMAT_H
