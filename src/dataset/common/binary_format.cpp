#include "binary_format.h"
#include "vector_file_io.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <vector>
#include <sys/stat.h>

namespace paraselect {
namespace dataset {

DatasetFormat BinaryFormat::detect_format(const std::string& filepath) {
    if (filepath.find(".bvecs") != std::string::npos) return DatasetFormat::BVECS;
    if (filepath.find(".fvecs") != std::string::npos) return DatasetFormat::FVECS;
    if (filepath.find(".ivecs") != std::string::npos) return DatasetFormat::IVECS;
    if (filepath.find(".fbin") != std::string::npos) return DatasetFormat::FBIN;
    if (filepath.find(".ibin") != std::string::npos) return DatasetFormat::IBIN;
    return DatasetFormat::UNKNOWN;
}

std::string BinaryFormat::extract_name(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of("/\\");
    size_t start = (last_slash == std::string::npos) ? 0 : last_slash + 1;
    size_t last_dot = filepath.find_last_of('.');
    if (last_dot == std::string::npos || last_dot < start) {
        return filepath.substr(start);
    }
    return filepath.substr(start, last_dot - start);
}

int BinaryFormat::read_dim_from_file(const std::string& filepath) {
    return vector_file_io::read_dim_from_file(filepath);
}

DatasetInfo BinaryFormat::read_header_fbin(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t n, dim;
    if (fread(&n, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    fclose(f);
    return DatasetInfo(n, static_cast<int>(dim), static_cast<size_t>(n) * dim * sizeof(float));
}

DatasetInfo BinaryFormat::read_header_ibin(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t n, dim;
    if (fread(&n, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    fclose(f);
    return DatasetInfo(n, static_cast<int>(dim), static_cast<size_t>(n) * dim * sizeof(int));
}


bool BinaryFormat::is_header_u8bin(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0;
    uint32_t dim = 0;
    const bool header_ok = fread(&n, sizeof(uint32_t), 1, f) == 1 &&
                           fread(&dim, sizeof(uint32_t), 1, f) == 1;
    fclose(f);
    if (!header_ok || n == 0 || dim == 0) return false;

    struct stat st;
    if (stat(filepath.c_str(), &st) != 0 || st.st_size < 8) return false;
    const uint64_t expected = 8ULL + static_cast<uint64_t>(n) * static_cast<uint64_t>(dim);
    return static_cast<uint64_t>(st.st_size) == expected;
}

size_t BinaryFormat::count_vectors(const std::string& filepath, int dim, int prefix_bytes) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    // bvecs=uint8(1 byte), fvecs/ivecs=float/int(4 bytes)
    int data_bytes_per_elem = (prefix_bytes == 4) ? 1 : 4;
    int bytes_per_vec = prefix_bytes + dim * data_bytes_per_elem;
    return static_cast<size_t>(file_size / bytes_per_vec);
}

DatasetInfo BinaryFormat::get_file_info(const std::string& filepath) {
    DatasetFormat fmt = detect_format(filepath);

    switch (fmt) {
        case DatasetFormat::BVECS: {
            int dim = read_dim_from_file(filepath);
            int bytes_per_vec = 4 + dim;
            FILE* f = fopen(filepath.c_str(), "rb");
            if (!f) {
                throw std::runtime_error("Cannot open file: " + filepath);
            }
            fseek(f, 0, SEEK_END);
            long file_size = ftell(f);
            fclose(f);
            size_t n = file_size / bytes_per_vec;
            return DatasetInfo(n, dim, n * dim * sizeof(float));
        }
        case DatasetFormat::FVECS: {
            int dim = read_dim_from_file(filepath);
            int bytes_per_vec = 4 + dim * sizeof(float);
            FILE* f = fopen(filepath.c_str(), "rb");
            if (!f) {
                throw std::runtime_error("Cannot open file: " + filepath);
            }
            fseek(f, 0, SEEK_END);
            long file_size = ftell(f);
            fclose(f);
            size_t n = file_size / bytes_per_vec;
            return DatasetInfo(n, dim, n * dim * sizeof(float));
        }
        case DatasetFormat::IVECS: {
            int dim = read_dim_from_file(filepath);
            int bytes_per_vec = 4 + dim * sizeof(int);
            FILE* f = fopen(filepath.c_str(), "rb");
            if (!f) {
                throw std::runtime_error("Cannot open file: " + filepath);
            }
            fseek(f, 0, SEEK_END);
            long file_size = ftell(f);
            fclose(f);
            size_t n = file_size / bytes_per_vec;
            return DatasetInfo(n, dim, n * dim * sizeof(int));
        }
        case DatasetFormat::FBIN:
            return read_header_fbin(filepath);
        case DatasetFormat::IBIN:
            return read_header_ibin(filepath);
        default:
            throw std::runtime_error("Unknown file format: " + filepath);
    }
}

void* BinaryFormat::allocate_pinned(size_t size_bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMallocHost failed: " + std::string(cudaGetErrorString(err)) +
            " (size=" + std::to_string(size_bytes / (1024*1024*1024)) + " GB, bytes=" + std::to_string(size_bytes) + ")"
        );
    }
    return ptr;
}

void BinaryFormat::free_pinned(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

bool BinaryFormat::load_bvecs_to_host(const std::string& filepath, float** data, int& n, int& dim, size_t n_max) {
    return vector_file_io::load_bvecs_to_host(filepath, data, n, dim, n_max);
}

bool BinaryFormat::load_fvecs_to_host(const std::string& filepath, float** data, int& n, int& dim) {
    return vector_file_io::load_fvecs_to_host(filepath, data, n, dim, vector_file_io::kLoadAllRows);
}

bool BinaryFormat::load_ivecs_to_host(const std::string& filepath, int** data, int& n, int& dim) {
    return vector_file_io::load_ivecs_to_host(filepath, data, n, dim, vector_file_io::kLoadAllRows);
}

bool BinaryFormat::load_fbin_to_host(const std::string& filepath, float** data, int& n, int& dim, size_t n_max) {
    return vector_file_io::load_fbin_to_host(filepath, data, n, dim, n_max);
}

bool BinaryFormat::load_ibin_to_host(const std::string& filepath, int** data, int& n, int& dim, size_t n_max) {
    return vector_file_io::load_ibin_to_host(filepath, data, n, dim, n_max);
}


bool BinaryFormat::load_groundtruth_bin_to_host(
    const std::string& filepath, int** ids, float** distances,
    int& n, int& k, size_t n_max) {
    if (!ids || !distances) return false;
    *ids = nullptr;
    *distances = nullptr;
    FILE* file = fopen(filepath.c_str(), "rb");
    if (!file) return false;

    uint32_t file_n = 0;
    uint32_t file_k = 0;
    if (fread(&file_n, sizeof(file_n), 1, file) != 1 ||
        fread(&file_k, sizeof(file_k), 1, file) != 1 ||
        file_n == 0 || file_k == 0 ||
        file_n > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        file_k > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        fclose(file);
        return false;
    }

    const size_t total_rows = file_n;
    const size_t cols = file_k;
    if (cols > std::numeric_limits<size_t>::max() / total_rows) {
        fclose(file);
        return false;
    }
    const size_t total_values = total_rows * cols;
    const size_t keep_rows = n_max ? std::min(n_max, total_rows) : total_rows;
    const size_t keep_values = keep_rows * cols;
    std::unique_ptr<int[]> loaded_ids(new (std::nothrow) int[keep_values]);
    std::unique_ptr<float[]> loaded_distances(new (std::nothrow) float[keep_values]);
    if (!loaded_ids || !loaded_distances ||
        fread(loaded_ids.get(), sizeof(int), keep_values, file) != keep_values) {
        fclose(file);
        return false;
    }

    const uint64_t distance_offset = 2ULL * sizeof(uint32_t) +
        static_cast<uint64_t>(total_values) * sizeof(int);
    if (fseek(file, static_cast<long>(distance_offset), SEEK_SET) != 0 ||
        fread(loaded_distances.get(), sizeof(float), keep_values, file) != keep_values) {
        fclose(file);
        return false;
    }
    fclose(file);
    n = static_cast<int>(keep_rows);
    k = static_cast<int>(cols);
    *ids = loaded_ids.release();
    *distances = loaded_distances.release();
    return true;
}

DatasetInfo BinaryFormat::load_bvecs_pinned(const std::string& filepath, float** h_data, size_t n_max) {
    int dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim;

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    size_t n_total = file_size / bytes_per_vec;
    size_t n = n_max > 0 ? std::min(n_max, n_total) : n_total;

    size_t data_bytes = n * dim * sizeof(float);
    *h_data = static_cast<float*>(allocate_pinned(data_bytes));

    //
    const size_t chunk_size = 100000;
    std::vector<uint8_t> raw_buffer(chunk_size * bytes_per_vec);

    f = fopen(filepath.c_str(), "rb");
    if (!f) {
        free_pinned(*h_data);
        *h_data = nullptr;
        throw std::runtime_error("Cannot reopen file: " + filepath);
    }

    for (size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        size_t chunk_n = std::min(chunk_size, n - chunk_start);
        size_t bytes_to_read = chunk_n * bytes_per_vec;

        if (fread(raw_buffer.data(), 1, bytes_to_read, f) != bytes_to_read) {
            fclose(f);
            free_pinned(*h_data);
            *h_data = nullptr;
            throw std::runtime_error("Failed to read chunk at offset " + std::to_string(chunk_start));
        }

        float* dst = *h_data + chunk_start * dim;
        for (size_t i = 0; i < chunk_n; ++i) {
            const uint8_t* src = raw_buffer.data() + i * bytes_per_vec + 4;
            for (int d = 0; d < dim; ++d) {
                dst[i * dim + d] = static_cast<float>(src[d]);
            }
        }
    }

    fclose(f);
    return DatasetInfo(n, dim, data_bytes);
}

DatasetInfo BinaryFormat::load_bvecs_pinned_u8(const std::string& filepath, uint8_t** h_data, size_t n_max) {
    int dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim;

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    size_t n_total = file_size / bytes_per_vec;
    size_t n = n_max > 0 ? std::min(n_max, n_total) : n_total;

    size_t data_bytes = n * (size_t)dim * sizeof(uint8_t);
    *h_data = static_cast<uint8_t*>(allocate_pinned(data_bytes));

    const size_t chunk_size = 100000;
    std::vector<uint8_t> raw_buffer(chunk_size * (size_t)bytes_per_vec);

    f = fopen(filepath.c_str(), "rb");
    if (!f) {
        free_pinned(*h_data);
        *h_data = nullptr;
        throw std::runtime_error("Cannot reopen file: " + filepath);
    }

    for (size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        size_t chunk_n = std::min(chunk_size, n - chunk_start);
        size_t bytes_to_read = chunk_n * (size_t)bytes_per_vec;
        if (fread(raw_buffer.data(), 1, bytes_to_read, f) != bytes_to_read) {
            fclose(f);
            free_pinned(*h_data);
            *h_data = nullptr;
            throw std::runtime_error("Failed to read chunk at offset " + std::to_string(chunk_start));
        }
        uint8_t* dst = *h_data + chunk_start * (size_t)dim;
        for (size_t i = 0; i < chunk_n; ++i) {
            const uint8_t* src = raw_buffer.data() + i * (size_t)bytes_per_vec + 4;
            std::memcpy(dst + i * (size_t)dim, src, (size_t)dim * sizeof(uint8_t));
        }
    }

    fclose(f);
    return DatasetInfo(n, dim, data_bytes);
}

DatasetInfo BinaryFormat::load_fvecs_pinned(const std::string& filepath, float** h_data, size_t n_max) {
    int dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim * sizeof(float);

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    size_t n_total = file_size / bytes_per_vec;
    size_t n = n_max > 0 ? std::min(n_max, n_total) : n_total;
    fclose(f);

    size_t data_bytes = n * dim * sizeof(float);
    *h_data = static_cast<float*>(allocate_pinned(data_bytes));

    f = fopen(filepath.c_str(), "rb");
    if (!f) {
        free_pinned(*h_data);
        *h_data = nullptr;
        throw std::runtime_error("Cannot reopen file: " + filepath);
    }

    for (size_t i = 0; i < n; ++i) {
        fseek(f, 4, SEEK_CUR);
        if (fread((*h_data) + i * dim, sizeof(float), dim, f) != static_cast<size_t>(dim)) {
            fclose(f);
            free_pinned(*h_data);
            *h_data = nullptr;
            throw std::runtime_error("Failed to read vector " + std::to_string(i));
        }
    }

    fclose(f);
    return DatasetInfo(n, dim, data_bytes);
}

DatasetInfo BinaryFormat::load_ivecs_pinned(const std::string& filepath, int** h_data, size_t n_max) {
    int dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim * sizeof(int);

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    size_t n_total = file_size / bytes_per_vec;
    size_t n = n_max > 0 ? std::min(n_max, n_total) : n_total;
    fclose(f);

    size_t data_bytes = n * dim * sizeof(int);
    *h_data = static_cast<int*>(allocate_pinned(data_bytes));

    f = fopen(filepath.c_str(), "rb");
    if (!f) {
        free_pinned(*h_data);
        *h_data = nullptr;
        throw std::runtime_error("Cannot reopen file: " + filepath);
    }

    for (size_t i = 0; i < n; ++i) {
        fseek(f, 4, SEEK_CUR);
        if (fread((*h_data) + i * dim, sizeof(int), dim, f) != static_cast<size_t>(dim)) {
            fclose(f);
            free_pinned(*h_data);
            *h_data = nullptr;
            throw std::runtime_error("Failed to read vector " + std::to_string(i));
        }
    }

    fclose(f);
    return DatasetInfo(n, dim, data_bytes);
}

DatasetInfo BinaryFormat::load_fbin_pinned(const std::string& filepath, float** h_data, size_t n_max) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t n_file, dim;
    if (fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    size_t n = n_max > 0 ? std::min(n_max, static_cast<size_t>(n_file)) : n_file;
    size_t data_bytes = n * dim * sizeof(float);
    *h_data = static_cast<float*>(allocate_pinned(data_bytes));

    //
    size_t n_to_read = n * dim;
    if (fread(*h_data, sizeof(float), n_to_read, f) != n_to_read) {
        fclose(f);
        free_pinned(*h_data);
        *h_data = nullptr;
        throw std::runtime_error("Failed to read data from: " + filepath);
    }

    fclose(f);
    return DatasetInfo(n, static_cast<int>(dim), data_bytes);
}

DatasetInfo BinaryFormat::load_ibin_pinned(const std::string& filepath, int** h_data, size_t n_max) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t n_file, dim;
    if (fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    size_t n = n_max > 0 ? std::min(n_max, static_cast<size_t>(n_file)) : n_file;
    size_t data_bytes = n * dim * sizeof(int);
    *h_data = static_cast<int*>(allocate_pinned(data_bytes));

    //
    size_t n_to_read = n * dim;
    if (fread(*h_data, sizeof(int), n_to_read, f) != n_to_read) {
        fclose(f);
        free_pinned(*h_data);
        *h_data = nullptr;
        throw std::runtime_error("Failed to read data from: " + filepath);
    }

    fclose(f);
    return DatasetInfo(n, static_cast<int>(dim), data_bytes);
}

size_t BinaryFormat::load_bvecs_chunk(const std::string& filepath, float* h_data, size_t n_start, size_t n_read) {
    int dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim;

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    long offset = static_cast<long>(n_start * bytes_per_vec);
    if (fseek(f, offset, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("Failed to seek to position " + std::to_string(offset));
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    size_t n_total = file_size / bytes_per_vec;
    size_t n_available = (n_start < n_total) ? (n_total - n_start) : 0;
    size_t n = std::min(n_read, n_available);

    fseek(f, offset, SEEK_SET);

    const size_t chunk_size = 100000;
    std::vector<uint8_t> raw_buffer(chunk_size * bytes_per_vec);

    for (size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        size_t chunk_n = std::min(chunk_size, n - chunk_start);
        size_t bytes_to_read = chunk_n * bytes_per_vec;

        if (fread(raw_buffer.data(), 1, bytes_to_read, f) != bytes_to_read) {
            fclose(f);
            throw std::runtime_error("Failed to read chunk at offset " + std::to_string(chunk_start));
        }

        float* dst = h_data + chunk_start * dim;
        for (size_t i = 0; i < chunk_n; ++i) {
            const uint8_t* src = raw_buffer.data() + i * bytes_per_vec + 4;
            for (int d = 0; d < dim; ++d) {
                dst[i * dim + d] = static_cast<float>(src[d]);
            }
        }
    }

    fclose(f);
    return n;
}

size_t BinaryFormat::load_bvecs_chunk_u8(const std::string& filepath, uint8_t* h_data, size_t n_start, size_t n_read) {
    int dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim;

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    long offset = static_cast<long>(n_start * (size_t)bytes_per_vec);
    if (fseek(f, offset, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("Failed to seek to position " + std::to_string(offset));
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    size_t n_total = file_size / (size_t)bytes_per_vec;
    size_t n_available = (n_start < n_total) ? (n_total - n_start) : 0;
    size_t n = std::min(n_read, n_available);
    fseek(f, offset, SEEK_SET);

    const size_t chunk_size = 100000;
    std::vector<uint8_t> raw_buffer(chunk_size * (size_t)bytes_per_vec);
    for (size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        size_t chunk_n = std::min(chunk_size, n - chunk_start);
        size_t bytes_to_read = chunk_n * (size_t)bytes_per_vec;
        if (fread(raw_buffer.data(), 1, bytes_to_read, f) != bytes_to_read) {
            fclose(f);
            throw std::runtime_error("Failed to read chunk at offset " + std::to_string(chunk_start));
        }
        uint8_t* dst = h_data + chunk_start * (size_t)dim;
        for (size_t i = 0; i < chunk_n; ++i) {
            const uint8_t* src = raw_buffer.data() + i * (size_t)bytes_per_vec + 4;
            std::memcpy(dst + i * (size_t)dim, src, (size_t)dim * sizeof(uint8_t));
        }
    }
    fclose(f);
    return n;
}


size_t BinaryFormat::load_header_u8bin_chunk(
    const std::string& filepath,
    float* h_data,
    size_t n_start,
    size_t n_read) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t n_file = 0;
    uint32_t dim = 0;
    if (fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    size_t n_available = (n_start < n_file) ? (n_file - n_start) : 0;
    size_t n = std::min(n_read, n_available);
    long data_offset = 8 + static_cast<long>(n_start * (size_t)dim);
    if (fseek(f, data_offset, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("Failed to seek to position " + std::to_string(data_offset));
    }

    const size_t chunk_size = 100000;
    std::vector<uint8_t> raw(chunk_size * (size_t)dim);
    for (size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        size_t chunk_n = std::min(chunk_size, n - chunk_start);
        size_t bytes = chunk_n * (size_t)dim;
        if (fread(raw.data(), 1, bytes, f) != bytes) {
            fclose(f);
            throw std::runtime_error("Failed to read header-u8bin chunk");
        }
        float* dst = h_data + chunk_start * (size_t)dim;
        for (size_t i = 0; i < bytes; ++i) dst[i] = static_cast<float>(raw[i]);
    }

    fclose(f);
    return n;
}

size_t BinaryFormat::load_header_u8bin_chunk_u8(
    const std::string& filepath, uint8_t* data, size_t start, size_t count) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open file: " + filepath);
    uint32_t n = 0, dim = 0;
    if (fread(&n, sizeof(n), 1, f) != 1 || fread(&dim, sizeof(dim), 1, f) != 1) { fclose(f); throw std::runtime_error("Failed to read u8bin header"); }
    const size_t rows = std::min(count, start < n ? static_cast<size_t>(n) - start : 0);
    const long offset = 8 + static_cast<long>(start * static_cast<size_t>(dim));
    if (fseek(f, offset, SEEK_SET) != 0 || fread(data, dim, rows, f) != rows) { fclose(f); throw std::runtime_error("Failed to read u8bin range"); }
    fclose(f);
    return rows;
}

bool BinaryFormat::load_selected_vectors_as_float(
    const std::string& filepath,
    bool bvecs_format,
    const long long* ids,
    size_t n_ids,
    int n_base_limit,
    float** out_vectors,
    long long** out_ids,
    size_t* out_count,
    int* dim_out) {
    if (!ids || !out_vectors || !out_ids || !out_count || !dim_out) return false;
    *out_vectors = nullptr;
    *out_ids = nullptr;
    *out_count = 0;
    *dim_out = 0;
    if (n_ids == 0) return false;

    const bool header_u8bin = bvecs_format && is_header_u8bin(filepath);
    DatasetInfo info = header_u8bin ? read_header_ibin(filepath) : get_file_info(filepath);
    if (info.dim <= 0) return false;
    const size_t n_limit = n_base_limit > 0 ? std::min((size_t)n_base_limit, info.n) : info.n;

    std::vector<long long> sorted_ids;
    sorted_ids.reserve(n_ids);
    for (size_t i = 0; i < n_ids; ++i) {
        const long long id = ids[i];
        if (id >= 0 && (size_t)id < n_limit) sorted_ids.push_back(id);
    }
    if (sorted_ids.empty()) return false;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    float* vectors = new float[sorted_ids.size() * (size_t)info.dim];
    long long* kept_ids = new long long[sorted_ids.size()];
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        delete[] vectors;
        delete[] kept_ids;
        return false;
    }

    std::vector<uint8_t> raw_u8((size_t)info.dim);
    size_t out_idx = 0;
    bool ok = true;
    for (long long id : sorted_ids) {
        const long long offset = header_u8bin
            ? 8LL + id * (long long)info.dim
            : (bvecs_format
                ? id * (long long)(4 + info.dim) + 4LL
                : 8LL + id * (long long)info.dim * (long long)sizeof(float));
        if (fseek(f, offset, SEEK_SET) != 0) {
            ok = false;
            break;
        }
        float* dst = vectors + out_idx * (size_t)info.dim;
        if (bvecs_format) {
            if (fread(raw_u8.data(), sizeof(uint8_t), (size_t)info.dim, f) != (size_t)info.dim) {
                ok = false;
                break;
            }
            for (int d = 0; d < info.dim; ++d) dst[d] = static_cast<float>(raw_u8[(size_t)d]);
        } else {
            if (fread(dst, sizeof(float), (size_t)info.dim, f) != (size_t)info.dim) {
                ok = false;
                break;
            }
        }
        kept_ids[out_idx] = id;
        ++out_idx;
    }
    fclose(f);

    if (!ok || out_idx == 0) {
        delete[] vectors;
        delete[] kept_ids;
        return false;
    }
    *out_vectors = vectors;
    *out_ids = kept_ids;
    *out_count = out_idx;
    *dim_out = info.dim;
    return true;
}

size_t BinaryFormat::load_fbin_chunk(const std::string& filepath, float* h_data, size_t n_start, size_t n_read) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    uint32_t n_file, dim;
    if (fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    size_t n_available = (n_start < n_file) ? (n_file - n_start) : 0;
    size_t n = std::min(n_read, n_available);

    //  (header 8 + )
    long data_offset = 8 + static_cast<long>(n_start * dim * sizeof(float));
    if (fseek(f, data_offset, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("Failed to seek to position " + std::to_string(data_offset));
    }

    //
    size_t n_to_read = n * dim;
    if (fread(h_data, sizeof(float), n_to_read, f) != n_to_read) {
        fclose(f);
        throw std::runtime_error("Failed to read chunk");
    }

    fclose(f);
    return n;
}

} // namespace dataset
} // namespace paraselect
