#include "vector_file_io.h"

#include <cstdlib>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace paraselect {
namespace dataset {
namespace vector_file_io {

namespace {

template <typename T>
void checked_write(FILE* f, const T* data, size_t count, const std::string& what) {
    if (std::fwrite(data, sizeof(T), count, f) != count) {
        throw std::runtime_error("failed to write " + what);
    }
}

} // namespace

int read_dim_from_file(const std::string& filepath) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    int dim = 0;
    if (std::fread(&dim, sizeof(int), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("Failed to read dimension from: " + filepath);
    }

    std::fclose(f);
    return dim;
}

bool load_bvecs_to_host(
    const std::string& filepath,
    float** data,
    int& n,
    int& dim,
    size_t n_max_rows) {
    dim = read_dim_from_file(filepath);
    const int bytes_per_vec = 4 + dim;

    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;

    std::fseek(f, 0, SEEK_END);
    const long file_size = std::ftell(f);
    const size_t n_file = static_cast<size_t>(file_size) / static_cast<size_t>(bytes_per_vec);
    const size_t n_load =
        (n_max_rows > 0) ? std::min(n_max_rows, n_file) : n_file;
    n = static_cast<int>(n_load);

    *data = new float[n_load * static_cast<size_t>(dim)];

    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buffer(static_cast<size_t>(bytes_per_vec));

    for (size_t i = 0; i < n_load; ++i) {
        if (std::fread(buffer.data(), 1, static_cast<size_t>(bytes_per_vec), f) !=
            static_cast<size_t>(bytes_per_vec)) {
            delete[] *data;
            *data = nullptr;
            std::fclose(f);
            return false;
        }
        for (int d = 0; d < dim; ++d) {
            (*data)[i * static_cast<size_t>(dim) + static_cast<size_t>(d)] =
                static_cast<float>(buffer[static_cast<size_t>(4 + d)]);
        }
    }

    std::fclose(f);
    return true;
}

bool load_fvecs_to_host(
    const std::string& filepath,
    float** data,
    int& n,
    int& dim,
    size_t n_max_rows) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;

    if (std::fread(&dim, sizeof(int), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    std::fseek(f, 0, SEEK_END);
    const long file_size = std::ftell(f);
    const int bytes_per_vec = 4 + dim * static_cast<int>(sizeof(float));
    const size_t n_file = static_cast<size_t>(file_size) / static_cast<size_t>(bytes_per_vec);
    const size_t n_load =
        (n_max_rows > 0) ? std::min(n_max_rows, n_file) : n_file;
    n = static_cast<int>(n_load);

    *data = new float[n_load * static_cast<size_t>(dim)];

    std::fseek(f, 0, SEEK_SET);
    for (size_t i = 0; i < n_load; ++i) {
        int d = 0;
        if (std::fread(&d, sizeof(int), 1, f) != 1 || d != dim) {
            delete[] *data;
            *data = nullptr;
            std::fclose(f);
            return false;
        }
        if (std::fread((*data) + i * static_cast<size_t>(dim), sizeof(float), static_cast<size_t>(dim), f) !=
            static_cast<size_t>(dim)) {
            delete[] *data;
            *data = nullptr;
            std::fclose(f);
            return false;
        }
    }

    std::fclose(f);
    return true;
}

bool load_ivecs_to_host(
    const std::string& filepath,
    int** data,
    int& n,
    int& dim,
    size_t n_max_rows) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;

    if (std::fread(&dim, sizeof(int), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    std::fseek(f, 0, SEEK_END);
    const long file_size = std::ftell(f);
    const int bytes_per_vec = 4 + dim * static_cast<int>(sizeof(int));
    const size_t n_file = static_cast<size_t>(file_size) / static_cast<size_t>(bytes_per_vec);
    const size_t n_load =
        (n_max_rows > 0) ? std::min(n_max_rows, n_file) : n_file;
    n = static_cast<int>(n_load);

    *data = new int[n_load * static_cast<size_t>(dim)];

    std::fseek(f, 0, SEEK_SET);
    for (size_t i = 0; i < n_load; ++i) {
        int d = 0;
        if (std::fread(&d, sizeof(int), 1, f) != 1 || d != dim) {
            delete[] *data;
            *data = nullptr;
            std::fclose(f);
            return false;
        }
        if (std::fread((*data) + i * static_cast<size_t>(dim), sizeof(int), static_cast<size_t>(dim), f) !=
            static_cast<size_t>(dim)) {
            delete[] *data;
            *data = nullptr;
            std::fclose(f);
            return false;
        }
    }

    std::fclose(f);
    return true;
}

bool load_fbin_to_host(
    const std::string& filepath,
    float** data,
    int& n,
    int& dim,
    size_t n_max_rows) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;

    uint32_t n_file = 0;
    uint32_t dim_file = 0;
    if (std::fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        std::fread(&dim_file, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    const size_t n_load = (n_max_rows > 0)
                              ? std::min(n_max_rows, static_cast<size_t>(n_file))
                              : static_cast<size_t>(n_file);
    n = static_cast<int>(n_load);
    dim = static_cast<int>(dim_file);

    *data = new float[n_load * static_cast<size_t>(dim)];

    const size_t n_to_read = n_load * static_cast<size_t>(dim);
    if (std::fread(*data, sizeof(float), n_to_read, f) != n_to_read) {
        delete[] *data;
        *data = nullptr;
        std::fclose(f);
        return false;
    }

    std::fclose(f);
    return true;
}

bool load_ibin_to_host(
    const std::string& filepath,
    int** data,
    int& n,
    int& dim,
    size_t n_max_rows) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;

    uint32_t n_file = 0;
    uint32_t dim_file = 0;
    if (std::fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        std::fread(&dim_file, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    const size_t n_load = (n_max_rows > 0)
                              ? std::min(n_max_rows, static_cast<size_t>(n_file))
                              : static_cast<size_t>(n_file);
    n = static_cast<int>(n_load);
    dim = static_cast<int>(dim_file);

    *data = new int[n_load * static_cast<size_t>(dim)];

    const size_t n_to_read = n_load * static_cast<size_t>(dim);
    if (std::fread(*data, sizeof(int), n_to_read, f) != n_to_read) {
        delete[] *data;
        *data = nullptr;
        std::fclose(f);
        return false;
    }

    std::fclose(f);
    return true;
}

void load_fbin_into_vector(
    const std::string& filepath,
    size_t n_max_rows,
    std::vector<float>& out_values,
    size_t& out_n,
    int& out_dim) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + filepath);
    }

    uint32_t n_file = 0;
    uint32_t dim_file = 0;
    if (std::fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        std::fread(&dim_file, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("failed to read fbin header: " + filepath);
    }

    out_n = (n_max_rows > 0)
                ? std::min(n_max_rows, static_cast<size_t>(n_file))
                : static_cast<size_t>(n_file);
    out_dim = static_cast<int>(dim_file);
    out_values.resize(out_n * static_cast<size_t>(out_dim));

    const size_t n_to_read = out_values.size();
    if (std::fread(out_values.data(), sizeof(float), n_to_read, f) != n_to_read) {
        std::fclose(f);
        throw std::runtime_error("failed to read fbin values: " + filepath);
    }

    std::fclose(f);
}

void load_bvecs_into_vector(
    const std::string& filepath,
    size_t n_max_rows,
    std::vector<float>& out_values,
    size_t& out_n,
    int& out_dim) {
    out_dim = read_dim_from_file(filepath);
    if (out_dim <= 0) {
        throw std::runtime_error("invalid bvecs dim in " + filepath);
    }

    const int bytes_per_vec = 4 + out_dim;
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + filepath);
    }

    std::fseek(f, 0, SEEK_END);
    const long file_size = std::ftell(f);
    const size_t n_file = static_cast<size_t>(file_size) / static_cast<size_t>(bytes_per_vec);
    out_n = (n_max_rows > 0) ? std::min(n_max_rows, n_file) : n_file;
    out_values.resize(out_n * static_cast<size_t>(out_dim));

    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> row(static_cast<size_t>(out_dim));
    for (size_t i = 0; i < out_n; ++i) {
        int32_t row_dim = 0;
        if (std::fread(&row_dim, sizeof(int32_t), 1, f) != 1 || row_dim != out_dim) {
            std::fclose(f);
            throw std::runtime_error("inconsistent bvecs row dim in " + filepath);
        }
        if (std::fread(row.data(), sizeof(uint8_t), row.size(), f) != row.size()) {
            std::fclose(f);
            throw std::runtime_error("failed to read bvecs row in " + filepath);
        }
        float* dst = out_values.data() + i * static_cast<size_t>(out_dim);
        for (int d = 0; d < out_dim; ++d) {
            dst[d] = static_cast<float>(row[static_cast<size_t>(d)]);
        }
    }

    std::fclose(f);
}

void load_u8bin_into_vector(
    const std::string& filepath,
    size_t n_max_rows,
    std::vector<float>& out_values,
    size_t& out_n,
    int& out_dim) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + filepath);
    }

    uint32_t n_file = 0;
    uint32_t dim_file = 0;
    if (std::fread(&n_file, sizeof(uint32_t), 1, f) != 1 ||
        std::fread(&dim_file, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("failed to read u8bin header: " + filepath);
    }

    out_n = (n_max_rows > 0)
                ? std::min(n_max_rows, static_cast<size_t>(n_file))
                : static_cast<size_t>(n_file);
    out_dim = static_cast<int>(dim_file);
    out_values.resize(out_n * static_cast<size_t>(out_dim));

    std::vector<uint8_t> buf(out_n * static_cast<size_t>(out_dim));
    if (std::fread(buf.data(), sizeof(uint8_t), buf.size(), f) != buf.size()) {
        std::fclose(f);
        throw std::runtime_error("failed to read u8bin values: " + filepath);
    }
    for (size_t i = 0; i < buf.size(); ++i) {
        out_values[i] = static_cast<float>(buf[i]);
    }

    std::fclose(f);
}

void save_ibin(const std::string& path, const std::vector<int>& values, int n, int dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
    const auto nn = static_cast<uint32_t>(n);
    const auto dd = static_cast<uint32_t>(dim);
    checked_write(f, &nn, 1, path);
    checked_write(f, &dd, 1, path);
    checked_write(f, values.data(), static_cast<size_t>(n) * dim, path);
    std::fclose(f);
}

void save_fbin(const std::string& path, const std::vector<float>& values, int n, int dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
    const auto nn = static_cast<uint32_t>(n);
    const auto dd = static_cast<uint32_t>(dim);
    checked_write(f, &nn, 1, path);
    checked_write(f, &dd, 1, path);
    checked_write(f, values.data(), static_cast<size_t>(n) * dim, path);
    std::fclose(f);
}

void save_ivecs(const std::string& path, const std::vector<int>& values, int n, int dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
    for (int i = 0; i < n; ++i) {
        const int32_t dd = dim;
        checked_write(f, &dd, 1, path);
        checked_write(f, values.data() + static_cast<size_t>(i) * dim, static_cast<size_t>(dim), path);
    }
    std::fclose(f);
}

void save_fvecs(const std::string& path, const std::vector<float>& values, int n, int dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
    for (int i = 0; i < n; ++i) {
        const int32_t dd = dim;
        checked_write(f, &dd, 1, path);
        checked_write(f, values.data() + static_cast<size_t>(i) * dim, static_cast<size_t>(dim), path);
    }
    std::fclose(f);
}

} // namespace vector_file_io
} // namespace dataset
} // namespace paraselect
