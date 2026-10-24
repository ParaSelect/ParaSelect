#include "npy_file.h"
#include <fstream>
#include <sstream>
#include <regex>

namespace paraselect {
namespace dataset {

NpyFile::DataType NpyFile::parse_dtype(const std::string& dtype_str) {
    //  numpy dtype , "<f4", ">i8"
    // < ,>
    // f4 = float32, f8 = float64, i4 = int32, i8 = int64

    //
    std::string dtype = dtype_str;
    if (!dtype.empty() && (dtype[0] == '<' || dtype[0] == '>' || dtype[0] == '|')) {
        dtype = dtype.substr(1);
    }

    if (dtype == "f4" || dtype == "float32") {
        return DataType::FLOAT32;
    } else if (dtype == "f8" || dtype == "float64") {
        return DataType::FLOAT64;
    } else if (dtype == "i4" || dtype == "int32") {
        return DataType::INT32;
    } else if (dtype == "i8" || dtype == "int64") {
        return DataType::INT64;
    }
    return DataType::UNKNOWN;
}

std::string NpyFile::dtype_to_str(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "<f4";
        case DataType::FLOAT64: return "<f8";
        case DataType::INT32: return "<i4";
        case DataType::INT64: return "<i8";
        default: return "<f4";
    }
}

bool NpyFile::parse_header(const std::string& filepath, ArrayInfo& info) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        std::cerr << "[NpyFile] Cannot open file: " << filepath << std::endl;
        return false;
    }

    //  magic number
    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fclose(f);
        return false;
    }

    //  magic number: 0x93 'N' 'U' 'M' 'P' 'Y'
    if (magic[0] != 0x93 || memcmp(magic + 1, "NUMPY", 5) != 0) {
        std::cerr << "[NpyFile] Invalid magic number" << std::endl;
        fclose(f);
        return false;
    }

    //
    unsigned char version[2];
    if (fread(version, 1, 2, f) != 2) {
        fclose(f);
        return false;
    }

    //  header
    uint16_t header_len;
    if (version[0] == 1) {
        //  1.0: 2
        unsigned char len_bytes[2];
        if (fread(len_bytes, 1, 2, f) != 2) {
            fclose(f);
            return false;
        }
        header_len = len_bytes[0] | (len_bytes[1] << 8);
    } else {
        //  2.0+: 4
        unsigned char len_bytes[4];
        if (fread(len_bytes, 1, 4, f) != 4) {
            fclose(f);
            return false;
        }
        header_len = len_bytes[0] | (len_bytes[1] << 8) |
                     (len_bytes[2] << 16) | (len_bytes[3] << 24);
    }

    //  header
    std::vector<char> header_buf(header_len + 1);
    if (fread(header_buf.data(), 1, header_len, f) != header_len) {
        fclose(f);
        return false;
    }
    header_buf[header_len] = '\0';
    std::string header(header_buf.data());

    //  shape
    info.shape.clear();
    std::regex shape_regex(R"(shape\s*:\s*\(([^)]*)\))");
    std::smatch shape_match;
    if (std::regex_search(header, shape_match, shape_regex)) {
        std::string shape_str = shape_match[1];
        std::stringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            //
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            if (!token.empty()) {
                info.shape.push_back(std::stoull(token));
            }
        }
    }

    //  dtype
    std::regex dtype_regex(R"(descr\s*:\s*['"]([^'"]+)['"])");
    std::smatch dtype_match;
    info.dtype = DataType::UNKNOWN;
    if (std::regex_search(header, dtype_match, dtype_regex)) {
        info.dtype = parse_dtype(dtype_match[1]);
    }

    //  C-order
    std::regex order_regex(R"(order\s*:\s*['"]([^'"]+)['"])");
    std::smatch order_match;
    std::string order = "C";
    if (std::regex_search(header, order_match, order_regex)) {
        order = order_match[1];
    }
    if (order != "C") {
        std::cerr << "[NpyFile] Only C-order arrays are supported" << std::endl;
        fclose(f);
        return false;
    }

    //
    info.total_elements = 1;
    for (size_t s : info.shape) {
        info.total_elements *= s;
    }

    //
    info.data_offset = ftell(f);
    switch (info.dtype) {
        case DataType::FLOAT32: info.data_size = info.total_elements * 4; break;
        case DataType::FLOAT64: info.data_size = info.total_elements * 8; break;
        case DataType::INT32:   info.data_size = info.total_elements * 4; break;
        case DataType::INT64:   info.data_size = info.total_elements * 8; break;
        default: info.data_size = 0;
    }

    fclose(f);
    return true;
}

bool NpyFile::load_float32(const std::string& filepath, float** data, int& n, int& dim) {
    ArrayInfo info;
    if (!parse_header(filepath, info)) {
        return false;
    }

    if (info.dtype != DataType::FLOAT32) {
        std::cerr << "[NpyFile] Expected float32, got different dtype" << std::endl;
        return false;
    }

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return false;

    //
    fseek(f, info.data_offset, SEEK_SET);

    //
    *data = (float*)malloc(info.data_size);
    if (!*data) {
        fclose(f);
        return false;
    }

    //
    if (fread(*data, 1, info.data_size, f) != info.data_size) {
        free(*data);
        *data = nullptr;
        fclose(f);
        return false;
    }

    fclose(f);

    //
    if (info.shape.size() >= 1) {
        n = static_cast<int>(info.shape[0]);
    } else {
        n = 1;
    }

    if (info.shape.size() >= 2) {
        dim = static_cast<int>(info.shape[1]);
    } else {
        dim = 0;
    }

    return true;
}

bool NpyFile::load_int32(const std::string& filepath, int** data, int& n, int& dim) {
    ArrayInfo info;
    if (!parse_header(filepath, info)) {
        return false;
    }

    if (info.dtype != DataType::INT32) {
        std::cerr << "[NpyFile] Expected int32, got different dtype" << std::endl;
        return false;
    }

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return false;

    fseek(f, info.data_offset, SEEK_SET);

    *data = (int*)malloc(info.data_size);
    if (!*data) {
        fclose(f);
        return false;
    }

    if (fread(*data, 1, info.data_size, f) != info.data_size) {
        free(*data);
        *data = nullptr;
        fclose(f);
        return false;
    }

    fclose(f);

    if (info.shape.size() >= 1) {
        n = static_cast<int>(info.shape[0]);
    } else {
        n = 1;
    }

    if (info.shape.size() >= 2) {
        dim = static_cast<int>(info.shape[1]);
    } else {
        dim = 0;
    }

    return true;
}

bool NpyFile::load_int64(const std::string& filepath, long long** data, int& n, int& dim) {
    ArrayInfo info;
    if (!parse_header(filepath, info)) {
        return false;
    }

    if (info.dtype != DataType::INT64) {
        std::cerr << "[NpyFile] Expected int64, got different dtype" << std::endl;
        return false;
    }

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return false;

    fseek(f, info.data_offset, SEEK_SET);

    *data = (long long*)malloc(info.data_size);
    if (!*data) {
        fclose(f);
        return false;
    }

    if (fread(*data, 1, info.data_size, f) != info.data_size) {
        free(*data);
        *data = nullptr;
        fclose(f);
        return false;
    }

    fclose(f);

    if (info.shape.size() >= 1) {
        n = static_cast<int>(info.shape[0]);
    } else {
        n = 1;
    }

    if (info.shape.size() >= 2) {
        dim = static_cast<int>(info.shape[1]);
    } else {
        dim = 0;
    }

    return true;
}

bool NpyFile::save_float32(const std::string& filepath, const float* data,
                           const std::vector<size_t>& shape) {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    //
    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;

    //  header
    std::stringstream header_ss;
    header_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) header_ss << ", ";
        header_ss << shape[i];
    }
    header_ss << "), }";

    std::string header = header_ss.str();

    //  64
    size_t header_len = header.length() + 1;  // +1 for newline
    size_t padding_needed = (64 - (8 + header_len) % 64) % 64;
    if (padding_needed < 1) padding_needed += 64;

    header_len += padding_needed;

    //  magic
    unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);

    //
    unsigned char version[2] = {1, 0};
    fwrite(version, 1, 2, f);

    //  header
    uint16_t len = static_cast<uint16_t>(header_len);
    fwrite(&len, 2, 1, f);

    //  header + padding
    fwrite(header.c_str(), 1, header.length(), f);
    for (size_t i = 0; i < padding_needed; ++i) {
        fputc(' ', f);  // padding with spaces
    }
    fputc('\n', f);

    //
    fwrite(data, sizeof(float), total_elements, f);

    fclose(f);
    return true;
}

bool NpyFile::save_int32(const std::string& filepath, const int* data,
                         const std::vector<size_t>& shape) {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;

    std::stringstream header_ss;
    header_ss << "{'descr': '<i4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) header_ss << ", ";
        header_ss << shape[i];
    }
    header_ss << "), }";

    std::string header = header_ss.str();
    size_t header_len = header.length() + 1;
    size_t padding_needed = (64 - (8 + header_len) % 64) % 64;
    if (padding_needed < 1) padding_needed += 64;
    header_len += padding_needed;

    unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);

    unsigned char version[2] = {1, 0};
    fwrite(version, 1, 2, f);

    uint16_t len = static_cast<uint16_t>(header_len);
    fwrite(&len, 2, 1, f);

    fwrite(header.c_str(), 1, header.length(), f);
    for (size_t i = 0; i < padding_needed; ++i) {
        fputc(' ', f);
    }
    fputc('\n', f);

    fwrite(data, sizeof(int), total_elements, f);

    fclose(f);
    return true;
}

bool NpyFile::save_int64(const std::string& filepath, const long long* data,
                         const std::vector<size_t>& shape) {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;

    std::stringstream header_ss;
    header_ss << "{'descr': '<i8', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) header_ss << ", ";
        header_ss << shape[i];
    }
    header_ss << "), }";

    std::string header = header_ss.str();
    size_t header_len = header.length() + 1;
    size_t padding_needed = (64 - (8 + header_len) % 64) % 64;
    if (padding_needed < 1) padding_needed += 64;
    header_len += padding_needed;

    unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);

    unsigned char version[2] = {1, 0};
    fwrite(version, 1, 2, f);

    uint16_t len = static_cast<uint16_t>(header_len);
    fwrite(&len, 2, 1, f);

    fwrite(header.c_str(), 1, header.length(), f);
    for (size_t i = 0; i < padding_needed; ++i) {
        fputc(' ', f);
    }
    fputc('\n', f);

    fwrite(data, sizeof(long long), total_elements, f);

    fclose(f);
    return true;
}

} // namespace dataset
} // namespace paraselect
