#ifndef PARASELECT_INDEX_IVF_CACHE_CACHE_FILE_H
#define PARASELECT_INDEX_IVF_CACHE_CACHE_FILE_H

#include "dataset/common/temporary_mapping.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace paraselect::ivf_cache::detail {

template <typename T>
void write_binary_atomic(
    const std::string& path,
    uint32_t n,
    uint32_t dim,
    const T* data) {
    const std::string tmp = path + ".tmp";
    FILE* file = std::fopen(tmp.c_str(), "wb");
    if (!file) throw std::runtime_error("failed to create cache file: " + tmp);
    const size_t count = static_cast<size_t>(n) * dim;
    bool ok = std::fwrite(&n, sizeof(n), 1, file) == 1 &&
              std::fwrite(&dim, sizeof(dim), 1, file) == 1 &&
              std::fwrite(data, sizeof(T), count, file) == count;
    if (std::fclose(file) != 0) ok = false;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to commit cache file: " + path);
    }
}

template <typename T>
class WritableBinaryMapping {
public:
    WritableBinaryMapping(const std::string& final_path, uint32_t n, uint32_t dim)
        : final_path_(final_path), tmp_path_(final_path + ".tmp") {
        const size_t count = static_cast<size_t>(n) * dim;
        bytes_ = sizeof(uint32_t) * 2 + count * sizeof(T);
        fd_ = ::open(tmp_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0 || ::ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
            cleanup();
            throw std::runtime_error("failed to allocate mapped cache file: " + tmp_path_);
        }
        mapping_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            cleanup();
            throw std::runtime_error("failed to mmap cache file: " + tmp_path_);
        }
        auto* header = static_cast<uint32_t*>(mapping_);
        header[0] = n;
        header[1] = dim;
        data_ = reinterpret_cast<T*>(header + 2);
    }

    ~WritableBinaryMapping() { cleanup(); }
    WritableBinaryMapping(const WritableBinaryMapping&) = delete;
    WritableBinaryMapping& operator=(const WritableBinaryMapping&) = delete;

    T* data() { return data_; }

    void commit() {
        if (!mapping_) throw std::runtime_error("cache mapping is not open");
        if (::msync(mapping_, bytes_, MS_SYNC) != 0) {
            throw std::runtime_error("failed to sync cache file: " + tmp_path_);
        }
        ::munmap(mapping_, bytes_);
        mapping_ = nullptr;
        data_ = nullptr;
        if (::fsync(fd_) != 0 || ::close(fd_) != 0) {
            fd_ = -1;
            throw std::runtime_error("failed to flush cache file: " + tmp_path_);
        }
        fd_ = -1;
        if (std::rename(tmp_path_.c_str(), final_path_.c_str()) != 0) {
            throw std::runtime_error("failed to commit cache file: " + final_path_);
        }
        committed_ = true;
    }

private:
    void cleanup() {
        if (mapping_) ::munmap(mapping_, bytes_);
        if (fd_ >= 0) ::close(fd_);
        if (!committed_) std::remove(tmp_path_.c_str());
        mapping_ = nullptr;
        data_ = nullptr;
        fd_ = -1;
    }

    std::string final_path_;
    std::string tmp_path_;
    int fd_ = -1;
    void* mapping_ = nullptr;
    T* data_ = nullptr;
    size_t bytes_ = 0;
    bool committed_ = false;
};

using paraselect::dataset::TemporaryMapping;


}  // namespace paraselect::ivf_cache::detail

#endif
