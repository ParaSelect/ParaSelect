#ifndef PARASELECT_DATASET_COMMON_MAPPED_ARRAY_H
#define PARASELECT_DATASET_COMMON_MAPPED_ARRAY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace paraselect::dataset {

class ReadOnlyMapping {
public:
    static std::shared_ptr<ReadOnlyMapping> open(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("failed to open mapped file: " + path);
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            throw std::runtime_error("failed to stat mapped file: " + path);
        }
        void* data = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (data == MAP_FAILED) {
            throw std::runtime_error("failed to mmap file: " + path);
        }
        return std::shared_ptr<ReadOnlyMapping>(
            new ReadOnlyMapping(data, static_cast<size_t>(st.st_size)));
    }

    ~ReadOnlyMapping() {
        if (data_) ::munmap(data_, bytes_);
    }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;

    const void* data() const { return data_; }
    size_t bytes() const { return bytes_; }

private:
    ReadOnlyMapping(void* data, size_t bytes) : data_(data), bytes_(bytes) {}
    void* data_ = nullptr;
    size_t bytes_ = 0;
};

template <typename T>
class MappedArray {
public:
    static MappedArray open_binary(
        const std::string& path,
        uint32_t expected_n,
        uint32_t expected_dim) {
        auto mapping = ReadOnlyMapping::open(path);
        constexpr size_t header_bytes = sizeof(uint32_t) * 2;
        if (mapping->bytes() < header_bytes) {
            throw std::runtime_error("mapped binary header is truncated: " + path);
        }
        const auto* header = static_cast<const uint32_t*>(mapping->data());
        if (header[0] != expected_n || header[1] != expected_dim) {
            throw std::runtime_error("mapped binary shape mismatch: " + path);
        }
        const size_t count = static_cast<size_t>(expected_n) * expected_dim;
        if (mapping->bytes() != header_bytes + count * sizeof(T)) {
            throw std::runtime_error("mapped binary size mismatch: " + path);
        }
        MappedArray out;
        out.mapping_ = std::move(mapping);
        out.data_ = reinterpret_cast<const T*>(
            static_cast<const unsigned char*>(out.mapping_->data()) + header_bytes);
        out.size_ = count;
        return out;
    }

    static MappedArray open_binary_prefix(
        const std::string& path,
        uint32_t prefix_n,
        uint32_t expected_dim) {
        auto mapping = ReadOnlyMapping::open(path);
        constexpr size_t header_bytes = sizeof(uint32_t) * 2;
        if (mapping->bytes() < header_bytes) {
            throw std::runtime_error("mapped binary header is truncated: " + path);
        }
        const auto* header = static_cast<const uint32_t*>(mapping->data());
        if (header[0] < prefix_n || header[1] != expected_dim) {
            throw std::runtime_error("mapped binary prefix shape mismatch: " + path);
        }
        const size_t file_count = static_cast<size_t>(header[0]) * header[1];
        if (mapping->bytes() != header_bytes + file_count * sizeof(T)) {
            throw std::runtime_error("mapped binary size mismatch: " + path);
        }
        MappedArray out;
        out.mapping_ = std::move(mapping);
        out.data_ = reinterpret_cast<const T*>(
            static_cast<const unsigned char*>(out.mapping_->data()) + header_bytes);
        out.size_ = static_cast<size_t>(prefix_n) * expected_dim;
        return out;
    }

    const T* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T& operator[](size_t index) const { return data_[index]; }

    void reset() {
        mapping_.reset();
        data_ = nullptr;
        size_ = 0;
    }

private:
    std::shared_ptr<ReadOnlyMapping> mapping_;
    const T* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace paraselect::dataset

#endif
