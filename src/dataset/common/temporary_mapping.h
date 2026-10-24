#pragma once

#include <cstddef>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace paraselect::dataset {

template <typename T>
class TemporaryMapping {
public:
    TemporaryMapping(const std::string& directory, const char* name, size_t count) {
        if (directory.empty() || !name || count == 0) {
            throw std::invalid_argument("invalid temporary mapping request");
        }
        if (count > static_cast<size_t>(std::numeric_limits<off_t>::max()) / sizeof(T)) {
            throw std::overflow_error("temporary mapping byte count overflow");
        }
        bytes_ = count * sizeof(T);
        std::string pattern = directory + "/." + name + "-XXXXXX";
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        fd_ = ::mkstemp(path.data());
        if (fd_ < 0) {
            throw std::runtime_error("failed to create temporary mapping in: " + directory);
        }
        if (::ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
            ::close(fd_);
            fd_ = -1;
            ::unlink(path.data());
            throw std::runtime_error("failed to allocate temporary mapping in: " + directory);
        }
        mapping_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            ::unlink(path.data());
            throw std::runtime_error("failed to mmap temporary file in: " + directory);
        }
        if (::unlink(path.data()) != 0) {
            ::munmap(mapping_, bytes_);
            mapping_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to unlink temporary mapping in: " + directory);
        }
    }

    ~TemporaryMapping() {
        if (mapping_) ::munmap(mapping_, bytes_);
        if (fd_ >= 0) ::close(fd_);
    }

    TemporaryMapping(const TemporaryMapping&) = delete;
    TemporaryMapping& operator=(const TemporaryMapping&) = delete;
    T* data() { return static_cast<T*>(mapping_); }

private:
    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace paraselect::dataset
