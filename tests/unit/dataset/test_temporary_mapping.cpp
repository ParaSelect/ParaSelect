#include "dataset/common/temporary_mapping.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() try {
    char pattern[] = "/tmp/paraselect-temp-mapping-XXXXXX";
    char* directory = ::mkdtemp(pattern);
    if (!directory) throw std::runtime_error("mkdtemp failed");
    const std::filesystem::path root(directory);
    {
        paraselect::dataset::TemporaryMapping<uint32_t> map(directory, "inverse", 4096);
        uint32_t* data = map.data();
        for (uint32_t row = 0; row < 4096; ++row) data[(row * 37) % 4096] = row;
        if (!std::filesystem::is_empty(root)) throw std::runtime_error("temporary name leaked");
        if (::msync(data, 4096 * sizeof(uint32_t), MS_SYNC) ||
            ::madvise(data, 4096 * sizeof(uint32_t), MADV_DONTNEED)) {
            throw std::runtime_error("reclaim simulation failed");
        }
        for (uint32_t row = 0; row < 4096; ++row) {
            if (data[(row * 37) % 4096] != row) {
                throw std::runtime_error("inverse map lost values after reclaim");
            }
        }
    }
    bool rejected = false;
    try {
        paraselect::dataset::TemporaryMapping<uint32_t> invalid(
            directory, "overflow", std::numeric_limits<size_t>::max());
    } catch (const std::overflow_error&) { rejected = true; }
    if (!rejected || !std::filesystem::is_empty(root)) {
        throw std::runtime_error("invalid mapping was accepted or left a file");
    }
    std::filesystem::remove(root);
    std::cout << "PASS temporary inverse mapping survives reclaim, unlinks, rejects overflow\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
