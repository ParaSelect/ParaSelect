#ifndef PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_READER_H
#define PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_READER_H

#include "search/ivf_common/ivf_storage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace paraselect::search {

class IVFFileReader {
public:
    struct Request {
        uint64_t block = 0;
        size_t bytes = 0;
    };

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t bytes_read = 0;
        uint64_t bytes_reused = 0;
        uint64_t prefill_bytes = 0;
        double prefill_ms = 0.0;
        size_t capacity_bytes = 0;
    };

    class Block {
    public:
        Block() = default;
        ~Block();
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
        Block(Block&& other) noexcept;
        Block& operator=(Block&& other) noexcept;

        uint8_t* data() const { return data_; }
        bool reused() const { return reused_; }

    private:
        friend class IVFFileReader;
        Block(IVFFileReader* owner, size_t slot, uint8_t* data, bool reused);
        Block(
            uint8_t* data,
            bool reused,
            std::shared_ptr<void> owned_data);
        void release() noexcept;

        IVFFileReader* owner_ = nullptr;
        size_t slot_ = 0;
        uint8_t* data_ = nullptr;
        bool reused_ = false;
        std::shared_ptr<void> owned_data_;

    };

    explicit IVFFileReader(
        const IVFFileStorageView& storage,
        const IVFFileReadConfig& config);
    ~IVFFileReader();
    IVFFileReader(const IVFFileReader&) = delete;
    IVFFileReader& operator=(const IVFFileReader&) = delete;

    std::vector<Block> read(const std::vector<Request>& requests);
    bool contains(uint64_t block) const;
    std::vector<uint8_t> contains_many(
        const std::vector<uint64_t>& blocks) const;
    Stats stats() const;
    size_t block_size() const noexcept;
    size_t capacity_blocks() const noexcept;
    int ring_count() const noexcept;
    int queue_depth_per_ring() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void release(size_t slot) noexcept;
};

std::shared_ptr<IVFFileReader> make_ivf_file_reader(
    const IVFFileStorageView& storage,
    const IVFFileReadConfig& config);

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_FILE_IVF_FILE_READER_H
