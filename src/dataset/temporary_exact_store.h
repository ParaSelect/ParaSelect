#ifndef PARASELECT_DATASET_TEMPORARY_EXACT_STORE_H
#define PARASELECT_DATASET_TEMPORARY_EXACT_STORE_H

#include "dataset/common/dataset_source.h"
#include "search/ivf_common/ivf_storage.h"

#include <cstddef>
#include <string>

namespace paraselect::dataset {

struct TemporaryExactStoreOptions {
    std::string directory;
    size_t source_chunk_rows = 1U << 16;
};

class TemporaryExactStore {
public:
    TemporaryExactStore() = default;
    ~TemporaryExactStore();
    TemporaryExactStore(const TemporaryExactStore&) = delete;
    TemporaryExactStore& operator=(const TemporaryExactStore&) = delete;
    TemporaryExactStore(TemporaryExactStore&& other) noexcept;
    TemporaryExactStore& operator=(TemporaryExactStore&& other) noexcept;

    static TemporaryExactStore build(
        const DatasetSource& source,
        const idx_t* reordered_ids,
        idx_t count,
        int dim,
        const TemporaryExactStoreOptions& options);

    search::IVFFileStorageView view() const;
    double build_ms() const { return build_ms_; }
    bool valid() const { return fd_ >= 0; }

private:
    void reset() noexcept;

    int fd_ = -1;
    search::IVFExactRerankDType dtype_ = search::IVFExactRerankDType::Disabled;
    idx_t count_ = 0;
    int dim_ = 0;
    size_t row_stride_ = 0;
    size_t payload_offset_ = 0;
    size_t block_size_ = 4096;
    double build_ms_ = 0.0;
};

}  // namespace paraselect::dataset

#endif  // PARASELECT_DATASET_TEMPORARY_EXACT_STORE_H
