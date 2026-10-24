#pragma once

#include "dataset/common/mapped_array.h"
#include "index/row_correction/row_correction.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace paraselect::dataset {

class RowCorrectionAsset {
public:
    static RowCorrectionAsset open(const std::string& path, std::uint64_t rows) {
        if (!rows || rows > (std::numeric_limits<size_t>::max() - sizeof(row_correction::U16Header)) / sizeof(float) ||
            rows > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::invalid_argument("invalid row correction row count");
        RowCorrectionAsset result;
        result.mapping_ = ReadOnlyMapping::open(path);
        auto& view = result.view_;
        view.count = static_cast<std::int64_t>(rows);
        if (result.mapping_->bytes() == rows * sizeof(float)) {
            view.fp32 = static_cast<const float*>(result.mapping_->data());
            return result;
        }
        if (result.mapping_->bytes() < sizeof(row_correction::U16Header))
            throw std::invalid_argument("row correction file size does not match reordered rows");
        const auto* h = static_cast<const row_correction::U16Header*>(result.mapping_->data());
        if (std::memcmp(h->magic, row_correction::kU16Magic, sizeof(h->magic)) ||
            h->version != row_correction::kU16Version || h->header_bytes != sizeof(*h) ||
            h->count != rows || !std::isfinite(h->scale) || !(h->scale > 0) ||
            !std::isfinite(h->offset) ||
            result.mapping_->bytes() != sizeof(*h) + rows * sizeof(std::uint16_t))
            throw std::invalid_argument("row correction U16 header or payload is invalid");
        view.u16 = reinterpret_cast<const std::uint16_t*>(
            static_cast<const std::uint8_t*>(result.mapping_->data()) + sizeof(*h));
        view.scale = h->scale;
        view.offset = h->offset;
        return result;
    }

    const row_correction::StorageView& view() const noexcept { return view_; }

private:
    std::shared_ptr<ReadOnlyMapping> mapping_;
    row_correction::StorageView view_;
};

}  // namespace paraselect::dataset
