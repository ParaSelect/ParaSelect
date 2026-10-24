#pragma once

#include "search/ivf_common/ivf_storage.h"
#include "search/ivf_common/quantized/res_query_lut_bias.h"
#include <cstdint>

namespace paraselect::search {

struct ResidualAssetsView {
    IVFExactStorageView exact;
    const std::uint32_t* tq_codes = nullptr;
    const float* tq_scales = nullptr;
    const float* residual_rotation = nullptr;
    QueryLutBias query_lut_bias;
};

}  // namespace paraselect::search
