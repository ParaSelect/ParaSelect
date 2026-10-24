#pragma once

#include "common/row_correction.h"

namespace paraselect::search {

using QueryLutBias = paraselect::row_correction::StorageView;

using paraselect::row_correction::value;
inline float query_lut_bias_value(const float* centroid, const float* reconstruction, int dim) {
    return paraselect::row_correction::value(centroid, reconstruction, dim);
}

}  // namespace paraselect::search
