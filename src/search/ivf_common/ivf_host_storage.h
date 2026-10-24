#pragma once
#include "utils/types.h"
#include <cstdint>
namespace paraselect::search {
struct IVFHostStorageView {
    const float* fp32 = nullptr;
    const uint8_t* u8 = nullptr;
    idx_t count = 0;
    int dim = 0;
    // Physical row -> original IVF row, borrowed for the storage lifetime.
    const int* source_rows = nullptr;
};
}
