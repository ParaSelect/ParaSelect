#pragma once
#include <cstddef>

namespace paraselect::coarse::current {
// One geometry contract for scratch allocation, producer launch and merge.
struct SimdShape {
    static constexpr int tile = 256;
    static constexpr int merge_tile = 1024;
    int tiles;
    int stride;
    std::size_t elements;

    SimdShape(int batch, int nlist, int k)
        : tiles(nlist / tile + (nlist % tile != 0)),
          stride(tiles * k), elements(std::size_t(batch) * stride) {}
    static int merge_blocks(int length) {
        return length / merge_tile + (length % merge_tile != 0);
    }
};
}
