#ifndef PARASELECT_SEARCH_IVFRESPQ_TIMING_H
#define PARASELECT_SEARCH_IVFRESPQ_TIMING_H

#include <algorithm>
#include <cmath>

namespace paraselect::search {

struct IVFResPQSerialTiming {
    double batch_ms = 0.0;
    double coarse_ms = 0.0;
    double fine_ms = 0.0;
    double map_ids_ms = 0.0;
    double fine_gpu_wall_ms = 0.0;
    double fine_cpu_rerank_ms = 0.0;

    double fine_overhead_ms() const {
        return fine_ms - fine_gpu_wall_ms - fine_cpu_rerank_ms;
    }

    double stage_sum_ms() const {
        return coarse_ms + fine_ms + map_ids_ms;
    }

    double closure_error_ms() const {
        return batch_ms - stage_sum_ms();
    }
};

}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVFRESPQ_TIMING_H
