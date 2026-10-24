#ifndef PARASELECT_SEARCH_IVF_COMMON_IVF_STORAGE_H
#define PARASELECT_SEARCH_IVF_COMMON_IVF_STORAGE_H

#include "search/ivf_common/ivf_search_types.h"
#include "utils/pch.h"
#include "search/ivf_common/ivf_host_storage.h"

#include <cstddef>
#include <cstdint>

namespace paraselect::cpu_fine {
struct CpuPqCodeStorageView;
}

namespace paraselect::search {

struct IVFPQCodeStorageView {
    const uint8_t* codes = nullptr;
    const paraselect::cpu_fine::CpuPqCodeStorageView* code_storage = nullptr;
    const float* codebook = nullptr;
    int subvectors = 0;
    int codebook_size = 0;
};

enum class IVFExactStorageKind : uint8_t {
    None = 0,
    Host = 1,
    Resident = 2,
    File = 3,
};

enum class IVFRefineBackend : uint8_t {
    None = 0,
    Cpu = 1,
    Gpu = 2,
};

inline const char* ivf_refine_backend_name(IVFRefineBackend backend) {
    switch (backend) {
        case IVFRefineBackend::None: return "none";
        case IVFRefineBackend::Cpu: return "cpu";
        case IVFRefineBackend::Gpu: return "gpu";
    }
    return "unknown";
}



struct IVFResidentStorageView {
    const float* fp32 = nullptr;
    const uint8_t* u8 = nullptr;
    const float* norms = nullptr;
    const idx_t* source_ids = nullptr;
    idx_t count = 0;
    int dim = 0;
};

struct IVFFileStorageView {
    IVFExactRerankDType dtype = IVFExactRerankDType::Disabled;
    int fd = -1;
    idx_t count = 0;
    int dim = 0;
    size_t row_stride = 0;
    size_t payload_offset = 0;
    size_t block_size = 4096;
};

struct IVFFileReadConfig {
    size_t read_block_bytes = 0;
    int ring_count = 0;
    int queue_depth_per_ring = 0;
    size_t cache_capacity_bytes = 0;
};

struct IVFExactStorageView {
    IVFExactStorageKind kind = IVFExactStorageKind::None;
    IVFHostStorageView host;
    IVFResidentStorageView resident;
    IVFFileStorageView file;

    static IVFExactStorageView from_host(IVFHostStorageView value) {
        IVFExactStorageView out;
        out.kind = IVFExactStorageKind::Host;
        out.host = value;
        return out;
    }

    static IVFExactStorageView from_resident(IVFResidentStorageView value) {
        IVFExactStorageView out;
        out.kind = IVFExactStorageKind::Resident;
        out.resident = value;
        return out;
    }

    static IVFExactStorageView from_file(IVFFileStorageView value) {
        IVFExactStorageView out;
        out.kind = IVFExactStorageKind::File;
        out.file = value;
        return out;
    }
};


}  // namespace paraselect::search

#endif  // PARASELECT_SEARCH_IVF_COMMON_IVF_STORAGE_H
