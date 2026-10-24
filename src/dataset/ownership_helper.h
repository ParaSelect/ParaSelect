#pragma once

#include <cstddef>
#include <cstdlib>
#include <utility>
#include <cuda_runtime.h>

namespace paraselect {
namespace ownership {

// .
// - None/Borrowed:
// - StdMalloc:  std::free ( memalign/malloc )
// - CudaHost:  cudaFreeHost ( cudaMallocHost )
// - Custom:  free_fn(data, free_ctx)
enum class BufferOwner : unsigned char {
    None = 0,
    StdMalloc = 1,
    CudaHost = 2,
    Borrowed = 3,
    Custom = 4
};

// ( BufferOwner::Custom)
using FreeFn = void (*)(void*, void*);

// move-only ,"//".
// :
// 1)  BufferHandle ;
// 2) /;
// 3)  detach() .
// ,.
struct BufferHandle {
    void* data = nullptr;
    std::size_t bytes = 0;
    BufferOwner owner = BufferOwner::None;
    FreeFn free_fn = nullptr;
    void* free_ctx = nullptr;

    BufferHandle() = default;
    BufferHandle(void* p, std::size_t n, BufferOwner o,
                 FreeFn fn = nullptr, void* ctx = nullptr)
        : data(p), bytes(n), owner(o), free_fn(fn), free_ctx(ctx) {}

    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;

    BufferHandle(BufferHandle&& other) noexcept { *this = std::move(other); }
    BufferHandle& operator=(BufferHandle&& other) noexcept {
        if (this != &other) {
            reset();
            data = other.data;
            bytes = other.bytes;
            owner = other.owner;
            free_fn = other.free_fn;
            free_ctx = other.free_ctx;
            other.data = nullptr;
            other.bytes = 0;
            other.owner = BufferOwner::None;
            other.free_fn = nullptr;
            other.free_ctx = nullptr;
        }
        return *this;
    }

    ~BufferHandle() { reset(); }

    template <typename T>
    T* as() const {
        return static_cast<T*>(data);
    }

    // , None.
    //  BufferHandle .
    void* detach() {
        void* p = data;
        data = nullptr;
        bytes = 0;
        owner = BufferOwner::None;
        free_fn = nullptr;
        free_ctx = nullptr;
        return p;
    }

    // ().
    // .
    void reset() {
        if (!data) return;
        switch (owner) {
            case BufferOwner::StdMalloc:
                std::free(data);
                break;
            case BufferOwner::CudaHost:
                cudaFreeHost(data);
                break;
            case BufferOwner::Custom:
                if (free_fn) free_fn(data, free_ctx);
                break;
            case BufferOwner::Borrowed:
            case BufferOwner::None:
            default:
                break;
        }
        data = nullptr;
        bytes = 0;
        owner = BufferOwner::None;
        free_fn = nullptr;
        free_ctx = nullptr;
    }
};

// ().
template <typename T>
inline T* take(BufferHandle& h) {
    return static_cast<T*>(h.detach());
}

// .
template <typename T>
inline void take_to(BufferHandle& h, T*& out) {
    out = take<T>(h);
}

// , owner.
template <typename T>
inline void take_to(BufferHandle& h, T*& out, BufferOwner& out_owner) {
    const BufferOwner owner = h.owner;
    out = take<T>(h);
    out_owner = owner;
}

//  host  cudaMallocHost.
//  fallback.
inline BufferOwner detect_owner(
    const void* ptr,
    BufferOwner fallback = BufferOwner::StdMalloc) {
#if defined(CUDART_VERSION)
    if (!ptr) return fallback;
    cudaPointerAttributes attr{};
    cudaError_t st = cudaPointerGetAttributes(&attr, ptr);
    if (st == cudaSuccess) {
#if CUDART_VERSION >= 10000
        if (attr.type == cudaMemoryTypeHost) return BufferOwner::CudaHost;
#else
        if (attr.memoryType == cudaMemoryTypeHost) return BufferOwner::CudaHost;
#endif
    } else {
        cudaGetLastError();
    }
#endif
    return fallback;
}

//  owner ; RAII .
inline void free_raw(
    void* ptr,
    BufferOwner owner,
    FreeFn free_fn = nullptr,
    void* free_ctx = nullptr) {
    if (!ptr) return;
    switch (owner) {
        case BufferOwner::StdMalloc:
            std::free(ptr);
            break;
        case BufferOwner::CudaHost:
            cudaFreeHost(ptr);
            break;
        case BufferOwner::Custom:
            if (free_fn) free_fn(ptr, free_ctx);
            break;
        case BufferOwner::Borrowed:
        case BufferOwner::None:
        default:
            break;
    }
}

}  // namespace ownership
}  // namespace paraselect
