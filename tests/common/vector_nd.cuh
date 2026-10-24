#ifndef VECTOR_ND_H
#define VECTOR_ND_H

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include "cpu/utils/cpu_array_utils.h"

// /**
//  *
//  */
// void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size) {
//     //
//     void* data = malloc(n_batch * n_dim * elem_size);
//     if (data == NULL) return NULL;

//     //
//     void** vector_list = (void**)malloc(n_batch * sizeof(void*));
//     if (vector_list == NULL) {
//         free(data);
//         return NULL;
//     }

//     //
//     for (size_t i = 0; i < n_batch; i++) {
//         vector_list[i] = (char*)data + i * n_dim * elem_size;
//     }

//     return vector_list;
// }

/**
 *
 * :[ T*  ] [ padding ] [ T  ]
 */
template <typename T>
T** malloc_vector_2D(int nx, int ny) {
    // 1.
    size_t ptr_section_size = nx * sizeof(T*);
    size_t data_section_size = nx * ny * sizeof(T);

    // 2.
    size_t alignment = alignof(T);
    size_t data_offset = (ptr_section_size + alignment - 1) & ~(alignment - 1);

    size_t total_size = data_offset + data_section_size;

    // 3.
    //  char*
    char* raw_memory = (char*)malloc(total_size);
    if (!raw_memory) return nullptr;

    // 4.
    T** ptr_head = reinterpret_cast<T**>(raw_memory);
    T* data_head = reinterpret_cast<T*>(raw_memory + data_offset);

    // 5. ( arr[i][j] )
    for (size_t i = 0; i < nx; ++i) {
        ptr_head[i] = data_head + (i * ny);
    }

    return ptr_head;
}
/**
*
* :[ T*  ] [ padding ] [ T  ]
*/
template <typename T>
T** malloc_vector_3D(int nx, int ny, int nz) {

    // 1.
    size_t size_L1 = nx * sizeof(T**);       //  ()
    size_t size_L2 = nx * ny * sizeof(T*);   //  ()
    size_t size_Data = nx * ny * nz * sizeof(T); //

    // 2.
    // , T
    size_t alignment = alignof(T);
    size_t ptr_total_size = size_L1 + size_L2;

    // ,()
    size_t data_offset = (ptr_total_size + alignment - 1) & ~(alignment - 1);
    size_t total_alloc_size = data_offset + size_Data;

    // 3.
    char* raw_memory = (char*)malloc(total_alloc_size);
    if (!raw_memory) return nullptr;

    // 4.
    T*** ptr_L1 = reinterpret_cast<T***>(raw_memory);
    T**  ptr_L2 = reinterpret_cast<T**>(raw_memory + size_L1);
    T*   data_head = reinterpret_cast<T*>(raw_memory + data_offset);

    // 5. (Wiring)
    for (size_t i = 0; i < nx; ++i) {
        ptr_L1[i] = ptr_L2 + (i * ny);
    }

    // 5.2  L2 -> Data
    // ptr_L2[j]  j  Data
    // : j , 0  nx*ny - 1
    for (size_t i = 0; i < nx * ny; ++i) {
        ptr_L2[i] = data_head + (i * nz);
    }
    return ptr_L1;
}



// template <typename T>
// T*** malloc_vector_3D(int nx, int ny, int nz) {
//     if (nx == 0 || ny == 0 || nz == 0) return nullptr;
//     T*** ptr_head = malloc_vector_3D<T>(nx, ny, nz);
//     return ptr_head;
// }

template <typename T>
T** generate_vector_2D(int nx, int ny) {
    if (nx == 0 || ny == 0) return nullptr;
    using RwT = std::remove_const_t<T>;
    RwT** ptr_head = malloc_vector_2D<RwT>(nx, ny);
    init_array_multithreaded(ptr_head[0], nx * ny);
    return (T**)ptr_head;
}

template <typename T>
T*** generate_vector_3D(int nx, int ny, int nz) {
    if (nx == 0 || ny == 0 || nz == 0) return nullptr;
    using RwT = std::remove_const_t<T>;
    RwT*** ptr_head = malloc_vector_3D<RwT>(nx, ny, nz);
    init_array_multithreaded(ptr_head[0][0], nx * ny * nz);
    return (T***)ptr_head;
}

/**
*
* :, free
*/
template <typename T>
void free_vector(const void* vector) {
    free(const_cast<void*>(vector));
}

/*
*
*/
template <typename T>
T** malloc_vector_list(int n_batch, int n_dim) {
    if (n_batch == 0 || n_dim == 0) return nullptr;
    T** vector_list = malloc_vector_2D<T>(n_batch, n_dim);
    return vector_list;
}

template <typename T>
T** generate_vector_list(int n_batch, int n_dim) {
    T** vector_list = generate_vector_2D<T>(n_batch, n_dim);
    return vector_list;
}

template <typename T>
void free_vector_list(T** vector_list) {
    free_vector<T>(vector_list);
}

#endif // VECTOR_ND_H
