# ParaSelect

ParaSelect is a GPU-native inverted-file (IVF) system for large-scale vector search.
IVF search first routes each query to selected centroids and then scans the associated
inverted lists. ParaSelect co-designs the algorithms and GPU operators across both stages
to expose parallelism while reducing synchronization, intermediate memory traffic, and
scheduling overhead.

The implementation includes fused coarse search, shared lookup-table reuse, parallel
fine-search scheduling, candidate selection, and exact reranking over host-resident
vectors. This heterogeneous-memory design keeps fine search on the GPU while allowing
the original vector collection to exceed device memory.

This source release contains the reusable CUDA implementation and focused tests. It does
not include experiment launchers, paper-specific configurations, measured results, paper
sources, generated figures, or private dataset metadata.

## Requirements

- CMake 3.26 or newer
- CUDA Toolkit with `nvcc`
- A C++20-capable compiler
- Optional RAPIDS dependencies, when enabled through CMake options

## Configure and build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To configure without test targets, pass `-DPARASELECT_SKIP_TESTS=ON`.

Core search APIs accept caller-owned index and query views directly.

## Layout

- `src/`: CUDA kernels and core host-side support code
- `cmake/`: build configuration and library definitions
- `tests/unit/`: focused CUDA and host unit tests
- `tests/smoke/`: lightweight CUDA smoke tests
- `third_party/`: source dependencies distributed with the project
