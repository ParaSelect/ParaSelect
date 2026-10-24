#ifndef PARASELECT_U8_ISA_RUNNERS_H
#define PARASELECT_U8_ISA_RUNNERS_H

#include "search/kernels/fine/cpu_fine/types/cpu_fine_run.h"

namespace paraselect {
namespace cpu_fine {

long long u8_rowmajor_cluster_dynamic_scratch_plain(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_metadata_plain(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_compact_metadata_plain(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_scratch_plain(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_metadata_plain(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_compact_metadata_plain(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_scratch_plain(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_metadata_plain(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_compact_metadata_plain(const CpuFineRunArgs& args);

long long u8_rowmajor_cluster_dynamic_scratch_avx2(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_metadata_avx2(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_compact_metadata_avx2(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_scratch_avx2(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_metadata_avx2(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_compact_metadata_avx2(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_scratch_avx2(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_metadata_avx2(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_compact_metadata_avx2(const CpuFineRunArgs& args);

long long u8_rowmajor_cluster_dynamic_scratch_avx512(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_metadata_avx512(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_compact_metadata_avx512(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_scratch_avx512(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_metadata_avx512(const CpuFineRunArgs& args);
long long u8_rowmajor_cluster_dynamic_numa_compact_metadata_avx512(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_scratch_avx512(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_metadata_avx512(const CpuFineRunArgs& args);
long long u8_paged_cluster_dynamic_compact_metadata_avx512(const CpuFineRunArgs& args);

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_U8_ISA_RUNNERS_H
