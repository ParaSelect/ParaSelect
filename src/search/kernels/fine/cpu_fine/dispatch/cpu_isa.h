#ifndef PARASELECT_CPU_ISA_H
#define PARASELECT_CPU_ISA_H

#include "cpu/distance/cpu_isa.h"

namespace paraselect {
namespace cpu_fine {

using ::paraselect::cpu_distance::host_supports_avx2;
using ::paraselect::cpu_distance::host_supports_avx512f;
using ::paraselect::cpu_distance::host_supports_fma;
using ::paraselect::cpu_distance::host_supports_avx512bw;
using ::paraselect::cpu_distance::host_supports_u8_avx512;
using ::paraselect::cpu_distance::host_supports_fp32_avx2;
using ::paraselect::cpu_distance::host_supports_fp32_avx512;

}  // namespace cpu_fine
}  // namespace paraselect

#endif  // PARASELECT_CPU_ISA_H
