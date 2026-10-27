// SPDX-License-Identifier: Apache-2.0
// One atomic for a full warp. Per-lane atomics otherwise.
#pragma once
#include <cuda_runtime.h>

namespace clad {
namespace cuda {
// key names the destination. Call from the lanes that will update it.
__device__ __forceinline__ bool full_warp_key(unsigned key) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  return __match_any_sync(__activemask(), key) == 0xffffffff;
#else
  return false;
#endif
}

__device__ __forceinline__ void
warp_accumulate(bool complete, float* destination, float value) {
  if (complete) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      value += __shfl_down_sync(0xffffffff, value, offset);
    if ((((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
         31) == 0)
      atomicAdd(destination, value);
  } else {
    atomicAdd(destination, value);
  }
}
} // namespace cuda
} // namespace clad
