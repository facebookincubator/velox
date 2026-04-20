/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include "velox/experimental/torchwave/KernelParams.h"

namespace torch::wave {

template <typename T>
__device__ inline T* param(const BlockInfo& block, int32_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(block.params) + offset);
}

__device__ inline uint32_t numEl(const Tensor& tensor) {
  uint32_t size = 1;
  for (auto i = 0; i < tensor.rank; ++i) {
    size = size * tensor.dims[i];
  }
  return size;
}

template <typename T>
__device__ inline T* storage(const Tensor* tensor) {
  return reinterpret_cast<T*>(tensor->storage);
}

template <typename T>
__device__ inline T elementRef(Tensor* t, uint32_t idx, uint32_t size) {
  return idx < size ? storage<T>(t)[idx] : T();
}

__device__ inline uint32_t
complexIdx(bool fast, const Tensor* t, uint32_t idx) {
  if (!fast) {
    auto translated = t->indexToOffset(idx);
    printf("complexIdx: idx=%u translated=%d\n", idx, translated);
    return translated;
  }
  return idx;
}

struct Int32X32 {
  uint64_t data[32];
  __host__ __device__ operator void*() {
    return static_cast<void*>(data);
  }
  __host__ __device__ operator void*() const {
    return const_cast<void*>(static_cast<const void*>(data));
  }
};

#define ENTRY                                                                  \
  __shared__ BlockInfo blockInfo;                                              \
  if (threadIdx.x == 0) {                                                      \
    blockInfo =                                                                \
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x]; \
    blockInfo.debugInfo =                                                      \
        params.debugInfo ? &params.debugInfo[blockIdx.x] : nullptr;            \
    if (blockInfo.debugInfo) {                                                 \
      blockInfo.start = clock64();                                             \
      memset(blockInfo.debugInfo, 0, sizeof(DebugInfo));                       \
    }                                                                          \
  }                                                                            \
  __syncthreads();

#define LEAVE()                                                \
  __syncthreads();                                             \
  if (threadIdx.x == 0 && blockInfo.debugInfo) {               \
    blockInfo.debugInfo->clocks = clock64() - blockInfo.start; \
    blockInfo.debugInfo->op = blockInfo.op;                    \
  }

constexpr int32_t kWarpThreads = 32;

template <typename T, typename U>
__host__ __device__ constexpr inline T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

/// Rounds 'x' up to the next multiple of 'y', where y must be a power of two.
template <typename T, typename U>
__host__ __device__ constexpr inline T roundUpPwr2(T x, U y) {
  return (x + (y - 1)) & ~(y - 1);
}

#define TRACE0(__x) do { if (threadIdx.x == 0) {__x;} } while (0)

} // namespace torch::wave
