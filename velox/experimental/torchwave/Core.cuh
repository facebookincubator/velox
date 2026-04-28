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

__device__ uint32_t numEl(const Tensor& tensor) {
  uint32_t size = 1;
  for (auto i = 0; i < tensor.rank; ++i) {
    size = size * tensor.dims[i];
  }
  return size;
}

template <typename T>
__device__ T* storage(const Tensor* tensor) {
  return reinterpret_cast<T*>(tensor->storage);
}

__device__ bool isFastPathTensor(const Tensor& tensor) {
  return tensor.rank == 1 && tensor.strides[0] == 1;
}

#define ENTRY                                                                  \
  __shared__ BlockInfo blockInfo;                                              \
  if (threadIdx.x == 0) {                                                      \
    blockInfo =                                                                \
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x]; \
  }                                                                            \
  __syncthreads();
} // namespace torch::wave
