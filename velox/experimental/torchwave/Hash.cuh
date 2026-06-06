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

#pragma once

#include "velox/experimental/torchwave/KernelParams.h"

namespace torch::wave {

// GPU hash table for the isin() operation. Uses 0 as the empty sentinel,
// so element 0 is handled separately: slot 0 of the table is reserved as
// a flag indicating whether 0 is actually in the set.

#ifdef __CUDACC__

constexpr int64_t kEmpty = 0;

__device__ inline uint32_t hashInt(int64_t val) {
  uint64_t v = static_cast<uint64_t>(val);
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<uint32_t>(v ^ (v >> 31));
}

__device__ inline int32_t
atomicCAS_T(int32_t* addr, int32_t compare, int32_t val) {
  return atomicCAS(addr, compare, val);
}

__device__ inline int64_t
atomicCAS_T(int64_t* addr, int64_t compare, int64_t val) {
  return static_cast<int64_t>(atomicCAS(
      reinterpret_cast<unsigned long long*>(addr),
      static_cast<unsigned long long>(compare),
      static_cast<unsigned long long>(val)));
}

// Element 0 of the table is a flag: non-zero means kEmpty is in the set.
// The hash table entries start at table[1]. The hash mask is tableSize - 2
// because the power-of-two portion is tableSize - 1 elements.
template <typename T>
__device__ inline void
tw_isin_head(Tensor* input, Tensor* output, BlockInfo& /*block*/) {
  T* table = storage<T>(output);
  int32_t tableSize = output->numEl;
  uint32_t mask = static_cast<uint32_t>(tableSize - 2);
  T* in = storage<T>(input);
  int32_t n = numEl(*input);

  for (auto i = threadIdx.x; i < tableSize; i += blockDim.x) {
    table[i] = 0;
  }
  __syncthreads();

  for (auto i = threadIdx.x; i < n; i += blockDim.x) {
    T val = in[i];
    if (val == static_cast<T>(kEmpty)) {
      table[0] = 1;
      continue;
    }
    uint32_t h = hashInt(static_cast<int64_t>(val)) & mask;
    for (uint32_t iter = 0; iter < static_cast<uint32_t>(tableSize); ++iter) {
      T old = atomicCAS_T(&table[h + 1], static_cast<T>(0), val);
      if (old == 0 || old == val) {
        break;
      }
      h = (h + 1) & mask;
      assert(iter + 1 < static_cast<uint32_t>(tableSize));
    }
  }
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__ inline bool
__isin_final(uint32_t idx, uint32_t size, T val, Tensor* hashTable, bool inv) {
  if (idx >= size) {
    return false;
  }
  T* table = storage<T>(hashTable);
  bool found;
  if (val == static_cast<T>(kEmpty)) {
    found = table[0] != 0;
  } else {
    found = false;
    int32_t tableSize = hashTable->numEl;
    uint32_t mask = static_cast<uint32_t>(tableSize - 2);
    uint32_t h = hashInt(static_cast<int64_t>(val)) & mask;
    for (uint32_t iter = 0; iter < static_cast<uint32_t>(tableSize); ++iter) {
      T entry = table[h + 1];
      if (entry == 0) {
        break;
      }
      if (entry == val) {
        found = true;
        break;
      }
      h = (h + 1) & mask;
      assert(iter + 1 < static_cast<uint32_t>(tableSize));
    }
  }
  return inv ? !found : found;
}

#endif

} // namespace torch::wave
