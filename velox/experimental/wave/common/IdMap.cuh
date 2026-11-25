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

#include "velox/experimental/wave/common/IdMap.h"

namespace facebook::velox::wave {

template <typename T, typename H>
__device__ void IdMap<T, H>::clearTable() {
  for (auto i = threadIdx.x; i < capacity_; i += blockDim.x) {
    ids_[i] = 0;
  }
}

template <typename T, typename H>
__device__ void IdMap<T, H>::storeNewId(volatile int32_t* id) {
  *id = atomicAdd(const_cast<int*>(&lastId_), 1) + 1;
}

// `ensureIdReady' cannot be executed in the same lockstep with `storeNewId'
// (e.g. in the else branch of `storeNewId'), which will cause deadlock if
// both branches are executed on the same warp (making one single thread wait
// will cause the whole warp to wait).
template <typename T, typename H>
__device__ void IdMap<T, H>::ensureIdReady(
    volatile const int32_t* id,
    int32_t placeholder) {
  if (*id != placeholder) {
    return;
  }
  auto t0 = clock64();
  while (*id == placeholder) {
    assert(clock64() - t0 < 1'000'000);
  }
}

template <typename T, typename H>
__device__ T IdMap<T, H>::loadValue(int32_t i) const {
  // The value may have been written by another SM after this SM cached it.
  if constexpr (sizeof(T) % sizeof(uint64_t) == 0) {
    T value;
    auto* words = reinterpret_cast<volatile const uint64_t*>(&values_[i]);
    auto* result = reinterpret_cast<uint64_t*>(&value);
    for (auto w = 0; w < sizeof(T) / sizeof(uint64_t); ++w) {
      result[w] = words[w];
    }
    return value;
  } else {
    return *reinterpret_cast<volatile const T*>(&values_[i]);
  }
}

template <typename T, typename H>
__device__ int32_t IdMap<T, H>::makeId(T value) {
  auto mask = capacity_ - 1;
  auto maxEntries = capacity_ - capacity_ / 4;
  volatile int32_t* ids = ids_;
  for (auto i = H()(value) & mask;; i = (i + 1) & mask) {
    if (lastId_ > maxEntries) {
      return -1;
    }
    // A slot is claimed by its id word. The claiming thread writes the value
    // and then the id. Values of any size work without atomics on them.
    if (ids[i] == 0 &&
        atomicCAS(const_cast<int32_t*>(&ids[i]), 0, kClaimed) == 0) {
      values_[i] = value;
      __threadfence();
      storeNewId(&ids[i]);
    }
    ensureIdReady(&ids[i], kClaimed);
    __threadfence();
    if (loadValue(i) == value) {
      return ids[i];
    }
  }
}

} // namespace facebook::velox::wave
