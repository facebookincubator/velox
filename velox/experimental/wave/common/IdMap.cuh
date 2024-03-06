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

#include "velox/experimental/wave/common/StringView.cuh"

namespace facebook::velox::wave {

template <typename T, typename H>
__device__ void IdMap<T, H>::clearTable() {
  for (int i = threadIdx.x; i < capacity_; i += blockDim.x) {
    values_[i] = kEmptyMarker;
    ids_[i] = 0;
  }
}

template<typename T>
inline __device__ T atomicCAS_soft( T *address, T compare, T val ) {
  // synthesize shorter atomic CAS than availble
  // by using aligned 32-bit atomic CAS 
  // with existing values in the other bytes
  // and restarting if other bytes change while trying to store target
  uint32_t const ofs = ((uintptr_t)address) & 3;
  uint32_t const val_shift = ofs * 8;
  uint32_t const val_mask = ((1<<sizeof(T)*8)-1) << val_shift;
  uint32_t * aligned_addr = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(address)-ofs);

  uint32_t old = *aligned_addr;
  while( (old & val_mask) == (((uint32_t)compare) << val_shift) ) {
    uint32_t store = (old & ~val_mask) | (((uint32_t)val) << val_shift);
    store = atomicCAS( aligned_addr, old, store );
    if( store == old ) {
      return compare;
    }
    old = store;
  }
  return (T)((old & val_mask) >> val_shift);
}

template <typename T, typename H>
__device__ T IdMap<T, H>::casValue(T* address, T compare, T val) {
  if constexpr (std::is_same_v<T, StringView>) {
    return address->cas(compare, val);
  } else if constexpr (sizeof(T) == 8) {
    using ULL = unsigned long long;
    return atomicCAS((ULL*)address, (ULL)compare, (ULL)val);
  } else if constexpr (sizeof(T) == 1) {
    return (T)atomicCAS_soft( (uint8_t *)address, (uint8_t)compare, (uint8_t)val);
  } else if constexpr (sizeof(T) == 2) {
    return (T)atomicCAS_soft( (uint16_t *)address, (uint16_t)compare, (uint16_t)val);
  } else {
    return atomicCAS(address, compare, val);
  }
  __builtin_unreachable();
}

template <typename T, typename H>
__device__ int32_t IdMap<T, H>::ensureIdReady(
    volatile int32_t* id) {
  volatile int32_t cand = *id;
  if (cand <= 0) {
    auto t0 = clock64();
    do {
      assert(clock64() - t0 < 10'000'000);
      cand = *id;
    } while (cand <= 0);
  }
  return cand;
}

template <typename T, typename H>
__device__ int32_t IdMap<T, H>::makeId(T value) {
  if (value == kEmptyMarker) {
    return kEmptyId;
  }
  auto const capacity = capacity_;
  auto end = H()(value) % capacity;
  auto i = (end+1) % capacity;
  for (; i != end; i = (i + 1) % capacity) {
    auto const cand_value = casValue(&values_[i], kEmptyMarker, value);
    if( cand_value == kEmptyMarker ) {
      // This is a critical section, and so we don't need to protect the write.
      // We wait on the value to be written later, so that we know the id is
      // set correctly on all threads
      ids_[i] = atomicAdd(&lastId_, 1) + 1;
      break;
    }
    if( cand_value == value ) {
      break;
    }
  }

  if (i != end) {
    return ensureIdReady(&ids_[i]);
  } else {
    return -1;
  }
}

} // namespace facebook::velox::wave
