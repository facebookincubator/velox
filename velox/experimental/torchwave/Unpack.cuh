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

#include "velox/experimental/torchwave/Core.cuh"
#include "velox/experimental/torchwave/PackFormat.h"

namespace torch::wave {

#ifdef __CUDACC__

// Decoding of a packed input, laid out as PackFormat.h says: a header, one
// descriptor row per slot, then the slots' values stored as unsigned offsets
// from the slot's base.

// A scalar argument arrives by value when it is a constant and by pointer
// when the device holds it (a SymInt input).
template <typename T>
__device__ inline int64_t unpackScalar(T value) {
  return static_cast<int64_t>(value);
}

template <typename T>
__device__ inline int64_t unpackScalar(T* value) {
  return static_cast<int64_t>(*value);
}

// One slot of a packed input, read where it lies. Besides tw.unpack_slot,
// a getter that reads a slot's values directly instead of an unpacked copy of
// them holds one of these. A slot with a hash (kPackDescHash) decodes each
// value v to v % hash, truncated toward zero as torch.fmod: the id hashing the
// trainer would otherwise have applied to the unpacked values after the copy.
struct PackedSlot {
  const uint8_t* data;
  uint64_t base;
  int32_t bits;
  uint64_t hash;
  // floor((2^64 - 1) / hash): the quotient estimate __umul64hi(x, hashInverse)
  // is at most 2 below x / hash, so two conditional subtractions give x % hash
  // without a divide.
  uint64_t hashInverse;

  __device__ void init(Tensor* packed, int64_t slot) {
    const int64_t* words = storage<int64_t>(packed);
    const int64_t* desc = words + kPackHeaderWords + kPackDescWords * slot;
    data = reinterpret_cast<const uint8_t*>(words) + desc[kPackDescByteOffset];
    bits = static_cast<int32_t>(desc[kPackDescBits]);
    base = static_cast<uint64_t>(desc[kPackDescBase]);
    hash = desc[kPackDescHash] > 0 ? static_cast<uint64_t>(desc[kPackDescHash])
                                   : 0;
    hashInverse = hash != 0 ? ~uint64_t{0} / hash : 0;
  }

  __device__ int64_t hashed(int64_t value) const {
    const uint64_t sign = static_cast<uint64_t>(value >> 63);
    const uint64_t magnitude = (static_cast<uint64_t>(value) ^ sign) - sign;
    uint64_t remainder = magnitude - __umul64hi(magnitude, hashInverse) * hash;
    remainder -= remainder >= hash ? hash : 0;
    remainder -= remainder >= hash ? hash : 0;
    return static_cast<int64_t>((remainder ^ sign) - sign);
  }

  __device__ int64_t operator[](int64_t i) const {
    uint64_t offset = 0;
    switch (bits) {
      case 8:
        offset = data[i];
        break;
      case 16:
        offset = reinterpret_cast<const uint16_t*>(data)[i];
        break;
      case 32:
        offset = reinterpret_cast<const uint32_t*>(data)[i];
        break;
      case 64:
        offset = reinterpret_cast<const uint64_t*>(data)[i];
        break;
      default:
        break;
    }
    const auto value = static_cast<int64_t>(base + offset);
    return hash != 0 ? hashed(value) : value;
  }
};

// tw.unpack_slot: writes the 'size' values of slot 'slot' of 'packed' to
// 'output' as int64, elements striding over the op's threads.
template <typename TSize, typename TSlot>
__device__ void unpack_slot(
    TSize size,
    Tensor* packed,
    TSlot slot,
    Tensor* output,
    BlockInfo& block) {
  PackedSlot source;
  source.init(packed, unpackScalar(slot));
  const int64_t n = unpackScalar(size);
  int64_t* out = storage<int64_t>(output);
  for (int64_t i = block.blockInOp * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(block.numBlocksInOp) * blockDim.x) {
    out[i] = source[i];
  }
}

#endif

} // namespace torch::wave
