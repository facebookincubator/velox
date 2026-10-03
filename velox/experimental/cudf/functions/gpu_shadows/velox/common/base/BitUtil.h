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

// GPU shadow for velox/common/base/BitUtil.h. Provides only the helpers simple
// function bodies call, such as bits::countBits for BitCountFunction, as
// __host__ __device__ functions.
#pragma once

#include "folly/CPortability.h"
#include "velox/common/base/Macros.h"

#include <cstdint>

namespace facebook::velox::bits {

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int32_t popcount64(uint64_t value) {
  return __builtin_popcountll(value);
}

// Counts the set bits in [begin, end), as velox::bits::countBits does. A
// negative bound yields 0 rather than shifting a negative operand.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int32_t
countBits(const uint64_t* bits, int32_t begin, int32_t end) {
  if (begin < 0 || end <= begin) {
    return 0;
  }
  constexpr int32_t kWordShift = 6; // log2(64)
  constexpr int32_t kWordMask = 63; // 64 - 1
  const int32_t firstWord = begin >> kWordShift;
  const int32_t lastWord = (end - 1) >> kWordShift;
  const int32_t firstBit = begin & kWordMask;
  const int32_t lastBit = end & kWordMask; // 0 == last word fully covered.

  if (firstWord == lastWord) {
    uint64_t mask = (~uint64_t{0}) << firstBit;
    if (lastBit != 0) {
      mask &= (uint64_t{1} << lastBit) - 1;
    }
    return popcount64(bits[firstWord] & mask);
  }

  int32_t count = popcount64(bits[firstWord] & ((~uint64_t{0}) << firstBit));
  for (int32_t i = firstWord + 1; i < lastWord; ++i) {
    count += popcount64(bits[i]);
  }
  const uint64_t tailMask =
      (lastBit == 0) ? (~uint64_t{0}) : ((uint64_t{1} << lastBit) - 1);
  count += popcount64(bits[lastWord] & tailMask);
  return count;
}

} // namespace facebook::velox::bits
