/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>

namespace facebook::nimble::benchmarks {

/// Default seed shared by the native benchmark and executable runner.
inline constexpr uint64_t kHuffmanBenchmarkDefaultSeed = 0xC0FFEE;

/// Returns one value from the deterministic 90% / 6% / 4% native profile.
inline constexpr uint32_t huffmanSkewedLowCardinalityValue(
    uint64_t row,
    uint64_t seed = kHuffmanBenchmarkDefaultSeed) {
  const uint64_t rotation = seed ^ kHuffmanBenchmarkDefaultSeed;
  const uint32_t phase = static_cast<uint32_t>((row + rotation % 100) % 100);
  if (phase < 90) {
    return 0;
  }
  if (phase < 96) {
    return 1;
  }
  const uint32_t tailRotation = static_cast<uint32_t>((rotation / 100) % 14);
  return 2 + static_cast<uint32_t>((row + tailRotation) % 14);
}

static_assert(huffmanSkewedLowCardinalityValue(0) == 0);
static_assert(huffmanSkewedLowCardinalityValue(89) == 0);
static_assert(huffmanSkewedLowCardinalityValue(90) == 1);
static_assert(huffmanSkewedLowCardinalityValue(95) == 1);
static_assert(huffmanSkewedLowCardinalityValue(96) == 14);

} // namespace facebook::nimble::benchmarks
