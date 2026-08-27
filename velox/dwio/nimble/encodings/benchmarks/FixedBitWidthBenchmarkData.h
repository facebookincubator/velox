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
#include <limits>

namespace facebook::nimble::benchmarks {

/// Baseline added to benchmark values with bit widths below 64.
inline constexpr uint64_t kFixedBitWidthBenchmarkBaseline = 12'345;

/// Odd multiplier used to traverse the bounded benchmark value space.
inline constexpr uint64_t kFixedBitWidthBenchmarkMultiplier = 1'000'003;

/// Returns the native benchmark mask for a bit width in [1, 64].
inline constexpr uint64_t fixedBitWidthBenchmarkMask(int bitWidth) {
  return bitWidth >= std::numeric_limits<uint64_t>::digits
      ? std::numeric_limits<uint64_t>::max()
      : (uint64_t{1} << bitWidth) - 1;
}

/// Returns the native benchmark baseline, or zero for a 64-bit value.
///
/// `bitWidth` must be in [1, 64].
inline constexpr uint64_t fixedBitWidthBenchmarkBaseline(int bitWidth) {
  return bitWidth >= std::numeric_limits<uint64_t>::digits
      ? 0
      : kFixedBitWidthBenchmarkBaseline;
}

/// Returns one deterministic native-profile value for a width in [1, 64].
inline constexpr uint64_t fixedBitWidthBenchmarkValue(
    uint64_t index,
    int bitWidth) {
  return ((index * kFixedBitWidthBenchmarkMultiplier) &
          fixedBitWidthBenchmarkMask(bitWidth)) +
      fixedBitWidthBenchmarkBaseline(bitWidth);
}

static_assert(
    fixedBitWidthBenchmarkMask(64) == std::numeric_limits<uint64_t>::max());
static_assert(fixedBitWidthBenchmarkBaseline(64) == 0);

} // namespace facebook::nimble::benchmarks
