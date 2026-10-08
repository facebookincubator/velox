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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace facebook::nimble::benchmarks {

inline constexpr std::string_view kVarintBenchmarkProfileName =
    "Uint32NonZeroBaselineMixedWidths";
inline constexpr uint64_t kVarintBenchmarkDefaultSeed = 0xC0FFEE;
inline constexpr uint32_t kVarintBenchmarkBaseValue = 50'000;
inline constexpr size_t kVarintBulkDecodePaddingBytes = 7;

inline constexpr uint32_t varintBenchmarkBaseline(
    uint64_t seed = kVarintBenchmarkDefaultSeed) {
  return kVarintBenchmarkBaseValue + static_cast<uint8_t>(seed);
}

inline constexpr uint64_t varintBenchmarkMix(uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

inline constexpr uint32_t varintBenchmarkResidual(
    uint32_t row,
    uint64_t seed = kVarintBenchmarkDefaultSeed) {
  const uint32_t maxResidual =
      std::numeric_limits<uint32_t>::max() - varintBenchmarkBaseline(seed);
  switch (row & 0xFF) {
    case 0:
      return 0;
    case 1:
      return 0x7F;
    case 2:
      return 0x80;
    case 3:
      return 0x3FFF;
    case 4:
      return 0x4000;
    case 5:
      return 0x1FFFFF;
    case 6:
      return 0x200000;
    case 7:
      return 0x0FFFFFFF;
    case 8:
      return 0x10000000;
    case 9:
      return maxResidual;
  }

  const uint64_t mixed = varintBenchmarkMix(seed + static_cast<uint64_t>(row));
  uint64_t lower;
  uint64_t upper;
  switch (row % 5) {
    case 0:
      lower = 0;
      upper = 0x7F;
      break;
    case 1:
      lower = 0x80;
      upper = 0x3FFF;
      break;
    case 2:
      lower = 0x4000;
      upper = 0x1FFFFF;
      break;
    case 3:
      lower = 0x200000;
      upper = 0x0FFFFFFF;
      break;
    default:
      lower = 0x10000000;
      upper = maxResidual;
      break;
  }
  return static_cast<uint32_t>(lower + mixed % (upper - lower + 1));
}

inline constexpr uint32_t varintBenchmarkValue(
    uint32_t row,
    uint64_t seed = kVarintBenchmarkDefaultSeed) {
  return varintBenchmarkBaseline(seed) + varintBenchmarkResidual(row, seed);
}

static_assert(varintBenchmarkResidual(0) == 0);
static_assert(varintBenchmarkResidual(1) == 0x7F);
static_assert(varintBenchmarkResidual(2) == 0x80);
static_assert(varintBenchmarkResidual(3) == 0x3FFF);
static_assert(varintBenchmarkResidual(4) == 0x4000);
static_assert(varintBenchmarkResidual(5) == 0x1FFFFF);
static_assert(varintBenchmarkResidual(6) == 0x200000);
static_assert(varintBenchmarkResidual(7) == 0x0FFFFFFF);
static_assert(varintBenchmarkResidual(8) == 0x10000000);
static_assert(varintBenchmarkValue(9) == std::numeric_limits<uint32_t>::max());

} // namespace facebook::nimble::benchmarks
