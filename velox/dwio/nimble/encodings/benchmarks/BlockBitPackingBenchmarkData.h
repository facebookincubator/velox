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

#include "velox/dwio/nimble/common/Constants.h"

namespace facebook::nimble::benchmarks {

/// Default seed shared by the native benchmark and executable runner.
inline constexpr uint64_t kBlockBitPackingBenchmarkDefaultSeed = 0xC0FFEE;

/// First per-block baseline in the representative unsigned corpus.
inline constexpr uint32_t kBlockBitPackingBenchmarkBaseline = 50'000;

/// Distance between adjacent block baselines in the representative corpus.
inline constexpr uint32_t kBlockBitPackingBenchmarkBaselineStride = 8'192;

/// Largest residual in each representative 12-bit block.
inline constexpr uint32_t kBlockBitPackingBenchmarkResidualMax = 0xFFF;

/// Returns one deterministic value from the representative block-local corpus.
inline constexpr uint32_t blockBitPackingBenchmarkValue(
    uint64_t index,
    uint64_t seed = kBlockBitPackingBenchmarkDefaultSeed) {
  const uint64_t block = index / kBlockBitPackingBlockSize;
  const uint64_t blockOffset = index % kBlockBitPackingBlockSize;
  const auto baseline = static_cast<uint32_t>(
      kBlockBitPackingBenchmarkBaseline +
      block * kBlockBitPackingBenchmarkBaselineStride + (seed & 0xFF));
  if (blockOffset == 0) {
    return baseline;
  }
  if (blockOffset == 1) {
    return baseline + kBlockBitPackingBenchmarkResidualMax;
  }
  const uint64_t mixed = (index * 1'000'003ULL) ^ seed ^ (seed >> 32);
  return baseline +
      static_cast<uint32_t>(mixed & kBlockBitPackingBenchmarkResidualMax);
}

static_assert(
    blockBitPackingBenchmarkValue(1) - blockBitPackingBenchmarkValue(0) ==
    kBlockBitPackingBenchmarkResidualMax);

} // namespace facebook::nimble::benchmarks
