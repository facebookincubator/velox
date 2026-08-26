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
inline constexpr uint64_t kSimdForBitpackBenchmarkDefaultSeed = 0xC0FFEE;

/// Baseline used by the representative unsigned corpus.
inline constexpr uint32_t kSimdForBitpackBenchmarkBaseline = 50'000;

/// Largest residual in the representative 16-bit corpus.
inline constexpr uint32_t kSimdForBitpackBenchmarkResidualMax = 0xFFFF;

/// Returns one deterministic value from the representative 16-bit corpus.
inline constexpr uint32_t simdForBitpackBenchmarkValue(
    uint64_t index,
    uint64_t seed = kSimdForBitpackBenchmarkDefaultSeed) {
  if (index == 0) {
    return kSimdForBitpackBenchmarkBaseline;
  }
  if (index == 1) {
    return kSimdForBitpackBenchmarkBaseline +
        kSimdForBitpackBenchmarkResidualMax;
  }
  const uint64_t mixed = (index * 1'000'003ULL) ^ seed ^ (seed >> 32);
  return kSimdForBitpackBenchmarkBaseline +
      static_cast<uint32_t>(mixed & kSimdForBitpackBenchmarkResidualMax);
}

static_assert(
    simdForBitpackBenchmarkValue(0) == kSimdForBitpackBenchmarkBaseline &&
    simdForBitpackBenchmarkValue(1) ==
        kSimdForBitpackBenchmarkBaseline + kSimdForBitpackBenchmarkResidualMax);

} // namespace facebook::nimble::benchmarks
