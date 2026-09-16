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
inline constexpr uint64_t kPforBenchmarkDefaultSeed = 0xC0FFEE;

/// Minimum value in the representative PFOR corpus.
inline constexpr uint32_t kPforBenchmarkBaseline = 50;

/// Largest residual in the representative corpus's narrow population.
inline constexpr uint32_t kPforBenchmarkNarrowMax = 127;

/// Minimum residual in the representative corpus's outlier population.
inline constexpr uint32_t kPforBenchmarkOutlierBase = 100'000;

/// Returns whether `index` belongs to the approximately-ten-percent outliers.
inline constexpr bool pforBenchmarkIsOutlier(uint64_t index, uint64_t seed) {
  return index > 1 && (index + seed % 10) % 10 == 7;
}

/// Returns one deterministic value from the representative PFOR corpus.
inline constexpr uint32_t pforBenchmarkValue(
    uint64_t index,
    uint64_t seed = kPforBenchmarkDefaultSeed) {
  if (index == 0) {
    return kPforBenchmarkBaseline;
  }
  if (index == 1) {
    return kPforBenchmarkBaseline + kPforBenchmarkNarrowMax;
  }
  const uint64_t mixed = (index * 1'000'003ULL) ^ seed ^ (seed >> 32);
  if (pforBenchmarkIsOutlier(index, seed)) {
    return kPforBenchmarkBaseline + kPforBenchmarkOutlierBase +
        static_cast<uint32_t>(mixed & 0xFFFF);
  }
  return kPforBenchmarkBaseline +
      static_cast<uint32_t>(mixed & kPforBenchmarkNarrowMax);
}

static_assert(
    pforBenchmarkValue(0) == kPforBenchmarkBaseline &&
    pforBenchmarkValue(1) == kPforBenchmarkBaseline + kPforBenchmarkNarrowMax);
static_assert(pforBenchmarkIsOutlier(7, kPforBenchmarkDefaultSeed));
static_assert(
    pforBenchmarkValue(7) >=
    kPforBenchmarkBaseline + kPforBenchmarkOutlierBase);

} // namespace facebook::nimble::benchmarks
