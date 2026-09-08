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
inline constexpr uint64_t kDeltaBenchmarkDefaultSeed = 0xC0FFEE;

/// Largest increment in the native benchmark's increasing-data profile.
inline constexpr uint32_t kDeltaBenchmarkMaxIncrement = 9;

/// Returns the first value for a deterministic, seed-aware Delta corpus.
inline constexpr uint32_t deltaBenchmarkInitialValue(uint64_t seed) {
  return static_cast<uint32_t>(seed & 0xFFFF);
}

/// Returns the small non-negative increment at `index` after the first row.
inline constexpr uint32_t deltaBenchmarkIncrement(
    uint64_t index,
    uint64_t seed) {
  return static_cast<uint32_t>(
      ((index * 1'000'003ULL) ^ seed ^ (seed >> 32)) %
      (kDeltaBenchmarkMaxIncrement + 1));
}

static_assert(deltaBenchmarkIncrement(1, kDeltaBenchmarkDefaultSeed) <= 9);

} // namespace facebook::nimble::benchmarks
