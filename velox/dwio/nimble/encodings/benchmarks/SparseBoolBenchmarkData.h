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
inline constexpr uint64_t kSparseBoolBenchmarkDefaultSeed = 0xC0FFEE;

/// Returns one deterministic approximately-five-percent-true benchmark value.
constexpr bool sparseBoolBenchmarkValue(uint32_t index, uint64_t seed) {
  if (index == 0) {
    return false;
  }
  if (index == 1) {
    return true;
  }
  uint64_t mixed = seed + static_cast<uint64_t>(index) * 0x9E3779B97F4A7C15ULL;
  mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
  mixed ^= mixed >> 31;
  return mixed % 100 < 5;
}

} // namespace facebook::nimble::benchmarks
