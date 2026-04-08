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

#include <folly/Random.h>

#include "velox/functions/Macros.h"
#include "velox/functions/lib/XORShiftRandom.h"

namespace facebook::velox::functions::sparksql {

/// Spark SQL rand([seed]) - Returns a random double in [0.0, 1.0).
/// Uses XORShift algorithm for Spark-compatible reproducibility.
/// Generator is initialized with (seed + sparkPartitionId) to match Spark's
/// per-partition determinism.
///
/// Note: Even with a constant seed, different rows produce different outputs
/// as the generator advances, so is_deterministic is set to false.
template <typename T>
struct RandFunction {
  static constexpr bool is_deterministic = false;

  /// Initialize for unseeded variant: rand().
  /// Uses a random seed from folly::Random to match Spark's behavior where
  /// unseeded rand() generates different values across executions.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config) {
    const auto partitionId = config.sparkPartitionId();
    // Use folly::Random to generate a random seed for unseeded rand().
    int64_t seed = folly::Random::rand64();
    generator_.setSeed(seed + partitionId);
  }

  /// Initialize for seeded variant: rand(seed).
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const TInput* seedInput) {
    const auto partitionId = config.sparkPartitionId();
    int64_t seed = seedInput ? static_cast<int64_t>(*seedInput) : 0;
    generator_.setSeed(seed + partitionId);
  }

  FOLLY_ALWAYS_INLINE void call(double& result) {
    result = generator_.nextDouble();
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void callNullable(double& result, TInput /*seedInput*/) {
    result = generator_.nextDouble();
  }

 private:
  functions::XORShiftRandom generator_;
};

} // namespace facebook::velox::functions::sparksql
