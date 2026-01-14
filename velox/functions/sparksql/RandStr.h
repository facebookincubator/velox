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
#include <random>

#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

// Spark SQL randstr(length[, seed]) - Returns a string of the specified length
// with characters chosen uniformly at random from 0-9, a-z, A-Z.
//
// Semantics:
// - length must be a constant SMALLINT/INT and >= 0.
// - seed (if provided) must be a constant INT/LONG.
// - With seed provided, generator is initialized with (seed + sparkPartitionId)
//   to match Spark's per-partition determinism.
// - Without seed, the function is non-deterministic and uses thread-local
//   randomness.
template <typename T>
struct RandStrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr bool is_deterministic = false;

  /// The implementation of randstr(length).
  template <typename TLen>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const TLen* length) {
    VELOX_CHECK_NOT_NULL(length, "length argument must be constant");
    VELOX_USER_CHECK_GE(
        static_cast<int64_t>(*length), 0, "length must be non-negative");
    hasSeed_ = false;
  }

  /// The implementation of randstr(length, seed).
  template <typename TLen, typename TSeed>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const TLen* length,
      const TSeed* seed) {
    VELOX_CHECK_NOT_NULL(length, "length argument must be constant");
    VELOX_USER_CHECK_GE(
        static_cast<int64_t>(*length), 0, "length must be non-negative");
    const int64_t s = seed ? static_cast<int64_t>(*seed) : 0;
    const int64_t partitionId = config.sparkPartitionId();
    generator_.seed(s + partitionId);
    hasSeed_ = true;
  }

  FOLLY_ALWAYS_INLINE void call(out_type<Varchar>& out, int32_t length) {
    generateUnseeded(out, length);
  }

  FOLLY_ALWAYS_INLINE void call(out_type<Varchar>& out, int16_t length) {
    generateUnseeded(out, static_cast<int32_t>(length));
  }

  template <typename TLen, typename TSeed>
  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varchar>& out,
      const TLen* length,
      const TSeed* /*seed*/) {
    // Null seed is treated as 0 (handled in initialize).
    // Length cannot be null (constant argument).
    generateSeeded(out, static_cast<int32_t>(*length));
    return true;
  }

 private:
  static constexpr char kPool[] =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  static constexpr int kPoolSize = 62;

  FOLLY_ALWAYS_INLINE void generateUnseeded(
      out_type<Varchar>& out,
      int32_t length) {
    // length validation is performed in initialize for Constant args. Guard
    // here as well for safety.
    VELOX_USER_CHECK_GE(length, 0, "length must be non-negative");
    out.resize(length);
    for (auto i = 0; i < length; ++i) {
      // Use thread-local randomness for non-deterministic behavior.
      uint32_t r = folly::Random::rand32();
      out.data()[i] = kPool[r % kPoolSize];
    }
  }

  FOLLY_ALWAYS_INLINE void generateSeeded(
      out_type<Varchar>& out,
      int32_t length) {
    VELOX_USER_CHECK_GE(length, 0, "length must be non-negative");
    out.resize(length);
    // Unbiased distribution over [0, kPoolSize-1].
    std::uniform_int_distribution<int> dist(0, kPoolSize - 1);
    for (auto i = 0; i < length; ++i) {
      out.data()[i] = kPool[dist(generator_)];
    }
  }

  std::mt19937 generator_;
  bool hasSeed_{false};
};

} // namespace facebook::velox::functions::sparksql
