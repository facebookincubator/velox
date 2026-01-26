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

namespace detail {
constexpr char kPool[] =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr int kPoolSize = 62;
} // namespace detail

/// Spark SQL randstr(length) - Returns a string of the specified length
/// with characters chosen uniformly at random from 0-9, a-z, A-Z.
/// This variant is non-deterministic (no seed provided).
template <typename T>
struct RandStrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr bool is_deterministic = false;

  template <typename TLen>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const TLen* length) {
    VELOX_USER_CHECK_NOT_NULL(length, "length argument must be constant");
    VELOX_USER_CHECK_GE(
        static_cast<int64_t>(*length), 0, "length must be non-negative");
  }

  FOLLY_ALWAYS_INLINE void call(out_type<Varchar>& out, int32_t length) {
    out.resize(length);
    for (auto i = 0; i < length; ++i) {
      out.data()[i] =
          detail::kPool[folly::Random::rand32() % detail::kPoolSize];
    }
  }

  FOLLY_ALWAYS_INLINE void call(out_type<Varchar>& out, int16_t length) {
    call(out, static_cast<int32_t>(length));
  }
};

/// Spark SQL randstr(length, seed) - Returns a string of the specified length
/// with characters chosen uniformly at random from 0-9, a-z, A-Z.
/// This variant uses a seed for reproducibility. With the same seed and
/// sparkPartitionId, results are deterministic.
/// Generator is initialized with (seed + sparkPartitionId) to match Spark's
/// per-partition determinism. If seed is null, it is treated as 0.
template <typename T>
struct RandStrSeededFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr bool is_deterministic = false;

  template <typename TLen, typename TSeed>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const TLen* length,
      const TSeed* seed) {
    VELOX_USER_CHECK_NOT_NULL(length, "length argument must be constant");
    VELOX_USER_CHECK_GE(
        static_cast<int64_t>(*length), 0, "length must be non-negative");
    generator_.seed(
        (seed ? static_cast<int64_t>(*seed) : 0) + config.sparkPartitionId());
  }

  template <typename TLen, typename TSeed>
  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varchar>& out,
      const TLen* length,
      const TSeed* /*seed*/) {
    // Null seed is treated as 0 (handled in initialize).
    // Length cannot be null (constant argument).
    const auto len = static_cast<int32_t>(*length);
    out.resize(len);
    for (auto i = 0; i < len; ++i) {
      // Match Spark's selection: use modulo similar to Java's approach.
      // Note: std::mt19937 produces unsigned 32-bit values, so no need for abs.
      out.data()[i] = detail::kPool[generator_() % detail::kPoolSize];
    }
    return true;
  }

 private:
  std::mt19937 generator_;
};

} // namespace facebook::velox::functions::sparksql
