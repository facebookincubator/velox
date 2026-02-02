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

#include <random>

#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

namespace detail {
constexpr char kPool[] =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr int kPoolSize = 62;
} // namespace detail

/// Spark SQL randstr(length, seed) - Returns a string of the specified
/// length with characters chosen uniformly at random from 0-9, a-z, A-Z.
/// Generator is initialized with (seed + sparkPartitionId) to match Spark's
/// per-partition determinism. If seed is NULL constant, it is treated as 0.
///
/// Note: Spark's analyzer always provides a seed (either user-specified or
/// auto-generated), so only the seeded variant is needed.
///
/// Note: Even with a constant seed, different rows produce different outputs
/// as the generator advances, so is_deterministic is set to false.
template <typename T>
struct RandStrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr bool is_deterministic = false;

  /// Initialize for seeded variant: randstr(length, seed).
  /// The seed argument is validated as constant by the Constant<TSeed> wrapper.
  /// A NULL seed constant (pointer is nullptr) is treated as 0 per Spark
  /// semantics.
  template <typename TLen, typename TSeed>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const TLen* length,
      const TSeed* seed) {
    // With Constant<TLen>, length is guaranteed to be constant.
    // nullptr means NULL constant, which is not allowed for length.
    VELOX_USER_CHECK_NOT_NULL(length, "length must not be null");
    VELOX_USER_CHECK_GE(
        static_cast<int64_t>(*length), 0, "length must be non-negative");
    // NULL seed constant (nullptr) is treated as 0 per Spark semantics.
    generator_.seed(
        (seed ? static_cast<int64_t>(*seed) : 0) + config.sparkPartitionId());
  }

  /// Called for seeded variant. Uses the seeded generator for reproducibility.
  template <typename TLen, typename TSeed>
  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varchar>& out,
      const TLen* length,
      const TSeed* /*seed*/) {
    // Length cannot be null (validated in initialize).
    const auto len = static_cast<int32_t>(*length);
    out.resize(len);
    for (auto i = 0; i < len; ++i) {
      out.data()[i] = detail::kPool[generator_() % detail::kPoolSize];
    }
    return true;
  }

 private:
  std::mt19937 generator_;
};

} // namespace facebook::velox::functions::sparksql
