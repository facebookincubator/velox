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

#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct RandFunction {
  static constexpr bool is_deterministic = false;

  std::optional<std::mt19937> generator;

  FOLLY_ALWAYS_INLINE void call(double& result) {
    result = folly::Random::randDouble01();
  }

  FOLLY_ALWAYS_INLINE void callNullable(
      double& result,
      const int32_t* seed,
      const int32_t* partitionIndex) {
    callNullable(result, (int64_t*)seed, partitionIndex);
  }

  /// To differentiate generator for each thread, seed plus partitionIndex is
  /// the actual seed used for generator.
  FOLLY_ALWAYS_INLINE void callNullable(
      double& result,
      const int64_t* seed,
      const int32_t* partitionIndex) {
    VELOX_USER_CHECK_NOT_NULL(partitionIndex, "partitionIndex cannot be null.");
    if (!generator.has_value()) {
      generator = std::mt19937{};
      if (seed) {
        generator->seed(*seed + *partitionIndex);
      } else {
        // For null input, 0 plus partitionIndex is the seed, consistent with
        // Spark.
        generator->seed(*partitionIndex);
      }
    }
    result = folly::Random::randDouble01(*generator);
  }

  // For NULL constant input of unknown type.
  FOLLY_ALWAYS_INLINE void callNullable(
      double& result,
      const UnknownValue* seed,
      const int32_t* partitionIndex) {
    VELOX_USER_CHECK_NOT_NULL(partitionIndex, "partitionIndex cannot be null.");
    if (!generator.has_value()) {
      generator = std::mt19937{};
      // For null input, 0 plus partitionIndex is the seed, consistent with
      // Spark.
      generator->seed(*partitionIndex);
    }
    result = folly::Random::randDouble01(*generator);
  }
};

void registerRandFunctions(const std::string& prefix) {
  // No input.
  registerFunction<RandFunction, double>({prefix + "rand", prefix + "random"});
  // Has seed & partition index as input.
  registerFunction<
      RandFunction,
      double,
      int32_t /*seed*/,
      int32_t /*partition index*/>({prefix + "rand", prefix + "random"});
  // Has seed & partition index as input.
  registerFunction<
      RandFunction,
      double,
      int64_t /*seed*/,
      int32_t /*partition index*/>({prefix + "rand", prefix + "random"});
  // NULL constant as seed of unknown type.
  registerFunction<
      RandFunction,
      double,
      UnknownValue /*seed*/,
      int32_t /*partition index*/>({prefix + "rand", prefix + "random"});
}
} // namespace facebook::velox::functions::sparksql
