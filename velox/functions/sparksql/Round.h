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

#include <cstdint>

#include "folly/CPortability.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

/// Rounds a REAL input while preventing binary scaling from introducing a
/// non-finite result.
void roundFloatingPoint(float input, int32_t scale, float& result);

/// Rounds a DOUBLE input while preventing binary scaling from introducing a
/// non-finite result.
void roundFloatingPoint(double input, int32_t scale, double& result);

} // namespace detail

/// Rounds floating-point values using Spark's HALF_UP direction while avoiding
/// non-finite results introduced solely by binary scale arithmetic.
template <typename T>
struct FloatingPointRoundFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const TInput& input) {
    detail::roundFloatingPoint(input, 0, result);
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(TInput& result, const TInput& input, int32_t scale) {
    detail::roundFloatingPoint(input, scale, result);
  }
};

} // namespace facebook::velox::functions::sparksql
