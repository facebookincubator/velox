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

#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

Status broundFloatingPoint(float value, int32_t scale, float& result);

Status broundFloatingPoint(double value, int32_t scale, double& result);

FOLLY_ALWAYS_INLINE int128_t divideHalfEven(int128_t value, int128_t divisor) {
  const int128_t quotient = value / divisor;
  const int128_t remainder = value % divisor;
  if (remainder == 0) {
    return quotient;
  }

  const int128_t absoluteRemainder = remainder < 0 ? -remainder : remainder;
  const int128_t half = divisor / 2;
  if (absoluteRemainder > half ||
      (absoluteRemainder == half && quotient % 2 != 0)) {
    return quotient + (value < 0 ? -1 : 1);
  }
  return quotient;
}

template <typename T>
FOLLY_ALWAYS_INLINE T wrapToSigned(int128_t value) {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
  using UnsignedT = std::make_unsigned_t<T>;
  return std::bit_cast<T>(static_cast<UnsignedT>(value));
}

template <typename T>
FOLLY_ALWAYS_INLINE Status
broundIntegral(T value, int32_t scale, bool ansiEnabled, T& result) {
  static_assert(
      std::is_integral_v<T> && std::is_signed_v<T> && !std::is_same_v<T, bool>);

  if (scale >= 0 || value == 0) {
    result = value;
    return Status::OK();
  }

  const int64_t roundingDigitCount = -static_cast<int64_t>(scale);
  if (roundingDigitCount > std::numeric_limits<T>::digits10 + 1) {
    result = 0;
    return Status::OK();
  }

  const int128_t divisor = DecimalUtil::kPowersOfTen[roundingDigitCount];
  const int128_t rounded =
      divideHalfEven(static_cast<int128_t>(value), divisor) * divisor;
  if (ansiEnabled &&
      (rounded < std::numeric_limits<T>::min() ||
       rounded > std::numeric_limits<T>::max())) {
    return threadSkipErrorDetails()
        ? Status::UserError()
        : Status::UserError(
              "Arithmetic overflow in bround({}, {})",
              static_cast<int64_t>(value),
              scale);
  }

  result = wrapToSigned<T>(rounded);
  return Status::OK();
}

} // namespace detail

template <typename TExec>
struct BRoundFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const T* /*value*/) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const T* /*value*/,
      const int32_t* /*scale*/) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
  }

  /// Rounds 'value' to zero decimal places using HALF_EVEN semantics.
  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T value) {
    return call(result, value, 0);
  }

  /// Rounds 'value' to 'scale' decimal places using HALF_EVEN semantics.
  template <typename T>
  FOLLY_ALWAYS_INLINE Status
  call(T& result, const T value, const int32_t scale) {
    if constexpr (std::is_floating_point_v<T>) {
      return detail::broundFloatingPoint(value, scale, result);
    } else {
      return detail::broundIntegral(value, scale, ansiEnabled_, result);
    }
  }

 private:
  bool ansiEnabled_{false};
};

/// Registers primitive bround functions.
void registerBRoundFunctions(const std::string& prefix);

} // namespace facebook::velox::functions::sparksql
