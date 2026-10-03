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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {

/// Matches Java BigInteger's maximum supported power-of-ten exponent.
inline constexpr int64_t kMaxJavaBigIntegerPowerOfTenExponent = 536'870'919;

namespace detail {

inline constexpr size_t kMaxRoundingDigitCount = 18;
inline constexpr uint64_t kTenToNineteen = 10'000'000'000'000'000'000ULL;

Status broundFloatingPoint(float value, int32_t scale, float& result);

Status broundFloatingPoint(double value, int32_t scale, double& result);

FOLLY_ALWAYS_INLINE int64_t
broundUnscaled(int64_t unscaled, size_t roundingDigitCount) {
  VELOX_CHECK_LE(roundingDigitCount, kMaxRoundingDigitCount);
  const int64_t divisor =
      static_cast<int64_t>(DecimalUtil::kPowersOfTen[roundingDigitCount]);
  const int64_t quotient = unscaled / divisor;
  const int64_t remainder = unscaled % divisor;
  if (remainder == 0) {
    return quotient;
  }

  const uint64_t absoluteRemainder = remainder < 0
      ? uint64_t{0} - static_cast<uint64_t>(remainder)
      : static_cast<uint64_t>(remainder);
  const uint64_t half = static_cast<uint64_t>(divisor) / 2;
  if (absoluteRemainder > half ||
      (absoluteRemainder == half && quotient % 2 != 0)) {
    return quotient + (unscaled < 0 ? -1 : 1);
  }
  return quotient;
}

template <typename T>
FOLLY_ALWAYS_INLINE T wrapToSigned(uint64_t value) {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
  using UnsignedT = std::make_unsigned_t<T>;
  return std::bit_cast<T>(static_cast<UnsignedT>(value));
}

template <typename T>
FOLLY_ALWAYS_INLINE Status broundIntegral(T value, int32_t scale, T& result) {
  static_assert(
      std::is_integral_v<T> && std::is_signed_v<T> && !std::is_same_v<T, bool>);

  if (scale >= 0 || value == 0) {
    result = value;
    return Status::OK();
  }

  const int64_t roundingDigitCount = -static_cast<int64_t>(scale);
  if (roundingDigitCount > kMaxJavaBigIntegerPowerOfTenExponent) {
    if (threadSkipErrorDetails()) {
      return Status::UserError();
    }
    return Status::UserError("Underflow while rounding to scale {}", scale);
  }

  if constexpr (sizeof(T) == sizeof(int64_t)) {
    if (roundingDigitCount == 19) {
      const uint64_t magnitude = value < 0
          ? uint64_t{0} - static_cast<uint64_t>(value)
          : static_cast<uint64_t>(value);
      if (magnitude <= kTenToNineteen / 2) {
        result = 0;
      } else {
        result = wrapToSigned<T>(
            value < 0 ? uint64_t{0} - kTenToNineteen : kTenToNineteen);
      }
      return Status::OK();
    }
  }

  if (roundingDigitCount > static_cast<int64_t>(kMaxRoundingDigitCount)) {
    result = 0;
    return Status::OK();
  }

  const auto digitCount = static_cast<size_t>(roundingDigitCount);
  const int64_t divisor =
      static_cast<int64_t>(DecimalUtil::kPowersOfTen[digitCount]);
  const int64_t rounded =
      broundUnscaled(static_cast<int64_t>(value), digitCount);
  const uint64_t scaled =
      static_cast<uint64_t>(rounded) * static_cast<uint64_t>(divisor);
  result = wrapToSigned<T>(scaled);
  return Status::OK();
}

} // namespace detail

template <typename TExec>
struct BRoundFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  /// Rounds 'value' to 'scale' decimal places using HALF_EVEN semantics.
  template <typename T>
  FOLLY_ALWAYS_INLINE Status
  call(T& result, const T value, const int32_t scale = 0) {
    if constexpr (std::is_floating_point_v<T>) {
      return detail::broundFloatingPoint(value, scale, result);
    } else {
      return detail::broundIntegral(value, scale, result);
    }
  }
};

} // namespace facebook::velox::functions::sparksql
