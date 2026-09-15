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

#include <array>
#include <cstdint>
#include <utility>

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"
#include "velox/type/TypeKind.h"

/// The scalar half of DecimalUtil: rescaling, range checks, and division.
///
/// Split out of DecimalUtil.h so that code which only does decimal arithmetic
/// does not also pull in the runtime type system. DecimalUtil.h itself needs
/// Type.h, Status.h, <string> and <charconv> for its string and cast helpers;
/// none of those are reachable from a CUDA translation unit, and none of them
/// are needed to add two decimals.
///
/// DecimalUtil inherits from this, so every existing DecimalUtil::kPowersOfTen
/// and DecimalUtil::valueInRange call site keeps working unchanged.
namespace facebook::velox {

class Type;

/// Precision and scale of a decimal type. Defined in Type.cpp; declared here
/// so that a header defining decimal simple functions can name it from an
/// initialize() body without including the whole runtime type system.
std::pair<uint8_t, uint8_t> getDecimalPrecisionScale(const Type& type);

namespace detail {

/// Maximum precision of the two decimal storage widths. These mirror
/// ShortDecimalType::kMaxPrecision and LongDecimalType::kMaxPrecision, which
/// live in Type.h and so cannot be named here.
inline constexpr uint8_t kMaxShortDecimalPrecision = 18;
inline constexpr uint8_t kMaxLongDecimalPrecision = 38;

/// 10^exponent. exponent must be <= kMaxLongDecimalPrecision.
///
/// The table is inside the function rather than a static member on purpose.
/// nvcc cannot index an object with static storage duration from device code
/// when the index is only known at run time -- it reports "identifier is
/// undefined in device code" -- and --expt-relaxed-constexpr does not lift
/// that. A function-local table has no such restriction, and a constant index
/// folds to the same immediate either way.
///
/// This is also the single place the literals are written; DecimalArithmetic's
/// kPowersOfTen is derived from it at compile time. It sits outside the class
/// because a class's own static members cannot call one of its constexpr
/// member functions while the class is still incomplete.
VELOX_GPU_COMPATIBLE constexpr int128_t decimalPowerOfTen(uint8_t exponent) {
  constexpr int128_t kTable[kMaxLongDecimalPrecision + 1] = {
      1,
      10,
      100,
      1'000,
      10'000,
      100'000,
      1'000'000,
      10'000'000,
      100'000'000,
      1'000'000'000,
      10'000'000'000,
      100'000'000'000,
      1'000'000'000'000,
      10'000'000'000'000,
      100'000'000'000'000,
      1'000'000'000'000'000,
      10'000'000'000'000'000,
      100'000'000'000'000'000,
      1'000'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10,
      1'000'000'000'000'000'000 * (int128_t)100,
      1'000'000'000'000'000'000 * (int128_t)1'000,
      1'000'000'000'000'000'000 * (int128_t)10'000,
      1'000'000'000'000'000'000 * (int128_t)100'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
          (int128_t)10,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
          (int128_t)100};
  return kTable[exponent];
}

} // namespace detail

struct DecimalArithmetic {
  static constexpr uint8_t kMaxShortPrecision =
      detail::kMaxShortDecimalPrecision;
  static constexpr uint8_t kMaxLongPrecision = detail::kMaxLongDecimalPrecision;

  /// 10^exponent. exponent must be <= kMaxLongPrecision. Safe to call from
  /// device code, unlike indexing kPowersOfTen.
  VELOX_GPU_COMPATIBLE static constexpr int128_t powerOfTen(uint8_t exponent) {
    return detail::decimalPowerOfTen(exponent);
  }

  /// kPowersOfTen[i] == 10^i, derived from powerOfTen() so the literals are
  /// written once. Host-only: indexing this from device code is the very thing
  /// powerOfTen() exists to avoid.
  static constexpr std::array<int128_t, kMaxLongPrecision + 1> kPowersOfTen =
      [] {
        std::array<int128_t, kMaxLongPrecision + 1> table{};
        for (uint8_t i = 0; i <= kMaxLongPrecision; ++i) {
          table[i] = detail::decimalPowerOfTen(i);
        }
        return table;
      }();

  static constexpr int128_t kLongDecimalMin =
      -detail::decimalPowerOfTen(kMaxLongPrecision) + 1;
  static constexpr int128_t kLongDecimalMax =
      detail::decimalPowerOfTen(kMaxLongPrecision) - 1;
  static constexpr int128_t kShortDecimalMin =
      -detail::decimalPowerOfTen(kMaxShortPrecision) + 1;
  static constexpr int128_t kShortDecimalMax =
      detail::decimalPowerOfTen(kMaxShortPrecision) - 1;

  VELOX_GPU_COMPATIBLE static void valueInRange(int128_t value) {
    VELOX_USER_CHECK(
        (value >= kLongDecimalMin && value <= kLongDecimalMax),
        "Decimal overflow. Value '{}' is not in the range of Decimal Type",
        value);
  }

  /// Returns true if the precision can represent the value.
  template <typename T>
  VELOX_GPU_COMPATIBLE static bool valueInPrecisionRange(
      T value,
      uint8_t precision) {
    return value < powerOfTen(precision) && value > -powerOfTen(precision);
  }

  template <typename R, typename A, typename B>
  VELOX_GPU_COMPATIBLE static R divideWithRoundUp(
      R& r,
      A a,
      B b,
      bool noRoundUp,
      uint8_t aRescale,
      uint8_t /*bRescale*/) {
    VELOX_USER_CHECK_NE(b, 0, "Division by zero");
    int resultSign = 1;
    R unsignedDividendRescaled(a);
    if (a < 0) {
      resultSign = -1;
      unsignedDividendRescaled *= -1;
    }
    B unsignedDivisor(b);
    if (b < 0) {
      resultSign *= -1;
      unsignedDivisor *= -1;
    }
    unsignedDividendRescaled = checkedMultiply<R>(
        unsignedDividendRescaled, R(powerOfTen(aRescale)), "Decimal");
    R quotient = unsignedDividendRescaled / unsignedDivisor;
    R remainder = unsignedDividendRescaled % unsignedDivisor;
    if (!noRoundUp && static_cast<const B>(remainder) * 2 >= unsignedDivisor) {
      ++quotient;
    }
    r = quotient * resultSign;
    return remainder * resultSign;
  }
};

} // namespace facebook::velox
