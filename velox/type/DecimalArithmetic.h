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
#include <type_traits>
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
/// Computed rather than tabulated because device code cannot read a table.
/// A table cannot be a static member or a static local there: nvcc
/// reports "identifier is undefined in device code" for a runtime index into
/// static storage, --expt-relaxed-constexpr does not lift it, and a static
/// local is ill-formed inside a constexpr function before C++23. That leaves a
/// function-local table, which every calling kernel then pays for -- 39 int128
/// stores and a local-memory load, 624 bytes of stack frame per thread
/// measured with -Xptxas -v on sm_80. Squaring costs at most six multiplies
/// and no memory at all.
///
/// It sits outside the class because a class's own static members cannot call
/// one of its constexpr member functions while the class is still incomplete;
/// DecimalArithmetic::kPowersOfTen is derived from this at compile time.
VELOX_GPU_COMPATIBLE constexpr int128_t decimalPowerOfTen(uint8_t exponent) {
  int128_t result = 1;
  int128_t base = 10;
  while (exponent > 0) {
    if (exponent & 1) {
      result *= base;
    }
    exponent >>= 1;
    // Squaring 10^32 would pass int128's range, and the loop has no use for
    // base once the last bit is consumed.
    if (exponent > 0) {
      base *= base;
    }
  }
  return result;
}

// Anchors on the algorithm: the bottom, the short-decimal boundary, and the
// widest value it has to reach.
static_assert(decimalPowerOfTen(0) == 1);
static_assert(
    decimalPowerOfTen(kMaxShortDecimalPrecision) == 1'000'000'000'000'000'000);
static_assert(
    decimalPowerOfTen(kMaxLongDecimalPrecision) ==
    1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
        (int128_t)100);

} // namespace detail

struct DecimalArithmetic {
  static constexpr uint8_t kMaxShortPrecision =
      detail::kMaxShortDecimalPrecision;
  static constexpr uint8_t kMaxLongPrecision = detail::kMaxLongDecimalPrecision;

  /// 10^exponent. exponent must be <= kMaxLongPrecision.
  ///
  /// Indexes the table on the host and computes on the device, where the
  /// table is not addressable. The two agree by construction: kPowersOfTen is
  /// filled by the same function the device branch calls. Carrying the split
  /// here is what lets one call() body compile for both.
  VELOX_GPU_COMPATIBLE static constexpr int128_t powerOfTen(uint8_t exponent) {
#ifdef __CUDA_ARCH__
    return detail::decimalPowerOfTen(exponent);
#else
    return kPowersOfTen[exponent];
#endif
  }

  /// kPowersOfTen[i] == 10^i, derived from detail::decimalPowerOfTen() so the
  /// literals are written once. Host-only; device code goes through
  /// powerOfTen().
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

  /// Magnitude of a decimal's unscaled value, as an unsigned type wide enough
  /// to hold it. Negating the minimum of a signed type is undefined, so the
  /// cast happens before the negation.
  ///
  /// Lives here rather than on DecimalUtil because callers that only do
  /// arithmetic on unscaled values -- sparksql/DecimalUtil.h among them --
  /// would otherwise have to reach the runtime type system for it. DecimalUtil
  /// derives from this, so DecimalUtil::absValue still resolves.
  template <class T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  VELOX_GPU_COMPATIBLE static uint64_t absValue(int64_t a) {
    return a < 0 ? -static_cast<uint64_t>(a) : static_cast<uint64_t>(a);
  }

  template <class T, typename = std::enable_if_t<std::is_same_v<T, int128_t>>>
  VELOX_GPU_COMPATIBLE static __uint128_t absValue(int128_t a) {
    return a < 0 ? -static_cast<__uint128_t>(a) : static_cast<__uint128_t>(a);
  }

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
