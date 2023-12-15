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

#include <boost/multiprecision/cpp_int.hpp>

#include "velox/type/DecimalUtil.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {
using int256_t = boost::multiprecision::int256_t;

// DecimalUtil holds the utility function for Spark sql.
class DecimalUtil {
 public:
  /// Scale adjustment implementation is based on Hive's one, which is itself
  /// inspired to SQLServer's one. In particular, when a result precision is
  /// greater than {LongDecimalType::kMaxPrecision}, the corresponding scale is
  /// reduced to prevent the integral part of a result from being truncated.
  ///
  /// This method is used only when
  /// `spark.sql.decimalOperations.allowPrecisionLoss` is set to true.
  inline static std::pair<uint8_t, uint8_t> adjustPrecisionScale(
      const uint8_t rPrecision,
      const uint8_t rScale) {
    if (rPrecision <= LongDecimalType::kMaxPrecision) {
      return {rPrecision, rScale};
    } else {
      int32_t minScale = std::min(static_cast<int32_t>(rScale), 6);
      int32_t delta = rPrecision - 38;
      return {38, std::max(rScale - delta, minScale)};
    }
  }

  /// @brief Convert int256 value to int64 or int128, set overflow to true if
  /// value cannot convert to specific type.
  /// @return The converted value.
  template <
      class T,
      typename = std::enable_if_t<
          std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>>>
  inline static T convert(int256_t in, bool& overflow) {
    typedef typename std::
        conditional<std::is_same_v<T, int64_t>, uint64_t, __uint128_t>::type UT;
    T result = 0;
    constexpr auto uintMask =
        static_cast<int256_t>(std::numeric_limits<UT>::max());

    int256_t inAbs = abs(in);
    bool isNegative = in < 0;

    UT unsignResult = (inAbs & uintMask).convert_to<UT>();
    inAbs >>= sizeof(T) * 8;

    if (inAbs > 0) {
      // We've shifted in by bit of T, so nothing should be left.
      overflow = true;
    } else if (unsignResult > std::numeric_limits<T>::max()) {
      overflow = true;
    } else {
      result = static_cast<T>(unsignResult);
    }
    return isNegative ? -result : result;
  }

  // Returns the abs value of input value.
  template <class T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  FOLLY_ALWAYS_INLINE static uint64_t absValue(int64_t a) {
    return a < 0 ? static_cast<uint64_t>(-a) : static_cast<uint64_t>(a);
  }

  // Returns the abs value of input value.
  template <class T, typename = std::enable_if_t<std::is_same_v<T, int128_t>>>
  FOLLY_ALWAYS_INLINE static uint128_t absValue(int128_t a) {
    return a < 0 ? static_cast<uint128_t>(-a) : static_cast<uint128_t>(a);
  }

  /// Multiply a and b, set overflow to true if overflow. The caller should
  /// treat overflow flag first.
  template <class T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  FOLLY_ALWAYS_INLINE static int64_t
  multiply(int64_t a, int64_t b, bool& overflow) {
    int64_t value;
    overflow = __builtin_mul_overflow(a, b, &value);
    return value;
  }

  /// Multiply a and b, set overflow to true if overflow. The caller should
  /// treat overflow flag first.
  template <class T, typename = std::enable_if_t<std::is_same_v<T, int128_t>>>
  FOLLY_ALWAYS_INLINE static int128_t
  multiply(int128_t a, int128_t b, bool& overflow) {
    int128_t value;
    overflow = __builtin_mul_overflow(a, b, &value);
    return value;
  }

  template <typename A, typename B>
  inline static int32_t
  minLeadingZeros(const A& a, const B& b, uint8_t aScale, uint8_t bScale) {
    int32_t aLeadingZeros = bits::countLeadingZeros(absValue<A>(a));
    int32_t bLeadingZeros = bits::countLeadingZeros(absValue<B>(b));
    if (aScale < bScale) {
      aLeadingZeros =
          minLeadingZerosAfterScaling(aLeadingZeros, bScale - aScale);
    } else if (aScale > bScale) {
      bLeadingZeros =
          minLeadingZerosAfterScaling(bLeadingZeros, aScale - bScale);
    }
    return std::min(aLeadingZeros, bLeadingZeros);
  }

  /// Derives from Arrow BasicDecimal128 Divide.
  /// https://github.com/apache/arrow/blob/release-12.0.1-rc1/cpp/src/gandiva/precompiled/decimal_ops.cc#L350
  ///
  /// Divide a and b, set overflow to true if the result overflows. The caller
  /// should treat the overflow flag first. Using HALF_UP rounding, the digit 5
  /// is rounded up.
  /// Compute the maximum bits required to increase, if it is less than
  /// result type bits, result type is enough, if not, we should introduce
  /// int256_t as intermediate type, and then convert to real result type with
  /// overflow flag.
  template <typename R, typename A, typename B>
  inline static R divideWithRoundUp(
      R& r,
      const A& a,
      const B& b,
      uint8_t aRescale,
      bool& overflow) {
    if (b == 0) {
      overflow = true;
      return R(-1);
    }
    int resultSign = 1;
    R unsignedDividendRescaled(a);
    int aSign = 1;
    int bSign = 1;
    if (a < 0) {
      resultSign = -1;
      unsignedDividendRescaled *= -1;
      aSign = -1;
    }
    R unsignedDivisor(b);
    if (b < 0) {
      resultSign *= -1;
      unsignedDivisor *= -1;
      bSign = -1;
    }
    auto bitsRequiredAfterScaling = maxBitsRequiredAfterScaling<A>(a, aRescale);
    if (bitsRequiredAfterScaling < sizeof(R) * 8) {
      // Fast-path. The dividend fits in 128-bit after scaling too.
      overflow = __builtin_mul_overflow(
          unsignedDividendRescaled,
          R(velox::DecimalUtil::kPowersOfTen[aRescale]),
          &unsignedDividendRescaled);
      VELOX_DCHECK(!overflow);
      R quotient = unsignedDividendRescaled / unsignedDivisor;
      R remainder = unsignedDividendRescaled % unsignedDivisor;
      if (remainder * 2 >= unsignedDivisor) {
        ++quotient;
      }
      r = quotient * resultSign;
      return remainder;
    } else {
      if (aRescale > 38 && bitsRequiredAfterScaling > 255) {
        overflow = true;
        return R(-1);
      }
      int256_t aLarge = a;
      int256_t aLargeScaledUp =
          aLarge * velox::DecimalUtil::kPowersOfTen[aRescale];
      int256_t bLarge = b;
      int256_t resultLarge = aLargeScaledUp / bLarge;
      int256_t remainderLarge = aLargeScaledUp % bLarge;
      /// Since we are scaling up and then, scaling down, round-up the result
      /// (+1 for +ve, -1 for -ve), if the remainder is >= 2 * divisor.
      if (abs(2 * remainderLarge) >= abs(bLarge)) {
        /// x +ve and y +ve, result is +ve =>   (1 ^ 1)  + 1 =  0 + 1 = +1
        /// x +ve and y -ve, result is -ve =>  (-1 ^ 1)  + 1 = -2 + 1 = -1
        /// x +ve and y -ve, result is -ve =>   (1 ^ -1) + 1 = -2 + 1 = -1
        /// x -ve and y -ve, result is +ve =>  (-1 ^ -1) + 1 =  0 + 1 = +1
        resultLarge += (aSign ^ bSign) + 1;
      }

      auto result = convert<R>(resultLarge, overflow);
      if (overflow) {
        return R(-1);
      }
      r = result;
      auto remainder = convert<R>(remainderLarge, overflow);
      return remainder;
    }
  }

 private:
  /// We rely on the following formula:
  /// bits_required(x * 10^y) <= bits_required(x) + floor(log2(10^y)) + 1
  /// We precompute floor(log2(10^x)) + 1 for x = 0, 1, 2...75, 76
  static constexpr int32_t kMaxBitsRequiredIncreaseAfterScaling[] = {
      0,   4,   7,   10,  14,  17,  20,  24,  27,  30,  34,  37,  40,
      44,  47,  50,  54,  57,  60,  64,  67,  70,  74,  77,  80,  84,
      87,  90,  94,  97,  100, 103, 107, 110, 113, 117, 120, 123, 127,
      130, 133, 137, 140, 143, 147, 150, 153, 157, 160, 163, 167, 170,
      173, 177, 180, 183, 187, 190, 193, 196, 200, 203, 206, 210, 213,
      216, 220, 223, 226, 230, 233, 236, 240, 243, 246, 250, 253};

  template <typename A>
  inline static int32_t maxBitsRequiredAfterScaling(
      const A& num,
      uint8_t aRescale) {
    auto valueAbs = absValue<A>(num);
    int32_t numOccupied = sizeof(A) * 8 - bits::countLeadingZeros(valueAbs);
    return numOccupied + kMaxBitsRequiredIncreaseAfterScaling[aRescale];
  }

  /// If we have a number with 'numLeadingZeros' leading zeros, and we scale it
  /// up by 10^scale_by, this function returns the minimum number of leading
  /// zeros the result can have.
  inline static int32_t minLeadingZerosAfterScaling(
      int32_t numLeadingZeros,
      int32_t scaleBy) {
    int32_t result =
        numLeadingZeros - kMaxBitsRequiredIncreaseAfterScaling[scaleBy];
    return result;
  }
};
} // namespace facebook::velox::functions::sparksql
