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

#include <charconv>
#include <string>
#include "velox/common/base/CheckedArithmetic.h"

#ifdef _MSC_VER
#include "velox/common/base/windows/BuiltinUtil.h"
#include "velox/type/HugeInt.h"
#endif
#include "velox/common/base/CountBits.h"
#include "velox/common/base/Doubles.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Nulls.h"
#include "velox/common/base/Status.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// A static class that holds helper functions for DECIMAL type.
class DecimalUtil {
 public:
#ifdef _WIN32
  static const int128_t kPowersOfTen[LongDecimalType::kMaxPrecision + 1];
#else
  static constexpr int128_t kPowersOfTen[LongDecimalType::kMaxPrecision + 1] = {
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
#endif

#ifdef _WIN32
  static const int128_t kLongDecimalMin;
  static const int128_t kLongDecimalMax;
  static const int128_t kShortDecimalMin;
  static const int128_t kShortDecimalMax;
#else
  static constexpr int128_t kLongDecimalMin =
      -kPowersOfTen[LongDecimalType::kMaxPrecision] + 1;
  static constexpr int128_t kLongDecimalMax =
      kPowersOfTen[LongDecimalType::kMaxPrecision] - 1;
  static constexpr int128_t kShortDecimalMin =
      -kPowersOfTen[ShortDecimalType::kMaxPrecision] + 1;
  static constexpr int128_t kShortDecimalMax =
      kPowersOfTen[ShortDecimalType::kMaxPrecision] - 1;
#endif

  /// Scale threshold for scientific notation.
  static constexpr int32_t kMinScientificNotationScale = 6;

  static constexpr uint64_t kInt64Mask = ~(static_cast<uint64_t>(1) << 63);
  static constexpr uint128_t kInt128Mask = (static_cast<uint128_t>(1) << 127);

  FOLLY_ALWAYS_INLINE static void valueInRange(int128_t value) {
    VELOX_USER_CHECK(
        (value >= kLongDecimalMin && value <= kLongDecimalMax),
        "Decimal overflow. Value '{}' is not in the range of Decimal Type",
        value);
  }

  // Returns true if the precision can represent the value.
  template <typename T>
  FOLLY_ALWAYS_INLINE static bool valueInPrecisionRange(
      T value,
      uint8_t precision) {
    const T powerOfTen = static_cast<T>(kPowersOfTen[precision]);
    return value < powerOfTen && value > -powerOfTen;
  }

  /// Helper function to convert a decimal value to string.
  static std::string toString(int128_t value, const Type& type);

  // TODO Remove.
  static std::string toString(int128_t value, const TypePtr& type) {
    return toString(value, *type);
  }

  template <typename T>
  inline static void fillDecimals(
      T* decimals,
      const uint64_t* nullsPtr,
      const T* values,
      const int64_t* scales,
      int32_t numValues,
      int32_t targetScale) {
    for (int32_t i = 0; i < numValues; i++) {
      if (!nullsPtr || !bits::isBitNull(nullsPtr, i)) {
        int32_t currentScale = scales[i];
        T value = values[i];
        if constexpr (std::is_same_v<T, std::int64_t>) { // Short Decimal
          if (targetScale > currentScale &&
              targetScale - currentScale <= ShortDecimalType::kMaxPrecision) {
            value *= static_cast<T>(kPowersOfTen[targetScale - currentScale]);
          } else if (
              targetScale < currentScale &&
              currentScale - targetScale <= ShortDecimalType::kMaxPrecision) {
            value /= static_cast<T>(kPowersOfTen[currentScale - targetScale]);
          } else if (targetScale != currentScale) {
            VELOX_FAIL("Decimal scale out of range");
          }
        } else { // Long Decimal
          if (targetScale > currentScale) {
            while (targetScale > currentScale) {
              int32_t scaleAdjust = std::min<int32_t>(
                  ShortDecimalType::kMaxPrecision, targetScale - currentScale);
              value *= kPowersOfTen[scaleAdjust];
              currentScale += scaleAdjust;
            }
          } else if (targetScale < currentScale) {
            while (currentScale > targetScale) {
              int32_t scaleAdjust = std::min<int32_t>(
                  ShortDecimalType::kMaxPrecision, currentScale - targetScale);
              value /= kPowersOfTen[scaleAdjust];
              currentScale -= scaleAdjust;
            }
          }
        }
        decimals[i] = value;
      }
    }
  }

  template <typename TInput, typename TOutput>
  inline static Status rescaleWithRoundUp(
      TInput inputValue,
      int fromPrecision,
      int fromScale,
      int toPrecision,
      int toScale,
      TOutput& output) {
    int128_t rescaledValue = inputValue;
    auto scaleDifference = toScale - fromScale;
    bool isOverflow = false;
    if (scaleDifference >= 0) {
      VELOX_RETURN_IF(
          scaleDifference > LongDecimalType::kMaxPrecision,
          Status::UserError(
              "Decimal scale difference is too large: {} vs max {}.",
              scaleDifference,
              LongDecimalType::kMaxPrecision));
#ifdef _MSC_VER
      if constexpr (std::is_same_v<TInput, int128_t>) {
        isOverflow = windows::builtin_mul_overflow(
            rescaledValue,
            DecimalUtil::kPowersOfTen[scaleDifference],
            &rescaledValue);
      } else {
        isOverflow = __builtin_mul_overflow(
            rescaledValue,
            DecimalUtil::kPowersOfTen[scaleDifference],
            &rescaledValue);
      }
#else
      isOverflow = __builtin_mul_overflow(
          rescaledValue,
          DecimalUtil::kPowersOfTen[scaleDifference],
          &rescaledValue);
#endif
    } else {
      scaleDifference = -scaleDifference;
      if (scaleDifference > LongDecimalType::kMaxPrecision) {
        rescaledValue = 0;
      } else {
        const auto scalingFactor = DecimalUtil::kPowersOfTen[scaleDifference];
#ifdef _MSC_VER
        if constexpr (std::is_same_v<TInput, int128_t>) {
          // Manual division and remainder for int128_t on MSVC.
          int128_t quotient = rescaledValue / scalingFactor;
          int128_t remainder = inputValue - (quotient * scalingFactor);
          int128_t halfScaling = scalingFactor / int128_t(2);
          int128_t zero = Int128(0);
          int128_t negHalfScaling = -halfScaling;
          rescaledValue = quotient;
          if (inputValue >= zero && remainder >= halfScaling) {
            ++rescaledValue;
          } else if (remainder <= negHalfScaling) {
            --rescaledValue;
          }
        } else {
          // Cast scalingFactor to TInput for operator compatibility.
          const TInput scalingFactorT = static_cast<TInput>(scalingFactor);
          rescaledValue /= scalingFactorT;
          TInput remainder = inputValue % scalingFactorT;
          if (inputValue >= 0 && remainder >= scalingFactorT / 2) {
            ++rescaledValue;
          } else if (remainder <= -scalingFactorT / 2) {
            --rescaledValue;
          }
        }
#else
        VELOX_DCHECK_LT(scaleDifference, std::size(DecimalUtil::kPowersOfTen));
        rescaledValue /= scalingFactor;
        int128_t remainder = inputValue % scalingFactor;
        if (inputValue >= 0 && remainder >= scalingFactor / 2) {
          ++rescaledValue;
        } else if (remainder <= -scalingFactor / 2) {
          --rescaledValue;
        }
#endif
      }
    }
    // Check overflow.
    if (!valueInPrecisionRange(rescaledValue, toPrecision) || isOverflow) {
      return Status::UserError(
          "Cannot cast DECIMAL '{}' to DECIMAL({}, {})",
          DecimalUtil::toString(inputValue, DECIMAL(fromPrecision, fromScale)),
          toPrecision,
          toScale);
    }
    output = static_cast<TOutput>(rescaledValue);
    return Status::OK();
  }

  template <typename TInput, typename TOutput>
  inline static std::optional<TOutput>
  rescaleInt(TInput inputValue, int toPrecision, int toScale) {
    int128_t rescaledValue = static_cast<int128_t>(inputValue);
#ifdef _MSC_VER
    if constexpr (std::is_same_v<TInput, int128_t>) {
      bool isOverflow = windows::builtin_mul_overflow(
          rescaledValue, DecimalUtil::kPowersOfTen[toScale], &rescaledValue);
      // Check overflow.
      if (!valueInPrecisionRange(rescaledValue, toPrecision) || isOverflow) {
        VELOX_USER_FAIL(
            "Cannot cast {} '{}' to DECIMAL({}, {})",
            SimpleTypeTrait<TInput>::name,
            inputValue,
            toPrecision,
            toScale);
      }
    } else {
      bool isOverflow = __builtin_mul_overflow(
          rescaledValue, DecimalUtil::kPowersOfTen[toScale], &rescaledValue);
      // Check overflow.
      if (!valueInPrecisionRange(rescaledValue, toPrecision) || isOverflow) {
        VELOX_USER_FAIL(
            "Cannot cast {} '{}' to DECIMAL({}, {})",
            SimpleTypeTrait<TInput>::name,
            inputValue,
            toPrecision,
            toScale);
      }
    }
#else
    bool isOverflow = __builtin_mul_overflow(
        rescaledValue, DecimalUtil::kPowersOfTen[toScale], &rescaledValue);
    // Check overflow.
    if (!valueInPrecisionRange(rescaledValue, toPrecision) || isOverflow) {
      VELOX_USER_FAIL(
          "Cannot cast {} '{}' to DECIMAL({}, {})",
          SimpleTypeTrait<TInput>::name,
          inputValue,
          toPrecision,
          toScale);
    }
#endif
    return static_cast<TOutput>(rescaledValue);
  }

  /// Rescales a decimal value to the target precision and scale with rounding.
  ///
  /// @tparam T The type of the decimal value.
  /// @param value The input decimal value.
  /// @param fromScale The original scale of the input value.
  /// @param toPrecision The target precision.
  /// @param toScale The target scale.
  /// @param output The output decimal value after rescaling.
  /// @return A Status indicating success or failure.
  template <typename T>
  static Status rescaleDecimal(
      T value,
      const int fromScale,
      const int toPrecision,
      const int toScale,
      T& output) {
    if (toScale > fromScale) {
#ifdef _MSC_VER
      if constexpr (std::is_same_v<T, int128_t>) {
        int128_t result;
        int128_t scaleFactor = kPowersOfTen[toScale - fromScale];
        bool isOverflow = windows::builtin_mul_overflow(
            value, scaleFactor, &result);
        if (isOverflow) {
          return Status::UserError("Result overflows.");
        }
        value = result;
      } else {
        T scaleFactor = static_cast<T>(kPowersOfTen[toScale - fromScale]);
        const bool isOverflow = __builtin_mul_overflow(
            value, scaleFactor, &value);
        if (isOverflow) {
          return Status::UserError("Result overflows.");
        }
      }
#else
      T scaleFactor = static_cast<T>(kPowersOfTen[toScale - fromScale]);
      const bool isOverflow = __builtin_mul_overflow(
          value, scaleFactor, &value);
      if (isOverflow) {
        return Status::UserError("Result overflows.");
      }
#endif
    } else if (toScale < fromScale) {
      const auto scalingFactor = kPowersOfTen[fromScale - toScale];
#ifdef _MSC_VER
      if constexpr (std::is_same_v<T, int128_t>) {
        // MSVC doesn't support int128_t operators directly
        // Compute quotient and remainder manually
        int128_t quotient = value / scalingFactor;
        int128_t remainder = value - (quotient * scalingFactor);
        int128_t halfScaling = scalingFactor / int128_t(2);
        int128_t zero = Int128(0);
        int128_t negHalfScaling = -halfScaling;
        
        // Check for rounding
        bool shouldRoundUp = (quotient >= zero) && (remainder >= halfScaling);
        bool shouldRoundDown = (remainder <= negHalfScaling);
        
        if (shouldRoundUp) {
          value = quotient + Int128(1);
        } else if (shouldRoundDown) {
          value = quotient - Int128(1);
        } else {
          value = quotient;
        }
      } else {
        const T scalingFactorT = static_cast<T>(scalingFactor);
        const T remainder = value % scalingFactorT;
        value /= scalingFactorT;
        if (value >= 0 && remainder >= scalingFactorT / 2) {
          ++value;
        } else if (remainder <= -scalingFactorT / 2) {
          --value;
        }
      }
#else
      const T scalingFactorT = static_cast<T>(scalingFactor);
      const T remainder = value % scalingFactorT;
      value /= scalingFactorT;
      if (value >= 0 && remainder >= scalingFactorT / 2) {
        ++value;
      } else if (remainder <= -scalingFactorT / 2) {
        --value;
      }
#endif
    }

    if (!valueInPrecisionRange<T>(value, toPrecision)) {
      return Status::UserError(
          "Result cannot fit in the given precision {}.", toPrecision);
    }
    output = value;
    return Status::OK();
  }

  /// Rescales a floating point value to decimal value of given precision and
  /// scale. Returns error status if fails.
  /// @tparam TInput Either float or double.
  /// @tparam TOutput Either int64_t or int128_t.
  template <typename TInput, typename TOutput>
  inline static Status rescaleFloatingPoint(
      TInput value,
      int precision,
      int scale,
      TOutput& output) {
    if (!std::isfinite(value)) {
      return Status::UserError("The input value should be finite.");
    }

    TInput maxValue;
    if constexpr (std::is_same_v<TOutput, int64_t>) {
      maxValue = kMaxDoubleBelowInt64Max;
    } else {
      maxValue = kMaxDoubleBelowInt128Max;
    }

#ifdef _MSC_VER
    // MSVC: Handle Int128 comparison. operator int64_t() only returns low_,
    // which is 0 for Int128::min(). Use operator double() for the full range.
    TInput minValue;
    if constexpr (std::is_same_v<TOutput, int128_t>) {
      minValue = static_cast<TInput>(
          static_cast<double>(std::numeric_limits<TOutput>::min()));
    } else {
      minValue = static_cast<TInput>(std::numeric_limits<TOutput>::min());
    }
    if (value <= minValue || 
        static_cast<long double>(value) > static_cast<long double>(maxValue)) {
      return Status::UserError("Result overflows.");
    }
#else
    if (value <= std::numeric_limits<TOutput>::min() || 
        static_cast<long double>(value) > static_cast<long double>(maxValue)) {
      return Status::UserError("Result overflows.");
    }
#endif

    uint8_t digits;
    if constexpr (std::is_same_v<TInput, float>) {
      // A float provides nearly 7 precise digits.
      digits = 7;
    } else {
      // A double provides from 15 to 17 decimal digits, so at least 15 digits
      // are precise.
      digits = 15;
    }

    // Calculate the precise fractional digits.
    const auto integralValue = static_cast<uint128_t>(std::abs(value));
    const auto integralDigits =
        integralValue == 0 ? 0 : countDigits(integralValue);
    const auto fractionDigits = std::max(digits - integralDigits, 0);

    // Scales up the input value with all the precise fractional digits kept.
    // Convert value as long double type because 1) double * int128_t returns
    // int128_t and fractional digits are lost. 2) we could also convert the
    // int128_t value as double to avoid 'double * int128_t', but double
    // multiplication gives inaccurate result on large numbers. For example,
    // -3333030000000000000 * 1e3 = -3333030000000000065536. No need to
    // consider the result becoming infinite as DOUBLE_MAX * 10^38 <
    // LONG_DOUBLE_MAX.
#ifdef _MSC_VER
    // MSVC doesn't support int128_t to long double cast, convert via double
    long double valueAsLongDouble;
    if constexpr (std::is_same_v<decltype(value), double> || std::is_same_v<decltype(value), float>) {
      valueAsLongDouble = static_cast<long double>(value);
    } else {
      // Convert integral types via double
      valueAsLongDouble = static_cast<long double>(static_cast<double>(value));
    }
    long double scaleAsLongDouble = static_cast<long double>(static_cast<int64_t>(DecimalUtil::kPowersOfTen[fractionDigits]));
    long double scaledValue = std::round(valueAsLongDouble * scaleAsLongDouble);
#else
    long double scaledValue = std::round(
        (long double)value * DecimalUtil::kPowersOfTen[fractionDigits]);
#endif
#ifdef _MSC_VER
    if constexpr (std::is_same_v<TOutput, int128_t>) {
      // MSVC: folly::tryTo doesn't support int128_t, and long double to int128_t cast not supported
      // Convert to int64_t first if within range, otherwise construct int128_t from parts
      if (scaledValue > static_cast<long double>(std::numeric_limits<int64_t>::max()) ||
          scaledValue < static_cast<long double>(std::numeric_limits<int64_t>::min())) {
        // For values outside int64_t range, construct Int128 from high/low parts
        long double absValue = std::abs(scaledValue);
        constexpr long double k2pow64 = 18446744073709551616.0L; // 2^64
        int64_t high = static_cast<int64_t>(absValue / k2pow64);
        uint64_t low = static_cast<uint64_t>(
            absValue - static_cast<long double>(high) * k2pow64);
        int128_t resultValue(high, low);
        if (scaledValue < 0) {
          resultValue = -resultValue;
        }
        return rescaleDecimal<TOutput>(resultValue, fractionDigits, precision, scale, output);
      } else {
        int128_t resultValue = static_cast<int128_t>(static_cast<int64_t>(scaledValue));
        return rescaleDecimal<TOutput>(resultValue, fractionDigits, precision, scale, output);
      }
    } else {
#endif
      const auto result = folly::tryTo<TOutput>(scaledValue);
      if (result.hasError()) {
        return Status::UserError("Result overflows.");
      }
      return rescaleDecimal<TOutput>(
          result.value(), fractionDigits, precision, scale, output);
#ifdef _MSC_VER
    }
#endif
  }

  template <typename R, typename A, typename B>
  inline static R divideWithRoundUp(
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
        unsignedDividendRescaled,
        R(DecimalUtil::kPowersOfTen[aRescale]),
        "Decimal");
#ifdef _MSC_VER
    // On Windows, if R is a smaller type than B (e.g., R=int64_t, B=int128_t),
    // the / and % operators don't work directly, so we need to cast
    if constexpr (sizeof(R) < sizeof(B)) {
      B dividendB = static_cast<B>(unsignedDividendRescaled);
      R quotient = static_cast<R>(dividendB / unsignedDivisor);
      R remainder = static_cast<R>(dividendB % unsignedDivisor);
#else
    R quotient = unsignedDividendRescaled / unsignedDivisor;
    R remainder = unsignedDividendRescaled % unsignedDivisor;
#endif
    if (!noRoundUp && static_cast<const B>(remainder) * 2 >= unsignedDivisor) {
      ++quotient;
    }
    r = quotient * resultSign;
    return remainder * resultSign;
#ifdef _MSC_VER
    } else {
      R quotient = unsignedDividendRescaled / static_cast<R>(unsignedDivisor);
      R remainder = unsignedDividendRescaled % static_cast<R>(unsignedDivisor);
      if (!noRoundUp && static_cast<const B>(remainder) * 2 >= unsignedDivisor) {
        ++quotient;
      }
      r = quotient * resultSign;
      return remainder * resultSign;
    }
#endif
  }

  /// Returns the max required size to convert the decimal of this precision and
  /// scale to varchar. A varchar's size is estimated with unscaled value
  /// digits, dot, leading zero, and possible minus sign.
  static int32_t maxStringViewSize(int precision, int scale);

#ifdef _MSC_VER
  // Helper function to convert int128_t to string for MSVC
  static inline size_t int128ToChars(char* buffer, size_t bufferSize, int128_t value) {
    if (value == 0) {
      buffer[0] = '0';
      return 1;
    }
    
    bool negative = value < 0;
    if (negative) {
      value = -value;
    }
    
    char temp[40]; // Enough for 128-bit number
    size_t pos = 0;
    
    while (value > 0) {
      temp[pos++] = '0' + static_cast<char>(static_cast<int64_t>(value % 10));
      value = value / 10;
    }
    
    size_t writePos = 0;
    if (negative) {
      buffer[writePos++] = '-';
    }
    
    // Reverse the digits
    for (size_t i = pos; i > 0; --i) {
      buffer[writePos++] = temp[i - 1];
    }
    
    return writePos;
  }
  
  // Helper function to convert uint128_t to string for MSVC
  static inline size_t uint128ToChars(char* buffer, size_t bufferSize, uint128_t value) {
    if (value == 0) {
      buffer[0] = '0';
      return 1;
    }
    
    char temp[40]; // Enough for 128-bit number
    size_t pos = 0;
    
    while (value > 0) {
      temp[pos++] = '0' + static_cast<char>(static_cast<uint64_t>(value % 10));
      value = value / 10;
    }
    
    // Reverse the digits
    size_t writePos = 0;
    for (size_t i = pos; i > 0; --i) {
      buffer[writePos++] = temp[i - 1];
    }
    
    return writePos;
  }
#endif

  /// @brief Convert the unscaled value of a decimal to string and write to raw
  /// string buffer from start position.
  /// @tparam T The type of input value.
  /// @param unscaledValue The input unscaled value.
  /// @param scale The scale of decimal.
  /// @param maxSize The estimated max size of string.
  /// @param startPosition The start position to write from.
  /// @param isScientific Whether to format small magnitude decimals using
  /// scientific notation (Spark-compatible). When true, absolute values less
  /// than 1e-6 are formatted in scientific notation. For example:
  /// - With scale=20 and value=1: "1E-20" (scientific) vs
  /// "0.00000000000000000001" (normal)
  /// @return The number of characters written starting from startPosition.
  template <typename T>
  static size_t castToString(
      T unscaledValue,
      int32_t scale,
      int32_t maxSize,
      char* const startPosition,
      bool isScientific = false) {
    char* writePosition = startPosition;
    if (unscaledValue == 0) {
      *writePosition++ = '0';
      if (isScientific && scale > kMinScientificNotationScale) {
        *writePosition++ = 'E';
        auto exp =
            std::to_chars(writePosition, startPosition + maxSize, -scale);
        VELOX_DCHECK_EQ(
            exp.ec,
            std::errc(),
            "Failed to cast exponent value to varchar: {}",
            std::make_error_code(exp.ec).message());
        VELOX_DCHECK_LE(exp.ptr - startPosition, maxSize);
        writePosition = exp.ptr;
      } else if (scale > 0) {
        *writePosition++ = '.';
        // Append trailing zeros.
        std::memset(writePosition, '0', scale);
        writePosition += scale;
      }
    } else {
      if (unscaledValue < 0) {
        *writePosition++ = '-';
        unscaledValue = -unscaledValue;
      }
#ifdef _MSC_VER
      // For int128_t on MSVC, use custom conversion.
      if constexpr (std::is_same_v<T, int128_t>) {
        int128_t integralPart = unscaledValue / DecimalUtil::kPowersOfTen[scale];
        size_t len = int128ToChars(writePosition, maxSize, integralPart);
        writePosition += len;

        if (scale > 0) {
          *writePosition++ = '.';
          uint128_t fraction = static_cast<uint128_t>(unscaledValue) %
              static_cast<uint128_t>(DecimalUtil::kPowersOfTen[scale]);
          // Append leading zeros.
          int numLeadingZeros = std::max(scale - countDigits(fraction), 0);
          std::memset(writePosition, '0', numLeadingZeros);
          writePosition += numLeadingZeros;
          // Append remaining fraction digits.
          size_t fracLen = uint128ToChars(writePosition, maxSize, fraction);
          writePosition += fracLen;
        }
      } else {
        // For non-int128_t types on MSVC, handle division with int128_t
        // scaleFactor.
        auto scaleFactor = DecimalUtil::kPowersOfTen[scale];
        T integralPart = static_cast<T>(
            static_cast<int64_t>(unscaledValue) /
            static_cast<int64_t>(scaleFactor));
        auto [position, errorCode] =
            std::to_chars(writePosition, writePosition + maxSize, integralPart);
        VELOX_DCHECK_EQ(
            errorCode,
            std::errc(),
            "Failed to cast decimal to varchar: {}",
            std::make_error_code(errorCode).message());
        writePosition = position;

        if (scale > 0) {
          *writePosition++ = '.';
          uint64_t fraction = static_cast<uint64_t>(
              static_cast<int64_t>(unscaledValue) %
              static_cast<int64_t>(scaleFactor));
          // Append leading zeros.
          int numLeadingZeros = std::max(scale - countDigits(fraction), 0);
          std::memset(writePosition, '0', numLeadingZeros);
          writePosition += numLeadingZeros;
          // Append remaining fraction digits.
          auto result =
              std::to_chars(writePosition, writePosition + maxSize, fraction);
          VELOX_DCHECK_EQ(
              result.ec,
              std::errc(),
              "Failed to cast decimal to varchar: {}",
              std::make_error_code(result.ec).message());
          writePosition = result.ptr;
        }
      }
#else
      if (isScientific) {
        if (scale >= kMinScientificNotationScale &&
            unscaledValue < DecimalUtil::kPowersOfTen
                                [scale - kMinScientificNotationScale]) {
          // Use scientific notation if the absolute value is less than 1e-6.
          // This is consistent with Spark's behavior.
          const auto digits = countDigits(unscaledValue);
          auto coefficientBuf = std::vector<char>(digits);
          const auto coefficient = std::to_chars(
              coefficientBuf.data(),
              coefficientBuf.data() + digits,
              unscaledValue);
          VELOX_DCHECK_EQ(
              coefficient.ec,
              std::errc(),
              "Failed to cast coefficient to varchar.");
          VELOX_DCHECK_EQ(coefficient.ptr, coefficientBuf.data() + digits);
          *writePosition++ = coefficientBuf[0];
          if (coefficient.ptr - coefficientBuf.data() > 1) {
            *writePosition++ = '.';
            size_t toCopy = digits - 1;
            std::memcpy(writePosition, coefficientBuf.data() + 1, toCopy);
            writePosition += toCopy;
          }
          *writePosition++ = 'E';
          const auto adjusted = digits - 1 - scale;
          auto exp = std::to_chars(
              writePosition, writePosition + maxSize - digits - 2, adjusted);
          VELOX_DCHECK_EQ(
              exp.ec,
              std::errc(),
              "Failed to cast exponent value to varchar: {}",
              std::make_error_code(exp.ec).message());
          writePosition = exp.ptr;
          return writePosition - startPosition;
        }
      }
      auto [position, errorCode] = std::to_chars(
          writePosition,
          writePosition + maxSize,
          unscaledValue / DecimalUtil::kPowersOfTen[scale]);
      VELOX_DCHECK_EQ(
          errorCode,
          std::errc(),
          "Failed to cast decimal to varchar: {}",
          std::make_error_code(errorCode).message());
      writePosition = position;

      if (scale > 0) {
        *writePosition++ = '.';
        uint128_t fraction = unscaledValue % DecimalUtil::kPowersOfTen[scale];
        // Append leading zeros.
        int numLeadingZeros = std::max(scale - countDigits(fraction), 0);
        std::memset(writePosition, '0', numLeadingZeros);
        writePosition += numLeadingZeros;
        // Append remaining fraction digits.
        auto result =
            std::to_chars(writePosition, writePosition + maxSize, fraction);
        VELOX_DCHECK_EQ(
            result.ec,
            std::errc(),
            "Failed to cast decimal to varchar: {}",
            std::make_error_code(result.ec).message());
        writePosition = result.ptr;
      }
#endif
    }
    return writePosition - startPosition;
  }

  /*
   * sum up and return overflow/underflow.
   */
  inline static int64_t addUnsignedValues(
      int128_t& sum,
      int128_t lhs,
      int128_t rhs,
      bool isResultNegative) {
#ifdef _MSC_VER
    UInt128 unsignedSum = UInt128(lhs) + UInt128(rhs);
    // Ignore overflow value.
    UInt128 overflow_mask = kOverflowMultiplier;
    UInt128 masked = unsignedSum & ~overflow_mask;
    sum = int128_t(static_cast<int64_t>(masked.high()), masked.low());
    sum = isResultNegative ? -sum : sum;
    return static_cast<int64_t>(unsignedSum >> static_cast<uint32_t>(127));
#else
    __uint128_t unsignedSum = (__uint128_t)lhs + (__uint128_t)rhs;
    // Ignore overflow value.
    sum = (int128_t)unsignedSum & ~kOverflowMultiplier;
    sum = isResultNegative ? -sum : sum;
    return (unsignedSum >> 127);
#endif
  }

  /// Adds two signed 128-bit numbers (int128_t), calculates the sum, and
  /// returns the overflow. It can be used to track the number of overflow when
  /// adding a batch of input numbers. It takes lhs and rhs as input, and stores
  /// their sum in result. overflow == 1 indicates upward overflow. overflow ==
  /// -1 indicates downward overflow. overflow == 0 indicates no overflow.
  /// Adding negative and non-negative numbers never overflows, so we can
  /// directly add them. Adding two negative or two positive numbers may
  /// overflow. To add numbers that may overflow, first convert both numbers to
  /// unsigned 128-bit number (uint128_t), and perform the addition. The highest
  /// bits in the result indicates overflow. Adjust the signs of sum and
  /// overflow based on the signs of the inputs. The caller must sum up overflow
  /// values and call adjustSumForOverflow after processing all inputs.
  inline static int64_t
  addWithOverflow(int128_t& result, int128_t lhs, int128_t rhs) {
    bool isLhsNegative = lhs < 0;
    bool isRhsNegative = rhs < 0;
    int64_t overflow = 0;
    if (isLhsNegative == isRhsNegative) {
      // Both inputs of same time.
      if (isLhsNegative) {
        // Both negative, ignore signs and add.
        VELOX_DCHECK_NE(lhs, std::numeric_limits<int128_t>::min());
        VELOX_DCHECK_NE(rhs, std::numeric_limits<int128_t>::min());
        overflow = addUnsignedValues(result, -lhs, -rhs, true);
        overflow = -overflow;
      } else {
        overflow = addUnsignedValues(result, lhs, rhs, false);
      }
    } else {
      // If one of them is negative, use addition.
      result = lhs + rhs;
    }
    return overflow;
  }

  /// Corrects the sum result calculated using addWithOverflow. Since the sum
  /// calculated by addWithOverflow only retains the lower 127 bits,
  /// it may miss one calculation of +(1 << 127) or -(1 << 127).
  /// Therefore, we need to make the following adjustments:
  /// 1. If overflow = 1 && sum < 0, the calculation missed +(1 << 127).
  /// Add 1 << 127 to the sum.
  /// 2. If overflow = -1 && sum > 0, the calculation missed -(1 << 127).
  /// Subtract 1 << 127 to the sum.
  /// If an overflow indeed occurs and the result cannot be adjusted,
  /// it will return std::nullopt.
  inline static std::optional<int128_t> adjustSumForOverflow(
      int128_t sum,
      int64_t overflow) {
    // Value is valid if the conditions below are true.
    if ((overflow == 1 && sum < 0) || (overflow == -1 && sum > 0)) {
#ifdef _MSC_VER
      // On Windows, construct int128_t from UInt128 using high() and low()
      return int128_t(
          int128_t(static_cast<int64_t>(DecimalUtil::kOverflowMultiplier.high()), DecimalUtil::kOverflowMultiplier.low()) * overflow + sum);
#else
      // On Linux, kOverflowMultiplier is __uint128_t which can be used directly
      return int128_t(kOverflowMultiplier) * overflow + sum;
#endif
    }
    if (overflow != 0) {
      // The actual overflow occurred.
      return std::nullopt;
    }

    return sum;
  }

  /// avg = (sum + overflow * kOverflowMultiplier) / count
  static void
  computeAverage(int128_t& avg, int128_t sum, int64_t count, int64_t overflow);

  /// Origins from java side BigInteger#bitLength.
  ///
  /// Returns the number of bits in the minimal two's-complement
  /// representation of this BigInteger, <em>excluding</em> a sign bit.
  /// For positive BigIntegers, this is equivalent to the number of bits in
  /// the ordinary binary representation.  For zero this method returns
  /// {@code 0}.  (Computes {@code (ceil(log2(this < 0 ? -this : this+1)))}.)
  ///
  /// @return number of bits in the minimal two's-complement
  ///         representation of this BigInteger, <em>excluding</em> a sign bit.
  static int32_t getByteArrayLength(int128_t value);

  /// This method return the same result with the BigInterger#toByteArray()
  /// method in Java side.
  ///
  /// Returns a byte array containing the two's-complement representation of
  /// this BigInteger. The byte array will be in big-endian byte-order: the most
  /// significant byte is in the zeroth element. The array will contain the
  /// minimum number of bytes required to represent this BigInteger, including
  /// at least one sign bit, which is (ceil((this.bitLength() + 1)/8)).
  ///
  /// @return The length of out.
  static int32_t toByteArray(int128_t value, char* out);

  /// Reverse byte order of an int128_t if native byte-order is little endian.
  /// If native byte-order is big endian, the value will be unchanged. This
  /// is similar to folly::Endian::big(), which does not support int128_t.
  ///
  /// \return A value with reversed byte-order for little endian platforms.
  inline static int128_t bigEndian(int128_t value) {
    if (folly::kIsLittleEndian) {
      auto upper = folly::Endian::big(HugeInt::upper(value));
      auto lower = folly::Endian::big(HugeInt::lower(value));
      return HugeInt::build(lower, upper);
    } else {
      return value;
    }
  }

  /// Converts string view to decimal value of given precision and scale.
  /// Derives from Arrow function DecimalFromString. Arrow implementation:
  /// https://github.com/apache/arrow/blob/56c0e2f508fdc5137d6734b406634386f9284a52/cpp/src/arrow/util/decimal.cc#L862.
  ///
  /// Firstly, it parses the varchar to DecimalComponents which contains the
  /// message that can represent a decimal value. Secondly, processes the
  /// exponent to get the scale. Thirdly, compute the rescaled value. Returns
  /// status for the outcome of computing.
  template <typename T>
  static Status castFromString(
      const StringView s,
      int32_t toPrecision,
      int32_t toScale,
      T& decimalValue) {
    int32_t parsedPrecision = 0;
    int32_t parsedScale = 0;
    int128_t out = 0;
    VELOX_RETURN_NOT_OK(parseStringToDecimalComponents(
        s, toScale, parsedPrecision, parsedScale, out));


    const auto status = rescaleWithRoundUp<int128_t, T>(
        out,
        std::min(
            static_cast<uint8_t>(parsedPrecision),
            LongDecimalType::kMaxPrecision),
        parsedScale,
        toPrecision,
        toScale,
        decimalValue);
    if (!status.ok()) {
      return Status::UserError("Value too large.");
    }
    return status;
  }

#ifdef _MSC_VER
  static const UInt128 kOverflowMultiplier;
#else
  static constexpr __uint128_t kOverflowMultiplier = ((__uint128_t)1 << 127);
#endif

 private:
  // Parses the string view to decimal components, which contains the
  // unscaled value, precision, and scale. The parsed precision and scale are
  // returned through the reference parameters. The unscaled value is returned
  // through the out parameter.
  static Status parseStringToDecimalComponents(
      const StringView& s,
      int32_t toScale,
      int32_t& parsedPrecision,
      int32_t& parsedScale,
      int128_t& out);
}; // DecimalUtil
} // namespace facebook::velox
