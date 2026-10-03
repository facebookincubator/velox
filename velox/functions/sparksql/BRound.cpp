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

#include "velox/functions/sparksql/BRound.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>

#include <boost/multiprecision/cpp_int.hpp>
#include <fast_float/fast_float.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::functions::sparksql {
namespace {

using boost::multiprecision::cpp_int;

constexpr uint64_t kDoubleExponentMask = 0x7FF0000000000000ULL;
constexpr uint64_t kDoubleSignificandMask = 0x000FFFFFFFFFFFFFULL;
constexpr uint64_t kDoubleSignMask = 0x8000000000000000ULL;
constexpr uint32_t kDoubleSignificandWidth = 53;
constexpr uint64_t kDoubleImplicitBit = 1ULL << 52;
constexpr int32_t kDoubleMaxExponent = 1023;
constexpr int32_t kMinSmallExponent = -21;
constexpr int32_t kMaxSmallExponent = 62;
// Match the plain-decimal formatting range used by JDK 8-18 FloatingDecimal.
constexpr int32_t kMinPlainDecimalExponent = -3;
constexpr int32_t kMaxPlainDecimalExponent = 8;
constexpr double kLog10OfEDiv1P5 = 0.289529654;
constexpr double kLog10Of1P5 = 0.176091259;
constexpr double kLog10Of2 = 0.301029995663981;

cpp_int powerOfFive(uint32_t exponent) {
  cpp_int result = 1;
  cpp_int base = 5;
  while (exponent != 0) {
    if ((exponent & 1) != 0) {
      result *= base;
    }
    exponent >>= 1;
    if (exponent != 0) {
      base *= base;
    }
  }
  return result;
}

int32_t estimateDecimalExponent(int32_t exponent, uint64_t significand) {
  const uint64_t normalizedBits = static_cast<uint64_t>(kDoubleMaxExponent)
          << 52 |
      (significand & kDoubleSignificandMask);
  const double normalizedSignificand = std::bit_cast<double>(normalizedBits);
  const double approximation = (normalizedSignificand - 1.5) * kLog10OfEDiv1P5 +
      kLog10Of1P5 + exponent * kLog10Of2;
  const uint64_t approximationBits = std::bit_cast<uint64_t>(approximation);
  const auto approximationExponent = static_cast<int32_t>(
      static_cast<int64_t>((approximationBits & kDoubleExponentMask) >> 52) -
      kDoubleMaxExponent);

  if (approximationExponent >= 0 && approximationExponent < 52) {
    const auto extractedExponent = static_cast<int32_t>(
        ((approximationBits & kDoubleSignificandMask) | kDoubleImplicitBit) >>
        (52 - approximationExponent));
    return (approximationBits >> 63) != 0
        ? ((approximationBits &
            kDoubleSignificandMask >> approximationExponent) == 0
               ? -extractedExponent
               : -extractedExponent - 1)
        : extractedExponent;
  }

  if (approximationExponent < 0) {
    return (approximationBits & ~kDoubleSignMask) == 0 ? 0
        : (approximationBits & kDoubleSignMask) != 0   ? -1
                                                       : 0;
  }

  return static_cast<int32_t>(
      static_cast<uint32_t>(static_cast<int64_t>(approximation)));
}

void convertJavaDigitsToDecimal(
    bool negative,
    const std::array<char, 20>& digits,
    size_t digitCount,
    int32_t decimalExponent,
    int64_t& unscaled,
    int32_t& scale) {
  int64_t magnitude = 0;
  for (size_t i = 0; i < digitCount; ++i) {
    magnitude = magnitude * 10 + digits[i] - '0';
  }

  const bool scientific = decimalExponent <= kMinPlainDecimalExponent ||
      decimalExponent >= kMaxPlainDecimalExponent;
  if (scientific && digitCount == 1) {
    magnitude *= 10;
    digitCount = 2;
  } else if (
      !scientific && decimalExponent >= static_cast<int32_t>(digitCount)) {
    const auto zeroCount =
        decimalExponent - static_cast<int32_t>(digitCount) + 1;
    magnitude *= static_cast<int64_t>(DecimalUtil::kPowersOfTen[zeroCount]);
    scale = 1;
    unscaled = negative ? -magnitude : magnitude;
    return;
  }

  scale = static_cast<int32_t>(digitCount) - decimalExponent;
  unscaled = negative ? -magnitude : magnitude;
}

uint32_t insignificantDecimalDigits(int32_t bitIndex) {
  if (bitIndex <= 1 || bitIndex >= 64) {
    return 0;
  }
  if (bitIndex == 63) {
    return 19;
  }

  uint32_t count = 0;
  for (uint64_t value = uint64_t{1} << bitIndex; value >= 10; value /= 10) {
    ++count;
  }
  return count;
}

void decomposeSmallExponentNumber(
    bool negative,
    int32_t exponent,
    uint64_t significand,
    uint32_t significantBitCount,
    int64_t& unscaled,
    int32_t& scale) {
  const int32_t adjustedExponent =
      exponent - static_cast<int32_t>(significantBitCount) - 1;
  const uint32_t insignificantDigits = exponent > significantBitCount
      ? insignificantDecimalDigits(adjustedExponent)
      : 0;

  if (exponent >= static_cast<int32_t>(kDoubleSignificandWidth - 1)) {
    significand <<= exponent - (kDoubleSignificandWidth - 1);
  } else {
    significand >>= kDoubleSignificandWidth - 1 - exponent;
  }

  int32_t decimalExponent = 0;
  if (insignificantDigits != 0) {
    const uint64_t powerOfTen =
        static_cast<uint64_t>(DecimalUtil::kPowersOfTen[insignificantDigits]);
    const uint64_t remainder = significand % powerOfTen;
    significand /= powerOfTen;
    decimalExponent += static_cast<int32_t>(insignificantDigits);
    if (remainder >= powerOfTen / 2) {
      ++significand;
    }
  }

  while (significand % 10 == 0) {
    ++decimalExponent;
    significand /= 10;
  }

  std::array<char, 20> reversedDigits;
  size_t digitCount = 0;
  while (significand != 0) {
    reversedDigits[digitCount++] = static_cast<char>('0' + significand % 10);
    significand /= 10;
  }
  decimalExponent += static_cast<int32_t>(digitCount);

  std::array<char, 20> digits;
  for (size_t i = 0; i < digitCount; ++i) {
    digits[i] = reversedDigits[digitCount - i - 1];
  }
  convertJavaDigitsToDecimal(
      negative, digits, digitCount, decimalExponent, unscaled, scale);
}

/// Reproduces JDK 8-18 FloatingDecimal.BinaryToASCIIBuffer.dtoa so Spark's
/// BigDecimal.valueOf rounding observes the same decimal digits. The int64
/// branch mirrors Java's wrapping long path; cpp_int mirrors FDBigInteger.
void decomposeFloatingPoint(double value, int64_t& unscaled, int32_t& scale) {
  const uint64_t bits = std::bit_cast<uint64_t>(value);
  const bool negative = (bits >> 63) != 0;
  auto exponent = static_cast<int32_t>((bits & kDoubleExponentMask) >> 52);
  uint64_t significand = bits & kDoubleSignificandMask;
  uint32_t significantBitCount;

  if (exponent == 0) {
    const uint32_t leadingZeroCount = std::countl_zero(significand);
    const uint32_t shift = leadingZeroCount - (64 - kDoubleSignificandWidth);
    exponent = 1 - static_cast<int32_t>(shift) - kDoubleMaxExponent;
    significand <<= shift;
    significantBitCount = 64 - leadingZeroCount;
  } else {
    exponent -= kDoubleMaxExponent;
    significand |= kDoubleImplicitBit;
    significantBitCount = kDoubleSignificandWidth;
  }

  const uint32_t trailingZeroCount = std::countr_zero(significand);
  const uint32_t fractionalBitCount =
      kDoubleSignificandWidth - trailingZeroCount;
  const uint32_t excessBitCount = static_cast<uint32_t>(
      std::max(0, static_cast<int32_t>(fractionalBitCount) - exponent - 1));
  if (exponent >= kMinSmallExponent && exponent <= kMaxSmallExponent &&
      excessBitCount == 0 && fractionalBitCount < 64) {
    decomposeSmallExponentNumber(
        negative, exponent, significand, significantBitCount, unscaled, scale);
    return;
  }

  int32_t decimalExponent = estimateDecimalExponent(exponent, significand);

  const uint32_t valuePowerOfFive =
      static_cast<uint32_t>(std::max(0, -decimalExponent));
  int32_t valuePowerOfTwo =
      static_cast<int32_t>(valuePowerOfFive + excessBitCount) + exponent;
  const uint32_t scalePowerOfFive =
      static_cast<uint32_t>(std::max(0, decimalExponent));
  int32_t scalePowerOfTwo =
      static_cast<int32_t>(scalePowerOfFive + excessBitCount);
  const uint32_t marginPowerOfFive = valuePowerOfFive;
  int32_t marginPowerOfTwo =
      valuePowerOfTwo - static_cast<int32_t>(significantBitCount);

  significand >>= trailingZeroCount;
  valuePowerOfTwo -= static_cast<int32_t>(fractionalBitCount - 1);
  const int32_t minPowerOfTwo = std::min(valuePowerOfTwo, scalePowerOfTwo);
  valuePowerOfTwo -= minPowerOfTwo;
  scalePowerOfTwo -= minPowerOfTwo;
  marginPowerOfTwo -= minPowerOfTwo;

  if (fractionalBitCount == 1) {
    --marginPowerOfTwo;
  }
  if (marginPowerOfTwo < 0) {
    valuePowerOfTwo -= marginPowerOfTwo;
    scalePowerOfTwo -= marginPowerOfTwo;
    marginPowerOfTwo = 0;
  }

  VELOX_DCHECK_GE(valuePowerOfTwo, 0);
  VELOX_DCHECK_GE(scalePowerOfTwo, 0);
  cpp_int decimalValue = powerOfFive(valuePowerOfFive) * significand;
  decimalValue <<= valuePowerOfTwo;
  cpp_int decimalScale = powerOfFive(scalePowerOfFive);
  decimalScale <<= scalePowerOfTwo;
  cpp_int margin = powerOfFive(marginPowerOfFive);
  margin <<= marginPowerOfTwo;
  margin *= 10;
  const cpp_int decimalScaleTimesTen = decimalScale * 10;
  const auto valueBitCount =
      static_cast<uint32_t>(boost::multiprecision::msb(decimalValue) + 1);
  const auto scaleTimesTenBitCount = static_cast<uint32_t>(
      boost::multiprecision::msb(decimalScaleTimesTen) + 1);
  const bool useInt64Path = valueBitCount < 64 && scaleTimesTenBitCount < 64;

  std::array<char, 20> digits;
  size_t digitCount = 0;
  auto finishRounding = [&](bool low, bool high, int roundingThreshold) {
    ++decimalExponent;
    if (high &&
        (!low || roundingThreshold > 0 ||
         (roundingThreshold == 0 && ((digits[digitCount - 1] - '0') & 1)))) {
      size_t index = digitCount;
      while (index != 0 && digits[index - 1] == '9') {
        digits[--index] = '0';
      }
      if (index == 0) {
        ++decimalExponent;
        digits[0] = '1';
      } else {
        ++digits[index - 1];
      }
    }
    convertJavaDigitsToDecimal(
        negative, digits, digitCount, decimalExponent, unscaled, scale);
  };

  if (useInt64Path) {
    auto javaAdd = [](int64_t left, int64_t right) {
      return std::bit_cast<int64_t>(
          static_cast<uint64_t>(left) + static_cast<uint64_t>(right));
    };
    auto javaSubtract = [](int64_t left, int64_t right) {
      return std::bit_cast<int64_t>(
          static_cast<uint64_t>(left) - static_cast<uint64_t>(right));
    };
    auto javaMultiply = [](int64_t left, int64_t right) {
      return std::bit_cast<int64_t>(
          static_cast<uint64_t>(left) * static_cast<uint64_t>(right));
    };

    int64_t longValue = decimalValue.convert_to<int64_t>();
    const int64_t longScale = decimalScale.convert_to<int64_t>();
    int64_t longMargin = margin.convert_to<int64_t>();
    const int64_t longScaleTimesTen =
        decimalScaleTimesTen.convert_to<int64_t>();
    auto nextLongDigit = [&]() {
      const auto digit = static_cast<uint32_t>(longValue / longScale);
      longValue = javaMultiply(longValue % longScale, 10);
      return digit;
    };

    const uint32_t firstDigit = nextLongDigit();
    bool low = longValue < longMargin;
    bool high = javaAdd(longValue, longMargin) > longScaleTimesTen;
    if (firstDigit == 0 && !high) {
      --decimalExponent;
    } else {
      digits[digitCount++] = static_cast<char>('0' + firstDigit);
    }

    if (decimalExponent < kMinPlainDecimalExponent ||
        decimalExponent >= kMaxPlainDecimalExponent) {
      low = false;
      high = false;
    }

    while (!low && !high) {
      VELOX_CHECK_LT(digitCount, digits.size());
      digits[digitCount++] = static_cast<char>('0' + nextLongDigit());
      longMargin = javaMultiply(longMargin, 10);
      if (longMargin > 0) {
        low = longValue < longMargin;
        high = javaAdd(longValue, longMargin) > longScaleTimesTen;
      } else {
        low = true;
        high = true;
      }
    }

    const int64_t difference =
        javaSubtract(javaMultiply(longValue, 2), longScaleTimesTen);
    finishRounding(low, high, difference < 0 ? -1 : difference > 0 ? 1 : 0);
    return;
  }

  auto nextDigit = [&]() {
    const auto digit = (decimalValue / decimalScale).convert_to<uint32_t>();
    decimalValue %= decimalScale;
    decimalValue *= 10;
    return digit;
  };

  const uint32_t firstDigit = nextDigit();
  bool low = decimalValue < margin;
  bool high = decimalValue + margin >= decimalScaleTimesTen;
  if (firstDigit == 0 && !high) {
    --decimalExponent;
  } else {
    digits[digitCount++] = static_cast<char>('0' + firstDigit);
  }

  if (decimalExponent < kMinPlainDecimalExponent ||
      decimalExponent >= kMaxPlainDecimalExponent) {
    low = false;
    high = false;
  }

  while (!low && !high) {
    VELOX_CHECK_LT(digitCount, digits.size());
    digits[digitCount++] = static_cast<char>('0' + nextDigit());
    margin *= 10;
    low = decimalValue < margin;
    high = decimalValue + margin >= decimalScaleTimesTen;
  }

  int roundingThreshold = 0;
  if (high && low) {
    roundingThreshold = decimalValue * 2 < decimalScaleTimesTen ? -1
        : decimalValue * 2 > decimalScaleTimesTen               ? 1
                                                                : 0;
  }

  finishRounding(low, high, roundingThreshold);
}

template <typename T>
T composeFloatingPoint(int64_t unscaled, int32_t scale) {
  if (unscaled == 0) {
    return 0;
  }

  std::array<char, 64> buffer;
  auto [position, integerError] =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), unscaled);
  VELOX_CHECK(integerError == std::errc(), "Failed to format decimal value");
  *position++ = 'e';
  const auto [end, exponentError] = std::to_chars(
      position, buffer.data() + buffer.size(), -static_cast<int64_t>(scale));
  VELOX_CHECK(
      exponentError == std::errc(), "Failed to format decimal exponent");

  T result;
  const auto [parseEnd, parseError] = fast_float::from_chars(
      buffer.data(), end, result, fast_float::chars_format::general);
  if (parseError == std::errc::result_out_of_range) {
    return std::copysign(
        scale > 0 ? static_cast<T>(0) : std::numeric_limits<T>::infinity(),
        static_cast<T>(unscaled));
  }
  VELOX_CHECK(parseError == std::errc(), "Failed to parse rounded value");
  VELOX_CHECK_EQ(parseEnd, end);
  return result;
}

template <typename T>
Status broundFloatingPointImpl(T value, int32_t scale, T& result) {
  static_assert(std::is_floating_point_v<T>);

  if (!std::isfinite(value)) {
    result = value;
    return Status::OK();
  }
  if (value == 0) {
    result = 0;
    return Status::OK();
  }

  int64_t unscaled;
  int32_t sourceScale;
  decomposeFloatingPoint(static_cast<double>(value), unscaled, sourceScale);
  const int64_t scaleDistance =
      std::abs(static_cast<int64_t>(sourceScale) - static_cast<int64_t>(scale));
  if (scaleDistance > kMaxJavaBigIntegerPowerOfTenExponent) {
    if (threadSkipErrorDetails()) {
      return Status::UserError();
    }
    return scale < sourceScale
        ? Status::UserError("Underflow while rounding to scale {}", scale)
        : Status::UserError(
              "BigInteger would overflow supported range while rounding to "
              "scale {}",
              scale);
  }
  if (scale >= sourceScale) {
    result = value;
    return Status::OK();
  }

  const int64_t roundingDigitCount =
      static_cast<int64_t>(sourceScale) - static_cast<int64_t>(scale);
  if (roundingDigitCount >
      static_cast<int64_t>(detail::kMaxRoundingDigitCount)) {
    result = 0;
    return Status::OK();
  }

  const int64_t rounded =
      detail::broundUnscaled(unscaled, static_cast<size_t>(roundingDigitCount));
  result = composeFloatingPoint<T>(rounded, scale);
  return Status::OK();
}

} // namespace

Status detail::broundFloatingPoint(float value, int32_t scale, float& result) {
  return broundFloatingPointImpl(value, scale, result);
}

Status
detail::broundFloatingPoint(double value, int32_t scale, double& result) {
  return broundFloatingPointImpl(value, scale, result);
}

} // namespace facebook::velox::functions::sparksql
