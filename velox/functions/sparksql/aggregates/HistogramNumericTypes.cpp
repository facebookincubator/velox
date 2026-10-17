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

#include "velox/functions/sparksql/aggregates/HistogramNumericTypes.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>

#include <boost/multiprecision/cpp_int.hpp>
#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

// Independently implements the numeric selection in the Java 21 public
// Double.toString specification, not a JDK implementation or display formatter:
// https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/Double.html#toString(double)
// Spark's corrected expression is public at revision
// b84dc909a8856388faddc154c6a1d3aba271474e, HistogramNumeric.scala (eval).
// Scala 2.13.17 BigDecimal.decimal selects this decimal, then MathContext(p)
// HALF_UP and setScale(s, HALF_UP) are two distinct rounding operations.
//
// Represent binary64 values in units of 2^-1074; interval midpoints then have
// denominator 2^1075. Even the virtual successor of max double needs only 2099
// bits. Decimal grids have exponents in [-341,309], coefficients < 10^17.
// Comparisons on a common decimal grid need at most 3300 bits (including
// scaling by 10^650 and the 1076-bit denominator). Final rescaling has at most
// 347 digits. A checked, fixed backend leaves ample room without a heap-based
// arbitrary-precision accumulator or unbounded exponent search.
using Wide = boost::multiprecision::number<
    boost::multiprecision::cpp_int_backend<
        4096,
        4096,
        boost::multiprecision::unsigned_magnitude,
        boost::multiprecision::checked,
        void>,
    boost::multiprecision::et_off>;

// Keep both division operands exact doubles. Widening either bound can
// introduce double rounding and break Spark parity.
constexpr int128_t kMaxExactDoubleInteger = int128_t{1} << 53;
constexpr std::array<double, 23> kExactPowersOfTen = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
};

// Holds a positive decimal coefficient without trailing zeros and its power
// of ten. A binary64 round-trip always needs at most 17 significant digits.
struct SelectedDecimal {
  // Retains the selected significant digits, at most 17.
  uint64_t coefficient;
  // Scales the coefficient by a power of ten, not a binary exponent.
  int32_t exponent;
};

// Computes bounded decimal powers using integer arithmetic only.
Wide powerOfTen(int32_t exponent) {
  Wide result = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    result *= 10;
  }
  return result;
}

// Decodes a nonnegative finite bit pattern, or the virtual successor 2^1024
// represented by the infinity bit pattern when constructing max double's edge.
Wide binaryUnits(uint64_t bits) {
  const auto exponent = (bits >> 52) & 0x7ff;
  const auto fraction = bits & 0x000fffffffffffffULL;
  if (exponent == 0) {
    return Wide(fraction);
  }
  return Wide(fraction | (1ULL << 52)) << (exponent - 1);
}

// Finds floor(log10(numerator / denominator)) without floating logarithms.
int32_t decimalExponent(const Wide& numerator, const Wide& denominator) {
  const Wide whole = numerator / denominator;
  if (whole != 0) {
    return static_cast<int32_t>(whole.str().size()) - 1;
  }
  Wide scaled = numerator;
  int32_t exponent = 0;
  // Positive binary64 values are at least 10^-324, so at most 324 iterations.
  while (scaled < denominator) {
    scaled *= 10;
    --exponent;
  }
  return exponent;
}

// Selects the nearest integer, breaking exact midpoint ties to even.
Wide nearestEven(const Wide& numerator, const Wide& denominator) {
  Wide quotient = numerator / denominator;
  const Wide twiceRemainder = (numerator % denominator) * 2;
  if (twiceRemainder > denominator ||
      (twiceRemainder == denominator && (quotient & 1) != 0)) {
    ++quotient;
  }
  return quotient;
}

// Compares exact distances to binary64 on a shared decimal grid. No rational
// cross-products of two large denominators are required.
bool nearer(
    const SelectedDecimal& candidate,
    const SelectedDecimal& current,
    const Wide& numerator,
    const Wide& denominator) {
  const auto commonExponent =
      std::min({0, candidate.exponent, current.exponent});
  const Wide scaledValue = numerator * powerOfTen(-commonExponent);
  const auto distance = [&](const SelectedDecimal& decimal) -> Wide {
    const Wide scaledDecimal = Wide(decimal.coefficient) * denominator *
        powerOfTen(decimal.exponent - commonExponent);
    return scaledDecimal >= scaledValue ? scaledDecimal - scaledValue
                                        : scaledValue - scaledDecimal;
  };
  const auto candidateDistance = distance(candidate);
  const auto currentDistance = distance(current);
  return candidateDistance < currentDistance ||
      (candidateDistance == currentDistance && candidate.coefficient % 2 == 0 &&
       current.coefficient % 2 != 0);
}

// Searches the round-to-nearest-even interval for the shortest decimal. When
// that length is one, Java 21 also considers length two, choosing the closest
// value and then an even significand. Grid neighbors cover powers-of-ten
// crossings, including subnormals and the finite-to-infinity rounding edge.
SelectedDecimal selectJavaDecimal(uint64_t magnitudeBits) {
  const Wide units = binaryUnits(magnitudeBits);
  const Wide numerator = units * 2;
  const Wide denominator = Wide(1) << 1075;
  const Wide lower = units + binaryUnits(magnitudeBits - 1);
  const Wide upper = units + binaryUnits(magnitudeBits + 1);
  const bool inclusive = (magnitudeBits & 1) == 0;
  const auto order = decimalExponent(numerator, denominator);
  std::optional<SelectedDecimal> best;

  for (int32_t length = 1; length <= 17; ++length) {
    const Wide minCoefficient = powerOfTen(length - 1);
    const Wide maxCoefficient = powerOfTen(length) - 1;
    for (int32_t exponent = order - length; exponent <= order - length + 2;
         ++exponent) {
      Wide scaledLower = lower;
      Wide scaledUpper = upper;
      Wide scaledValue = numerator;
      Wide divisor = denominator;
      if (exponent >= 0) {
        divisor *= powerOfTen(exponent);
      } else {
        const auto multiplier = powerOfTen(-exponent);
        scaledLower *= multiplier;
        scaledUpper *= multiplier;
        scaledValue *= multiplier;
      }
      Wide first = scaledLower / divisor;
      if (!inclusive || scaledLower % divisor != 0) {
        ++first;
      }
      Wide last = scaledUpper / divisor;
      if (!inclusive && scaledUpper % divisor == 0) {
        --last;
      }
      first = std::max(first, minCoefficient);
      last = std::min(last, maxCoefficient);
      if (first > last) {
        continue;
      }
      const Wide rounded =
          std::clamp(nearestEven(scaledValue, divisor), first, last);
      SelectedDecimal candidate{rounded.convert_to<uint64_t>(), exponent};
      while (candidate.coefficient % 10 == 0) {
        candidate.coefficient /= 10;
        ++candidate.exponent;
      }
      if (!best.has_value() ||
          nearer(candidate, *best, numerator, denominator)) {
        best = candidate;
      }
    }
    if (best.has_value() && length >= 2) {
      return *best;
    }
  }
  VELOX_UNREACHABLE("A binary64 value requires at most 17 decimal digits");
}

// Rounds a positive integer quotient with ties away from zero. The sign is
// applied only after both rounding stages, so negative ties are also HALF_UP.
Wide halfUp(const Wide& value, const Wide& divisor) {
  Wide result = value / divisor;
  if ((value % divisor) * 2 >= divisor) {
    ++result;
  }
  return result;
}

// Reconstructs the absolute unscaled integer without imposing the native
// declared precision after Spark's two rounds.
Wide reconstructedMagnitude(double center, int32_t precision, int32_t scale) {
  VELOX_USER_CHECK(
      std::isfinite(center),
      "Cannot reconstruct histogram_numeric Decimal from a non-finite center");
  const auto magnitudeBits =
      std::bit_cast<uint64_t>(center) & 0x7fffffffffffffffULL;
  if (magnitudeBits == 0) {
    return Wide(0);
  }
  auto decimal = selectJavaDecimal(magnitudeBits);
  Wide coefficient = decimal.coefficient;
  const auto numDigits =
      static_cast<int32_t>(std::to_string(decimal.coefficient).size());
  if (numDigits > precision) {
    const auto removed = numDigits - precision;
    coefficient = halfUp(coefficient, powerOfTen(removed));
    decimal.exponent += removed;
  }
  const auto scaleShift = decimal.exponent + scale;
  if (scaleShift >= 0) {
    return coefficient * powerOfTen(scaleShift);
  }
  return halfUp(coefficient, powerOfTen(-scaleShift));
}

// Rounds a 53-bit positive significand after removing 29..53 low bits.
uint64_t roundedFloatSignificand(uint64_t significand, uint32_t shift) {
  const auto truncated = significand >> shift;
  const auto remainder = significand & ((1ULL << shift) - 1);
  const auto half = 1ULL << (shift - 1);
  return truncated +
      (remainder > half || (remainder == half && (truncated & 1) != 0));
}

} // namespace

HistogramNumericDecimal::HistogramNumericDecimal(
    int32_t precision,
    int32_t scale)
    : precision_(precision), scale_(scale) {
  VELOX_USER_CHECK(
      precision >= 1 && precision <= 38,
      "histogram_numeric Decimal precision must be in [1, 38]: {}",
      precision);
  VELOX_USER_CHECK(
      scale >= 0 && scale <= precision,
      "histogram_numeric Decimal scale must be in [0, precision]: {}",
      scale);
  type_ = DECIMAL(precision, scale);
}

double HistogramNumericDecimal::decimalToDouble(int128_t unscaled) const {
  if (scale_ <= 22 && unscaled >= -kMaxExactDoubleInteger &&
      unscaled <= kMaxExactDoubleInteger) {
    return static_cast<double>(unscaled) / kExactPowersOfTen[scale_];
  }
  return folly::to<double>(DecimalUtil::toString(unscaled, *type_));
}

std::string HistogramNumericDecimal::reconstructDecimal(double center) const {
  const auto magnitude = reconstructedMagnitude(center, precision_, scale_);
  auto text = magnitude.str();
  if (scale_ > 0) {
    if (text.size() <= scale_) {
      text.insert(0, scale_ + 1 - text.size(), '0');
    }
    text.insert(text.size() - scale_, 1, '.');
  }
  if (std::signbit(center) && magnitude != 0) {
    text.insert(0, 1, '-');
  }
  return text;
}

std::optional<int128_t> HistogramNumericDecimal::tryReconstructNativeDecimal(
    double center) const {
  const auto magnitude = reconstructedMagnitude(center, precision_, scale_);
  if (magnitude >= powerOfTen(precision_)) {
    return std::nullopt;
  }
  const auto unscaled = magnitude.convert_to<int128_t>();
  return std::signbit(center) ? -unscaled : unscaled;
}

double HistogramNumericTypes::timestampToDouble(const Timestamp& value) {
  VELOX_USER_CHECK_EQ(
      value.getNanos() % 1'000,
      0,
      "histogram_numeric timestamp requires microsecond precision");
  return numericToDouble(value.toMicros());
}

int32_t HistogramNumericTypes::javaDoubleToInt(double value) {
  if (std::isnan(value)) {
    return 0;
  }
  if (value >= 0x1p31) {
    return std::numeric_limits<int32_t>::max();
  }
  if (value <= -0x1p31) {
    return std::numeric_limits<int32_t>::min();
  }
  return static_cast<int32_t>(value);
}

int64_t HistogramNumericTypes::javaDoubleToLong(double value) {
  if (std::isnan(value)) {
    return 0;
  }
  // double(INT64_MAX) rounds UP to 2^63: equality must saturate before casting.
  if (value >= 0x1p63) {
    return std::numeric_limits<int64_t>::max();
  }
  if (value <= -0x1p63) {
    return std::numeric_limits<int64_t>::min();
  }
  return static_cast<int64_t>(value);
}

int8_t HistogramNumericTypes::javaDoubleToByte(double value) {
  const auto low = static_cast<uint32_t>(javaDoubleToInt(value)) & 0xff;
  const auto signedValue = low >= 0x80 ? static_cast<int32_t>(low) - 0x100
                                       : static_cast<int32_t>(low);
  return static_cast<int8_t>(signedValue);
}

int16_t HistogramNumericTypes::javaDoubleToShort(double value) {
  const auto low = static_cast<uint32_t>(javaDoubleToInt(value)) & 0xffff;
  const auto signedValue = low >= 0x8000 ? static_cast<int32_t>(low) - 0x10000
                                         : static_cast<int32_t>(low);
  return static_cast<int16_t>(signedValue);
}

float HistogramNumericTypes::javaDoubleToFloat(double value) {
  const auto bits = std::bit_cast<uint64_t>(value);
  const auto sign = static_cast<uint32_t>(bits >> 32) & 0x80000000U;
  const auto biasedExponent = (bits >> 52) & 0x7ff;
  const auto fraction = bits & 0x000fffffffffffffULL;
  if (biasedExponent == 0x7ff) {
    const auto payload = fraction == 0
        ? 0U
        : static_cast<uint32_t>(fraction >> 29) | 0x00400000U;
    return std::bit_cast<float>(sign | 0x7f800000U | payload);
  }
  auto exponent = static_cast<int32_t>(biasedExponent) - 1023;
  if (exponent < -150) {
    return std::bit_cast<float>(sign);
  }
  if (exponent > 127) {
    return std::bit_cast<float>(sign | 0x7f800000U);
  }
  const auto significand = fraction | (1ULL << 52);
  if (exponent < -126) {
    // The rounded subnormal can carry into the smallest normal (bit 23).
    const auto rounded = roundedFloatSignificand(significand, -exponent - 97);
    return std::bit_cast<float>(sign | static_cast<uint32_t>(rounded));
  }
  auto rounded = roundedFloatSignificand(significand, 29);
  if (rounded == (1ULL << 24)) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent > 127) {
    return std::bit_cast<float>(sign | 0x7f800000U);
  }
  return std::bit_cast<float>(
      sign | (static_cast<uint32_t>(exponent + 127) << 23) |
      (static_cast<uint32_t>(rounded) & 0x007fffffU));
}

Timestamp HistogramNumericTypes::javaDoubleToTimestamp(double value) {
  const auto micros = javaDoubleToLong(value);
  auto seconds = micros / 1'000'000;
  auto remainder = micros % 1'000'000;
  if (remainder < 0) {
    --seconds;
    remainder += 1'000'000;
  }
  return Timestamp(seconds, static_cast<uint64_t>(remainder) * 1'000);
}

} // namespace facebook::velox::functions::aggregate::sparksql
