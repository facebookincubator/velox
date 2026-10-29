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

#include <array>
#include <bit>
#include <cmath>
#include <limits>

#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

// Keep both division operands exact doubles. Widening either bound can
// introduce double rounding and break Spark parity.
constexpr int128_t kMaxExactDoubleInteger = int128_t{1} << 53;
constexpr std::array<double, 23> kExactPowersOfTen = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
};

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
    : scale_(scale) {
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
