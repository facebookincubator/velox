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

#include "velox/type/DecimalUtils.h"

namespace facebook::velox {
namespace {
std::string formatDecimal(uint8_t scale, int128_t unscaledValue) {
  VELOX_DCHECK_GE(scale, 0);
  VELOX_DCHECK_LT(
      static_cast<size_t>(scale), sizeof(DecimalUtil::kPowersOfTen));
  if (unscaledValue == 0) {
    return "0";
  }

  bool isNegative = (unscaledValue < 0);
  if (isNegative) {
    unscaledValue = ~unscaledValue + 1;
  }
  int128_t integralPart = unscaledValue / DecimalUtil::kPowersOfTen[scale];

  bool isFraction = (scale > 0);
  std::string fractionString;
  if (isFraction) {
    auto fraction =
        std::to_string(unscaledValue % DecimalUtil::kPowersOfTen[scale]);
    fractionString += ".";
    // Append leading zeros.
    fractionString += std::string(
        std::max(scale - static_cast<int>(fraction.size()), 0), '0');
    // Append the fraction part.
    fractionString += fraction;
  }

  return fmt::format(
      "{}{}{}", isNegative ? "-" : "", integralPart, fractionString);
}

int128_t toInt128(
    const char* data,
    size_t size,
    const uint8_t precision,
    const uint8_t scale) {
  if (size == 0) {
    return 0;
  }
  bool isNegative = (data[0] == '-');
  int32_t start = (isNegative || data[0] == '+') ? 1 : 0;
  int32_t pos = start;
  int128_t result = 0;
  size_t integerLen = 0;
  size_t fractionLen = 0;
  while (pos < size) {
    char c = data[pos++];
    if (c == '.') {
      integerLen = pos - start;
      while (pos < size) {
        char digit = data[pos++];
        VELOX_DCHECK(isdigit(digit));
        result = result * 10 + (digit - '0');
        ++fractionLen;
      }
    } else if (isdigit(c)) {
      result = result * 10 + (c - '0');
    } else {
      VELOX_DCHECK(false, "Invalid character at pos {}", pos)
    }
  }
  VELOX_DCHECK((integerLen + fractionLen) <= precision);
  VELOX_DCHECK(fractionLen <= scale);
  result = result * DecimalUtil::kPowersOfTen[scale - fractionLen];
  return (isNegative) ? result * -1 : result;
}

} // namespace

const int128_t DecimalUtil::kPowersOfTen[]{
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
    10000000000000000,
    100000000000000000,
    1000000000000000000,
    1000000000000000000 * (int128_t)10,
    1000000000000000000 * (int128_t)100,
    1000000000000000000 * (int128_t)1000,
    1000000000000000000 * (int128_t)10000,
    1000000000000000000 * (int128_t)100000,
    1000000000000000000 * (int128_t)1000000,
    1000000000000000000 * (int128_t)10000000,
    1000000000000000000 * (int128_t)100000000,
    1000000000000000000 * (int128_t)1000000000,
    1000000000000000000 * (int128_t)10000000000,
    1000000000000000000 * (int128_t)100000000000,
    1000000000000000000 * (int128_t)1000000000000,
    1000000000000000000 * (int128_t)10000000000000,
    1000000000000000000 * (int128_t)100000000000000,
    1000000000000000000 * (int128_t)1000000000000000,
    1000000000000000000 * (int128_t)10000000000000000,
    1000000000000000000 * (int128_t)100000000000000000,
    1000000000000000000 * (int128_t)1000000000000000000,
    1000000000000000000 * (int128_t)1000000000000000000 * (int128_t)10};

template <typename T>
T DecimalUtil::stringToDecimal(
    const char* data,
    const size_t size,
    const TypePtr& type) {
  return nullptr;
}

template <>
std::string DecimalUtil::toString<LongDecimal>(
    const LongDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asLongDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}

template <>
std::string DecimalUtil::toString<ShortDecimal>(
    const ShortDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asShortDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}

template <typename T>
std::string DecimalUtil::toString(const T& value, const TypePtr& type) {
  VELOX_UNSUPPORTED();
}

template <>
ShortDecimal DecimalUtil::stringToDecimal(
    const char* data,
    const size_t size,
    const TypePtr& toType) {
  auto [precision, scale] = getDecimalPrecisionScale(toType);
  int128_t unscaledValue = toInt128(data, size, precision, scale);
  return ShortDecimal(unscaledValue);
}
} // namespace facebook::velox
