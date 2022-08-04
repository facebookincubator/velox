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

#include "velox/type/DecimalUtil.h"

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

int128_t rescale(
    const int128_t& unscaledValue,
    const uint8_t fromScale,
    const std::pair<uint8_t, uint8_t>& toPrecisionScale) {
  auto rescaledValue = unscaledValue;
  auto rf = toPrecisionScale.second - fromScale;
  if (rf >= 0) {
    rescaledValue = unscaledValue * DecimalUtil::kPowersOfTen[rf];
  } else {
    rf = -1 * rf;
    rescaledValue = unscaledValue / DecimalUtil::kPowersOfTen[rf];
    int128_t remainder = unscaledValue % DecimalUtil::kPowersOfTen[rf];
    if (unscaledValue >= 0) {
      if (remainder > DecimalUtil::kPowersOfTen[rf] / 2) {
        rescaledValue++;
      }
    } else {
      if (remainder < -DecimalUtil::kPowersOfTen[rf] / 2) {
        rescaledValue--;
      }
    }
  }
  // Check overflowing
  VELOX_CHECK_LE(
      rescaledValue, DecimalUtil::kPowersOfTen[toPrecisionScale.first]);
  return rescaledValue;
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

template <>
std::string DecimalUtil::toString<UnscaledLongDecimal>(
    const UnscaledLongDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asLongDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}

template <>
std::string DecimalUtil::toString<UnscaledShortDecimal>(
    const UnscaledShortDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asShortDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}

template <>
void DecimalUtil::rescaleWithRoundUp(
    ShortDecimal& output,
    const int128_t& unscaledValue,
    const uint8_t fromScale,
    const std::pair<uint8_t, uint8_t>& toPrecisionScale) {
  output.setUnscaledValue(rescale(unscaledValue, fromScale, toPrecisionScale));
}

template <>
void DecimalUtil::rescaleWithRoundUp(
    LongDecimal& output,
    const int128_t& unscaledValue,
    const uint8_t fromScale,
    const std::pair<uint8_t, uint8_t>& toPrecisionScale) {
  output.setUnscaledValue(rescale(unscaledValue, fromScale, toPrecisionScale));
}
} // namespace facebook::velox
