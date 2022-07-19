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

#include "DecimalUtils.h"

namespace {
std::string formatDecimal(
    uint8_t scale,
    facebook::velox::int128_t unscaledValue) {
  VELOX_CHECK_LE(0, scale);
  VELOX_CHECK_LT(
      static_cast<size_t>(scale), sizeof(facebook::velox::POWERS_OF_TEN));
  if (unscaledValue == 0)
    return "0";
  std::string sign = "";
  if (unscaledValue < 0) {
    sign = "-";
    unscaledValue = ~unscaledValue + 1;
  }
  facebook::velox::int128_t integralPart =
      unscaledValue / facebook::velox::POWERS_OF_TEN[scale];
  facebook::velox::int128_t fractionPart =
      unscaledValue % facebook::velox::POWERS_OF_TEN[scale];
  std::string fractionStr = "";
  if (fractionPart != 0) {
    fractionStr = std::to_string(fractionPart);
  }
  std::string leadingZeros;
  std::string decimal = "";
  if (scale > 0) {
    decimal = ".";
    if (fractionStr.length() < scale) {
      leadingZeros = std::string(scale - fractionStr.length(), '0');
    }
  }
  return fmt::format(
      "{}{}{}{}{}", sign, integralPart, decimal, leadingZeros, fractionStr);
}
} // namespace

namespace facebook::velox {

template <>
std::string decimalToString<LongDecimal>(
    const LongDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asLongDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}

template <>
std::string decimalToString<ShortDecimal>(
    const ShortDecimal& value,
    const TypePtr& type) {
  auto decimalType = type->asShortDecimal();
  return formatDecimal(decimalType.scale(), value.unscaledValue());
}
} // namespace facebook::velox
