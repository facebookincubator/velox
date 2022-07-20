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
  VELOX_DCHECK_LT(static_cast<size_t>(scale), sizeof(kPowersOfTen));
  if (unscaledValue == 0) {
    return "0";
  }

  bool isNegative = (unscaledValue < 0);
  if (isNegative) {
    unscaledValue = ~unscaledValue + 1;
  }
  int128_t integralPart = unscaledValue / kPowersOfTen[scale];

  bool isFraction = (scale > 0);
  std::string fractionString;
  if (isFraction) {
    int128_t fractionPart = unscaledValue % kPowersOfTen[scale];
    // Calculate the string length of fractional part.
    int fractionSize = 1;
    if (fractionPart > 0) {
      // Log can be used to count the number of digits of a positive number.
      // Log is likely an assembly instruction, so efficient to use.
      fractionSize += log10(fractionPart);
    }

    fractionString += ".";
    // Append leading zeros.
    fractionString += std::string(std::max(scale - fractionSize, 0), '0');
    // Append the fraction part.
    fractionString += std::to_string(fractionPart);
  }

  return fmt::format(
      "{}{}{}", isNegative ? "-" : "", integralPart, fractionString);
}
} // namespace

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
