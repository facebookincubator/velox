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

namespace facebook::velox {

std::string formatAsDecimal(const uint8_t scale, int128_t number) {
  std::stringstream ss;
  std::string result;
  if (number == 0)
    return "0";
  while (number) {
    uint64_t remainder = number % 10;
    number = number / 10;
    ss << remainder;
    result.insert(0, ss.str());
    ss.str("");
  }
  std::string formattedStr;
  if (result.length() <= scale) {
    formattedStr.append("0");
  } else {
    formattedStr.append(result.substr(0, result.length() - scale));
  }

  if (scale > 0) {
    formattedStr.append(".");
    if (result.length() < scale) {
      for (auto i = 0; i < scale - result.length(); ++i) {
        formattedStr.append("0");
      }
      formattedStr.append(result);
    } else {
      formattedStr.append(result.substr(result.length() - scale));
    }
  }
  return formattedStr;
}

std::string shortDecimalToString(
    uint8_t precision,
    uint8_t scale,
    const ShortDecimal& shortDecimalValue) {
  auto unscaledValue = shortDecimalValue.unscaledValue();
  std::string result;
  if (unscaledValue < 0) {
    result.append("-");
    unscaledValue = ~unscaledValue + 1;
  }
  return result.append(formatAsDecimal(scale, unscaledValue));
}

std::string longDecimalToString(
    uint8_t precision,
    uint8_t scale,
    const LongDecimal& longDecimal) {
  std::string result;
  auto unscaledValue = longDecimal.unscaledValue();
  if (unscaledValue < 0) {
    result.append("-");
    unscaledValue = ~unscaledValue + 1;
  }
  return result.append(formatAsDecimal(scale, unscaledValue));
}
} // namespace facebook::velox
