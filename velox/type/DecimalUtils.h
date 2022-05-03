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

#include <string>
#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"
namespace facebook::velox {

static const int128_t POWERS_OF_TEN[]{
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
    1000000000000000000 * (int128_t)1000000000000000000 * (int128_t)10,
    1000000000000000000 * (int128_t)1000000000000000000 * (int128_t)100};

class DecimalCasts {
 public:
  static std::string ShortDecimalToString(
      uint8_t precision,
      uint8_t scale,
      int64_t unscaledValue) {
    std::string result;
    if (unscaledValue < 0) {
      result.append("-");
      unscaledValue = ~unscaledValue + 1;
    }
    std::stringstream ss;
    ss << unscaledValue;
    std::string unscaledValueStr = ss.str();
    return result.append(UnscaledValueToDecimalString(scale, unscaledValueStr));
  }

  static std::string LongDecimalToString(
      uint8_t precision,
      uint8_t scale,
      int128_t unscaledValue) {
    std::string result;
    if (unscaledValue < 0) {
      result.append("-");
      unscaledValue = ~unscaledValue + 1;
    }
    auto unscaledValueStr = Int128ToString(unscaledValue);
    return result.append(UnscaledValueToDecimalString(scale, unscaledValueStr));
  }

  static std::string Int128ToString(uint128_t number) {
    std::stringstream ss;
    std::string result;
    if (number == 0)
      return "0";
    while (number) {
      uint64_t remainder = number % POWERS_OF_TEN[18];
      number = number / POWERS_OF_TEN[18];
      ss << remainder;
      result.insert(0, ss.str());
      ss.str("");
    }
    return result;
  }

  static std::string UnscaledValueToDecimalString(
      uint8_t scale,
      std::string& unscaledValueStr) {
    std::string result;
    if (unscaledValueStr.length() <= scale) {
      result.append("0");
    } else {
      result.append(
          unscaledValueStr.substr(0, unscaledValueStr.length() - scale));
    }

    if (scale > 0) {
      result.append(".");
      if (unscaledValueStr.length() < scale) {
        for (auto i = 0; i < scale - unscaledValueStr.length(); ++i) {
          result.append("0");
        }
        result.append(unscaledValueStr);
      } else {
        result.append(
            unscaledValueStr.substr(unscaledValueStr.length() - scale));
      }
    }
    return result;
  }
}; // DecimalCasts
} // namespace facebook::velox
