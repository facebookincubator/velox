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
#include "velox/type/UnscaledLongDecimal.h"
#include "velox/type/UnscaledShortDecimal.h"

namespace facebook::velox {

/// A static class that holds helper functions for DECIMAL type.
class DecimalUtil {
 public:
  static const int128_t kPowersOfTen[LongDecimalType::kMaxPrecision];

  /// Helper function to convert a decimal value to string.
  template <typename T>
  static std::string toString(const T& value, const TypePtr& type);

  inline static int128_t rescaleWithRoundUp(
      const int128_t& unscaledValue,
      const uint8_t fromScale,
      const std::pair<uint8_t, uint8_t>& toPrecisionScale,
      bool& isNullOut) {
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
    auto valueToCompare = rescaledValue;
    if (rescaledValue < 0) {
      valueToCompare *= -1;
    }
    if (valueToCompare > DecimalUtil::kPowersOfTen[toPrecisionScale.first]) {
      isNullOut = true;
    }
    return rescaledValue;
  }
};
} // namespace facebook::velox
