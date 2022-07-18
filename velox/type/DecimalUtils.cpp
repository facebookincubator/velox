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

std::string formatAsDecimal(uint8_t scale, int128_t unscaledValue) {
  if (unscaledValue == 0)
    return "0";
  std::string result;
  if (unscaledValue < 0) {
    result.append("-");
    unscaledValue = ~unscaledValue + 1;
  }
  std::string unscaledStr = std::to_string(unscaledValue);
  std::string formattedStr;
  if (unscaledStr.length() <= scale) {
    formattedStr.append("0");
  } else {
    formattedStr.append(unscaledStr.substr(0, unscaledStr.length() - scale));
  }
  if (scale > 0) {
    formattedStr.append(".");
    if (unscaledStr.length() < scale) {
      for (auto i = 0; i < scale - unscaledStr.length(); ++i) {
        formattedStr.append("0");
      }
      formattedStr.append(unscaledStr);
    } else {
      formattedStr.append(unscaledStr.substr(unscaledStr.length() - scale));
    }
  }
  return result.append(formattedStr);
}
} // namespace facebook::velox
