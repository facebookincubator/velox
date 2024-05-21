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

#include "velox/type/Conversions.h"

DEFINE_bool(
    experimental_enable_legacy_cast,
    false,
    "Experimental feature flag for backward compatibility with previous output"
    " format of type conversions used for casting. This is a temporary solution"
    " that aims to facilitate a seamless transition for users who rely on the"
    " legacy behavior and hence can change in the future.");

namespace facebook::velox::util {
/// Normalize the given floating-point standard notation string in place, by
/// appending '.0' if it has only the integer part but no fractional part. For
/// example, for the given string '12345', replace it with '12345.0'.
void normalizeStandardNotation(std::string& str) {
  if (!FLAGS_experimental_enable_legacy_cast &&
      str.find(".") == std::string::npos && isdigit(str[str.length() - 1])) {
    str += ".0";
  }
}

/// Normalize the given floating-point scientific notation string in place, by
/// removing the trailing 0s of the coefficient as well as the leading '+' and
/// 0s of the exponent. For example, for the given string '3.0000000E+005',
/// replace it with '3.0E5'. For '-1.2340000E-010', replace it with
/// '-1.234E-10'.
void normalizeScientificNotation(std::string& str) {
  size_t idxE = str.find('E');

  VELOX_DCHECK_NE(
      idxE,
      std::string::npos,
      "Expect a character 'E' in scientific notation.");

  int endCoef = idxE - 1;
  while (endCoef >= 0 && str[endCoef] == '0') {
    --endCoef;
  }
  VELOX_DCHECK_GT(endCoef, 0, "Coefficient should not be all zeros.");

  int pos = endCoef + 1;
  if (str[endCoef] == '.') {
    pos++;
  }
  str[pos++] = 'E';

  int startExp = idxE + 1;
  if (str[startExp] == '-') {
    str[pos++] = '-';
    startExp++;
  }
  while (startExp < str.length() &&
         (str[startExp] == '0' || str[startExp] == '+')) {
    startExp++;
  }
  VELOX_DCHECK_LT(startExp, str.length(), "Exponent should not be all zeros.");
  str.replace(pos, str.length() - startExp, str, startExp);
  pos += str.length() - startExp;

  str.resize(pos);
}

template <typename T>
std::string floatingToString(const T& val) {
  using doubleToStringConv = double_conversion::DoubleToStringConverter;

  static auto dconvertor = doubleToStringConv(
      doubleToStringConv::EMIT_TRAILING_DECIMAL_POINT |
          doubleToStringConv::EMIT_TRAILING_ZERO_AFTER_POINT |
          doubleToStringConv::EMIT_TRAILING_ZERO_AFTER_POINT,
      "Infinity",
      "NaN",
      'E',
      std::is_same_v<T, float> ? -5 : -7,
      std::is_same_v<T, float> ? 5 : 7,
      0,
      0);

  // Implementation below is close to String.of(double) of Java. For
  // example, for some rare cases the result differs in precision by
  // the least significant bit.
  if (FOLLY_UNLIKELY(std::isinf(val) || std::isnan(val))) {
    return folly::to<std::string>(val);
  }
  if ((val > -10'000'000 && val <= -0.001) ||
      (val >= 0.001 && val < 10'000'000) || val == 0.0) {
    auto str = fmt::format("{}", val);
    normalizeStandardNotation(str);
    return str;
  }

  const auto kBufferSize = 64;
  char buffer[kBufferSize];
  double_conversion::StringBuilder builder{buffer, kBufferSize};
  if constexpr (std::is_same_v<T, float>) {
    dconvertor.ToPrecision(val, 8, &builder);
  } else {
    dconvertor.ToExponential(val, 15, &builder);
  }

  auto convertedResult = std::string{builder.Finalize()};

  normalizeScientificNotation(convertedResult);
  return convertedResult;
}

std::string floatToString(const float& val) {
  return floatingToString<float>(val);
}

std::string doubleToString(const double& val) {
  return floatingToString<double>(val);
}

} // namespace facebook::velox::util
