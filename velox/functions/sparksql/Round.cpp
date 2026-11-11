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

#include "velox/functions/sparksql/Round.h"

#include <cmath>
#include <limits>

#include <fmt/format.h>
#include <folly/Conv.h>

#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <typename T>
T canonicalizeZero(T value) {
  return value == 0 ? T{0} : value;
}

template <typename T>
T finiteOrOriginal(double candidate, T original) {
  if (!std::isfinite(candidate) ||
      candidate > static_cast<double>(std::numeric_limits<T>::max()) ||
      candidate < static_cast<double>(std::numeric_limits<T>::lowest())) {
    return original;
  }
  return canonicalizeZero(static_cast<T>(candidate));
}

double roundWithOverflowingFactor(double input, int32_t scale) {
  // Bound decimal conversion to tiny DOUBLEs. Even 17 significant digits end
  // before scale 309 for magnitudes at or above 1e-291.
  // The smallest decimal exponent is -324, so 17 digits need at most scale 340.
  constexpr int32_t kMaxDecimalScale = 340;
  const auto magnitude = std::abs(input);
  if (magnitude >= 1e-291 || scale >= kMaxDecimalScale) {
    return input;
  }

  auto text = fmt::format("{}", magnitude);
  // Match Java's minimum of two significant digits in scientific notation.
  // In particular, Double.MIN_VALUE is 4.9e-324, not the shortest 5e-324.
  if (text.find('e') == 1) {
    text = fmt::format("{:.1e}", magnitude);
  }

  int32_t precision{0};
  int32_t decimalScale{0};
  int128_t coefficient{0};
  VELOX_CHECK_OK(
      facebook::velox::DecimalUtil::parseStringToDecimalComponents(
          StringView(text),
          kMaxDecimalScale,
          precision,
          decimalScale,
          coefficient));
  if (scale >= decimalScale) {
    return input;
  }

  int64_t rounded{0};
  VELOX_CHECK_OK(
      (facebook::velox::DecimalUtil::rescaleWithRoundUp<int128_t, int64_t>(
          coefficient, precision, decimalScale, precision, scale, rounded)));
  if (rounded == 0) {
    return 0;
  }
  // Parse the decimal result once instead of performing a lossy split-factor
  // binary round trip in the subnormal range.
  return std::copysign(
      folly::to<double>(fmt::format("{}e-{}", rounded, scale)), input);
}

template <typename T>
T roundFloatingPointImpl(T input, int32_t scale) {
  if (!std::isfinite(input)) {
    return input;
  }
  if (input == 0) {
    return 0;
  }
  if (scale == 0) {
    return canonicalizeZero(std::round(input));
  }

  if (scale < 0) {
    const double factor = std::pow(10.0, scale);
    if (factor == 0) {
      return 0;
    }
    const double scaled = static_cast<double>(input) * factor;
    const double rounded = std::round(scaled);
    if (rounded == 0) {
      return 0;
    }
    return finiteOrOriginal(rounded / factor, input);
  }

  if (scale > std::numeric_limits<double>::max_exponent10) {
    return static_cast<T>(roundWithOverflowingFactor(input, scale));
  }

  return finiteOrOriginal(
      facebook::velox::functions::round(input, scale), input);
}

} // namespace

namespace detail {

void roundFloatingPoint(float input, int32_t scale, float& result) {
  result = roundFloatingPointImpl(input, scale);
}

void roundFloatingPoint(double input, int32_t scale, double& result) {
  result = roundFloatingPointImpl(input, scale);
}

} // namespace detail
} // namespace facebook::velox::functions::sparksql
