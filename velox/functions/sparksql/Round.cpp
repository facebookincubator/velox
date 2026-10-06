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

namespace facebook::velox::functions::sparksql {
namespace {

template <typename T>
T canonicalizeZero(T value) {
  return value == 0 ? T{0} : value;
}

template <typename T>
T finiteOrOriginal(double candidate, T original) {
  const auto result = static_cast<T>(candidate);
  return std::isfinite(result) ? canonicalizeZero(result) : original;
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

  const double factor = std::pow(10.0, scale);
  if (scale < 0) {
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

  if (!std::isfinite(factor)) {
    return input;
  }
  const double scaled = static_cast<double>(input) * factor;
  if (!std::isfinite(scaled)) {
    return input;
  }

  const T truncated = std::trunc(input);
  const T fraction = input - truncated;
  if (fraction == 0) {
    return input;
  }

  double candidate;
  if (std::abs(input) < 17'592'186'044'415.F) {
    candidate = std::round(scaled) / factor;
  } else {
    candidate =
        truncated + std::round(static_cast<double>(fraction) * factor) / factor;
  }
  return finiteOrOriginal(candidate, input);
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
