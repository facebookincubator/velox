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

// GPU shadow for velox/functions/prestosql/ArithmeticImpl.h
// Adds __host__ __device__ to helper functions that the original leaves
// unannotated (plus, minus, multiply, divide, modulus, negate, abs, floor,
// ceil). Functions already marked FOLLY_ALWAYS_INLINE in the original get the
// annotation via our CPortability.h shadow.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include "folly/CPortability.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {

template <typename TNum, typename TDecimals, bool alwaysRoundNegDec = false>
FOLLY_ALWAYS_INLINE TNum
round(const TNum& number, const TDecimals& decimals = 0) {
  static_assert(!std::is_same_v<TNum, bool> && "round not supported for bool");

  if constexpr (std::is_integral_v<TNum>) {
    if constexpr (alwaysRoundNegDec) {
      if (decimals >= 0)
        return number;
    } else {
      return number;
    }
  }
  if (!std::isfinite(number)) {
    return number;
  }

  if (decimals == 0) {
    return std::round(number);
  }

  if (decimals < 0) {
    const double factor = std::pow(10, decimals);
    return std::round(number * factor) / factor;
  }

  const TNum trancated = std::trunc(number);
  const TNum fraction = number - trancated;
  if (fraction == 0.0)
    return number;

  const double factor = std::pow(10, decimals);

  if constexpr (!std::is_integral_v<TNum>) {
    if (fabs(number) < 17592186044415.F) {
      return std::round(number * factor) / factor;
    }
  }

  const TNum roundedFractions = std::round(fraction * factor) / factor;
  return trancated + roundedFractions;
}

template <typename T>
__host__ __device__ T plus(const T a, const T b) {
  return a + b;
}

template <typename T>
__host__ __device__ T minus(const T a, const T b) {
  return a - b;
}

template <typename T>
__host__ __device__ T multiply(const T a, const T b) {
  return a * b;
}

template <typename T>
__host__ __device__ T divide(const T& a, const T& b) {
  T result = a / b;
  return result;
}

template <typename T>
__host__ __device__ T modulus(const T a, const T b) {
  if (b == 0) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  return std::fmod(a, b);
}

template <typename T>
__host__ __device__ T negate(const T& arg) {
  return -arg;
}

template <typename T>
__host__ __device__ T abs(const T& arg) {
  if constexpr (std::is_integral_v<T>) {
    if (arg < 0) {
      return -arg;
    }
    return arg;
  } else {
    return std::abs(arg);
  }
}

template <typename T>
__host__ __device__ T floor(const T& arg) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::floor(arg);
  } else {
    return arg;
  }
}

template <typename T>
__host__ __device__ T ceil(const T& arg) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::ceil(arg);
  } else {
    return arg;
  }
}

FOLLY_ALWAYS_INLINE double truncate(double number, int32_t decimals) {
  const bool decNegative = (decimals < 0);
  const auto log10Size = DoubleUtil::kPowersOfTen.size(); // 309
  if (decNegative && decimals <= -static_cast<int32_t>(log10Size)) {
    return 0.0;
  }

  const uint64_t absDec = std::abs(decimals);
  const double tmp = (absDec < log10Size) ? DoubleUtil::kPowersOfTen[absDec]
                                          : std::pow(10.0, (double)absDec);

  const double valueMulTmp = number * tmp;
  if (!decNegative && !std::isfinite(valueMulTmp)) {
    return number;
  }

  const double valueDivTmp = number / tmp;
  if (number >= 0.0) {
    return decimals < 0 ? std::floor(valueDivTmp) * tmp
                        : std::floor(valueMulTmp) / tmp;
  } else {
    return decimals < 0 ? std::ceil(valueDivTmp) * tmp
                        : std::ceil(valueMulTmp) / tmp;
  }
}

template <bool isUpper>
FOLLY_ALWAYS_INLINE double
wilsonInterval(int64_t successes, int64_t trials, double z) {
  VELOX_USER_CHECK_GE(successes, 0, "number of successes must not be negative");
  VELOX_USER_CHECK_GT(trials, 0, "number of trials must be positive");
  VELOX_USER_CHECK_LE(
      successes,
      trials,
      "number of successes must not be larger than number of trials");
  VELOX_USER_CHECK_GE(z, 0, "z-score must not be negative");

  double s{static_cast<double>(successes)};
  double n{static_cast<double>(trials)};
  double p{s / n};

  double a, c, r;

  if (z < 1) {
    a = n + z * z;
    c = s * p;
    r = 2 * s + z * z + z * std::sqrt(z * z + 4 * s * (1 - p));
  } else {
    a = n / (z * z) + 1;
    c = s * p / (z * z);
    r = 2 * s / (z * z) + 1 + std::sqrt(1 + 4 * s * (1 - p) / (z * z));
  }

  if constexpr (isUpper) {
    return r / (2 * a);
  } else {
    return (r > 0) ? (2 * c) / r : 0;
  }
}

} // namespace facebook::velox::functions
