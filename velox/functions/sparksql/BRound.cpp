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

#include "velox/functions/sparksql/BRound.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <typename T>
FOLLY_ALWAYS_INLINE T wrapToSigned(int128_t value) {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
  using UnsignedT = std::make_unsigned_t<T>;
  return std::bit_cast<T>(static_cast<UnsignedT>(value));
}

template <typename T>
FOLLY_ALWAYS_INLINE Status
broundIntegral(T value, int32_t scale, bool ansiEnabled, T& result) {
  static_assert(
      std::is_integral_v<T> && std::is_signed_v<T> && !std::is_same_v<T, bool>);

  if (scale >= 0 || value == 0) {
    result = value;
    return Status::OK();
  }

  const int64_t roundingDigitCount = -static_cast<int64_t>(scale);
  if (roundingDigitCount > std::numeric_limits<T>::digits10 + 1) {
    result = 0;
    return Status::OK();
  }

  const int128_t divisor = DecimalUtil::kPowersOfTen[roundingDigitCount];
  const int128_t rounded = DecimalUtil::divideWithRoundHalfEven(
                               static_cast<int128_t>(value), divisor) *
      divisor;
  if (ansiEnabled &&
      (rounded < std::numeric_limits<T>::min() ||
       rounded > std::numeric_limits<T>::max())) {
    return threadSkipErrorDetails()
        ? Status::UserError()
        : Status::UserError(
              "Arithmetic overflow in bround({}, {})",
              static_cast<int64_t>(value),
              scale);
  }

  result = wrapToSigned<T>(rounded);
  return Status::OK();
}

template <typename T>
Status
broundFloatingPointImpl(T value, int32_t scale, double factor, T& result) {
  static_assert(std::is_floating_point_v<T>);

  // Spark rounds a decimal representation produced by the Java runtime.
  // Velox intentionally rounds the binary value directly to avoid reproducing
  // runtime-specific floating-to-decimal conversion algorithms.
  if (!std::isfinite(value)) {
    result = value;
    return Status::OK();
  }
  if (value == 0) {
    result = 0;
    return Status::OK();
  }

  if (scale >= 0) {
    // Preserve the input when the power-of-ten scale is not representable.
    // Native scale-round-unscale cannot define a reliable HALF_EVEN result
    // without a finite factor.
    if (!std::isfinite(factor)) {
      result = value;
      return Status::OK();
    }
    const double scaled = static_cast<double>(value) * factor;
    if (!std::isfinite(scaled)) {
      result = value;
      return Status::OK();
    }
    result = static_cast<T>(std::nearbyint(scaled) / factor);
  } else {
    if (!std::isfinite(factor)) {
      result = 0;
      return Status::OK();
    }
    result = static_cast<T>(
        std::nearbyint(static_cast<double>(value) / factor) * factor);
  }

  if (result == 0) {
    result = 0;
  }
  return Status::OK();
}

template <typename TExec>
struct BRoundFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const T* /*value*/) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const T* /*value*/,
      const int32_t* scale) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
    if constexpr (std::is_floating_point_v<T>) {
      if (scale != nullptr) {
        floatingPointScale_ = *scale;
        const int64_t absoluteScale = floatingPointScale_ < 0
            ? -static_cast<int64_t>(floatingPointScale_)
            : floatingPointScale_;
        floatingPointFactor_ =
            std::pow(10.0, static_cast<double>(absoluteScale));
      }
    }
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T value) {
    return call(result, value, 0);
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE Status
  call(T& result, const T value, const int32_t scale) {
    if constexpr (std::is_floating_point_v<T>) {
      VELOX_DCHECK_EQ(scale, floatingPointScale_);
      return broundFloatingPointImpl(
          value, scale, floatingPointFactor_, result);
    } else {
      return broundIntegral(value, scale, ansiEnabled_, result);
    }
  }

 private:
  bool ansiEnabled_{false};
  int32_t floatingPointScale_{0};
  double floatingPointFactor_{1};
};

} // namespace

void registerBRoundFunctions(const std::string& prefix) {
  registerUnaryNumeric<BRoundFunction>({prefix + "bround"});
  registerFunction<BRoundFunction, int8_t, int8_t, Constant<int32_t>>(
      {prefix + "bround"});
  registerFunction<BRoundFunction, int16_t, int16_t, Constant<int32_t>>(
      {prefix + "bround"});
  registerFunction<BRoundFunction, int32_t, int32_t, Constant<int32_t>>(
      {prefix + "bround"});
  registerFunction<BRoundFunction, int64_t, int64_t, Constant<int32_t>>(
      {prefix + "bround"});
  registerFunction<BRoundFunction, float, float, Constant<int32_t>>(
      {prefix + "bround"});
  registerFunction<BRoundFunction, double, double, Constant<int32_t>>(
      {prefix + "bround"});
}

} // namespace facebook::velox::functions::sparksql
