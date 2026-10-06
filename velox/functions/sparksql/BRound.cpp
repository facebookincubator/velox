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

#include <cmath>

#include "velox/functions/lib/RegistrationHelpers.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <typename T>
Status broundFloatingPointImpl(T value, int32_t scale, T& result) {
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
    const double factor = std::pow(10.0, static_cast<double>(scale));
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
    const double factor =
        std::pow(10.0, static_cast<double>(-static_cast<int64_t>(scale)));
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

} // namespace

Status detail::broundFloatingPoint(float value, int32_t scale, float& result) {
  return broundFloatingPointImpl(value, scale, result);
}

Status
detail::broundFloatingPoint(double value, int32_t scale, double& result) {
  return broundFloatingPointImpl(value, scale, result);
}

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
