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

#include "velox/functions/sparksql/Rounding.h"

#include <bit>
#include <limits>
#include <type_traits>

#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/specialforms/DecimalRound.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Rounds in a wider domain before applying Spark's integral narrowing policy.
template <typename T>
Status roundIntegral(T& result, T value, int32_t scale, bool ansiEnabled) {
  int128_t rounded = value;
  if (scale < 0) {
    const auto digitsToDrop = -static_cast<int64_t>(scale);
    if (digitsToDrop > std::numeric_limits<T>::digits10 + 1) {
      rounded = 0;
    } else {
      const auto divisor = DecimalUtil::kPowersOfTen[digitsToDrop];
      DecimalUtil::divideWithRoundUp<int128_t, T, int128_t>(
          rounded, value, divisor, false, 0, 0);
      rounded *= divisor;
    }
  }
  if (ansiEnabled &&
      (rounded < std::numeric_limits<T>::min() ||
       rounded > std::numeric_limits<T>::max())) {
    return threadSkipErrorDetails() ? Status::UserError()
                                    : Status::UserError(
                                          "Arithmetic overflow in {}({}, {})",
                                          "round",
                                          static_cast<int64_t>(value),
                                          scale);
  }
  // Conversion to an unsigned type is modulo 2^N; bit_cast preserves the
  // resulting Java two's-complement representation without signed overflow.
  result = std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(rounded));
  return Status::OK();
}

// Captures integral ROUND's ANSI mode when the expression is initialized.
template <typename TExec>
struct SparkRoundFunction {
  template <typename T>
  void initialize(
      const std::vector<TypePtr>&,
      const core::QueryConfig& config,
      const T*) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
  }

  template <typename T>
  void initialize(
      const std::vector<TypePtr>& types,
      const core::QueryConfig& config,
      const T* value,
      const int32_t* scale) {
    VELOX_USER_CHECK_NOT_NULL(
        scale, "The second argument of round must be a constant INTEGER.");
    initialize(types, config, value);
  }

  template <typename T>
  void initialize(
      const std::vector<TypePtr>&,
      const core::QueryConfig&,
      const T*,
      const int32_t* scale,
      const bool* ansiEnabled) {
    VELOX_USER_CHECK_NOT_NULL(
        scale, "The second argument of round must be a constant INTEGER.");
    VELOX_USER_CHECK_NOT_NULL(
        ansiEnabled, "The third argument of round must be a constant BOOLEAN.");
    ansiEnabled_ = *ansiEnabled;
  }

  template <typename T>
  Status call(T& result, const T& value) {
    return call(result, value, 0);
  }

  template <typename T>
  Status call(T& result, const T& value, int32_t scale) {
    return roundIntegral(result, value, scale, ansiEnabled_);
  }

  template <typename T>
  Status call(T& result, const T& value, int32_t scale, bool /*ansiEnabled*/) {
    return roundIntegral(result, value, scale, ansiEnabled_);
  }

 private:
  bool ansiEnabled_{false};
};

template <typename T>
void registerIntegralRound(const std::string& name) {
  registerFunction<SparkRoundFunction, T, T, Constant<int32_t>>({name});
  registerFunction<SparkRoundFunction, T, T, Constant<int32_t>, Constant<bool>>(
      {name});
}

} // namespace

void registerRoundFunctions(const std::string& prefix) {
  registerUnaryIntegral<SparkRoundFunction>({prefix + "round"});
  for (const auto& name : {prefix + "round", prefix + kSparkRound}) {
    registerIntegralRound<int8_t>(name);
    registerIntegralRound<int16_t>(name);
    registerIntegralRound<int32_t>(name);
    registerIntegralRound<int64_t>(name);
  }
  // Preserve the existing binary floating-point implementation and signatures.
  registerFunction<RoundFunction, float, float>({prefix + "round"});
  registerFunction<RoundFunction, double, double>({prefix + "round"});
  registerFunction<RoundFunction, float, float, int32_t>({prefix + "round"});
  registerFunction<RoundFunction, double, double, int32_t>({prefix + "round"});
  registerDecimalRoundSpecialForm(prefix + kRoundDecimal);
  registerDecimalRoundSpecialForm(prefix + kSparkRoundDecimal);
}

} // namespace facebook::velox::functions::sparksql
