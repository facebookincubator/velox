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

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <type_traits>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/SpecialFormRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Keeps native support bounded without emulating JVM-specific extreme-scale
// arithmetic and allocation failures.
void validateScale(int32_t scale) {
  VELOX_USER_CHECK(
      scale >= -400 && scale <= 400,
      "The scale for bround must be between -400 and 400, got: {}.",
      scale);
}

// Initialize errors are deferred until a non-null row is evaluated, so a
// constant NULL scale still propagates NULL.
void validateScaleArgument(const int32_t* scale) {
  VELOX_USER_CHECK_NOT_NULL(
      scale, "The second argument of bround must be a constant INTEGER.");
  validateScale(*scale);
}

// Resolves an exact decimal midpoint toward the even quotient. Divisors are
// powers of ten, so comparing with divisor / 2 cannot overflow.
int128_t divideHalfEven(int128_t value, int128_t divisor) {
  const auto quotient = value / divisor;
  const auto remainder = value % divisor;
  if (remainder == 0) {
    return quotient;
  }
  const auto absoluteRemainder = remainder < 0 ? -remainder : remainder;
  const auto half = divisor / 2;
  if (absoluteRemainder > half ||
      (absoluteRemainder == half && quotient % 2 != 0)) {
    return quotient + (value < 0 ? -1 : 1);
  }
  return quotient;
}

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
      rounded = divideHalfEven(value, divisor) * divisor;
    }
  }
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
  // Conversion to an unsigned type is modulo 2^N; bit_cast preserves the
  // resulting Java two's-complement representation without signed overflow.
  result = std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(rounded));
  return Status::OK();
}

// Represents a shortest decimal as coefficient * 10^(-scale).
struct DecimalComponents {
  int64_t coefficient;
  int32_t scale;
};

// Converts a finite nonzero DOUBLE, including widened REAL values, without
// depending on private formatting-library interfaces or the current locale.
DecimalComponents shortestDecimal(double value) {
  std::array<char, 32> buffer;
  const auto conversion = std::to_chars(
      buffer.data(),
      buffer.data() + buffer.size(),
      value,
      std::chars_format::scientific);
  VELOX_CHECK(conversion.ec == std::errc{});
  const char* current = buffer.data();
  const bool negative = *current == '-';
  current += negative;
  int64_t coefficient = *current++ - '0';
  int32_t fractionalDigits{0};
  if (*current == '.') {
    ++current;
    while (*current != 'e') {
      coefficient = coefficient * 10 + (*current++ - '0');
      ++fractionalDigits;
    }
  }
  VELOX_CHECK_EQ(*current++, 'e');
  if (*current == '+') {
    ++current;
  }
  int32_t exponent{0};
  const auto parsed = std::from_chars(current, conversion.ptr, exponent);
  VELOX_CHECK(parsed.ec == std::errc{} && parsed.ptr == conversion.ptr);
  return {negative ? -coefficient : coefficient, fractionalDigits - exponent};
}

// Reconstructs directly in the output type, avoiding double rounding for REAL.
template <typename T>
T decimalToFloating(int64_t coefficient, int32_t scale) {
  if (coefficient == 0) {
    return T{0};
  }
  std::array<char, 48> buffer;
  auto coefficientConversion =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), coefficient);
  VELOX_CHECK(coefficientConversion.ec == std::errc{});
  *coefficientConversion.ptr++ = 'e';
  const auto exponentConversion = std::to_chars(
      coefficientConversion.ptr,
      buffer.data() + buffer.size(),
      -static_cast<int64_t>(scale));
  VELOX_CHECK(exponentConversion.ec == std::errc{});
  T result;
  const auto conversion = std::from_chars(
      buffer.data(),
      exponentConversion.ptr,
      result,
      std::chars_format::scientific);
  if (conversion.ec == std::errc::result_out_of_range) {
    // Coefficients have at most 17 digits, so negative scales can only
    // overflow and nonnegative scales can only underflow.
    return scale < 0 ? (coefficient < 0 ? -std::numeric_limits<T>::infinity()
                                        : std::numeric_limits<T>::infinity())
                     : T{0};
  }
  VELOX_CHECK(
      conversion.ec == std::errc{} && conversion.ptr == exponentConversion.ptr);
  return result == 0 ? T{0} : result;
}

// Avoids decimal conversion at scale zero. A nonzero fractional part implies
// that the integral part is small enough to convert safely to int64_t.
template <typename T>
T roundToInteger(T value) {
  const T integral = std::trunc(value);
  const T fraction = std::abs(value - integral);
  if (fraction < T{0.5} ||
      (fraction == T{0.5} && static_cast<int64_t>(integral) % 2 == 0)) {
    return integral == 0 ? T{0} : integral;
  }
  return integral + std::copysign(T{1}, value);
}

// Applies decimal HALF_EVEN semantics rather than rounding value * 10^scale
// in binary, which would introduce additional rounding near decimal midpoints.
template <typename T>
Status roundFloating(T& result, T value, int32_t scale) {
  if (!std::isfinite(value)) {
    result = value;
    return Status::OK();
  }
  if (value == 0) {
    result = T{0};
    return Status::OK();
  }
  if (scale == 0) {
    result = roundToInteger(value);
    return Status::OK();
  }
  const auto components = shortestDecimal(static_cast<double>(value));
  const auto digitsToDrop =
      static_cast<int64_t>(components.scale) - static_cast<int64_t>(scale);
  if (digitsToDrop <= 0) {
    result = value;
  } else if (digitsToDrop > 17) {
    result = T{0};
  } else {
    const auto coefficient = divideHalfEven(
        components.coefficient, DecimalUtil::kPowersOfTen[digitsToDrop]);
    result = decimalToFloating<T>(static_cast<int64_t>(coefficient), scale);
  }
  return Status::OK();
}

// Adapts primitive HALF_EVEN rounding to the simple-function evaluator.
template <typename TExec>
struct BRoundFunction {
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
    validateScaleArgument(scale);
    initialize(types, config, value);
  }

  template <typename T>
  void initialize(
      const std::vector<TypePtr>&,
      const core::QueryConfig&,
      const T*,
      const int32_t* scale,
      const bool* ansiEnabled) {
    validateScaleArgument(scale);
    VELOX_USER_CHECK_NOT_NULL(
        ansiEnabled,
        "The third argument of bround must be a constant BOOLEAN.");
    ansiEnabled_ = *ansiEnabled;
  }

  template <typename T>
  Status call(T& result, const T& value) {
    return call(result, value, 0);
  }

  template <typename T>
  Status call(T& result, const T& value, int32_t scale) {
    if constexpr (std::is_integral_v<T>) {
      return roundIntegral(result, value, scale, ansiEnabled_);
    } else {
      return roundFloating(result, value, scale);
    }
  }

  template <typename T>
  Status call(T& result, const T& value, int32_t scale, bool /*ansiEnabled*/) {
    // The constant expression mode was validated and captured by initialize().
    return roundIntegral(result, value, scale, ansiEnabled_);
  }

 private:
  bool ansiEnabled_{false};
};

// Rounds decimal values without sharing or changing ROUND/ceil/floor policies.
template <typename TInput, typename TResult>
class DecimalBRoundFunction : public exec::VectorFunction {
 public:
  DecimalBRoundFunction(
      uint8_t inputScale,
      int32_t requestedScale,
      uint8_t resultPrecision)
      : digitsToDrop_(static_cast<int64_t>(inputScale) - requestedScale),
        roundsToZero_(digitsToDrop_ > LongDecimalType::kMaxPrecision),
        divisor_(
            digitsToDrop_ > 0 && !roundsToZero_
                ? DecimalUtil::kPowersOfTen[digitsToDrop_]
                : 1),
        multiplier_(
            requestedScale < 0 && !roundsToZero_
                ? DecimalUtil::kPowersOfTen[-static_cast<int64_t>(
                      requestedScale)]
                : 1),
        limitBeforeMultiply_(
            (DecimalUtil::kPowersOfTen[resultPrecision] - 1) / multiplier_) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    auto* output = result->asUnchecked<FlatVector<TResult>>();
    output->clearNulls(rows);
    auto* values = output->mutableRawValues();
    DecodedVector input(*args[0], rows);
    rows.applyToSelected([&](auto row) {
      int128_t rounded{0};
      if (!roundsToZero_) {
        const auto value = input.valueAt<TInput>(row);
        rounded = digitsToDrop_ > 0 ? divideHalfEven(value, divisor_) : value;
      }
      if (rounded > limitBeforeMultiply_ || rounded < -limitBeforeMultiply_) {
        context.setStatus(
            row,
            threadSkipErrorDetails()
                ? Status::UserError()
                : Status::UserError("Decimal overflow in bround."));
        return;
      }
      values[row] = static_cast<TResult>(rounded * multiplier_);
    });
  }

 private:
  const int64_t digitsToDrop_;
  const bool roundsToZero_;
  const int128_t divisor_;
  const int128_t multiplier_;
  const int128_t limitBeforeMultiply_;
};

// Computes Spark's resolved decimal type with widened scale arithmetic.
TypePtr decimalResultType(const TypePtr& inputType, int32_t requestedScale) {
  const auto [precision, scale] = getDecimalPrecisionScale(*inputType);
  const int64_t integralDigits = precision - scale + 1;
  if (requestedScale < 0) {
    return DECIMAL(
        std::min<int64_t>(
            std::max(integralDigits, 1 - static_cast<int64_t>(requestedScale)),
            LongDecimalType::kMaxPrecision),
        0);
  }
  const auto resultScale = std::min<int32_t>(scale, requestedScale);
  return DECIMAL(
      std::min<int64_t>(
          integralDigits + resultScale, LongDecimalType::kMaxPrecision),
      resultScale);
}

// Requires an explicitly resolved output type because scale is a value rather
// than a type parameter. A constant null scale bypasses evaluation of the
// value.
class DecimalBRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  explicit DecimalBRoundCallToSpecialForm(std::string name)
      : name_(std::move(name)) {}

  TypePtr resolveType(const std::vector<TypePtr>&) override {
    VELOX_USER_FAIL("{} requires an explicitly resolved result type.", name_);
  }

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig&) override {
    VELOX_USER_CHECK(
        args.size() == 1 || args.size() == 2,
        "{} expects one or two arguments.",
        name_);
    VELOX_USER_CHECK(
        args[0]->type()->isDecimal(),
        "The first argument of {} must be decimal.",
        name_);
    int32_t scale{0};
    std::shared_ptr<exec::ConstantExpr> scaleExpression;
    bool nullScale{false};
    if (args.size() == 2) {
      VELOX_USER_CHECK(
          args[1]->type()->isInteger(),
          "The second argument of {} must be INTEGER.",
          name_);
      scaleExpression = std::dynamic_pointer_cast<exec::ConstantExpr>(args[1]);
      VELOX_USER_CHECK_NOT_NULL(
          scaleExpression,
          "The second argument of {} must be a constant expression.",
          name_);
      const auto* constant =
          scaleExpression->value()->asUnchecked<ConstantVector<int32_t>>();
      nullScale = constant->isNullAt(0);
      if (!nullScale) {
        scale = constant->valueAt(0);
        validateScale(scale);
      }
    }
    const auto expectedType = decimalResultType(args[0]->type(), scale);
    VELOX_USER_CHECK(
        type->equivalent(*expectedType),
        "Invalid result type for {}: expected {}, got {}.",
        name_,
        expectedType->toString(),
        type->toString());
    if (nullScale) {
      return std::make_shared<exec::ConstantExpr>(
          BaseVector::createNullConstant(
              type, 1, scaleExpression->value()->pool()));
    }
    const auto inputScale = getDecimalPrecisionScale(*args[0]->type()).second;
    const auto resultPrecision = getDecimalPrecisionScale(*type).first;
    std::shared_ptr<exec::VectorFunction> function;
    if (args[0]->type()->isShortDecimal()) {
      if (type->isShortDecimal()) {
        function = std::make_shared<DecimalBRoundFunction<int64_t, int64_t>>(
            inputScale, scale, resultPrecision);
      } else {
        function = std::make_shared<DecimalBRoundFunction<int64_t, int128_t>>(
            inputScale, scale, resultPrecision);
      }
    } else if (type->isShortDecimal()) {
      function = std::make_shared<DecimalBRoundFunction<int128_t, int64_t>>(
          inputScale, scale, resultPrecision);
    } else {
      function = std::make_shared<DecimalBRoundFunction<int128_t, int128_t>>(
          inputScale, scale, resultPrecision);
    }
    return std::make_shared<exec::Expr>(
        type,
        std::move(args),
        std::move(function),
        exec::VectorFunctionMetadata{},
        name_,
        trackCpuUsage);
  }

 private:
  const std::string name_;
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
  registerFunction<
      BRoundFunction,
      int8_t,
      int8_t,
      Constant<int32_t>,
      Constant<bool>>({prefix + "bround"});
  registerFunction<
      BRoundFunction,
      int16_t,
      int16_t,
      Constant<int32_t>,
      Constant<bool>>({prefix + "bround"});
  registerFunction<
      BRoundFunction,
      int32_t,
      int32_t,
      Constant<int32_t>,
      Constant<bool>>({prefix + "bround"});
  registerFunction<
      BRoundFunction,
      int64_t,
      int64_t,
      Constant<int32_t>,
      Constant<bool>>({prefix + "bround"});
  registerFunction<BRoundFunction, float, float, Constant<int32_t>>(
      {prefix + "bround"});
  registerFunction<BRoundFunction, double, double, Constant<int32_t>>(
      {prefix + "bround"});
  const auto decimalName = prefix + kBRoundDecimal;
  exec::registerFunctionCallToSpecialForm(
      decimalName,
      std::make_unique<DecimalBRoundCallToSpecialForm>(decimalName));
}

} // namespace facebook::velox::functions::sparksql
