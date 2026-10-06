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
#include <limits>
#include <utility>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/SpecialFormRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/vector/DecodedVector.h"

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
    // Once 10^scale overflows to infinity (scale >= 309) the scale/round/
    // unscale approach can no longer be applied. Computing the rounded value
    // directly would require dividing by the quantum 10^-scale, which for these
    // scales is itself a rounded subnormal and distorts HALF_EVEN ties, or
    // staged finite scaling, which perturbs ordinary values by an ULP. Rather
    // than reproduce those errors, 'bround' leaves the value unchanged at these
    // extreme scales. This is a deliberate simplification: true decimal
    // rounding would snap the smallest subnormals toward zero, so results here
    // can differ from Spark (see the documented floating-point divergence).
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

template <typename TInput, typename TResult>
class DecimalBRoundFunction : public exec::VectorFunction {
 public:
  DecimalBRoundFunction(
      uint8_t inputScale,
      int32_t requestedScale,
      uint8_t resultPrecision)
      : roundingDigitCount_(
            static_cast<int64_t>(inputScale) -
            static_cast<int64_t>(requestedScale)),
        requestedScale_(requestedScale),
        scaleUnderflows_(
            roundingDigitCount_ > detail::kMaxJavaBigIntegerPowerOfTenExponent),
        roundsToZero_(roundingDigitCount_ > LongDecimalType::kMaxPrecision),
        divisor_(
            roundingDigitCount_ > 0 && !roundsToZero_
                ? DecimalUtil::kPowersOfTen[roundingDigitCount_]
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
      const int128_t value = input.valueAt<TInput>(row);
      if (scaleUnderflows_ && value != 0) {
        output->setNull(row, true);
        context.setStatus(row, detail::broundUnderflowError(requestedScale_));
        return;
      }

      int128_t rounded{0};
      if (!roundsToZero_) {
        rounded = roundingDigitCount_ > 0
            ? detail::divideHalfEven(value, divisor_)
            : value;
      }
      if (rounded > limitBeforeMultiply_ || rounded < -limitBeforeMultiply_) {
        output->setNull(row, true);
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
  const int64_t roundingDigitCount_;
  const int32_t requestedScale_;
  const bool scaleUnderflows_;
  const bool roundsToZero_;
  const int128_t divisor_;
  const int128_t multiplier_;
  const int128_t limitBeforeMultiply_;
};

TypePtr decimalResultType(const TypePtr& inputType, int32_t requestedScale) {
  const auto [precision, scale] = getDecimalPrecisionScale(*inputType);
  const int64_t integralDigits = static_cast<int64_t>(precision) - scale + 1;
  if (requestedScale < 0) {
    const int32_t requiredPrecision = std::bit_cast<int32_t>(
        uint32_t{0} - static_cast<uint32_t>(requestedScale) + uint32_t{1});
    return DECIMAL(
        std::min<int64_t>(
            std::max<int64_t>(integralDigits, requiredPrecision),
            LongDecimalType::kMaxPrecision),
        0);
  }
  const int32_t resultScale = std::min<int32_t>(scale, requestedScale);
  return DECIMAL(
      std::min<int64_t>(
          integralDigits + resultScale, LongDecimalType::kMaxPrecision),
      resultScale);
}

class DecimalBRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  explicit DecimalBRoundCallToSpecialForm(std::string functionName)
      : functionName_(std::move(functionName)) {}

  TypePtr resolveType(const std::vector<TypePtr>& /*argTypes*/) override {
    VELOX_USER_FAIL(
        "{} requires an explicitly resolved result type.", functionName_);
  }

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& /*config*/) override {
    VELOX_USER_CHECK(
        args.size() == 1 || args.size() == 2,
        "{} expects one or two arguments.",
        functionName_);
    VELOX_USER_CHECK(
        args[0]->type()->isDecimal(),
        "The first argument of {} must be decimal.",
        functionName_);

    int32_t scale{0};
    std::shared_ptr<exec::ConstantExpr> scaleExpression;
    bool nullScale{false};
    if (args.size() == 2) {
      VELOX_USER_CHECK(
          args[1]->type()->isInteger(),
          "The second argument of {} must be INTEGER.",
          functionName_);
      scaleExpression = std::dynamic_pointer_cast<exec::ConstantExpr>(args[1]);
      VELOX_USER_CHECK_NOT_NULL(
          scaleExpression,
          "The second argument of {} must be a constant expression.",
          functionName_);
      const auto* constant =
          scaleExpression->value()->asUnchecked<ConstantVector<int32_t>>();
      nullScale = constant->isNullAt(0);
      if (!nullScale) {
        scale = constant->valueAt(0);
      }
    }

    const auto expectedType = decimalResultType(args[0]->type(), scale);
    VELOX_USER_CHECK(
        type->equivalent(*expectedType),
        "Invalid result type for {}: expected {}, got {}.",
        functionName_,
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
        functionName_,
        trackCpuUsage);
  }

 private:
  const std::string functionName_;
};

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

  const auto decimalName = prefix + kBRoundDecimal;
  exec::registerFunctionCallToSpecialForm(
      decimalName,
      std::make_unique<DecimalBRoundCallToSpecialForm>(decimalName));
}

} // namespace facebook::velox::functions::sparksql
