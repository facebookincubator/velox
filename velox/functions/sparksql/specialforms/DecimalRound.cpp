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

#include "velox/functions/sparksql/specialforms/DecimalRound.h"

#include <optional>
#include <string_view>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/SpecialFormRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction;

// Shares scale arithmetic, result typing and physical dispatch across policies.
class DecimalRoundOps {
 public:
  template <typename T>
  struct TypeTag {
    using type = T;
  };

  struct ScaleFactors {
    // A missing divisor denotes dropping more than 38 digits.
    std::optional<int128_t> divideFactor;
    int128_t multiplyFactor;
    // Check the rounded quotient before multiplication or physical narrowing.
    int128_t limitBeforeMultiply;
  };

  static ScaleFactors
  computeFactors(int32_t scale, uint8_t inputScale, uint8_t resultPrecision) {
    const int64_t digitsToDrop = static_cast<int64_t>(inputScale) - scale;
    const int64_t multiplyDigits = scale < 0 ? -static_cast<int64_t>(scale) : 0;
    ScaleFactors factors;
    if (digitsToDrop <= LongDecimalType::kMaxPrecision) {
      factors.divideFactor =
          DecimalUtil::kPowersOfTen[std::max<int64_t>(0, digitsToDrop)];
    }
    if (multiplyDigits > LongDecimalType::kMaxPrecision) {
      // Only zero fits. Avoid constructing an unrepresentable power of ten.
      factors.multiplyFactor = 1;
      factors.limitBeforeMultiply = 0;
    } else {
      factors.multiplyFactor = DecimalUtil::kPowersOfTen[multiplyDigits];
      factors.limitBeforeMultiply =
          (DecimalUtil::kPowersOfTen[resultPrecision] - 1) /
          factors.multiplyFactor;
    }
    return factors;
  }

  static TypePtr resultType(const TypePtr& inputType, int32_t scale) {
    const auto [precision, inputScale] = getDecimalPrecisionScale(*inputType);
    const int64_t integralDigits = precision - inputScale + 1;
    if (scale < 0) {
      return DECIMAL(
          std::min<int64_t>(
              std::max(integralDigits, 1 - static_cast<int64_t>(scale)),
              LongDecimalType::kMaxPrecision),
          0);
    }
    const auto resultScale = std::min<int32_t>(inputScale, scale);
    return DECIMAL(
        std::min<int64_t>(
            integralDigits + resultScale, LongDecimalType::kMaxPrecision),
        resultScale);
  }

  static std::optional<int32_t> extractConstantScaleArg(
      const exec::ExprPtr& expr,
      std::string_view name) {
    VELOX_USER_CHECK(
        expr->type()->isInteger(),
        "The second argument of {} must be INTEGER.",
        name);
    const auto constant = std::dynamic_pointer_cast<exec::ConstantExpr>(expr);
    VELOX_USER_CHECK_NOT_NULL(
        constant,
        "The second argument of {} must be a constant expression.",
        name);
    const auto* value =
        constant->value()->asUnchecked<ConstantVector<int32_t>>();
    return value->isNullAt(0) ? std::nullopt
                              : std::optional<int32_t>(value->valueAt(0));
  }

  template <typename Policy>
  static std::shared_ptr<exec::VectorFunction> createFunction(
      const TypePtr& inputType,
      int32_t scale,
      const TypePtr& resultType,
      const std::string& name) {
    const auto factors = computeFactors(
        scale,
        getDecimalPrecisionScale(*inputType).second,
        getDecimalPrecisionScale(*resultType).first);
    return dispatchTypes(
        inputType,
        resultType,
        [&](auto resultTag,
            auto inputTag) -> std::shared_ptr<exec::VectorFunction> {
          using TResult = typename decltype(resultTag)::type;
          using TInput = typename decltype(inputTag)::type;
          return std::make_shared<
              DecimalRoundFunction<TResult, TInput, Policy>>(factors, name);
        });
  }

 private:
  template <typename Fn>
  static auto
  dispatchTypes(const TypePtr& inputType, const TypePtr& resultType, Fn&& fn) {
    if (inputType->isShortDecimal()) {
      if (resultType->isShortDecimal()) {
        return fn(TypeTag<int64_t>{}, TypeTag<int64_t>{});
      }
      return fn(TypeTag<int128_t>{}, TypeTag<int64_t>{});
    }
    if (resultType->isShortDecimal()) {
      return fn(TypeTag<int64_t>{}, TypeTag<int128_t>{});
    }
    return fn(TypeTag<int128_t>{}, TypeTag<int128_t>{});
  }
};

// Applies the policy in int128, then checks precision before storing the
// result.
template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction : public exec::VectorFunction {
 public:
  DecimalRoundFunction(
      const DecimalRoundOps::ScaleFactors& factors,
      std::string name)
      : factors_(factors), name_(std::move(name)) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    auto* flat = result->asUnchecked<FlatVector<TResult>>();
    flat->clearNulls(rows);
    auto* rawResults = flat->mutableRawValues();
    DecodedVector input(*args[0], rows);
    rows.applyToSelected([&](auto row) {
      const auto rounded =
          Policy::divide(input.valueAt<TInput>(row), factors_.divideFactor);
      if (rounded > factors_.limitBeforeMultiply ||
          rounded < -factors_.limitBeforeMultiply) {
        context.setStatus(
            row,
            threadSkipErrorDetails()
                ? Status::UserError()
                : Status::UserError("Decimal overflow in {}.", name_));
        return;
      }
      rawResults[row] = static_cast<TResult>(rounded * factors_.multiplyFactor);
    });
  }

 private:
  const DecimalRoundOps::ScaleFactors factors_;
  const std::string name_;
};

// Rounds ties away from zero using the shared decimal division implementation.
struct RoundHalfUpPolicy {
  static int128_t divide(int128_t input, std::optional<int128_t> divisor) {
    if (!divisor.has_value()) {
      return 0;
    }
    int128_t result;
    DecimalUtil::divideWithRoundUp(result, input, divisor.value(), false, 0, 0);
    return result;
  }
};

// Preserves the signed remainder even when the divisor exceeds 10^38.
template <bool ceiling>
struct DirectionalRoundPolicy {
  static int128_t divide(int128_t input, std::optional<int128_t> divisor) {
    const auto quotient = divisor.has_value() ? input / divisor.value() : 0;
    const auto remainder =
        divisor.has_value() ? input % divisor.value() : input;
    if constexpr (ceiling) {
      return quotient + (remainder > 0 ? 1 : 0);
    } else {
      return quotient - (remainder < 0 ? 1 : 0);
    }
  }
};

// Validates all decimal forms before creating their policy-based evaluator.
template <typename Policy>
class DecimalRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  DecimalRoundCallToSpecialForm(std::string name, bool allowUnary)
      : name_(std::move(name)), allowUnary_(allowUnary) {}

  TypePtr resolveType(const std::vector<TypePtr>&) override {
    VELOX_USER_FAIL("{} requires an explicitly resolved result type.", name_);
  }

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig&) override {
    VELOX_USER_CHECK(
        args.size() == 2 || (allowUnary_ && args.size() == 1),
        "{} expects {} arguments.",
        name_,
        allowUnary_ ? "one or two" : "two");
    VELOX_USER_CHECK(
        args[0]->type()->isDecimal(),
        "The first argument of {} must be decimal.",
        name_);
    const auto scale = args.size() == 2
        ? DecimalRoundOps::extractConstantScaleArg(args[1], name_)
        : std::optional<int32_t>{0};
    // Spark's resolved type uses scale zero when the constant scale is NULL.
    const auto expected =
        DecimalRoundOps::resultType(args[0]->type(), scale.value_or(0));
    VELOX_USER_CHECK(
        type->equivalent(*expected),
        "Invalid result type for {}: expected {}, got {}.",
        name_,
        expected->toString(),
        type->toString());
    if (!scale.has_value()) {
      const auto constant =
          std::static_pointer_cast<exec::ConstantExpr>(args[1]);
      return std::make_shared<exec::ConstantExpr>(
          BaseVector::createNullConstant(type, 1, constant->value()->pool()));
    }
    auto function = DecimalRoundOps::createFunction<Policy>(
        args[0]->type(), scale.value(), type, name_);
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
  const bool allowUnary_;
};

} // namespace

void registerDecimalRoundSpecialForm(const std::string& name) {
  exec::registerFunctionCallToSpecialForm(
      name,
      std::make_unique<DecimalRoundCallToSpecialForm<RoundHalfUpPolicy>>(
          name, true));
}

void registerDecimalRoundingForms() {
  registerDecimalRoundSpecialForm(kRoundDecimal);
  registerDecimalRoundSpecialForm(kSparkRoundDecimal);
  exec::registerFunctionCallToSpecialForm(
      kCeilDecimal,
      std::make_unique<
          DecimalRoundCallToSpecialForm<DirectionalRoundPolicy<true>>>(
          kCeilDecimal, false));
  exec::registerFunctionCallToSpecialForm(
      kFloorDecimal,
      std::make_unique<
          DecimalRoundCallToSpecialForm<DirectionalRoundPolicy<false>>>(
          kFloorDecimal, false));
}

} // namespace facebook::velox::functions::sparksql
