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

#include "velox/expression/SpecialFormRegistry.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Half-up rounding (Spark's ROUND_HALF_UP). Never overflows because
// the result type is guaranteed to accommodate the rounded value.
template <typename TResult, typename TInput>
struct RoundHalfUpPolicy {
  static constexpr bool canOverflow = false;

  explicit RoundHalfUpPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  std::optional<TResult> applyOne(const TInput& input) const {
    if (factors_.scale >= 0) {
      TResult rescaledValue;
      const auto status = DecimalUtil::rescaleWithRoundUp<TInput, TResult>(
          input,
          factors_.inputPrecision,
          factors_.inputScale,
          factors_.resultPrecision,
          factors_.resultScale,
          rescaledValue);
      VELOX_DCHECK(status.ok());
      return rescaledValue;
    }
    // When scale < 0, divideFactor is always set (scale < inputScale holds).
    // multiplyFactor is always set when scale < 0. Use value_or(1) as a
    // defensive default matching the no-op identity.
    TResult rescaledValue;
    DecimalUtil::divideWithRoundUp<TResult, TInput, int128_t>(
        rescaledValue, input, factors_.divideFactor.value_or(1), false, 0, 0);
    rescaledValue *= factors_.multiplyFactor.value_or(1);
    return rescaledValue;
  }

 private:
  const DecimalRoundOps::ScaleFactors factors_;
};

// Directional rounding (ceil toward +∞, floor toward -∞). May overflow
// because rounding away from zero can push a value past the maximum
// representable precision. The 'ceiling' parameter selects the direction:
// true rounds toward +∞, false rounds toward -∞.
template <typename TResult, typename TInput, bool ceiling>
struct DirectionalRoundPolicy {
  static constexpr bool canOverflow = true;

  explicit DirectionalRoundPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  std::optional<TResult> applyOne(const TInput& input) const {
    if (!factors_.divideFactor.has_value()) {
      auto out = static_cast<int128_t>(input);
      if (out >= factors_.overflowBound || out <= -factors_.overflowBound) {
        return std::nullopt;
      }
      return static_cast<TResult>(out);
    }
    auto in = static_cast<int128_t>(input);
    const int128_t divisor = factors_.divideFactor.value();
    const int128_t quotient = in / divisor;
    const int128_t remainder = in % divisor;
    int128_t rounded = quotient + adjustment(remainder);
    if (factors_.multiplyFactor.has_value()) {
      const int128_t multiplier = factors_.multiplyFactor.value();
      const int128_t maxAbs = factors_.overflowBound / multiplier;
      if (rounded >= maxAbs || rounded <= -maxAbs) {
        return std::nullopt;
      }
      rounded *= multiplier;
    }
    if (rounded >= factors_.overflowBound ||
        rounded <= -factors_.overflowBound) {
      return std::nullopt;
    }
    return static_cast<TResult>(rounded);
  }

 private:
  static int128_t adjustment(int128_t remainder) {
    if constexpr (ceiling) {
      return remainder > 0 ? 1 : 0;
    } else {
      return remainder < 0 ? -1 : 0;
    }
  }

  const DecimalRoundOps::ScaleFactors factors_;
};

template <typename TResult, typename TInput>
struct CeilPolicy : DirectionalRoundPolicy<TResult, TInput, true> {
  using DirectionalRoundPolicy<TResult, TInput, true>::DirectionalRoundPolicy;
};

template <typename TResult, typename TInput>
struct FloorPolicy : DirectionalRoundPolicy<TResult, TInput, false> {
  using DirectionalRoundPolicy<TResult, TInput, false>::DirectionalRoundPolicy;
};

} // namespace

DecimalRoundOps::ScaleFactors DecimalRoundOps::computeFactors(
    int32_t scale,
    uint8_t inputPrecision,
    uint8_t inputScale,
    uint8_t resultPrecision,
    uint8_t resultScale) {
  ScaleFactors factors{};
  factors.scale = clampScale(scale);
  factors.inputPrecision = inputPrecision;
  factors.inputScale = inputScale;
  factors.resultPrecision = resultPrecision;
  factors.resultScale = resultScale;
  factors.overflowBound = DecimalUtil::kPowersOfTen[resultPrecision];

  if (factors.scale < static_cast<int32_t>(inputScale)) {
    const int32_t divDigits = static_cast<int32_t>(inputScale) - factors.scale;
    VELOX_DCHECK_GT(divDigits, 0);
    const int32_t cappedDivDigits = std::min(
        divDigits, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
    factors.divideFactor = DecimalUtil::kPowersOfTen[cappedDivDigits];
    if (factors.scale < 0) {
      VELOX_DCHECK_LE(
          -factors.scale, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
      factors.multiplyFactor = DecimalUtil::kPowersOfTen[-factors.scale];
    }
  }
  return factors;
}

int32_t DecimalRoundOps::extractConstantScaleArg(
    const exec::ExprPtr& expr,
    std::string_view funcName) {
  VELOX_USER_CHECK_EQ(
      expr->type()->kind(),
      TypeKind::INTEGER,
      "The second argument of {} must be INTEGER, got: {}.",
      funcName,
      expr->type()->toString());
  auto constantExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(expr);
  VELOX_USER_CHECK_NOT_NULL(
      constantExpr,
      "The second argument of {} must be a constant expression.",
      funcName);
  VELOX_CHECK(
      constantExpr->value()->isConstantEncoding(),
      "ConstantExpr must hold a constant-encoded vector.");
  auto* constantVector =
      constantExpr->value()->asUnchecked<ConstantVector<int32_t>>();
  VELOX_USER_CHECK(
      !constantVector->isNullAt(0),
      "The second argument of {} must not be NULL.",
      funcName);
  return constantVector->valueAt(0);
}

std::pair<uint8_t, uint8_t>
DecimalRoundCallToSpecialForm::getResultPrecisionScale(
    uint8_t precision,
    uint8_t scale,
    int32_t roundScale) {
  const int32_t integralLeastNumDigits = precision - scale + 1;
  if (roundScale < 0) {
    const auto newPrecision = std::max(
        integralLeastNumDigits,
        -std::max(
            roundScale, -static_cast<int32_t>(LongDecimalType::kMaxPrecision)) +
            1);
    return {
        std::min(
            newPrecision, static_cast<int32_t>(LongDecimalType::kMaxPrecision)),
        0};
  }
  const uint8_t newScale = std::min(static_cast<int32_t>(scale), roundScale);
  return {
      std::min(
          integralLeastNumDigits + newScale,
          static_cast<int32_t>(LongDecimalType::kMaxPrecision)),
      newScale};
}

TypePtr DecimalRoundCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL("Decimal round function does not support type resolution.");
}

exec::ExprPtr DecimalRoundCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK(
      type->isDecimal(),
      "The result type of {} must be decimal.",
      kRoundDecimal);
  VELOX_USER_CHECK(
      args.size() >= 1 && args.size() <= 2,
      "{} expects one or two arguments.",
      kRoundDecimal);
  VELOX_USER_CHECK(
      args[0]->type()->isDecimal(),
      "The first argument of {} must be decimal.",
      kRoundDecimal);

  int32_t scale = 0;
  if (args.size() > 1) {
    scale = DecimalRoundOps::extractConstantScaleArg(args[1], kRoundDecimal);
  }

  auto func = DecimalRoundOps::createFunction<RoundHalfUpPolicy>(
      args[0]->type(), scale, type);

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(func),
      exec::VectorFunctionMetadata{},
      std::string(kRoundDecimal),
      trackCpuUsage);
}

namespace {

// Special form for decimal_ceil and decimal_floor. Shares
// getResultPrecisionScale and validation with decimal_round.
class DecimalCeilFloorCallToSpecialForm
    : public exec::FunctionCallToSpecialForm {
 public:
  DecimalCeilFloorCallToSpecialForm(bool ceiling, std::string_view funcName)
      : ceiling_(ceiling), funcName_(funcName) {}

  TypePtr resolveType(const std::vector<TypePtr>& /*argTypes*/) override {
    VELOX_FAIL("{} special form does not support type resolution.", funcName_);
  }

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& /*config*/) override {
    VELOX_USER_CHECK(
        type->isDecimal(), "The result type of {} must be decimal.", funcName_);
    VELOX_USER_CHECK_EQ(
        args.size(),
        2,
        "{} expects two arguments (decimal value and target scale).",
        funcName_);
    VELOX_USER_CHECK(
        args[0]->type()->isDecimal(),
        "The first argument of {} must be decimal.",
        funcName_);

    const int32_t scale =
        DecimalRoundOps::extractConstantScaleArg(args[1], funcName_);

    auto func = ceiling_ ? DecimalRoundOps::createFunction<CeilPolicy>(
                               args[0]->type(), scale, type)
                         : DecimalRoundOps::createFunction<FloorPolicy>(
                               args[0]->type(), scale, type);

    return std::make_shared<exec::Expr>(
        type,
        std::move(args),
        std::move(func),
        exec::VectorFunctionMetadata{},
        std::string(funcName_),
        trackCpuUsage);
  }

 private:
  const bool ceiling_;
  const std::string funcName_;
};

} // namespace

void registerDecimalRoundingForms() {
  exec::registerFunctionCallToSpecialForm(
      DecimalRoundCallToSpecialForm::kRoundDecimal,
      std::make_unique<DecimalRoundCallToSpecialForm>());
  exec::registerFunctionCallToSpecialForm(
      DecimalRoundCallToSpecialForm::kCeilDecimal,
      std::make_unique<DecimalCeilFloorCallToSpecialForm>(
          true, DecimalRoundCallToSpecialForm::kCeilDecimal));
  exec::registerFunctionCallToSpecialForm(
      DecimalRoundCallToSpecialForm::kFloorDecimal,
      std::make_unique<DecimalCeilFloorCallToSpecialForm>(
          false, DecimalRoundCallToSpecialForm::kFloorDecimal));
}

} // namespace facebook::velox::functions::sparksql
