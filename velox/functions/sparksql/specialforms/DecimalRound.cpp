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

#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include <folly/Expected.h>

#include "velox/common/base/Status.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/SpecialFormRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/BRound.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {

// Forward declaration — defined below after DecimalRoundOps.
template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction;

/// Shared infrastructure for decimal rounding special forms.
class DecimalRoundOps {
 public:
  template <typename T>
  struct TypeTag {
    using type = T;
  };

  /// Precomputed scale factors shared by all rounding policies.
  struct ScaleFactors {
    int32_t scale;
    int32_t requestedScale;
    int64_t roundingDigitCount;
    bool scaleUnderflows;
    uint8_t inputPrecision;
    uint8_t inputScale;
    uint8_t resultPrecision;
    uint8_t resultScale;
    std::optional<int128_t> divideFactor;
    std::optional<int128_t> multiplyFactor;
    int128_t overflowBound;
  };

  static int32_t clampScale(int32_t scale) {
    constexpr int32_t kMax = LongDecimalType::kMaxPrecision;
    return std::max(-kMax, std::min(scale, kMax));
  }

  static ScaleFactors computeFactors(
      int32_t scale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision,
      uint8_t resultScale);

  static int32_t extractConstantScaleArg(
      const exec::ExprPtr& expr,
      std::string_view funcName);

  template <template <typename, typename> class Policy>
  static std::shared_ptr<exec::VectorFunction> createFunction(
      const TypePtr& inputType,
      int32_t scale,
      const TypePtr& resultType) {
    const auto [inputPrecision, inputScale] =
        getDecimalPrecisionScale(*inputType);
    const auto [resultPrecision, resultScale] =
        getDecimalPrecisionScale(*resultType);
    const auto factors = computeFactors(
        scale, inputPrecision, inputScale, resultPrecision, resultScale);
    return dispatchTypes(
        inputType,
        resultType,
        [&](auto resultTag,
            auto inputTag) -> std::shared_ptr<exec::VectorFunction> {
          using TResult = typename decltype(resultTag)::type;
          using TInput = typename decltype(inputTag)::type;
          return std::make_shared<
              DecimalRoundFunction<TResult, TInput, Policy<TResult, TInput>>>(
              factors);
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

/// Generic VectorFunction for decimal rounding, parameterized on a Policy.
template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction : public exec::VectorFunction {
 public:
  explicit DecimalRoundFunction(const DecimalRoundOps::ScaleFactors& factors)
      : policy_(factors) {}

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

    if (args[0]->isConstantEncoding()) {
      const auto value =
          args[0]->template asUnchecked<ConstantVector<TInput>>()->valueAt(0);
      const auto rounded = policy_.applyOne(value);
      if (FOLLY_UNLIKELY(rounded.hasError())) {
        flat->addNulls(rows);
        context.setStatuses(rows, rounded.error());
        return;
      }
      if (FOLLY_UNLIKELY(!rounded.value().has_value())) {
        flat->addNulls(rows);
        return;
      }
      rows.applyToSelected(
          [&](auto row) { rawResults[row] = rounded.value().value(); });
    } else {
      const auto* rawValues =
          args[0]->template asUnchecked<FlatVector<TInput>>()->rawValues();
      rows.applyToSelected([&](auto row) {
        const auto rounded = policy_.applyOne(rawValues[row]);
        if (FOLLY_UNLIKELY(rounded.hasError())) {
          flat->setNull(row, true);
          context.setStatus(row, rounded.error());
        } else if (FOLLY_UNLIKELY(!rounded.value().has_value())) {
          flat->setNull(row, true);
        } else {
          rawResults[row] = rounded.value().value();
        }
      });
    }
  }

  bool supportsFlatNoNullsFastPath() const override {
    return !Policy::kMayReturnNull;
  }

 private:
  Policy policy_;
};

namespace {

int128_t broundUnscaled(int128_t unscaled, int32_t roundingDigitCount) {
  const int128_t divisor = DecimalUtil::kPowersOfTen[roundingDigitCount];
  const int128_t quotient = unscaled / divisor;
  const int128_t remainder = unscaled % divisor;
  if (remainder == 0) {
    return quotient;
  }

  const int128_t absoluteRemainder = remainder < 0 ? -remainder : remainder;
  const int128_t half = divisor / 2;
  if (absoluteRemainder > half ||
      (absoluteRemainder == half && quotient % 2 != 0)) {
    return quotient + (unscaled < 0 ? -1 : 1);
  }
  return quotient;
}

template <typename TInput>
Status checkScaleUnderflow(
    const TInput input,
    const DecimalRoundOps::ScaleFactors& factors) {
  if (!factors.scaleUnderflows || input == 0) {
    return Status::OK();
  }
  if (threadSkipErrorDetails()) {
    return Status::UserError();
  }
  return Status::UserError(
      "Underflow while rounding to scale {}", factors.requestedScale);
}

Status decimalOverflowStatus(const DecimalRoundOps::ScaleFactors& factors) {
  if (threadSkipErrorDetails()) {
    return Status::UserError();
  }
  return Status::UserError(
      "Overflow while rounding decimal to precision {} and scale {}",
      factors.resultPrecision,
      factors.resultScale);
}

// Half-up rounding (Spark's ROUND_HALF_UP).
template <typename TResult, typename TInput>
struct RoundHalfUpPolicy {
  static constexpr bool kMayReturnNull = false;

  explicit RoundHalfUpPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  Expected<std::optional<TResult>> applyOne(const TInput& input) const {
    const auto underflowStatus = checkScaleUnderflow(input, factors_);
    if (FOLLY_UNLIKELY(!underflowStatus.ok())) {
      return folly::makeUnexpected(underflowStatus);
    }
    if (factors_.scaleUnderflows) {
      return std::optional<TResult>{0};
    }

    if (factors_.requestedScale >= 0) {
      TResult rescaledValue;
      const auto status = DecimalUtil::rescaleWithRoundUp<TInput, TResult>(
          input,
          factors_.inputPrecision,
          factors_.inputScale,
          factors_.resultPrecision,
          factors_.resultScale,
          rescaledValue);
      if (FOLLY_UNLIKELY(!status.ok())) {
        return folly::makeUnexpected(decimalOverflowStatus(factors_));
      }
      return std::optional<TResult>{rescaledValue};
    }

    if (factors_.roundingDigitCount > LongDecimalType::kMaxPrecision) {
      return std::optional<TResult>{0};
    }

    TResult rescaledValue;
    DecimalUtil::divideWithRoundUp<TResult, TInput, int128_t>(
        rescaledValue, input, factors_.divideFactor.value(), false, 0, 0);
    rescaledValue *= factors_.multiplyFactor.value();
    if (FOLLY_UNLIKELY(!DecimalUtil::valueInPrecisionRange(
            rescaledValue, factors_.resultPrecision))) {
      return folly::makeUnexpected(decimalOverflowStatus(factors_));
    }
    return std::optional<TResult>{rescaledValue};
  }

 private:
  const DecimalRoundOps::ScaleFactors factors_;
};

// Half-even rounding (Spark's ROUND_HALF_EVEN).
template <typename TResult, typename TInput>
struct RoundHalfEvenPolicy {
  static constexpr bool kMayReturnNull = false;

  explicit RoundHalfEvenPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  Expected<std::optional<TResult>> applyOne(const TInput& input) const {
    const auto underflowStatus = checkScaleUnderflow(input, factors_);
    if (FOLLY_UNLIKELY(!underflowStatus.ok())) {
      return folly::makeUnexpected(underflowStatus);
    }
    if (factors_.scaleUnderflows) {
      return std::optional<TResult>{0};
    }
    if (factors_.roundingDigitCount > LongDecimalType::kMaxPrecision) {
      return std::optional<TResult>{0};
    }

    int128_t rounded = static_cast<int128_t>(input);
    if (factors_.roundingDigitCount > 0) {
      rounded = broundUnscaled(
          rounded, static_cast<int32_t>(factors_.roundingDigitCount));
    }
    if (factors_.requestedScale < 0) {
      rounded *= factors_.multiplyFactor.value();
    }
    if (FOLLY_UNLIKELY(!DecimalUtil::valueInPrecisionRange(
            rounded, factors_.resultPrecision))) {
      return folly::makeUnexpected(decimalOverflowStatus(factors_));
    }
    return std::optional<TResult>{static_cast<TResult>(rounded)};
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
  static constexpr bool kMayReturnNull = true;

  explicit DirectionalRoundPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  Expected<std::optional<TResult>> applyOne(const TInput& input) const {
    if (!factors_.divideFactor.has_value()) {
      auto out = static_cast<int128_t>(input);
      if (out >= factors_.overflowBound || out <= -factors_.overflowBound) {
        return std::optional<TResult>{};
      }
      return std::optional<TResult>{static_cast<TResult>(out)};
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
        return std::optional<TResult>{};
      }
      rounded *= multiplier;
    }
    if (rounded >= factors_.overflowBound ||
        rounded <= -factors_.overflowBound) {
      return std::optional<TResult>{};
    }
    return std::optional<TResult>{static_cast<TResult>(rounded)};
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

/// Spark decimal rounding special form. Defined in .cpp because external
/// callers only need registerDecimalRoundingForms().
class DecimalRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  DecimalRoundCallToSpecialForm(bool halfEven, std::string_view functionName)
      : halfEven_(halfEven), functionName_(functionName) {}

  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static std::pair<uint8_t, uint8_t>
  getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale);

 private:
  const bool halfEven_;
  const std::string functionName_;
};

DecimalRoundOps::ScaleFactors DecimalRoundOps::computeFactors(
    int32_t scale,
    uint8_t inputPrecision,
    uint8_t inputScale,
    uint8_t resultPrecision,
    uint8_t resultScale) {
  ScaleFactors factors{};
  factors.scale = clampScale(scale);
  factors.requestedScale = scale;
  factors.roundingDigitCount = scale >= 0
      ? static_cast<int64_t>(inputScale) - resultScale
      : static_cast<int64_t>(inputScale) - scale;
  factors.scaleUnderflows =
      factors.roundingDigitCount > kMaxJavaBigIntegerPowerOfTenExponent;
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
    // Keep this term negative where Spark's -roundScale + 1 overflows Int.
    const int32_t requiredPrecision =
        roundScale <= std::numeric_limits<int32_t>::min() + 1
        ? std::numeric_limits<int32_t>::min()
        : -roundScale + 1;
    const auto newPrecision =
        std::max(integralLeastNumDigits, requiredPrecision);
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
      functionName_);
  VELOX_USER_CHECK(
      args.size() >= 1 && args.size() <= 2,
      "{} expects one or two arguments.",
      functionName_);
  VELOX_USER_CHECK(
      args[0]->type()->isDecimal(),
      "The first argument of {} must be decimal.",
      functionName_);

  int32_t scale = 0;
  if (args.size() > 1) {
    scale = DecimalRoundOps::extractConstantScaleArg(args[1], functionName_);
  }

  auto func = halfEven_ ? DecimalRoundOps::createFunction<RoundHalfEvenPolicy>(
                              args[0]->type(), scale, type)
                        : DecimalRoundOps::createFunction<RoundHalfUpPolicy>(
                              args[0]->type(), scale, type);

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(func),
      exec::VectorFunctionMetadata{},
      functionName_,
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
      kRoundDecimal,
      std::make_unique<DecimalRoundCallToSpecialForm>(false, kRoundDecimal));
  exec::registerFunctionCallToSpecialForm(
      kBRoundDecimal,
      std::make_unique<DecimalRoundCallToSpecialForm>(true, kBRoundDecimal));
  exec::registerFunctionCallToSpecialForm(
      kCeilDecimal,
      std::make_unique<DecimalCeilFloorCallToSpecialForm>(true, kCeilDecimal));
  exec::registerFunctionCallToSpecialForm(
      kFloorDecimal,
      std::make_unique<DecimalCeilFloorCallToSpecialForm>(
          false, kFloorDecimal));
}

} // namespace facebook::velox::functions::sparksql
