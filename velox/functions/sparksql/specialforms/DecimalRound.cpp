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
    std::optional<int128_t> divideFactor;
    std::optional<int128_t> multiplyFactor;
    int128_t overflowBound;
    bool roundsToZero;
    bool mayOverflow;
  };

  static int32_t clampScale(int32_t scale) {
    constexpr int32_t kMax = LongDecimalType::kMaxPrecision;
    return std::max(-kMax, std::min(scale, kMax));
  }

  static ScaleFactors computeFactors(
      int32_t scale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision);

  static int32_t extractConstantScaleArg(
      const exec::ExprPtr& expr,
      std::string_view funcName);

  template <template <typename, typename> class Policy>
  static std::shared_ptr<exec::VectorFunction> createFunction(
      const TypePtr& inputType,
      int32_t scale,
      const TypePtr& resultType) {
    const auto inputPrecisionScale = getDecimalPrecisionScale(*inputType);
    const auto resultPrecisionScale = getDecimalPrecisionScale(*resultType);
    const auto factors = computeFactors(
        scale,
        inputPrecisionScale.first,
        inputPrecisionScale.second,
        resultPrecisionScale.first);
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
      auto value =
          args[0]->template asUnchecked<ConstantVector<TInput>>()->valueAt(0);
      auto rounded = policy_.applyOne(value);
      rows.applyToSelected([&](auto row) {
        if constexpr (Policy::canOverflow) {
          if (FOLLY_UNLIKELY(!rounded.has_value())) {
            flat->setNull(row, true);
            if constexpr (Policy::reportOverflow) {
              context.setStatus(row, Policy::overflowStatus());
            }
            return;
          }
        }
        rawResults[row] = rounded.value();
      });
    } else {
      auto* rawValues =
          args[0]->template asUnchecked<FlatVector<TInput>>()->rawValues();
      rows.applyToSelected([&](auto row) {
        auto rounded = policy_.applyOne(rawValues[row]);
        if constexpr (Policy::canOverflow) {
          if (FOLLY_UNLIKELY(!rounded.has_value())) {
            flat->setNull(row, true);
            if constexpr (Policy::reportOverflow) {
              context.setStatus(row, Policy::overflowStatus());
            }
            return;
          }
        }
        rawResults[row] = rounded.value();
      });
    }
  }

  bool supportsFlatNoNullsFastPath() const override {
    return !policy_.mayOverflow();
  }

 private:
  Policy policy_;
};

namespace {

FOLLY_ALWAYS_INLINE bool overflowsAfterMultiplication(
    int128_t value,
    int128_t multiplier,
    int128_t overflowBound) {
  VELOX_DCHECK_GT(multiplier, 0);
  VELOX_DCHECK_GT(overflowBound, 0);
  const int128_t maxMagnitude = (overflowBound - 1) / multiplier;
  return value > maxMagnitude || value < -maxMagnitude;
}

template <typename TResult, typename TInput, typename Divide>
std::optional<TResult> applyRoundingPolicy(
    const TInput& input,
    const DecimalRoundOps::ScaleFactors& factors,
    Divide divide) {
  if (factors.roundsToZero) {
    return TResult{0};
  }

  int128_t rounded = input;
  if (factors.divideFactor.has_value()) {
    rounded = divide(rounded, factors.divideFactor.value());
  }

  if (factors.multiplyFactor.has_value()) {
    const int128_t multiplier = factors.multiplyFactor.value();
    if (overflowsAfterMultiplication(
            rounded, multiplier, factors.overflowBound)) {
      return std::nullopt;
    }
    rounded *= multiplier;
  }
  if (rounded >= factors.overflowBound || rounded <= -factors.overflowBound) {
    return std::nullopt;
  }
  return static_cast<TResult>(rounded);
}

// Half-up rounding (Spark's ROUND_HALF_UP).
template <typename TResult, typename TInput>
struct RoundHalfUpPolicy {
  static constexpr bool canOverflow = true;
  static constexpr bool reportOverflow = true;

  explicit RoundHalfUpPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  std::optional<TResult> applyOne(const TInput& input) const {
    return applyRoundingPolicy<TResult>(
        input, factors_, [](int128_t value, int128_t divisor) {
          int128_t rounded;
          DecimalUtil::divideWithRoundUp<int128_t, int128_t, int128_t>(
              rounded, value, divisor, false, 0, 0);
          return rounded;
        });
  }

  bool mayOverflow() const {
    return !factors_.roundsToZero && factors_.mayOverflow;
  }

  static Status overflowStatus() {
    return threadSkipErrorDetails()
        ? Status::UserError()
        : Status::UserError("Decimal overflow in round.");
  }

 private:
  const DecimalRoundOps::ScaleFactors factors_;
};

// Half-even rounding (Spark's ROUND_HALF_EVEN).
template <typename TResult, typename TInput>
struct RoundHalfEvenPolicy {
  static constexpr bool canOverflow = true;
  static constexpr bool reportOverflow = true;

  explicit RoundHalfEvenPolicy(const DecimalRoundOps::ScaleFactors& factors)
      : factors_(factors) {}

  std::optional<TResult> applyOne(const TInput& input) const {
    return applyRoundingPolicy<TResult>(
        input, factors_, [](int128_t value, int128_t divisor) {
          return DecimalUtil::divideWithRoundHalfEven(value, divisor);
        });
  }

  bool mayOverflow() const {
    return !factors_.roundsToZero && factors_.mayOverflow;
  }

  static Status overflowStatus() {
    return threadSkipErrorDetails()
        ? Status::UserError()
        : Status::UserError("Decimal overflow in bround.");
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
  static constexpr bool reportOverflow = false;

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
      if (overflowsAfterMultiplication(
              rounded, multiplier, factors_.overflowBound)) {
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

  bool mayOverflow() const {
    return factors_.mayOverflow;
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

enum class RoundMode {
  kHalfUp,
  kHalfEven,
};

/// Spark decimal round special form. Defined in .cpp because external callers
/// only need registerDecimalRoundingForms().
class DecimalRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  DecimalRoundCallToSpecialForm(RoundMode mode, std::string_view functionName)
      : mode_(mode), functionName_(functionName) {}

  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static std::pair<uint8_t, uint8_t>
  getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale);

 private:
  const RoundMode mode_;
  const std::string functionName_;
};

DecimalRoundOps::ScaleFactors DecimalRoundOps::computeFactors(
    int32_t scale,
    uint8_t inputPrecision,
    uint8_t inputScale,
    uint8_t resultPrecision) {
  ScaleFactors factors{};
  factors.overflowBound = DecimalUtil::kPowersOfTen[resultPrecision];
  factors.roundsToZero =
      static_cast<int64_t>(inputScale) - static_cast<int64_t>(scale) >
      LongDecimalType::kMaxPrecision;

  const int32_t clampedScale = clampScale(scale);
  if (clampedScale < static_cast<int32_t>(inputScale)) {
    const int32_t divDigits = static_cast<int32_t>(inputScale) - clampedScale;
    VELOX_DCHECK_GT(divDigits, 0);
    const int32_t cappedDivDigits = std::min(
        divDigits, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
    factors.divideFactor = DecimalUtil::kPowersOfTen[cappedDivDigits];
    if (clampedScale < 0) {
      VELOX_DCHECK_LE(
          -clampedScale, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
      factors.multiplyFactor = DecimalUtil::kPowersOfTen[-clampedScale];
    }
  }

  int128_t maxRoundedMagnitude = DecimalUtil::kPowersOfTen[inputPrecision] - 1;
  if (factors.divideFactor.has_value()) {
    const int128_t divisor = factors.divideFactor.value();
    const int128_t remainder = maxRoundedMagnitude % divisor;
    maxRoundedMagnitude /= divisor;
    if (remainder != 0) {
      ++maxRoundedMagnitude;
    }
  }
  factors.mayOverflow = factors.multiplyFactor.has_value()
      ? overflowsAfterMultiplication(
            maxRoundedMagnitude,
            factors.multiplyFactor.value(),
            factors.overflowBound)
      : maxRoundedMagnitude >= factors.overflowBound;
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
  VELOX_FAIL(
      "{} special form does not support type resolution.", functionName_);
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

  auto func = mode_ == RoundMode::kHalfUp
      ? DecimalRoundOps::createFunction<RoundHalfUpPolicy>(
            args[0]->type(), scale, type)
      : DecimalRoundOps::createFunction<RoundHalfEvenPolicy>(
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
      std::make_unique<DecimalRoundCallToSpecialForm>(
          RoundMode::kHalfUp, kRoundDecimal));
  exec::registerFunctionCallToSpecialForm(
      kBRoundDecimal,
      std::make_unique<DecimalRoundCallToSpecialForm>(
          RoundMode::kHalfEven, kBRoundDecimal));
  exec::registerFunctionCallToSpecialForm(
      kCeilDecimal,
      std::make_unique<DecimalCeilFloorCallToSpecialForm>(true, kCeilDecimal));
  exec::registerFunctionCallToSpecialForm(
      kFloorDecimal,
      std::make_unique<DecimalCeilFloorCallToSpecialForm>(
          false, kFloorDecimal));
}

} // namespace facebook::velox::functions::sparksql
