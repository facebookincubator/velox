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

#include "velox/expression/ConstantExpr.h"

namespace facebook::velox::functions::sparksql {
namespace {
template <typename R, typename A>
class DecimalRoundFunction : public exec::VectorFunction {
 public:
  explicit DecimalRoundFunction(
      int32_t scale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision,
      uint8_t resultScale)
      : scale_(scale),
        inputPrecision_(inputPrecision),
        inputScale_(inputScale),
        resultPrecision_(resultPrecision),
        resultScale_(resultScale) {
    // auto [p, s] = getResultPrecisionScale(inputPrecision, inputScale, scale);
    // VELOX_USER_CHECK_EQ(p, resultPrecision);
    // VELOX_USER_CHECK_EQ(s, resultScale);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    applyRoundRows(rows, args, resultType, context, result);
  }

  bool supportsFlatNoNullsFastPath() const override {
    return true;
  }

 private:
  R* prepareResults(
      const SelectivityVector& rows,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const {
    context.ensureWritable(rows, resultType, result);
    result->clearNulls(rows);
    return result->asUnchecked<FlatVector<R>>()->mutableRawValues();
  }

  void applyRoundRows(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const {
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Single-arg deterministic functions receive their only argument as flat or constant vector.");
    auto rawResults = prepareResults(rows, resultType, context, result);
    if (args[0]->isConstantEncoding()) {
      // Fast path for constant vectors.
      auto constant = args[0]->asUnchecked<ConstantVector<A>>()->valueAt(0);
      context.applyToSelectedNoThrow(
          rows, [&](auto row) { rawResults[row] = applyRound(constant); });
    } else {
      // Fast path for flat.
      auto rawA = args[0]->asUnchecked<FlatVector<A>>()->mutableRawValues();
      context.applyToSelectedNoThrow(
          rows, [&](auto row) { rawResults[row] = applyRound(rawA[row]); });
    }
  }

  inline R applyRound(const A input) const {
    if (scale_ >= 0) {
      bool overflow = false;
      auto rescaledValue = DecimalUtil::rescaleWithRoundUp<A, R>(
          input,
          inputPrecision_,
          inputScale_,
          resultPrecision_,
          resultScale_,
          overflow,
          false);
      VELOX_DCHECK(rescaledValue.has_value());
      return rescaledValue.value();
    } else {
      auto rescaleFactor = DecimalUtil::kPowersOfTen[inputScale_ - scale_];
      R rescaledValue;
      DecimalUtil::divideWithRoundUp<R, A, int128_t>(
          rescaledValue, input, rescaleFactor, false, 0, 0);
      rescaledValue *= DecimalUtil::kPowersOfTen[-scale_];
      return rescaledValue;
    }
  }

  const int32_t scale_;
  const uint8_t inputPrecision_;
  const uint8_t inputScale_;
  const uint8_t resultPrecision_;
  const uint8_t resultScale_;
};

std::shared_ptr<exec::VectorFunction> createDecimalRound(
    int32_t scale,
    const TypePtr& inputType,
    const TypePtr& resultType) {
  auto [inputPrecision, inputScale] = getDecimalPrecisionScale(*inputType);
  auto [resultPrecision, resultScale] = getDecimalPrecisionScale(*resultType);
  if (inputType->isShortDecimal()) {
    if (resultType->isShortDecimal()) {
      return std::make_shared<DecimalRoundFunction<int64_t, int64_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    } else {
      return std::make_shared<DecimalRoundFunction<int128_t, int64_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    }
  } else {
    if (resultType->isShortDecimal()) {
      return std::make_shared<DecimalRoundFunction<int64_t, int128_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    } else {
      return std::make_shared<DecimalRoundFunction<int128_t, int128_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    }
  }
}
}; // namespace

std::pair<uint8_t, uint8_t>
getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale) {
  // After rounding we may need one more digit in the integral part.
  int32_t integralLeastNumDigits = precision - scale + 1;
  if (roundScale < 0) {
    // Negative scale means we need to adjust `-scale` number of digits before
    // the decimal point, which means we need at least `-scale + 1` digits after
    // rounding, and the result scale is 0.
    auto newPrecision = std::max(integralLeastNumDigits, -roundScale + 1);
    // We have to accept the risk of overflow as we can't exceed the max
    // precision.
    return {std::min(newPrecision, 38), 0};
  } else {
    uint8_t newScale = std::min((int32_t)scale, roundScale);
    // We have to accept the risk of overflow as we can't exceed the max
    // precision.
    return {std::min(integralLeastNumDigits + newScale, 38), newScale};
  }
}

TypePtr DecimalRoundCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  VELOX_FAIL("Decimal round function does not support type resolution.");
}

exec::ExprPtr DecimalRoundCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_USER_CHECK_GE(
      compiledChildren.size(),
      1,
      "Decimal_round expects one or two arguments.");
  VELOX_USER_CHECK_LE(
      compiledChildren.size(),
      2,
      "Decimal_round expects one or two arguments.");
  VELOX_USER_CHECK(
      compiledChildren[0]->type()->isDecimal(),
      "The first argument of decimal_round should be of decimal type.");
  int32_t scale = 0;
  if (compiledChildren.size() > 1) {
    VELOX_USER_CHECK_EQ(
        compiledChildren[1]->type()->kind(),
        TypeKind::INTEGER,
        "The second argument of decimal_round should be of integer type.");
    auto constantExpr =
        std::dynamic_pointer_cast<exec::ConstantExpr>(compiledChildren[1]);
    VELOX_USER_CHECK_NOT_NULL(constantExpr);
    VELOX_USER_CHECK(constantExpr->value()->isConstantEncoding());
    scale =
        constantExpr->value()->asUnchecked<ConstantVector<int32_t>>()->valueAt(
            0);
  }
  return std::make_shared<exec::Expr>(
      type,
      std::move(compiledChildren),
      createDecimalRound(scale, compiledChildren[0]->type(), type),
      kRoundDecimal,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
