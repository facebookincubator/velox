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

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {
namespace {

std::pair<uint8_t, uint8_t>
getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale) {
  // After rounding we may need one more digit in the integral part,
  // e.g. 'decimal_round(9.9, 0)' -> '10', 'decimal_round(99, -1)' -> '100'.
  const int32_t integralLeastNumDigits = precision - scale + 1;
  if (roundScale < 0) {
    // Negative scale means we need to adjust `-scale` number of digits before
    // the decimal point, which means we need at least `-scale + 1` digits after
    // rounding, and the result scale is 0.
    const auto newPrecision = std::max(
        integralLeastNumDigits,
        -std::max(roundScale, -(int32_t)LongDecimalType::kMaxPrecision) + 1);
    // We have to accept the risk of overflow as we can't exceed the max
    // precision.
    return {std::min(newPrecision, (int32_t)LongDecimalType::kMaxPrecision), 0};
  }
  const uint8_t newScale = std::min((int32_t)scale, roundScale);
  // We have to accept the risk of overflow as we cannot exceed the max
  // precision.
  return {
      std::min(
          integralLeastNumDigits + newScale,
          (int32_t)LongDecimalType::kMaxPrecision),
      newScale};
}

template <typename TResult, typename TInput>
class DecimalRoundFunction : public exec::VectorFunction {
 public:
  DecimalRoundFunction(
      int32_t scale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision,
      uint8_t resultScale)
      : scale_(
            scale >= 0
                ? std::min(scale, (int32_t)LongDecimalType::kMaxPrecision)
                : std::max(scale, -(int32_t)LongDecimalType::kMaxPrecision)),
        inputPrecision_(inputPrecision),
        inputScale_(inputScale),
        resultPrecision_(resultPrecision),
        resultScale_(resultScale) {
    // Decide the rescale factor of divide and multiply when rounding to a
    // negative scale.
    auto rescaleFactor = [&](int32_t rescale) {
      VELOX_USER_CHECK_GT(
          rescale, 0, "A non-negative rescale value is expected.");
      return velox::DecimalUtil::kPowersOfTen[std::min(
          rescale, (int32_t)LongDecimalType::kMaxPrecision)];
    };
    if (scale_ < 0) {
      divideFactor_ = rescaleFactor(inputScale_ - scale_);
      multiplyFactor_ = rescaleFactor(-scale_);
    }
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Single-arg deterministic functions receive their only argument as flat or constant vector.");
    context.ensureWritable(rows, resultType, result);
    result->clearNulls(rows);
    auto rawResults =
        result->asUnchecked<FlatVector<TResult>>()->mutableRawValues();
    if (args[0]->isConstantEncoding()) {
      // Fast path for constant vector.
      applyConstant(rows, args[0], rawResults);
    } else {
      // Fast path for flat vector.
      applyFlat(rows, args[0], rawResults);
    }
  }

  bool supportsFlatNoNullsFastPath() const override {
    return true;
  }

 private:
  inline TResult applyRound(const TInput& input) const {
    if (scale_ >= 0) {
      TResult rescaledValue;
      const auto status =
          velox::DecimalUtil::rescaleWithRoundUp<TInput, TResult>(
              input,
              inputPrecision_,
              inputScale_,
              resultPrecision_,
              resultScale_,
              rescaledValue);
      VELOX_DCHECK(status.ok());
      return rescaledValue;
    } else {
      TResult rescaledValue;
      bool overflow;
      DecimalUtil::divideWithRoundUp<TResult, TInput, int128_t>(
          rescaledValue, input, divideFactor_.value(), 0, overflow);
      VELOX_USER_CHECK(!overflow);
      rescaledValue *= multiplyFactor_.value();
      return rescaledValue;
    }
  }

  void applyConstant(
      const SelectivityVector& rows,
      const VectorPtr& arg,
      TResult* rawResults) const {
    const TResult rounded =
        applyRound(arg->asUnchecked<ConstantVector<TInput>>()->valueAt(0));
    rows.applyToSelected([&](auto row) { rawResults[row] = rounded; });
  }

  void applyFlat(
      const SelectivityVector& rows,
      const VectorPtr& arg,
      TResult* rawResults) const {
    auto rawValues = arg->asUnchecked<FlatVector<TInput>>()->mutableRawValues();
    rows.applyToSelected(
        [&](auto row) { rawResults[row] = applyRound(rawValues[row]); });
  }

  const int32_t scale_;
  const uint8_t inputPrecision_;
  const uint8_t inputScale_;
  const uint8_t resultPrecision_;
  const uint8_t resultScale_;
  std::optional<int128_t> divideFactor_ = std::nullopt;
  std::optional<int128_t> multiplyFactor_ = std::nullopt;
};

std::shared_ptr<exec::VectorFunction> createDecimalRoundFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  int32_t scale = 0;
  if (inputArgs.size() > 1) {
    VELOX_CHECK(!inputArgs[1].constantValue->isNullAt(0));
    scale = inputArgs[1]
                .constantValue->template as<ConstantVector<int32_t>>()
                ->valueAt(0);
  }
  const auto inputType = inputArgs[0].type;
  auto [inputPrecision, inputScale] = getDecimalPrecisionScale(*inputType);
  auto [resultPrecision, resultScale] =
      getResultPrecisionScale(inputPrecision, inputScale, scale);

  if (inputType->isShortDecimal()) {
    if (resultPrecision <= velox::ShortDecimalType::kMaxPrecision) {
      return std::make_shared<DecimalRoundFunction<int64_t, int64_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    } else {
      return std::make_shared<DecimalRoundFunction<int128_t, int64_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    }
  } else {
    if (resultPrecision <= velox::ShortDecimalType::kMaxPrecision) {
      return std::make_shared<DecimalRoundFunction<int64_t, int128_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    } else {
      return std::make_shared<DecimalRoundFunction<int128_t, int128_t>>(
          scale, inputPrecision, inputScale, resultPrecision, resultScale);
    }
  }
}

std::vector<std::shared_ptr<exec::FunctionSignature>> decimalSignature() {
  return {
      exec::FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .build(),
      exec::FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .constantArgumentType("integer")
          .build()};
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_decimal_round,
    decimalSignature(),
    createDecimalRoundFunction);
} // namespace facebook::velox::functions::sparksql
