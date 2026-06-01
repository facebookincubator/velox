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
#include "velox/functions/sparksql/specialforms/DecimalRoundBase.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Half-up rounding (Spark's ROUND_HALF_UP). Never overflows.
template <typename TResult, typename TInput>
struct RoundHalfUpPolicy {
  static constexpr bool canOverflow = false;

  using Factors = DecimalRoundBase::ScaleFactors;

  explicit RoundHalfUpPolicy(const Factors& f)
      : scaleNonNegative_(f.scale >= 0),
        inputPrecision_(f.inputPrecision),
        inputScale_(f.inputScale),
        resultPrecision_(f.resultPrecision),
        resultScale_(f.resultScale),
        divideFactor_(f.divideFactor.value_or(1)),
        multiplyFactor_(f.multiplyFactor.value_or(1)) {
    const auto [p, s] = DecimalRoundCallToSpecialForm::getResultPrecisionScale(
        f.inputPrecision, f.inputScale, f.scale);
    VELOX_DCHECK_EQ(p, f.resultPrecision);
    VELOX_DCHECK_EQ(s, f.resultScale);
  }

  std::optional<TResult> applyOne(const TInput& input) const {
    if (scaleNonNegative_) {
      TResult rescaledValue;
      const auto status = DecimalUtil::rescaleWithRoundUp<TInput, TResult>(
          input,
          inputPrecision_,
          inputScale_,
          resultPrecision_,
          resultScale_,
          rescaledValue);
      VELOX_DCHECK(status.ok());
      return rescaledValue;
    }
    TResult rescaledValue;
    DecimalUtil::divideWithRoundUp<TResult, TInput, int128_t>(
        rescaledValue, input, divideFactor_, false, 0, 0);
    rescaledValue *= multiplyFactor_;
    return rescaledValue;
  }

 private:
  const bool scaleNonNegative_;
  const uint8_t inputPrecision_;
  const uint8_t inputScale_;
  const uint8_t resultPrecision_;
  const uint8_t resultScale_;
  const int128_t divideFactor_;
  const int128_t multiplyFactor_;
};

} // namespace

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
      "The result type of decimal_round should be decimal type.");
  VELOX_USER_CHECK_GE(
      args.size(), 1, "Decimal_round expects one or two arguments.");
  VELOX_USER_CHECK_LE(
      args.size(), 2, "Decimal_round expects one or two arguments.");
  VELOX_USER_CHECK(
      args[0]->type()->isDecimal(),
      "The first argument of decimal_round should be of decimal type.");

  int32_t scale = 0;
  if (args.size() > 1) {
    scale = DecimalRoundBase::extractConstantScaleArg(args[1], kRoundDecimal);
  }

  auto func = DecimalRoundBase::createFunction(
      args[0]->type(),
      scale,
      type,
      [](auto resultTag, auto inputTag, const auto& factors) {
        using TResult = typename decltype(resultTag)::type;
        using TInput = typename decltype(inputTag)::type;
        using Policy = RoundHalfUpPolicy<TResult, TInput>;
        return std::make_shared<DecimalRoundFunction<TResult, TInput, Policy>>(
            factors);
      });

  return DecimalRoundBase::buildExpr(
      type, std::move(args), std::move(func), kRoundDecimal, trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
