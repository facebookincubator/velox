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
#include "velox/functions/sparksql/specialforms/DecimalRoundOps.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Half-up rounding (Spark's ROUND_HALF_UP). Never overflows.
template <typename TResult, typename TInput>
struct RoundHalfUpPolicy {
  static constexpr bool canOverflow = false;

  explicit RoundHalfUpPolicy(const DecimalRoundOps::ScaleFactors& f)
      : factors_(f) {
    const auto [expectedPrecision, expectedScale] =
        DecimalRoundCallToSpecialForm::getResultPrecisionScale(
            f.inputPrecision, f.inputScale, f.scale);
    VELOX_DCHECK_EQ(expectedPrecision, f.resultPrecision);
    VELOX_DCHECK_EQ(expectedScale, f.resultScale);
  }

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
    TResult rescaledValue;
    DecimalUtil::divideWithRoundUp<TResult, TInput, int128_t>(
        rescaledValue,
        input,
        factors_.divideFactor.value_or(1),
        false,
        0,
        0);
    rescaledValue *= factors_.multiplyFactor.value_or(1);
    return rescaledValue;
  }

 private:
  const DecimalRoundOps::ScaleFactors factors_;
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
    scale = DecimalRoundOps::extractConstantScaleArg(args[1], kRoundDecimal);
  }

  auto func = DecimalRoundOps::createFunction(
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

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(func),
      exec::VectorFunctionMetadata{},
      std::string(kRoundDecimal),
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
