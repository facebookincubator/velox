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

#include "velox/functions/sparksql/specialforms/DecimalCeilFloor.h"

#include "velox/functions/sparksql/specialforms/DecimalRoundOps.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Directional rounding (ceil toward +∞, floor toward -∞). May overflow.
template <typename TResult, typename TInput, bool ceiling>
struct CeilFloorPolicy {
  static constexpr bool canOverflow = true;

  explicit CeilFloorPolicy(const DecimalRoundOps::ScaleFactors& f)
      : factors_(f) {
    const auto [expectedP, expectedS] =
        DecimalCeilFloorCallToSpecialFormBase::getResultPrecisionScale(
            f.inputPrecision, f.inputScale, f.scale);
    VELOX_DCHECK_EQ(expectedP, f.resultPrecision);
    VELOX_DCHECK_EQ(expectedS, f.resultScale);
  }

  std::optional<TResult> applyOne(const TInput& input) const {
    int128_t out = 0;
    if (!factors_.divideFactor.has_value()) {
      out = static_cast<int128_t>(input);
    } else {
      auto in = static_cast<int128_t>(input);
      const int128_t divisor = factors_.divideFactor.value();
      const int128_t quotient = in / divisor;
      const int128_t remainder = in % divisor;
      int128_t rounded = 0;
      if constexpr (ceiling) {
        rounded = quotient + (remainder > 0 ? 1 : 0);
      } else {
        rounded = quotient + (remainder < 0 ? -1 : 0);
      }
      if (factors_.multiplyFactor.has_value()) {
        const int128_t multiplier = factors_.multiplyFactor.value();
        const int128_t maxAbs = factors_.overflowBound / multiplier;
        if (rounded >= maxAbs || rounded <= -maxAbs) {
          return std::nullopt;
        }
        rounded *= multiplier;
      }
      out = rounded;
    }
    if (out >= factors_.overflowBound || out <= -factors_.overflowBound) {
      return std::nullopt;
    }
    return static_cast<TResult>(out);
  }

 private:
  const DecimalRoundOps::ScaleFactors factors_;
};

// Each instantiation of createCeilFloorFunction produces only one template
// variant of CeilFloorPolicy (true or false), avoiding redundant codegen.
template <bool ceiling>
std::shared_ptr<exec::VectorFunction> createCeilFloorFunction(
    const TypePtr& inputType,
    int32_t scale,
    const TypePtr& resultType) {
  return DecimalRoundOps::createFunction(
      inputType,
      scale,
      resultType,
      [](auto resultTag, auto inputTag, const auto& factors) {
        using TResult = typename decltype(resultTag)::type;
        using TInput = typename decltype(inputTag)::type;
        using Policy = CeilFloorPolicy<TResult, TInput, ceiling>;
        return std::make_shared<DecimalRoundFunction<TResult, TInput, Policy>>(
            factors);
      });
}

} // namespace

TypePtr DecimalCeilFloorCallToSpecialFormBase::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL(
      "Decimal ceil/floor with scale special form does not support type resolution.");
}

exec::ExprPtr DecimalCeilFloorCallToSpecialFormBase::makeSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/,
    bool ceiling,
    std::string_view funcName) {
  VELOX_USER_CHECK(
      type->isDecimal(), "The result type of {} must be decimal.", funcName);
  VELOX_USER_CHECK_EQ(
      args.size(),
      2,
      "{} expects two arguments (decimal value and target scale).",
      funcName);
  VELOX_USER_CHECK(
      args[0]->type()->isDecimal(),
      "The first argument of {} must be decimal.",
      funcName);

  const int32_t scale =
      DecimalRoundOps::extractConstantScaleArg(args[1], funcName);

  auto func = ceiling
      ? createCeilFloorFunction<true>(args[0]->type(), scale, type)
      : createCeilFloorFunction<false>(args[0]->type(), scale, type);

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(func),
      exec::VectorFunctionMetadata{},
      std::string(funcName),
      trackCpuUsage);
}

exec::ExprPtr DecimalCeilCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  return makeSpecialForm(
      type, std::move(args), trackCpuUsage, config, true, kCeilDecimal);
}

exec::ExprPtr DecimalFloorCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  return makeSpecialForm(
      type, std::move(args), trackCpuUsage, config, false, kFloorDecimal);
}

} // namespace facebook::velox::functions::sparksql
