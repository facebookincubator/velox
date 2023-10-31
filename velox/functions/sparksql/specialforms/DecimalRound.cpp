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
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

namespace {

std::pair<uint8_t, uint8_t> computeRoundDecimalResultPrecisionScale(
    uint8_t aPrecision,
    uint8_t aScale,
    int32_t scale) {
  auto integralLeastNumDigits = aPrecision - aScale + 1;
  if (scale < 0) {
    auto newPrecision = std::max(integralLeastNumDigits, -scale + 1);
    return {std::min(newPrecision, 38), 0};
  } else {
    return {
        std::min(
            static_cast<int32_t>(
                integralLeastNumDigits +
                std::min(static_cast<int32_t>(aScale), scale)),
            38),
        std::min(static_cast<int32_t>(aScale), scale)};
  }
}

int32_t getRoundDecimalScale(core::TypedExprPtr input) {
  std::shared_ptr<memory::MemoryPool> pool = memory::addDefaultLeafMemoryPool();
  if (auto cast = dynamic_cast<const core::CastTypedExpr*>(input.get())) {
    VELOX_USER_CHECK(
        input->type()->kind() == TypeKind::INTEGER,
        "Second argument of decimal_round should be cast value as integer");
    auto scaleInput = cast->inputs()[0];
    auto constant =
        dynamic_cast<const core::ConstantTypedExpr*>(scaleInput.get());
    VELOX_USER_CHECK_NOT_NULL(constant);
    VELOX_USER_CHECK(
        constant->type()->kind() == TypeKind::BIGINT,
        "Second argument of decimal_round should be cast bigint value as integer");
    return constant->toConstantVector(pool.get())
        ->asUnchecked<ConstantVector<int64_t>>()
        ->value();
  } else {
    VELOX_USER_FAIL(
        "Second argument of decimal_round should be cast value as integer");
  }
}
} // namespace

TypePtr DecimalRoundCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  return nullptr;
}

TypePtr DecimalRoundCallToSpecialForm::resolveType(
    const std::vector<std::shared_ptr<const core::ITypedExpr>>& inputs) {
  auto numInput = inputs.size();
  int32_t scale = 0;
  if (numInput > 1) {
    scale = getRoundDecimalScale(inputs[1]);
  }
  auto [aPrecision, aScale] = getDecimalPrecisionScale(*inputs[0]->type());
  auto [rPrecision, rScale] =
      computeRoundDecimalResultPrecisionScale(aPrecision, aScale, scale);
  return DECIMAL(rPrecision, rScale);
}

exec::ExprPtr DecimalRoundCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_USER_CHECK(
      compiledChildren.size() <= 2 && compiledChildren.size() > 0,
      "RoundDecimal statements expect 1 or 2 arguments, received {}",
      compiledChildren.size());
  VELOX_USER_CHECK(
      compiledChildren[0]->type()->isDecimal(),
      "First argument of decimal_round should be decimal");
  if (compiledChildren.size() > 1) {
    VELOX_USER_CHECK_EQ(
        compiledChildren[1]->type()->kind(),
        TypeKind::INTEGER,
        "Second argument of decimal_round should be integer");
  }
  auto roundDecimalVectorFunction =
      exec::vectorFunctionFactories().withRLock([&config](auto& functionMap) {
        auto functionIterator = functionMap.find(kRoundDecimal);
        return functionIterator->second.factory(kRoundDecimal, {}, config);
      });
  return std::make_shared<exec::Expr>(
      type,
      std::move(compiledChildren),
      roundDecimalVectorFunction,
      kRoundDecimal,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
