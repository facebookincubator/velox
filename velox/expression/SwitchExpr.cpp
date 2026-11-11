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
#include "velox/expression/SwitchExpr.h"
#include "velox/expression/ExprConstants.h"
#include "velox/type/TypeCoercer.h"

namespace facebook::velox::exec {

namespace {
TypePtr resolveTypeInt(
    const std::vector<TypePtr>& argTypes,
    bool allowedCoercions,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer);

TypePtr resolveTypeInt(const std::vector<TypePtr>& argTypes) {
  std::vector<TypePtr> coercions;
  return resolveTypeInt(argTypes, false, coercions, TypeCoercer::defaults());
};
} // namespace

SwitchExpr::SwitchExpr(
    TypePtr type,
    const std::vector<ExprPtr>& inputs,
    bool inputsSupportFlatNoNullsFastPath)
    : CaseExprBase(
          SpecialFormKind::kSwitch,
          std::move(type),
          inputs,
          expression::kSwitch,
          /*hasSubject=*/false,
          inputsSupportFlatNoNullsFastPath) {
  std::vector<TypePtr> inputTypes;
  inputTypes.reserve(inputs_.size());
  std::transform(
      inputs_.begin(),
      inputs_.end(),
      std::back_inserter(inputTypes),
      [](const ExprPtr& expr) { return expr->type(); });

  // Apply type checking.
  auto typeExpected = resolveTypeInt(inputTypes);
  VELOX_CHECK(
      *typeExpected == *this->type(),
      "Switch expression type different than then clause. Expected {}, but got {}.",
      typeExpected->toString(),
      this->type()->toString());
}

void SwitchExpr::evalCondition(
    size_t index,
    const SelectivityVector& rows,
    const VectorPtr& /*subject*/,
    EvalCtx& context,
    VectorPtr& result) {
  whenClause(index)->eval(rows, context, result);
}

namespace {
TypePtr resolveTypeInt(
    const std::vector<TypePtr>& argTypes,
    bool allowedCoercions,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  VELOX_CHECK_GT(
      argTypes.size(),
      1,
      "Switch statements expect at least 2 arguments, received {}",
      argTypes.size());

  const size_t numArgs = argTypes.size();
  const size_t numCases = numArgs / 2;

  TypePtr resultType = argTypes[1];

  if (allowedCoercions) {
    coercions.clear();
    coercions.resize(numArgs);
  }

  // Records the coercion every then clause up to 'index' needs to reach
  // 'type'. A clause already of that type needs none.
  auto setCoercionsUpTo = [&](int index, const TypePtr& type) {
    for (auto i = 1; i <= index; i += 2) {
      coercions[i] = *argTypes[i] == *type ? nullptr : type;
    }
  };

  for (auto i = 0; i < numCases; i++) {
    const auto& conditionType = argTypes[i * 2];
    const auto& thenType = argTypes[i * 2 + 1];

    if (!conditionType->isBoolean()) {
      VELOX_CHECK(
          allowedCoercions,
          "Condition of  SWITCH statement is not bool: {}",
          conditionType->toString());
      // A null literal condition types as UNKNOWN, which coerces to boolean.
      VELOX_CHECK(
          coercer.coerce(conditionType, BOOLEAN()).has_value(),
          "Condition of  SWITCH statement is not coercible to bool: {}",
          conditionType->toString());
      coercions[i * 2] = BOOLEAN();
    }

    if (*thenType != *resultType) {
      const auto common = allowedCoercions
          ? coercer.leastCommonSuperType(resultType, thenType)
          : nullptr;
      VELOX_CHECK_NOT_NULL(
          common,
          "All then clauses of a SWITCH statement must have the same type. "
          "Expected {}, but got {}.",
          resultType->toString(),
          thenType->toString());

      resultType = common;
      setCoercionsUpTo(i * 2 + 1, resultType);
    }
  }

  const bool hasElse = argTypes.size() % 2 == 1;
  if (hasElse) {
    const auto& elseType = argTypes.back();

    if (*elseType != *resultType) {
      const auto common = allowedCoercions
          ? coercer.leastCommonSuperType(resultType, elseType)
          : nullptr;
      VELOX_CHECK_NOT_NULL(
          common,
          "Else clause of a SWITCH statement must have the same type as 'then' clauses. "
          "Expected {}, but got {}.",
          resultType->toString(),
          elseType->toString());

      resultType = common;
      setCoercionsUpTo(numArgs - 2, resultType);
      coercions.back() = *elseType == *resultType ? nullptr : resultType;
    }
  }

  return resultType;
}
} // namespace

TypePtr SwitchCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  return resolveTypeInt(argTypes);
}

TypePtr SwitchCallToSpecialForm::resolveTypeWithCoercions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  return resolveTypeInt(argTypes, true, coercions, coercer);
}

ExprPtr SwitchCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool /* trackCpuUsage */,
    const core::QueryConfig& /*config*/) {
  bool inputsSupportFlatNoNullsFastPath =
      Expr::allSupportFlatNoNullsFastPath(compiledChildren);
  return std::make_shared<SwitchExpr>(
      type, std::move(compiledChildren), inputsSupportFlatNoNullsFastPath);
}

TypePtr IfCallToSpecialForm::resolveType(const std::vector<TypePtr>& argTypes) {
  VELOX_CHECK(
      argTypes.size() == 2 || argTypes.size() == 3,
      "An IF statement must have 2 or 3 clauses: 'if', 'then', and optional 'else'.");

  return resolveTypeInt(argTypes);
}

TypePtr IfCallToSpecialForm::resolveTypeWithCoercions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  VELOX_CHECK_EQ(
      argTypes.size(),
      3,
      "An IF statement must have 3 clauses: 'if', 'then', and 'else'.");

  return resolveTypeInt(argTypes, true, coercions, coercer);
}
} // namespace facebook::velox::exec
