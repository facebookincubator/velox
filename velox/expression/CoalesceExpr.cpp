/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/expression/CoalesceExpr.h"
#include "velox/expression/ExprConstants.h"
#include "velox/type/TypeCoercer.h"

namespace facebook::velox::exec {

namespace {

TypePtr resolveTypeInt(
    const std::vector<TypePtr>& argTypes,
    bool allowedCoercions,
    std::vector<TypePtr>& coercions) {
  VELOX_CHECK_GT(
      argTypes.size(),
      0,
      "COALESCE statements expect to receive at least 1 argument, but did not receive any.");

  const auto numArgs = argTypes.size();

  if (allowedCoercions) {
    coercions.clear();
    coercions.resize(numArgs);
  }

  auto resultType = argTypes[0];

  for (auto i = 1; i < numArgs; i++) {
    if (*resultType == *argTypes[i]) {
      continue;
    }

    if (allowedCoercions && TypeCoercer::coercible(argTypes[i], resultType)) {
      coercions[i] = resultType;
      continue;
    }

    if (allowedCoercions && TypeCoercer::coercible(resultType, argTypes[i])) {
      resultType = argTypes[i];
      for (auto j = 0; j < i; j++) {
        coercions[j] = resultType;
      }
      continue;
    }

    VELOX_USER_CHECK(
        *argTypes[0] == *argTypes[i],
        "Inputs to coalesce must have the same type. Expected {}, but got {}.",
        argTypes[0]->toString(),
        argTypes[i]->toString());
  }

  return resultType;
}

TypePtr resolveTypeInt(const std::vector<TypePtr>& argTypes) {
  std::vector<TypePtr> coercions;
  return resolveTypeInt(argTypes, false, coercions);
}
} // namespace

CoalesceExpr::CoalesceExpr(
    TypePtr type,
    std::vector<ExprPtr>&& inputs,
    bool inputsSupportFlatNoNullsFastPath)
    : SpecialForm(
          SpecialFormKind::kCoalesce,
          std::move(type),
          std::move(inputs),
          expression::kCoalesce,
          inputsSupportFlatNoNullsFastPath,
          false /* trackCpuUsage */) {
  std::vector<TypePtr> inputTypes;
  inputTypes.reserve(inputs_.size());
  std::transform(
      inputs_.begin(),
      inputs_.end(),
      std::back_inserter(inputTypes),
      [](const ExprPtr& expr) { return expr->type(); });

  // Apply type checks.
  auto expectedType = resolveTypeInt(inputTypes);
  VELOX_CHECK(
      *expectedType == *this->type(),
      "Coalesce expression type different than its inputs. Expected {}, but got {}.",
      expectedType->toString(),
      this->type()->toString());
}

void CoalesceExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  // Null positions to populate.
  exec::LocalSelectivityVector activeRowsHolder(context, rows.end());
  auto activeRows = activeRowsHolder.get();
  assert(activeRows); // for lint
  *activeRows = rows;

  // Fix finalSelection at "rows" unless already fixed.
  ScopedFinalSelectionSetter scopedFinalSelectionSetter(context, &rows);

  exec::LocalDecodedVector decodedVector(context);
  for (int i = 0; i < inputs_.size(); i++) {
    inputs_[i]->eval(*activeRows, context, result);

    if (!result->mayHaveNulls()) {
      // No nulls left.
      return;
    }

    if (context.errors()) {
      context.deselectErrors(*activeRows);
    }

    decodedVector.get()->decode(*result, *activeRows);
    const uint64_t* rawNulls = decodedVector->nulls(activeRows);
    if (!rawNulls) {
      // No nulls left.
      return;
    }

    activeRows->deselectNonNulls(rawNulls, 0, activeRows->end());
    if (!activeRows->hasSelections()) {
      // No nulls left.
      return;
    }
  }
}

TypePtr CoalesceCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  return resolveTypeInt(argTypes);
}

TypePtr CoalesceCallToSpecialForm::resolveTypeWithCorsions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions) {
  return resolveTypeInt(argTypes, true, coercions);
}

ExprPtr CoalesceCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool /* trackCpuUsage */,
    const core::QueryConfig& /*config*/) {
  bool inputsSupportFlatNoNullsFastPath =
      Expr::allSupportFlatNoNullsFastPath(compiledChildren);
  return std::make_shared<CoalesceExpr>(
      type, std::move(compiledChildren), inputsSupportFlatNoNullsFastPath);
}
} // namespace facebook::velox::exec
