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
#include "velox/expression/CoalesceExpr.h"
#include "velox/expression/ConstantExpr.h"

namespace facebook::velox::exec {

CoalesceExpr::CoalesceExpr(
    TypePtr type,
    std::vector<ExprPtr>&& inputs,
    bool inputsSupportFlatNoNullsFastPath)
    : SpecialForm(
          std::move(type),
          std::move(inputs),
          kCoalesce,
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
  auto expectedType = resolveType(inputTypes);
  VELOX_CHECK(
      *expectedType == *this->type(),
      "Coalesce expression type different than its inputs. Expected {} but got Actual {}.",
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

// static
TypePtr CoalesceExpr::resolveType(const std::vector<TypePtr>& argTypes) {
  VELOX_CHECK_GT(
      argTypes.size(),
      0,
      "COALESCE statements expect to receive at least 1 argument, but did not receive any.");
  for (auto i = 1; i < argTypes.size(); i++) {
    VELOX_USER_CHECK(
        *argTypes[0] == *argTypes[i],
        "Inputs to coalesce must have the same type. Expected {}, but got {}.",
        argTypes[0]->toString(),
        argTypes[i]->toString());
  }

  return argTypes[0];
}

TypePtr CoalesceCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  return CoalesceExpr::resolveType(argTypes);
}

ExprPtr CoalesceCallToSpecialForm::optimize(
    std::vector<ExprPtr>& compiledChildren) {
  auto numChildren = compiledChildren.size();
  std::vector<column_index_t> nullIndices;
  for (auto i = 0; i < numChildren; i++) {
    if (auto constantExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(
            compiledChildren[i])) {
      if (constantExpr->value()->isNullAt(0)) {
        nullIndices.push_back(i);
      } else if (nullIndices.size() == i) {
        return constantExpr;
      }
    }
  }

  if (!nullIndices.empty()) {
    if (nullIndices.size() == compiledChildren.size()) {
      auto nullChild = compiledChildren.at(nullIndices.front());
      return std::dynamic_pointer_cast<exec::ConstantExpr>(nullChild);
    } else {
      auto nullCount = nullIndices.size();
      for (vector_size_t j = nullCount - 1; j >= 0; j--) {
        compiledChildren.erase(compiledChildren.begin() + nullIndices[j]);
      }
    }
  }
  return nullptr;
}

ExprPtr CoalesceCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool /* trackCpuUsage */,
    const core::QueryConfig& /*config*/) {
  auto children = std::move(compiledChildren);
  if (auto result = optimize(children)) {
    return result;
  }

  bool inputsSupportFlatNoNullsFastPath =
      Expr::allSupportFlatNoNullsFastPath(children);
  return std::make_shared<CoalesceExpr>(
      type, std::move(children), inputsSupportFlatNoNullsFastPath);
}
} // namespace facebook::velox::exec
