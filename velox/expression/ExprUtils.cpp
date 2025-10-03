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

#include "velox/expression/ExprUtils.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/ConstantVector.h"

namespace facebook::velox::expression::utils {

ConstantEvalResult evalExprAsConstant(const core::TypedExprPtr& expr) {
  if (expr->isConstantKind()) {
    auto constantExpr = expr->asUnchecked<core::ConstantTypedExpr>();
    if (constantExpr->isNull()) {
      return ConstantEvalResult::kNull;
    }
    if (constantExpr->type()->isBoolean()) {
      auto value = constantExpr->hasValueVector()
          ? constantExpr->valueVector()->as<ConstantVector<bool>>()->valueAt(0)
          : constantExpr->value().value<TypeKind::BOOLEAN>();
      if (value) {
        return ConstantEvalResult::kTrue;
      }
      return ConstantEvalResult::kFalse;
    }
    return ConstantEvalResult::kNonBoolConstant;
  }
  return ConstantEvalResult::kNotConstant;
}

bool isCall(const core::TypedExprPtr& expr, const std::string& name) {
  if (expr->isCallKind()) {
    return expr->asUnchecked<core::CallTypedExpr>()->name() == name;
  }
  return false;
}

ExprInputsKind getExprInputsKind(const core::TypedExprPtr& expr) {
  bool allConst = true;
  bool allField = true;
  for (const auto& input : expr->inputs()) {
    if (!input->isConstantKind() && !input->isFieldAccessKind()) {
      return kDefault;
    }
    allConst = allConst && input->isConstantKind();
    allField = allField && input->isFieldAccessKind();
  }
  return allConst ? kAllConstant : allField ? kAllField : kConstantOrField;
}

bool isAssociativeOperator(const core::CallTypedExpr* call) {
  auto sanitizedName = exec::sanitizeName(call->name());
  std::vector<std::string> parts;
  folly::split('.', sanitizedName, parts, true);
  VELOX_CHECK(
      parts.size() == 1 || parts.size() == 3,
      "Invalid CallExpr name ",
      sanitizedName);
  sanitizedName = (parts.size() == 1) ? parts[0] : parts[2];
  if (sanitizedName == kOr || sanitizedName == kAnd || sanitizedName == kPlus ||
      sanitizedName == kMultiply) {
    return true;
  }
  return false;
}

void flattenInput(
    const core::TypedExprPtr& input,
    const std::string& flattenCall,
    std::vector<core::TypedExprPtr>& flat) {
  if (isCall(input, flattenCall) && allInputTypesEquivalent(input)) {
    for (auto& child : input->inputs()) {
      flattenInput(child, flattenCall, flat);
    }
  } else {
    flat.emplace_back(input);
  }
}

bool allInputTypesEquivalent(const core::TypedExprPtr& expr) {
  const auto& inputs = expr->inputs();
  for (int i = 1; i < inputs.size(); i++) {
    if (!inputs[0]->type()->equivalent(*inputs[i]->type())) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::expression::utils
