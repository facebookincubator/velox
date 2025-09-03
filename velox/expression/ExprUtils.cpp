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
#include "velox/core/Expressions.h"
#include "velox/expression/ExprConstants.h"

namespace facebook::velox::expression::utils {

namespace {

// Helper function to constant fold the expression and return a fail expression
// in case constant folding throws an exception.
core::TypedExprPtr tryConstantFold(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvalErrorWithFailExpr) {
  try {
    if (auto results =
            exec::tryEvaluateConstantExpression(expr, pool, queryCtx, false)) {
      return std::make_shared<core::ConstantTypedExpr>(results);
    }
    // Return the expression unevaluated.
    return expr;
  } catch (VeloxUserError& e) {
    if (replaceEvalErrorWithFailExpr) {
      return std::make_shared<core::CallTypedExpr>(
          UNKNOWN(),
          kFail,
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), e.what()));
    }
    return expr;
  }
}
} // namespace

bool isCall(const core::TypedExprPtr& expr, const std::string& name) {
  if (expr->isCallKind()) {
    return expr->asUnchecked<core::CallTypedExpr>()->name() == name;
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

core::TypedExprPtr constantFold(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvaluationErrWithFailFunction) {
  std::vector<core::TypedExprPtr> foldedInputs;
  for (const auto& input : expr->inputs()) {
    foldedInputs.push_back(constantFold(
        input, queryCtx, pool, replaceEvaluationErrWithFailFunction));
  }

  core::TypedExprPtr result;
  if (const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>()) {
    result = std::make_shared<core::CallTypedExpr>(
        callExpr->type(), foldedInputs, callExpr->name());
  } else if (const auto* castExpr = expr->asUnchecked<core::CastTypedExpr>()) {
    result = std::make_shared<core::CastTypedExpr>(
        expr->type(), foldedInputs, castExpr->isTryCast());
  } else if (
      const auto* lambdaExpr = expr->asUnchecked<core::LambdaTypedExpr>()) {
    const auto foldedBody = constantFold(
        lambdaExpr->body(),
        queryCtx,
        pool,
        replaceEvaluationErrWithFailFunction);
    result = std::make_shared<core::LambdaTypedExpr>(
        lambdaExpr->signature(), foldedBody);
  } else if (
      const auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr)) {
    return constantExpr;
  } else if (
      const auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    return field;
  } else {
    result = expr;
  }

  return tryConstantFold(
      result, queryCtx, pool, replaceEvaluationErrWithFailFunction);
}

} // namespace facebook::velox::expression::utils
