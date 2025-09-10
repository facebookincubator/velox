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

#include <set>

#include "velox/expression/Expr.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprRewrite.h"
#include "velox/expression/ExprRewriteRegistry.h"

namespace facebook::velox::expression {

namespace {
core::TypedExprPtr tryConstantFold(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvaluationErrWithFailFunction) {
  try {
    if (auto results =
            exec::tryEvaluateConstantExpression(expr, pool, queryCtx, false)) {
      return std::make_shared<core::ConstantTypedExpr>(results);
    }
    // Return the expression unevaluated.
    return expr;
  } catch (VeloxUserError& e) {
    if (replaceEvaluationErrWithFailFunction) {
      return std::make_shared<core::CallTypedExpr>(
          UNKNOWN(),
          kFail,
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), e.what()));
    }
    return expr;
  }
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
} // namespace

core::TypedExprPtr rewriteExpression(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool enableConstantFolding,
    bool replaceEvaluationErrWithFailFunction) {
  const auto& rewriteRegistry = expressionRewriteRegistry();
  const auto& rewriteNames = rewriteRegistry.getExpressionRewriteNames();
  auto rewrittenExpr = expr;
  for (const auto& name : rewriteNames) {
    auto expressionRewriteFunc = *rewriteRegistry.getExpressionRewrite(name);
    if (expressionRewriteFunc != nullptr) {
      if (auto rewritten = expressionRewriteFunc(expr)) {
        rewrittenExpr = rewritten;
        break;
      }
    }
  }
  return enableConstantFolding
      ? constantFold(
            rewrittenExpr, queryCtx, pool, replaceEvaluationErrWithFailFunction)
      : rewrittenExpr;
}

} // namespace facebook::velox::expression
