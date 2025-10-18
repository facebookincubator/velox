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
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"

namespace facebook::velox::expression {

namespace {

// Tries constant folding expression with tryEvaluateConstantExpression API.
// If constant folding throws VeloxUserError, returns original expression when
// replaceEvalErrorWithFailExpr is false, otherwise returns a fail function
// with the error message.
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
  } catch (VeloxUserError& e) {
    if (replaceEvalErrorWithFailExpr) {
      return std::make_shared<core::CallTypedExpr>(
          UNKNOWN(),
          expression::kFail,
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), e.message()));
    }
  }
  // Return the expression unevaluated.
  return expr;
}

// Optimizes all inputs to expr and returns an expression that is of the same
// kind as expr but with optimized inputs. When 'replaceEvalErrorWithFailExpr'
// is 'true' and an exception is encountered when optimizing any input to expr,
// the respective input is replaced with a FAIL expression with UNKNOWN return
// type. Users should be mindful of this change and must ensure the optimized
// expression is not evaluated in Velox without adding the necessary type casts
// as expected by the registered functions.
core::TypedExprPtr optimizeInputs(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvalErrorWithFailExpr) {
  if (expr->isCallKind()) {
    std::vector<core::TypedExprPtr> optimizedInputs;
    optimizedInputs.reserve(expr->inputs().size());
    for (const auto& input : expr->inputs()) {
      optimizedInputs.push_back(
          optimize(input, queryCtx, pool, replaceEvalErrorWithFailExpr));
    }
    const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();

    return std::make_shared<core::CallTypedExpr>(
        callExpr->type(), optimizedInputs, callExpr->name());
  }

  if (expr->isCastKind()) {
    const auto optimizedInput = optimize(
        expr->inputs().at(0), queryCtx, pool, replaceEvalErrorWithFailExpr);
    if (*expr->type() == *optimizedInput->type()) {
      return optimizedInput;
    }
    const auto* castExpr = expr->asUnchecked<core::CastTypedExpr>();

    return std::make_shared<core::CastTypedExpr>(
        expr->type(), optimizedInput, castExpr->isTryCast());
  }

  if (expr->isLambdaKind()) {
    const auto* lambdaExpr = expr->asUnchecked<core::LambdaTypedExpr>();
    const auto foldedBody = optimize(
        lambdaExpr->body(), queryCtx, pool, replaceEvalErrorWithFailExpr);
    return std::make_shared<core::LambdaTypedExpr>(
        lambdaExpr->signature(), foldedBody);
  }

  return expr;
}
} // namespace

core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvalErrorWithFailExpr) {
  if (expr->isConstantKind() || expr->isFieldAccessKind()) {
    return expr;
  }

  core::TypedExprPtr result = expr;
  const auto kind = utils::getExprInputsKind(expr);
  switch (kind) {
    case utils::ExprInputsKind::kAllConstant: {
      result =
          tryConstantFold(expr, queryCtx, pool, replaceEvalErrorWithFailExpr);
      break;
    }
    case utils::ExprInputsKind::kAllField: {
      if (expr->isCastKind() &&
          *expr->inputs().at(0)->type() == *expr->type()) {
        result = expr->inputs().at(0);
      }
      break;
    }
    case utils::ExprInputsKind::kAny: {
      result =
          optimizeInputs(expr, queryCtx, pool, replaceEvalErrorWithFailExpr);
      break;
    }
    // TODO: For expressions representing an associative operation like plus,
    //  where plus(a, b) == plus(b, a) for all inputs a, b, constant inputs
    //  to the expression can be evaluated to return a partially constant
    //  folded expression. Metadata should be added to ITypedExpr to indicate
    //  expression associativity.
    case utils::ExprInputsKind::kConstantOrField:
      break;
  }

  const auto resultKind = utils::getExprInputsKind(result);
  // Try evaluating whole expression after folding subtrees, eg: (1 - 1) * a,
  // between(cast(null as integer), 2, 4), cast(cast(123 as BIGINT) as VARCHAR).
  if (resultKind == utils::ExprInputsKind::kAllConstant) {
    result =
        tryConstantFold(result, queryCtx, pool, replaceEvalErrorWithFailExpr);
  }

  // Try rewrite after folding subtrees and trying to constant fold.
  return ExprRewriteRegistry::instance().rewrite(result);
}

} // namespace facebook::velox::expression
