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
#include "velox/expression/ExprRewriteRegistry.h"

namespace facebook::velox::expression {

namespace {

// Tries constant folding input `expr` with `tryEvaluateConstantExpression` API
// and returns the evaluated expression if constant folding succeeds. If
// `tryEvaluateConstantExpression` throws a `VeloxUserError`, returns the
// result of applying `makeFailExpr` to `expr`, otherwise returns `expr`.
core::TypedExprPtr tryConstantFold(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    const MakeFailExpr& makeFailExpr) {
  try {
    if (auto results =
            exec::tryEvaluateConstantExpression(expr, pool, queryCtx, false)) {
      return std::make_shared<core::ConstantTypedExpr>(results);
    }
  } catch (VeloxUserError& e) {
    if (makeFailExpr != nullptr) {
      const auto result = makeFailExpr(e.message(), expr->type());
      VELOX_USER_CHECK(
          *result->type() == *expr->type(),
          "makeFailExpr returned expression: {} of type: {}, which does not match type: {} of failing expression: {}",
          result->toString(),
          result->type()->toString(),
          expr->type()->toString(),
          expr->toString());
      return result;
    }
  }
  // Return the expression unmodified.
  return expr;
}

// Optimizes all inputs to expr and returns an expression that is of the same
// kind as expr but with optimized inputs.
core::TypedExprPtr optimizeInputs(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    const MakeFailExpr& makeFailExpr) {
  if (expr->isCallKind()) {
    std::vector<core::TypedExprPtr> optimizedInputs;
    optimizedInputs.reserve(expr->inputs().size());
    for (const auto& input : expr->inputs()) {
      optimizedInputs.push_back(optimize(input, queryCtx, pool, makeFailExpr));
    }
    const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();

    return std::make_shared<core::CallTypedExpr>(
        callExpr->type(), optimizedInputs, callExpr->name());
  }

  if (expr->isCastKind()) {
    const auto optimizedInput =
        optimize(expr->inputs().at(0), queryCtx, pool, makeFailExpr);
    const auto* castExpr = expr->asUnchecked<core::CastTypedExpr>();
    return std::make_shared<core::CastTypedExpr>(
        expr->type(), optimizedInput, castExpr->isTryCast());
  }

  if (expr->isLambdaKind()) {
    const auto* lambdaExpr = expr->asUnchecked<core::LambdaTypedExpr>();
    const auto foldedBody =
        optimize(lambdaExpr->body(), queryCtx, pool, makeFailExpr);
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
    const MakeFailExpr& makeFailExpr) {
  auto result = expr;
  // cast(1 AS BIGINT) -> 1.
  // cast(a AS BIGINT) -> a ; when type(a) == BIGINT.
  // cast(concat(a, 'test') AS VARCHAR) -> concat(a, 'test') ; when type(a) ==
  //  VARCHAR.
  if (result->isCastKind() &&
      *result->type() == *result->inputs().at(0)->type()) {
    result = result->inputs().at(0);
  }
  // 1 -> 1, a -> a.
  if (result->isConstantKind() || result->isFieldAccessKind()) {
    return result;
  }

  result = optimizeInputs(result, queryCtx, pool, makeFailExpr);
  bool allInputsConstant = true;
  for (const auto& input : result->inputs()) {
    if (!input->isConstantKind()) {
      allInputsConstant = false;
      break;
    }
  }

  if (allInputsConstant) {
    return tryConstantFold(result, queryCtx, pool, makeFailExpr);
  }
  return ExprRewriteRegistry::instance().rewrite(result, pool);
}

} // namespace facebook::velox::expression
