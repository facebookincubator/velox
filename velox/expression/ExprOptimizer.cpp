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
#include "velox/core/Expressions.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"

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

// Optimizes all inputs to expr. Returns expr itself when no input changed, and
// otherwise the same expression over the optimized inputs.
core::TypedExprPtr optimizeInputs(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    const MakeFailExpr& makeFailExpr) {
  // A lambda's body is not one of its inputs: rebuild the lambda only if the
  // body changed.
  if (expr->isLambdaKind()) {
    const auto* lambdaExpr = expr->asUnchecked<core::LambdaTypedExpr>();
    auto foldedBody =
        optimize(lambdaExpr->body(), queryCtx, pool, makeFailExpr);
    if (foldedBody.get() == lambdaExpr->body().get()) {
      return expr;
    }
    return std::make_shared<core::LambdaTypedExpr>(
        lambdaExpr->signature(), std::move(foldedBody));
  }

  // Optimize every input.
  std::vector<core::TypedExprPtr> optimizedInputs;
  optimizedInputs.reserve(expr->inputs().size());
  for (const auto& input : expr->inputs()) {
    optimizedInputs.push_back(optimize(input, queryCtx, pool, makeFailExpr));
  }

  // Rebuild the node only if an input changed.
  if (utils::sameInputs(expr, optimizedInputs)) {
    return expr;
  }
  return expr->withInputs(std::move(optimizedInputs));
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
  // Constants, the input row and column references have nothing to optimize.
  if (result->isConstantKind() || result->isInputKind() ||
      (result->isFieldAccessKind() && result->inputs().empty())) {
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
  return ExprRewriteRegistry::instance().rewrite(result);
}

} // namespace facebook::velox::expression
