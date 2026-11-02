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

namespace facebook::velox::expression {

namespace {

// Folds an all-constant expression to a single-row vector. Returns nullptr if
// the expression cannot be folded. The two public optimize() overloads supply
// an evaluator backed by a QueryCtx or an ExpressionEvaluator.
using ConstantEvaluator = std::function<
    VectorPtr(const core::TypedExprPtr& expr, bool suppressEvaluationFailures)>;

core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    const ConstantEvaluator& constantEvaluator,
    const MakeFailExpr& makeFailExpr);

// Tries constant folding input `expr` with `constantEvaluator` and returns
// the evaluated expression if constant folding succeeds. If
// `constantEvaluator` throws a `VeloxUserError`, returns the result of
// applying `makeFailExpr` to `expr`, otherwise returns `expr`.
core::TypedExprPtr tryConstantFold(
    const core::TypedExprPtr& expr,
    const ConstantEvaluator& constantEvaluator,
    const MakeFailExpr& makeFailExpr) {
  try {
    if (auto results =
            constantEvaluator(expr, /*suppressEvaluationFailures=*/false)) {
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
    const ConstantEvaluator& constantEvaluator,
    const MakeFailExpr& makeFailExpr) {
  if (expr->isCallKind() || expr->isNullIfKind()) {
    std::vector<core::TypedExprPtr> optimizedInputs;
    optimizedInputs.reserve(expr->inputs().size());
    for (const auto& input : expr->inputs()) {
      optimizedInputs.push_back(
          optimize(input, constantEvaluator, makeFailExpr));
    }

    if (expr->isCallKind()) {
      const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
      return std::make_shared<core::CallTypedExpr>(
          callExpr->type(), optimizedInputs, callExpr->name());
    }

    const auto* nullIfExpr = expr->asUnchecked<core::NullIfTypedExpr>();
    return std::make_shared<core::NullIfTypedExpr>(
        optimizedInputs[0], optimizedInputs[1], nullIfExpr->commonType());
  }

  if (expr->isCastKind()) {
    const auto optimizedInput =
        optimize(expr->inputs().at(0), constantEvaluator, makeFailExpr);
    const auto* castExpr = expr->asUnchecked<core::CastTypedExpr>();
    return std::make_shared<core::CastTypedExpr>(
        expr->type(), optimizedInput, castExpr->isTryCast());
  }

  if (expr->isLambdaKind()) {
    const auto* lambdaExpr = expr->asUnchecked<core::LambdaTypedExpr>();
    const auto foldedBody =
        optimize(lambdaExpr->body(), constantEvaluator, makeFailExpr);
    return std::make_shared<core::LambdaTypedExpr>(
        lambdaExpr->signature(), foldedBody);
  }

  return expr;
}
core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    const ConstantEvaluator& constantEvaluator,
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

  result = optimizeInputs(result, constantEvaluator, makeFailExpr);
  bool allInputsConstant = true;
  for (const auto& input : result->inputs()) {
    if (!input->isConstantKind()) {
      allInputsConstant = false;
      break;
    }
  }

  if (allInputsConstant) {
    return tryConstantFold(result, constantEvaluator, makeFailExpr);
  }
  return ExprRewriteRegistry::instance().rewrite(result);
}

} // namespace

core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    const MakeFailExpr& makeFailExpr) {
  return optimize(
      expr,
      [queryCtx, pool](
          const core::TypedExprPtr& foldExpr, bool suppressEvaluationFailures) {
        return exec::tryEvaluateConstantExpression(
            foldExpr, pool, queryCtx, suppressEvaluationFailures);
      },
      makeFailExpr);
}

core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    const MakeFailExpr& makeFailExpr) {
  return optimize(
      expr,
      [evaluator](
          const core::TypedExprPtr& foldExpr, bool suppressEvaluationFailures) {
        return exec::tryEvaluateConstantExpression(
            foldExpr, evaluator, suppressEvaluationFailures);
      },
      makeFailExpr);
}

} // namespace facebook::velox::expression
