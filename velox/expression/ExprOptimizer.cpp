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

#define HANDLE_FAIL_FUNCTION_RESULT(result)                           \
  if (replaceEvalErrorWithFailExpr && utils::isCall(result, kFail)) { \
    return result;                                                    \
  }

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
    // Return the expression unevaluated.
    return expr;
  } catch (VeloxUserError& e) {
    if (replaceEvalErrorWithFailExpr) {
      return std::make_shared<core::CallTypedExpr>(
          UNKNOWN(),
          expression::kFail,
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), e.what()));
    }
    return expr;
  }
}

const core::TypedExprPtr foldInputs(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvalErrorWithFailExpr) {
  if (expr->isCallKind()) {
    std::vector<core::TypedExprPtr> foldedInputs;
    for (const auto& input : expr->inputs()) {
      const auto foldedInput =
          optimize(input, queryCtx, pool, replaceEvalErrorWithFailExpr);
      HANDLE_FAIL_FUNCTION_RESULT(foldedInput);
      foldedInputs.push_back(foldedInput);
    }
    const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();

    return std::make_shared<core::CallTypedExpr>(
        callExpr->type(), foldedInputs, callExpr->name());
  } else if (expr->isCastKind()) {
    const auto foldedInput = optimize(
        expr->inputs().at(0), queryCtx, pool, replaceEvalErrorWithFailExpr);
    HANDLE_FAIL_FUNCTION_RESULT(foldedInput);
    if (expr->type() == foldedInput->type()) {
      return foldedInput;
    }
    const auto* castExpr = expr->asUnchecked<core::CastTypedExpr>();

    return std::make_shared<core::CastTypedExpr>(
        expr->type(), foldedInput, castExpr->isTryCast());
  } else if (expr->isLambdaKind()) {
    const auto* lambdaExpr = expr->asUnchecked<core::LambdaTypedExpr>();
    const auto foldedBody = optimize(
        lambdaExpr->body(), queryCtx, pool, replaceEvalErrorWithFailExpr);
    HANDLE_FAIL_FUNCTION_RESULT(foldedBody);

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
  if (expr->isConstantKind()) {
    return expr;
  } else if (expr->isFieldAccessKind()) {
    return expr;
  }

  core::TypedExprPtr result;
  const auto kind = utils::getExprInputsKind(expr);
  switch (kind) {
    case utils::kAllConstant: {
      result =
          tryConstantFold(expr, queryCtx, pool, replaceEvalErrorWithFailExpr);
      break;
    }
    case utils::kAllField: {
      if (expr->isCastKind() && expr->inputs().at(0)->type() == expr->type()) {
        result = expr->inputs().at(0);
      } else {
        result = expr;
      }
      break;
    }
    case utils::kConstantOrField: {
      const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
      if (callExpr && utils::isAssociativeOperator(callExpr)) {
        std::vector<core::TypedExprPtr> constantInputs;
        std::vector<core::TypedExprPtr> fieldInputs;
        for (const auto& input : expr->inputs()) {
          if (input->isConstantKind()) {
            constantInputs.push_back(input);
          } else {
            fieldInputs.push_back(input);
          }
        }

        if (constantInputs.size() == 1) {
          result = expr;
        } else {
          std::vector<core::TypedExprPtr> newInputs = fieldInputs;
          const auto foldableCallExpr = std::make_shared<core::CallTypedExpr>(
              callExpr->type(), constantInputs, callExpr->name());
          const auto folded = tryConstantFold(
              foldableCallExpr, queryCtx, pool, replaceEvalErrorWithFailExpr);
          newInputs.push_back(folded);
          result = std::make_shared<core::CallTypedExpr>(
              callExpr->type(), newInputs, callExpr->name());
        }
      } else {
        result = expr;
      }
      break;
    }
    case utils::kDefault: {
      result = foldInputs(expr, queryCtx, pool, replaceEvalErrorWithFailExpr);
      break;
    }
    default:
      VELOX_UNREACHABLE("Invalid ExprInputKind.");
  }

  HANDLE_FAIL_FUNCTION_RESULT(result);
  // Try evaluating whole expression after folding subtrees, eg: (1 - 1) * a,
  // between(cast(null as integer), 2, 4), cast(cast(123 as BIGINT) as VARCHAR),
  // row_constructor(1,a,true)[3].
  if (utils::getExprInputsKind(result) != utils::kAllField) {
    result =
        tryConstantFold(result, queryCtx, pool, replaceEvalErrorWithFailExpr);
    HANDLE_FAIL_FUNCTION_RESULT(result);
  }

  // Try rewrite after folding subtrees and trying to constant fold.
  return ExprRewriteRegistry::instance().rewrite(result);
}

#undef HANDLE_FAIL_FUNCTION_RESULT
} // namespace facebook::velox::expression
