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

#include "velox/experimental/cudf/expression/CudfExpressionCompiler.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"

#include "velox/expression/ExprOptimizer.h"

namespace facebook::velox::cudf_velox {

CudfExpressionCompiler::CudfExpressionCompiler(
    const RowTypePtr& inputRowSchema,
    CudfExprCtx exprCtx)
    : schema_(inputRowSchema), exprCtx_(exprCtx) {}

std::shared_ptr<CudfExpression> CudfExpressionCompiler::compileImpl(
    const core::TypedExprPtr& expr) {
  // Select the best evaluator and create the expression.  Each evaluator
  // handles its own sub-tree compilation internally.
  const auto* best = findBestEvaluator(expr);
  VELOX_CHECK_NOT_NULL(
      best,
      "No cuDF expression evaluator can handle: {}",
      expr->toString());
  return best->create(expr, schema_, exprCtx_);
}

std::shared_ptr<CudfExpression> CudfExpressionCompiler::compile(
    const core::TypedExprPtr& expr) {
  const auto compiledExpr =
      expression::optimize(expr, exprCtx_.queryCtx, exprCtx_.pool);
  rootOptimizedExpr_ = compiledExpr;
  return compileImpl(compiledExpr);
}

std::shared_ptr<CudfExpression> CudfExpressionCompiler::compileSubExpression(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    CudfExprCtx exprCtx) {
  CudfExpressionCompiler compiler(inputRowSchema, exprCtx);
  return compiler.compileImpl(expr);
}

} // namespace facebook::velox::cudf_velox
