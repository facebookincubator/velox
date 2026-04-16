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

#pragma once

#include "velox/experimental/cudf/expression/CudfExprCtx.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

namespace facebook::velox::cudf_velox {

/// Compiler that transforms TypedExpr trees into CudfExpressions by
/// selecting the best evaluator for each sub-tree.
///
/// This class is intended for top-level operator compilation scopes
/// (e.g. operator initialization). Recursive sub-expression compilation should
/// use createCudfExpression().
///
/// The compile() method:
///   1. Optimizes the expression (constant folding / rewrites) when QueryCtx
///      and MemoryPool are available.
///   2. Selects the best evaluator for the root expression.
///   3. Creates the evaluator.  Each evaluator handles its own sub-tree
///      compilation internally (e.g. FunctionExpression recursively calls
///      createCudfExpression for children, AST calls createCudfExpression).
///
/// Usage:
///   CudfExpressionCompiler compiler(schema, exprCtx);
///   auto filter = compiler.compile(filterExpr);
///   auto proj   = compiler.compile(projectExpr);
class CudfExpressionCompiler {
 public:
  /// Construct a compiler for the given input schema.
  ///
  /// @param inputRowSchema  The schema of the input row.
  /// @param exprCtx         Expression compilation context.
  CudfExpressionCompiler(
      const RowTypePtr& inputRowSchema,
      CudfExprCtx exprCtx);

  /// Compile a single expression.
  ///
  /// If queryCtx and pool were provided at construction, the expression is
  /// optimized (constant folding / rewrites) before compilation.  The
  /// optimized root expression is retained and accessible via optimizedExpr().
  std::shared_ptr<CudfExpression> compile(const core::TypedExprPtr& expr);

  /// Compile a sub-expression without optimization.
  ///
  /// This is used by recursive evaluator internals after top-level compile()
  /// has already optimized the root expression.
  static std::shared_ptr<CudfExpression> compileSubExpression(
      const core::TypedExprPtr& expr,
      const RowTypePtr& inputRowSchema,
      CudfExprCtx exprCtx);

  /// The optimized expression from the most recent top-level compile() call.
  ///
  /// Compatibility note: this API exists for legacy CudfHashJoin usage.
  /// New code should avoid depending on this stateful accessor.
  const core::TypedExprPtr& optimizedExpr() const {
    return rootOptimizedExpr_;
  }

  const RowTypePtr& inputRowSchema() const {
    return schema_;
  }

 private:
  std::shared_ptr<CudfExpression> compileImpl(const core::TypedExprPtr& expr);

  RowTypePtr schema_;
  CudfExprCtx exprCtx_;

  /// The optimized expression from the most recent top-level compile() call.
  core::TypedExprPtr rootOptimizedExpr_;
};

} // namespace facebook::velox::cudf_velox
