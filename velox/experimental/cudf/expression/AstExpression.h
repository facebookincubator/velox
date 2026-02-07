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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrecomputeInstruction.h"

#include <cudf/ast/expressions.hpp>

namespace facebook::velox::cudf_velox {

const std::string kAstEvaluatorName = "ast";

cudf::ast::expression const& createAstTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<PrecomputeInstruction>& precomputeInstructions);

cudf::ast::expression const& createAstTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& leftRowSchema,
    const RowTypePtr& rightRowSchema,
    std::vector<PrecomputeInstruction>& leftPrecomputeInstructions,
    std::vector<PrecomputeInstruction>& rightPrecomputeInstructions);

/// Executes precompute instructions and returns computed columns.
/// @param inputColumnViews The input columns as views
/// @param precomputeInstructions Instructions for precomputation
/// @param scalars Scalar values used in precompute operations
/// @param inputRowSchema The schema of the input table
/// @param stream CUDA stream for operations
/// @return Vector of precomputed columns (either views or owned columns)
std::vector<ColumnOrView> precomputeSubexpressions(
    const std::vector<cudf::column_view>& inputColumnViews,
    const std::vector<PrecomputeInstruction>& precomputeInstructions,
    const std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    rmm::cuda_stream_view stream);

// Evaluates the expression tree
class ASTExpression : public CudfExpression {
 public:
  ASTExpression() = default;
  // Converts velox expressions to cudf::ast::tree, scalars and
  // precompute instructions and stores them
  ASTExpression(
      std::shared_ptr<velox::exec::Expr> expr,
      const RowTypePtr& inputRowSchema);

  // Evaluates the expression tree for the given input columns
  ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false) override;

  void close() override;

  // Check if this specific operation (not its children) can be evaluated by
  // ASTExpression
  static bool canEvaluate(std::shared_ptr<velox::exec::Expr> expr);

 private:
  std::shared_ptr<velox::exec::Expr> expr_;

  cudf::ast::tree cudfTree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  // instruction on dependent column to get new column index on non-ast
  // supported operations in expressions
  // <dependent_column_index, "instruction", new_column_index>
  std::vector<PrecomputeInstruction> precomputeInstructions_;
  RowTypePtr inputRowSchema_;

  friend class JitExpression;
};

void registerAstEvaluator(int priority);

} // namespace facebook::velox::cudf_velox
