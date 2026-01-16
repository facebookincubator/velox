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

#include <cudf/ast/expressions.hpp>

namespace facebook::velox::cudf_velox {

const std::string kAstEvaluatorName = "ast";

// Pre-compute instructions for the expression,
// for ops that are not supported by cudf::ast
struct PrecomputeInstruction {
  int dependent_column_index;
  std::string ins_name;
  int new_column_index;
  std::vector<int> nested_dependent_column_indices;
  std::shared_ptr<CudfExpression> cudf_expression;

  // Constructor to initialize the struct with values
  PrecomputeInstruction(
      int depIndex,
      const std::string& name,
      int newIndex,
      const std::shared_ptr<CudfExpression>& node = nullptr)
      : dependent_column_index(depIndex),
        ins_name(name),
        new_column_index(newIndex),
        cudf_expression(node) {}

  // TODO (dm): This two ctor situation is crazy.
  PrecomputeInstruction(
      int depIndex,
      const std::string& name,
      int newIndex,
      const std::vector<int>& nestedIndices,
      const std::shared_ptr<CudfExpression>& node = nullptr)
      : dependent_column_index(depIndex),
        ins_name(name),
        new_column_index(newIndex),
        nested_dependent_column_indices(nestedIndices),
        cudf_expression(node) {}
};

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
    std::vector<PrecomputeInstruction>& rightPrecomputeInstructions,
    const bool allowPureAstOnly);

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
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
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
};

void registerAstEvaluator(int priority);

} // namespace facebook::velox::cudf_velox
