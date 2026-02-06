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

#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include <cudf/ast/expressions.hpp>

namespace facebook::velox::cudf_velox {

const std::string kJitEvaluatorName = "jit";

// Evaluates the JIT expression tree
class JitExpression : public CudfExpression {
 public:
  JitExpression() = default;
  // Converts velox expressions to cudf::ast::tree, scalars and
  // precompute instructions and stores them
  JitExpression(
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
  // JitExpression
  static bool canEvaluate(std::shared_ptr<velox::exec::Expr> expr);

 private:
  ASTExpression expr_;
};

void registerJitEvaluator(int priority);

} // namespace facebook::velox::cudf_velox
