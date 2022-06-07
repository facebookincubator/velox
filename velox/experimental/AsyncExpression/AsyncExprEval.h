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
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>

#include "velox/experimental/AsyncExpression/AsyncVectorFunction.h"
#include "velox/expression/ControlExpr.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

class AsyncExprEval {
 public:
  // Async path to execute an expression that is composed of supported async
  // expressions. For now only AsyncVectorFunctions.
  static folly::coro::Task<void> evalExprAsync(
      Expr& expression_,
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) {
    VELOX_CHECK(
        isSupportedAsyncExpression(expression_),
        "expression is not supported in asyncEval")

    co_return co_await evalExprAsyncIntenral(
        expression_, rows, context, result);
  }

  // Async path to execute an expression that is composed of supported async
  // expressions. For now only AsyncVectorFunctions.
  static folly::coro::Task<void> evalExprSetAsync(
      ExprSet& exprSet,
      const SelectivityVector& rows,
      EvalCtx& context,
      std::vector<VectorPtr>& result) {
    result.resize(exprSet.exprs_.size());

    std::vector<folly::coro::Task<void>> tasks;

    for (int32_t i = 0; i < exprSet.exprs_.size(); ++i) {
      tasks.push_back(
          evalExprAsync(*exprSet.exprs_[i], rows, context, result[i]));
    }
    co_await folly::coro::collectAllRange(std::move(tasks));
    co_return;
  }

 private:
  static folly::coro::Task<void> evalExprAsyncIntenral(
      Expr& expr,
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result);

  static bool isSupportedAsyncExpression(const Expr& expr) {
    if (expr.isSpecialForm() && !isFieldReference(expr) && !isConstant(expr)) {
      return false;
    }
    bool result = true;
    for (auto& input : expr.inputs()) {
      result = result && isSupportedAsyncExpression(*input);
    }
    return result;
  }

  static bool isFieldReference(const Expr& expr) {
    return dynamic_cast<const FieldReference*>(&expr) != nullptr;
  }

  static bool isConstant(const Expr& expr) {
    return dynamic_cast<const ConstantExpr*>(&expr) != nullptr;
  }
};

} // namespace facebook::velox::exec
