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

#include <folly/experimental/coro/Collect.h>

#include "velox/experimental/AsyncExpression/AsyncExprEval.h"
#include "velox/experimental/AsyncExpression/AsyncVectorFunction.h"
namespace facebook::velox::exec {

folly::coro::Task<void> AsyncExprEval::evalExprAsyncIntenral(
    Expr& expr,
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  if (!rows.hasSelections()) {
    // Empty input, return an empty vector of the right type.ß
    result = BaseVector::createNullConstant(expr.type(), 0, context.pool());
    co_return;
  }

  if (auto fieldReference = dynamic_cast<FieldReference*>(&expr)) {
    fieldReference->evalSpecialForm(rows, context, result);
    co_return;
  } else if (auto constantExpr = dynamic_cast<ConstantExpr*>(&expr)) {
    fieldReference->evalSpecialForm(rows, context, result);
    co_return;
  }

  // Expression is vector function.
  expr.inputValues_.resize(expr.inputs_.size());

  std::vector<folly::coro::Task<void>> tasks;
  for (auto i = 0; i < expr.inputs_.size(); i++) {
    tasks.push_back(evalExprAsyncIntenral(
        *expr.inputs_[i], rows, context, expr.inputValues_[i]));
  }
  co_await folly::coro::collectAllRange(std::move(tasks));

  if (auto asyncFunction = std::dynamic_pointer_cast<AsyncVectorFunction>(
          expr.vectorFunction_)) {
    co_await asyncFunction->applyAsync(
        rows, expr.inputValues_, expr.type(), &context, &result);
  } else {
    expr.vectorFunction_->apply(
        rows, expr.inputValues_, expr.type(), &context, &result);
  }

  expr.inputValues_.clear();
  co_return;
}
} // namespace facebook::velox::exec
