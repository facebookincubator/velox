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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"

namespace facebook::velox::cudf_velox {

namespace {

std::unordered_map<std::string, CudfExpressionEvaluatorEntry>&
getRegistryImpl() {
  static std::unordered_map<std::string, CudfExpressionEvaluatorEntry> registry;
  return registry;
}

bool registeredBuiltins = false;

} // namespace

bool registerCudfExpressionEvaluator(
    const std::string& name,
    int priority,
    CudfExpressionEvaluatorCanEvaluate canEvaluate,
    CudfExpressionEvaluatorCreate create,
    bool overwrite) {
  auto& registry = getCudfExpressionEvaluatorRegistry();
  if (!overwrite && registry.find(name) != registry.end()) {
    return false;
  }
  registry[name] = CudfExpressionEvaluatorEntry{
      priority, std::move(canEvaluate), std::move(create)};
  return true;
}

std::unordered_map<std::string, CudfExpressionEvaluatorEntry>&
getCudfExpressionEvaluatorRegistry() {
  return getRegistryImpl();
}

void ensureBuiltinExpressionEvaluatorsRegistered() {
  if (registeredBuiltins) {
    return;
  }

  // Default priority for function evaluator
  const int kFunctionPriority = 50;

  // Function evaluator
  registerCudfExpressionEvaluator(
      "function",
      kFunctionPriority,
      [](const core::TypedExprPtr& expr) {
        return FunctionExpression::canEvaluate(expr);
      },
      [](const core::TypedExprPtr& expr,
         const RowTypePtr& row,
         CudfExprCtx exprCtx) {
        return FunctionExpression::create(expr, row, exprCtx);
      },
      /*overwrite=*/false);

  registeredBuiltins = true;
}

} // namespace facebook::velox::cudf_velox
