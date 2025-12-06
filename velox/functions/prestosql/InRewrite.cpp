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
#include "velox/functions/prestosql/InRewrite.h"
#include "velox/expression/ExprRewriteRegistry.h"

namespace facebook::velox::functions {

core::TypedExprPtr InRewrite::rewrite(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  // IN expression with a non-NULL constant value can be rewritten.
  if (expr->inputs().empty() || !expr->inputs()[0]->isConstantKind()) {
    return nullptr;
  }

  if (!expr->isCallKind()) {
    return nullptr;
  }
  static const char* kIn = "in";
  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  if (callExpr->name() != kIn) {
    return nullptr;
  }

  const auto& valueExpr = callExpr->inputs().at(0);
  const auto* constValueExpr =
      valueExpr->asUnchecked<core::ConstantTypedExpr>();
  const auto& inputs = callExpr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;
  optimizedInputs.emplace_back(valueExpr);

  for (auto i = 1; i < numInputs; i++) {
    const auto& input = inputs[i];
    if (input->isConstantKind()) {
      // Ensure comparison of constants follows Presto's semantics for NULL
      // comparison and accounts for indeterminate result.
      static constexpr auto kNullAsIndeterminate =
          CompareFlags::NullHandlingMode::kNullAsIndeterminate;
      const auto* constInput = input->asUnchecked<core::ConstantTypedExpr>();
      auto compareResult =
          constInput->equals(*constValueExpr, kNullAsIndeterminate, pool);

      if UNLIKELY (!compareResult.has_value()) {
        optimizedInputs.push_back(input);
      } else if (compareResult.value()) {
        return std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true);
      }
    } else {
      optimizedInputs.push_back(input);
    }
  }

  VELOX_CHECK(!optimizedInputs.empty(), "No inputs to IN after rewrite.");
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), callExpr->name());
}

void InRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      functions::InRewrite::rewrite);
}

} // namespace facebook::velox::functions
