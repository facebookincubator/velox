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
#include <set>

#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"
#include "velox/functions/prestosql/InRewrite.h"

namespace facebook::velox::functions {

core::TypedExprPtr InRewrite::rewrite(const core::TypedExprPtr& expr) {
  // IN expression with a constant value can be rewritten.
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
  if (constValueExpr->isNull()) {
    return core::ConstantTypedExpr::makeNull(BOOLEAN());
  }

  const auto& inputs = callExpr->inputs();
  const auto numInputs = inputs.size();
  bool hasNullInput = false;
  std::vector<core::TypedExprPtr> optimizedInputs;
  optimizedInputs.emplace_back(valueExpr);

  for (auto i = 1; i < numInputs; i++) {
    const auto& input = inputs[i];
    if (input->isConstantKind()) {
      const auto* constInput = input->asUnchecked<core::ConstantTypedExpr>();
      if (constInput->isNull()) {
        if (!hasNullInput) {
          optimizedInputs.push_back(input);
          hasNullInput = true;
        }
      } else if (*constInput == *constValueExpr) {
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
