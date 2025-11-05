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

#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"
#include "velox/expression/InRewrite.h"

namespace facebook::velox::expression {

core::TypedExprPtr InRewrite::rewrite(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return nullptr;
  }
  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  if (callExpr->name() != kIn) {
    return nullptr;
  }

  const auto& valueExpr = callExpr->inputs().at(0);
  if (!valueExpr->isConstantKind()) {
    return nullptr;
  }
  // IN expression with a constant IN-list can be evaluated directly and will
  // not be rewritten.
  const auto inList = callExpr->inputs().at(1);
  if (inList->isConstantKind() && callExpr->inputs().size() == 2) {
    return nullptr;
  }

  const auto* constValueExpr =
      valueExpr->asUnchecked<core::ConstantTypedExpr>();
  const auto inputs = callExpr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;
  optimizedInputs.emplace_back(valueExpr);

  for (auto i = 1; i < numInputs; i++) {
    if (inputs[i]->isConstantKind()) {
      const auto* constInput =
          inputs[i]->asUnchecked<core::ConstantTypedExpr>();
      if (*constInput == *constValueExpr) {
        return std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true);
      }
    } else {
      optimizedInputs.push_back(inputs[i]);
    }
  }

  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), callExpr->name());
}

void InRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      expression::InRewrite::rewrite);
}

} // namespace facebook::velox::expression
