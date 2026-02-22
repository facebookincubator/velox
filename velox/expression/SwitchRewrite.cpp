/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/expression/SwitchRewrite.h"

namespace facebook::velox::expression {

core::TypedExprPtr SwitchRewrite::rewrite(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return nullptr;
  }
  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  if (!(callExpr->name() == kSwitch) && !(callExpr->name() == kIf)) {
    return nullptr;
  }

  const auto& inputs = expr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;

  // Iterate over all `(condition, value)` pairs. Handle `else` value at the
  // end.
  for (auto i = 0; i < numInputs - 1; i += 2) {
    const auto& condition = inputs.at(i);
    const auto& value = inputs.at(i + 1);

    if (condition->isConstantKind()) {
      const auto* constCondition =
          condition->asUnchecked<core::ConstantTypedExpr>();
      if (auto boolCondition = constCondition->toBool()) {
        if (boolCondition.value()) {
          // If this condition is true and all conditions before this are false,
          // simplify `switch` expression to the corresponding `value`.
          if (optimizedInputs.empty()) {
            return value;
          }
          // If this condition is true and all conditions before this can only
          // be evaluated at runtime, make this the new `else` value and stop
          // checking further conditions.
          optimizedInputs.push_back(value);
          break;
        } else {
          // Skip false conditions.
          continue;
        }
      } else {
        // Skip NULL conditions.
        continue;
      }
    } else {
      optimizedInputs.push_back(condition);
      optimizedInputs.push_back(value);
    }
  }

  // Handle `else` value if present.
  if (optimizedInputs.size() % 2 == 0 && numInputs % 2 == 1) {
    const auto elseValue = inputs.at(numInputs - 1);
    // Return `else` value if there are no conditions.
    if (optimizedInputs.empty()) {
      return elseValue;
    }
    optimizedInputs.emplace_back(elseValue);
  }
  // Return NULL if there are no conditions and `else` value is not present.
  if (optimizedInputs.empty()) {
    return core::ConstantTypedExpr::makeNull(expr->type());
  }

  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), callExpr->name());
}

void SwitchRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      expression::SwitchRewrite::rewrite);
}

} // namespace facebook::velox::expression
