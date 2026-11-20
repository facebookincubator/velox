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
#include "velox/expression/SwitchRewrite.h"

namespace facebook::velox::expression {
namespace {

// Removes trailing (condition, value) pairs in 'inputs[begin, end - 1)' whose
// value equals the else value 'inputs[end - 1]'. Such a pair returns the else
// value whether its condition is true or not. Returns the new end of 'inputs'.
size_t dropCasesEqualToElse(
    std::vector<core::TypedExprPtr>& inputs,
    size_t begin) {
  const auto& elseValue = inputs.back();
  auto end = inputs.size() - 1;
  // When the expression runs, a failing condition raises its error unless the
  // values it can select are equal. Values that aren't comparable, e.g.
  // lambdas, can't be checked, so keep their conditions.
  if (!elseValue->type()->isComparable()) {
    return end;
  }
  while (end - begin >= 2 && *inputs[end - 1] == *elseValue) {
    end -= 2;
  }
  inputs.erase(inputs.begin() + end, inputs.end() - 1);
  return end;
}

// Rewrites a simple CASE expression with inputs `subject, when1, value1, ..,
// whenN, valueN, elseValue`. Returns nullptr if nothing changes.
core::TypedExprPtr rewriteCase(const core::CallTypedExpr& callExpr) {
  const auto& inputs = callExpr.inputs();
  if (inputs.size() % 2 == 1) {
    // No else value.
    return nullptr;
  }

  // The subject is compared to the WHEN values in their common supertype. A
  // WHEN value of another type can widen it, e.g. to DOUBLE for a BIGINT
  // subject, while WHEN values of the subject's type don't change it. So only
  // drop cases after the last WHEN value of another type.
  const auto& subjectType = inputs[0]->type();
  size_t begin = 1;
  for (size_t i = 1; i < inputs.size() - 1; i += 2) {
    if (*inputs[i]->type() != *subjectType) {
      begin = i + 2;
    }
  }

  auto optimizedInputs = inputs;
  if (dropCasesEqualToElse(optimizedInputs, begin) == 1) {
    return optimizedInputs.back();
  }
  if (optimizedInputs.size() == inputs.size()) {
    return nullptr;
  }
  return std::make_shared<core::CallTypedExpr>(
      callExpr.type(), std::move(optimizedInputs), callExpr.name());
}

} // namespace

core::TypedExprPtr SwitchRewrite::rewrite(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return nullptr;
  }
  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  if (callExpr->name() == kCase) {
    return rewriteCase(*callExpr);
  }
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

  if (optimizedInputs.size() % 2 == 1 &&
      dropCasesEqualToElse(optimizedInputs, 0) == 0) {
    return optimizedInputs.back();
  }

  // No condition was dropped: report no rewrite.
  if (utils::sameInputs(expr, optimizedInputs)) {
    return nullptr;
  }

  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), callExpr->name());
}

void SwitchRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      expression::SwitchRewrite::rewrite);
}

} // namespace facebook::velox::expression
