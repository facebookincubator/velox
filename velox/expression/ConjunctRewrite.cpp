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

#include "velox/expression/ConjunctRewrite.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"

namespace facebook::velox::expression {

core::TypedExprPtr ConjunctRewrite::rewrite(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return nullptr;
  }

  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  const bool conjunct = callExpr->name() == kAnd;
  const bool disjunct = callExpr->name() == kOr;
  if (!conjunct && !disjunct) {
    return nullptr;
  }

  std::vector<core::TypedExprPtr> flat;
  utils::flattenInput(expr, callExpr->name(), flat);
  std::vector<core::TypedExprPtr> optimizedInputs;
  bool nullInput = false;

  for (const auto& input : flat) {
    if (input->isConstantKind()) {
      const auto* constInput = input->asUnchecked<core::ConstantTypedExpr>();
      if (auto boolInput = constInput->toBool()) {
        if (!boolInput.value() && conjunct) {
          return input;
        }
        if (boolInput.value() && disjunct) {
          return input;
        }
      } else if (!nullInput) {
        nullInput = true;
        optimizedInputs.push_back(input);
      }
    } else {
      optimizedInputs.push_back(input);
    }
  }

  // optimizedInputs can be empty for expressions like `true OR true`, returns
  // the first input in this case.
  if (optimizedInputs.empty()) {
    return expr->inputs().front();
  }
  if (optimizedInputs.size() == 1) {
    return optimizedInputs.front();
  }
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), callExpr->name());
}

void ConjunctRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      expression::ConjunctRewrite::rewrite);
}

} // namespace facebook::velox::expression
