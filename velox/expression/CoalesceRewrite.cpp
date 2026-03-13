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

#include "velox/expression/CoalesceRewrite.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"

namespace facebook::velox::expression {

core::TypedExprPtr CoalesceRewrite::rewrite(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/) {
  if (!expr->isCallKind()) {
    return nullptr;
  }
  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  if (callExpr->name() != kCoalesce) {
    return nullptr;
  }

  // Deduplicate inputs to COALESCE and remove NULL inputs.
  std::vector<core::TypedExprPtr> flat;
  utils::flattenInput(expr, kCoalesce, flat);
  folly::F14FastSet<
      const core::ITypedExpr*,
      core::ITypedExprHasher,
      core::ITypedExprComparer>
      uniqueInputs;
  std::vector<core::TypedExprPtr> deduplicatedInputs;

  for (const auto& input : flat) {
    if (input->isConstantKind()) {
      const auto* constantExpr = input->asUnchecked<core::ConstantTypedExpr>();
      if (constantExpr->isNull()) {
        // Drop NULL constant.
        continue;
      }

      if (uniqueInputs.empty()) {
        // Return first non-NULL constant input.
        return input;
      }

      // Drop inputs after non-NULL constant input.
      uniqueInputs.insert(input.get());
      deduplicatedInputs.push_back(input);
      break;
    } else {
      if (uniqueInputs.emplace(input.get()).second) {
        deduplicatedInputs.push_back(input);
      }
    }
  }

  // Return NULL if all inputs to COALESCE are NULL.
  if (uniqueInputs.empty()) {
    return flat.front();
  }
  // If there is a single input to COALESCE, return this expression.
  if (uniqueInputs.size() == 1) {
    return deduplicatedInputs.front();
  }

  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(deduplicatedInputs), kCoalesce);
}

void CoalesceRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      expression::CoalesceRewrite::rewrite);
}

} // namespace facebook::velox::expression
