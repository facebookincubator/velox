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
#include "velox/expression/SpecialFormRewrites.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprUtils.h"

namespace facebook::velox::expression {

core::TypedExprPtr rewriteConjunctExpression(
    const core::TypedExprPtr& input,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  if (!queryCtx->queryConfig().exprApplySpecialFormRewrites() ||
      (!utils::isCall(input, kAnd) && !utils::isCall(input, kOr))) {
    return nullptr;
  }

  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  const bool isAnd = (expr->name() == kAnd) ? true : false;
  // If all inputs are AND or OR then we can flatten the inputs into a vector
  // before further optimizing.
  auto canFlatten = utils::allInputTypesEquivalent(expr);
  std::vector<core::TypedExprPtr> flat;
  if (canFlatten) {
    utils::flattenInput(input, expr->name(), flat);
  }

  const auto& inputsToOptimize = canFlatten ? flat : expr->inputs();
  bool allInputsConstant = true;
  bool hasNullInput = false;
  std::vector<core::TypedExprPtr> optimizedInputs;
  core::TypedExprPtr nullInput = nullptr;
  for (const auto& inputExpr : inputsToOptimize) {
    auto foldedExpr = utils::tryConstantFold(inputExpr, queryCtx, pool);
    if (utils::isCall(foldedExpr, kFail)) {
      return foldedExpr;
    }

    const auto result = utils::evalExprAsConstant(foldedExpr);
    switch (result) {
      case utils::ConstantEvalResult::IS_NULL:
        if (!hasNullInput) {
          hasNullInput = true;
          nullInput = inputExpr;
        }
        break;
      case utils::ConstantEvalResult::IS_TRUE:
        if (!isAnd) {
          // OR (.., true, ..) -> true
          return foldedExpr;
        }
        break;
      case utils::ConstantEvalResult::IS_FALSE:
        if (isAnd) {
          // AND (.., false, ..) -> false
          return foldedExpr;
        }
        break;
      case utils::ConstantEvalResult::IS_NOT_CONSTANT:
        allInputsConstant = false;
        optimizedInputs.push_back(inputExpr);
        break;
      default:
        return nullptr;
    }
  }

  if (allInputsConstant && hasNullInput) {
    return nullInput;
  } else if (optimizedInputs.empty()) {
    return expr->inputs().front();
  } else if (optimizedInputs.size() == 1) {
    return optimizedInputs.front();
  }
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), expr->name());
}
} // namespace facebook::velox::expression
