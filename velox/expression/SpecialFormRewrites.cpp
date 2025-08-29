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
#include "velox/expression/ExprUtils.h"
#include "velox/expression/SpecialFormRewrites.h"

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

/// Comparator for core::TypedExprPtr; used to deduplicate arguments to
/// COALESCE special form expression.
struct TypedExprComparator {
  bool operator()(const core::TypedExprPtr& a, const core::TypedExprPtr& b)
      const {
    return a->hash() < b->hash();
  }
};

core::TypedExprPtr addCoalesceArgument(
    const core::TypedExprPtr& input,
    std::set<core::TypedExprPtr, TypedExprComparator>& deduplicatedInputSet,
    std::vector<core::TypedExprPtr>& deduplicatedInputList,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  auto foldedExpr = utils::tryConstantFold(input, queryCtx, pool);
  if (utils::isCall(foldedExpr, kFail)) {
    return foldedExpr;
  }

  // First non-NULL constant input to COALESCE returns non-NULL value.
  const auto result = utils::evalExprAsConstant(foldedExpr);
  switch (result) {
    case utils::ConstantEvalResult::IS_NULL:
      break;
    case utils::ConstantEvalResult::IS_NON_BOOL_CONSTANT:
    case utils::ConstantEvalResult::IS_TRUE:
    case utils::ConstantEvalResult::IS_FALSE:
      return foldedExpr;
    case utils::ConstantEvalResult::IS_NOT_CONSTANT: {
      if (deduplicatedInputSet.find(foldedExpr) == deduplicatedInputSet.end()) {
        deduplicatedInputSet.insert(foldedExpr);
        deduplicatedInputList.emplace_back(foldedExpr);
      }
      break;
    }
    default:
      VELOX_UNREACHABLE("Err");
  }

  return nullptr;
}

core::TypedExprPtr rewriteCoalesceExpression(
    const core::TypedExprPtr& input,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  if (!queryCtx->queryConfig().exprApplySpecialFormRewrites() ||
      !utils::isCall(input, kCoalesce)) {
    return nullptr;
  }

  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  // Deduplicate inputs to COALESCE and remove NULL inputs, returning a list of
  // optimized inputs to COALESCE.
  std::set<core::TypedExprPtr, TypedExprComparator> deduplicatedInputSet;
  std::vector<core::TypedExprPtr> deduplicatedInputList;
  auto canFlatten = utils::allInputTypesEquivalent(expr);
  std::vector<core::TypedExprPtr> flat;
  if (canFlatten) {
    utils::flattenInput(input, expr->name(), flat);
  }

  const auto& inputsToOptimize = canFlatten ? flat : expr->inputs();
  for (const auto& exprInput : inputsToOptimize) {
    // Once a constant input is seen, subsequent inputs to COALESCE expression
    // can be ignored.
    if (auto optimized = addCoalesceArgument(
            exprInput,
            deduplicatedInputSet,
            deduplicatedInputList,
            queryCtx,
            pool)) {
      if (deduplicatedInputSet.empty()) {
        return optimized;
      }
      deduplicatedInputSet.insert(optimized);
      deduplicatedInputList.emplace_back(optimized);
      break;
    }
  }

  // Return NULL if all inputs to COALESCE are NULL. If there is a single input
  // to COALESCE after deduplication, return this expression. Otherwise, return
  // COALESCE expression with deduplicated inputs.
  if (deduplicatedInputSet.empty()) {
    return inputsToOptimize.front();
  } else if (deduplicatedInputSet.size() == 1) {
    return deduplicatedInputList.front();
  }
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(deduplicatedInputList), expr->name());
}

// Input expression should be of form: IF(condition, then, else).
core::TypedExprPtr rewriteIfExpression(
    const core::TypedExprPtr& input,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  if (!utils::isCall(input, kIf) || input->inputs().size() != 3) {
    return nullptr;
  }

  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  const auto& condition = expr->inputs().at(0);
  auto foldedCondition = utils::tryConstantFold(condition, queryCtx, pool);
  // The folded expression could be the fail function. In this case,
  // we don't want to futher analyze the expression and instead return the fail
  // function expression.
  if (utils::isCall(foldedCondition, kFail)) {
    return foldedCondition;
  }
  const auto result = utils::evalExprAsConstant(foldedCondition);
  switch (result) {
    case utils::ConstantEvalResult::IS_NULL:
      [[fallthrough]];
    case utils::ConstantEvalResult::IS_TRUE:
      return expr->inputs().at(1);
    case utils::ConstantEvalResult::IS_FALSE:
      return expr->inputs().at(2);
    case utils::ConstantEvalResult::IS_NOT_CONSTANT:
      [[fallthrough]];
    default:
      return expr;
  }
  return expr;
}

// Input expression should be of form: SWITCH(condition1, value1, condition2,
//   value2, ...., defaultValue).
core::TypedExprPtr rewriteSwitchExpression(
    const core::TypedExprPtr& input,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  if (!queryCtx->queryConfig().exprApplySpecialFormRewrites() ||
      !utils::isCall(input, kSwitch)) {
    return nullptr;
  }

  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  const auto& inputs = expr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;
  // If a case evaluates to true, it will be the new else clause.
  bool hasOptimizedElseValue = false;
  for (auto i = 0; i < numInputs - 1; i += 2) {
    const auto& condition = inputs.at(i);
    const auto foldedCondition =
        utils::tryConstantFold(condition, queryCtx, pool);
    if (utils::isCall(foldedCondition, kFail)) {
      return foldedCondition;
    }

    const auto& value = inputs.at(i + 1);
    const auto foldedValue = utils::tryConstantFold(value, queryCtx, pool);
    if (utils::isCall(foldedValue, kFail)) {
      return foldedValue;
    }

    const auto result = utils::evalExprAsConstant(foldedCondition);
    switch (result) {
      case utils::ConstantEvalResult::IS_NULL:
        continue;
      case utils::ConstantEvalResult::IS_TRUE:
        if (optimizedInputs.empty()) {
          return foldedValue;
        }
        hasOptimizedElseValue = true;
        optimizedInputs.emplace_back(foldedValue);
        break;
      case utils::ConstantEvalResult::IS_FALSE:
        continue;
      case utils::ConstantEvalResult::IS_NOT_CONSTANT:
        optimizedInputs.emplace_back(foldedCondition);
        optimizedInputs.emplace_back(foldedValue);
        continue;
      default:
        return expr;
    }
    break;

    /*
        if (auto constantExpr =
                std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                    foldedCondition)) {
          if (auto constVector = constantExpr->toConstantVector(pool)) {
            if (!constVector->isNullAt(0) &&
                constVector->as<ConstantVector<bool>>()->valueAt(0)) {
              if (optimizedInputs.empty()) {
                return foldedValue;
              }
              hasOptimizedElseValue = true;
              optimizedInputs.emplace_back(foldedValue);
              break;
            }
          }
        } else {
          optimizedInputs.emplace_back(foldedCondition);
          optimizedInputs.emplace_back(foldedValue);
        }
    */
  }

  if (!hasOptimizedElseValue) {
    const auto foldedElseValue =
        utils::tryConstantFold(inputs.at(numInputs - 1), queryCtx, pool);
    if (utils::isCall(foldedElseValue, kFail)) {
      return foldedElseValue;
    }
    if (optimizedInputs.empty()) {
      return foldedElseValue;
    }
    optimizedInputs.emplace_back(foldedElseValue);
  }
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), expr->name());
}

// When all input literals in IN-list are constant, the expression is expected
// to be of type IN(value, arrayVector<literal1, ....., literalN>). When any
// input literal in IN-list is non-constant, the expression is expected to be
// of type IN(value, literal1, ....., literalN). The latter case is optimized
// by this function and the former is handled during constant folding.
core::TypedExprPtr rewriteInExpression(
    const core::TypedExprPtr& input,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  if (!queryCtx->queryConfig().exprApplySpecialFormRewrites() ||
      !utils::isCall(input, kIn) || input->inputs().size() < 2) {
    return nullptr;
  }

  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  const auto& value = expr->inputs().at(0);
  const auto foldedExpr = utils::tryConstantFold(value, queryCtx, pool);
  if (utils::isCall(foldedExpr, kFail)) {
    return foldedExpr;
  }

  if (auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
              foldedExpr)) {
    const auto& inList = expr->inputs().at(1);
    if (std::dynamic_pointer_cast<const core::ConstantTypedExpr>(inList) ==
        nullptr) {
      const auto& inputs = expr->inputs();
      const auto numInputs = inputs.size();
      std::vector<core::TypedExprPtr> optimizedInputs;
      optimizedInputs.emplace_back(foldedExpr);

      for (auto i = 1; i < numInputs; i++) {
        const auto& literal = inputs.at(i);
        const auto foldedLiteral =
            utils::tryConstantFold(literal, queryCtx, pool);
        if (utils::isCall(foldedLiteral, kFail)) {
          return foldedLiteral;
        }
        if (auto constantLiteral =
                std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                    foldedLiteral)) {
          if (constantExpr->toConstantVector(pool)->equalValueAt(
                  constantLiteral->toConstantVector(pool).get(), 0, 0)) {
            return std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true);
          }
        } else {
          optimizedInputs.emplace_back(foldedLiteral);
        }
      }
      return std::make_shared<core::CallTypedExpr>(
          expr->type(), std::move(optimizedInputs), expr->name());
    }
  }
  return expr;
}
} // namespace facebook::velox::expression
