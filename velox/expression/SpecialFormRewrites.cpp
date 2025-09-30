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
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/ExprUtils.h"
#include "velox/expression/SpecialFormRewrites.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::expression {

core::TypedExprPtr rewriteConjunctExpression(
    const core::TypedExprPtr& expr) {
  if (!utils::isCall(expr, kAnd) && !utils::isCall(expr, kOr)) {
    return nullptr;
  }

  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  const bool isAnd = (callExpr->name() == kAnd) ? true : false;
  // If all inputs are AND or OR then we can flatten the inputs into a vector
  // before further optimizing.
  auto canFlatten = utils::allInputTypesEquivalent(expr);
  std::vector<core::TypedExprPtr> flat;
  if (canFlatten) {
    utils::flattenInput(expr, callExpr->name(), flat);
  }

  const auto& inputsToOptimize = canFlatten ? flat : expr->inputs();
  bool allInputsConstant = true;
  bool hasNullInput = false;
  std::vector<core::TypedExprPtr> optimizedInputs;
  core::TypedExprPtr nullInput = nullptr;
  for (const auto& inputExpr : inputsToOptimize) {
    if (utils::isCall(inputExpr, kFail)) {
      return inputExpr;
    }

    const auto result = utils::evalExprAsConstant(inputExpr);
    switch (result) {
      case utils::ConstantEvalResult::kNull:
        if (!hasNullInput) {
          hasNullInput = true;
          nullInput = inputExpr;
        }
        break;
      case utils::ConstantEvalResult::kTrue:
        if (!isAnd) {
          // OR (.., true, ..) -> true
          return inputExpr;
        }
        break;
      case utils::ConstantEvalResult::kFalse:
        if (isAnd) {
          // AND (.., false, ..) -> false
          return inputExpr;
        }
        break;
      case utils::ConstantEvalResult::kNotConstant:
        allInputsConstant = false;
        optimizedInputs.push_back(inputExpr);
        break;
      default:
        return expr;
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
      expr->type(), std::move(optimizedInputs), callExpr->name());
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
    const core::TypedExprPtr& expr,
    std::set<core::TypedExprPtr, TypedExprComparator>& deduplicatedInputSet,
    std::vector<core::TypedExprPtr>& deduplicatedInputList) {
  if (utils::isCall(expr, kFail)) {
    return expr;
  }

  // First non-NULL constant input to COALESCE returns non-NULL value.
  const auto result = utils::evalExprAsConstant(expr);
  switch (result) {
    case utils::ConstantEvalResult::kNull:
      break;
    case utils::ConstantEvalResult::kNonBoolConstant:
    case utils::ConstantEvalResult::kTrue:
    case utils::ConstantEvalResult::kFalse:
      return expr;
    case utils::ConstantEvalResult::kNotConstant: {
      if (deduplicatedInputSet.find(expr) == deduplicatedInputSet.end()) {
        deduplicatedInputSet.insert(expr);
        deduplicatedInputList.emplace_back(expr);
      }
      break;
    }
    default:
      VELOX_UNREACHABLE("Err");
  }

  return nullptr;
}

core::TypedExprPtr rewriteCoalesceExpression(const core::TypedExprPtr& expr) {
  if (!utils::isCall(expr, kCoalesce)) {
    return nullptr;
  }

  // Deduplicate inputs to COALESCE and remove NULL inputs, returning a list of
  // optimized inputs to COALESCE.
  std::set<core::TypedExprPtr, TypedExprComparator> deduplicatedInputSet;
  std::vector<core::TypedExprPtr> deduplicatedInputList;
  auto canFlatten = utils::allInputTypesEquivalent(expr);
  std::vector<core::TypedExprPtr> flat;
  if (canFlatten) {
    utils::flattenInput(expr, kCoalesce, flat);
  }

  const auto& inputsToOptimize = canFlatten ? flat : expr->inputs();
  for (const auto& exprInput : inputsToOptimize) {
    // Once a constant input is seen, subsequent inputs to COALESCE expression
    // can be ignored.
    if (auto optimized = addCoalesceArgument(
            exprInput,
            deduplicatedInputSet,
            deduplicatedInputList)) {
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
      expr->type(), std::move(deduplicatedInputList), kCoalesce);
}

// Input expression should be of form: IF(condition, then, else).
core::TypedExprPtr rewriteIfExpression(const core::TypedExprPtr& expr) {
  if (!utils::isCall(expr, kIf) || expr->inputs().size() != 3) {
    return nullptr;
  }

  const auto& foldedCondition = expr->inputs().at(0);
  // The folded expression could be the fail function. In this case,
  // we don't want to futher analyze the expression and instead return the fail
  // function expression.
  if (utils::isCall(foldedCondition, kFail)) {
    return foldedCondition;
  }
  const auto result = utils::evalExprAsConstant(foldedCondition);
  switch (result) {
    case utils::ConstantEvalResult::kNull:
      [[fallthrough]];
    case utils::ConstantEvalResult::kTrue:
      return expr->inputs().at(1);
    case utils::ConstantEvalResult::kFalse:
      return expr->inputs().at(2);
    case utils::ConstantEvalResult::kNotConstant:
      [[fallthrough]];
    default:
      return expr;
  }
  return expr;
}

// Input expression should be of form: SWITCH(condition1, value1, condition2,
//   value2, ...., defaultValue).
core::TypedExprPtr rewriteSwitchExpression(const core::TypedExprPtr& input) {
  if (!utils::isCall(input, kSwitch)) {
    return nullptr;
  }

  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  const auto& inputs = expr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;
  // If a case evaluates to true, it will be the new else clause.
  bool hasOptimizedElseValue = false;

  for (auto i = 0; i < numInputs - 1; i += 2) {
    const auto& foldedCondition = inputs.at(i);
    if (utils::isCall(foldedCondition, kFail)) {
      return foldedCondition;
    }
    const auto& foldedValue = expr->inputs().at(i + 1);
    if (utils::isCall(foldedValue, kFail)) {
      return foldedValue;
    }

    const auto result = utils::evalExprAsConstant(foldedCondition);
    switch (result) {
      case utils::ConstantEvalResult::kNull:
        continue;
      case utils::ConstantEvalResult::kTrue:
        if (optimizedInputs.empty()) {
          return foldedValue;
        }
        hasOptimizedElseValue = true;
        optimizedInputs.emplace_back(foldedValue);
        break;
      case utils::ConstantEvalResult::kFalse:
        continue;
      case utils::ConstantEvalResult::kNotConstant:
        optimizedInputs.emplace_back(foldedCondition);
        optimizedInputs.emplace_back(foldedValue);
        continue;
      default:
        return expr;
    }
    break;
  }

  if (!hasOptimizedElseValue) {
    const auto elseValue = inputs.at(numInputs - 1);
    if (utils::isCall(elseValue, kFail)) {
      return elseValue;
    }
    if (optimizedInputs.empty()) {
      return elseValue;
    }
    optimizedInputs.emplace_back(elseValue);
  }
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), expr->name());
}

// When all input literals in IN-list are constant, the expression is expected
// to be of type IN(value, arrayVector<literal1, ....., literalN>). When any
// input literal in IN-list is non-constant, the expression is expected to be
// of type IN(value, literal1, ....., literalN). The latter case is optimized
// by this function and the former is handled during constant folding. This
// rewrite also prunes constants from IN-list that will not match the value.
core::TypedExprPtr rewriteInExpression(const core::TypedExprPtr& expr) {
  if (!utils::isCall(expr, kIn) || expr->inputs().size() < 2) {
    return nullptr;
  }

  const auto& valueExpr = expr->inputs().at(0);
  if (utils::isCall(valueExpr, kFail)) {
    return valueExpr;
  }

  if (valueExpr->isConstantKind()) {
    const auto& inList = expr->inputs().at(1);
    const auto inListAsConstExpr =
        inList->asUnchecked<core::ConstantTypedExpr>();
    bool canApplyRewrite = false;
    if (inList->isConstantKind()) {
      const auto constVector = inListAsConstExpr->valueVector();
      if (constVector == nullptr) {
        canApplyRewrite = true;
      } else {
        const auto isArrayVector = constVector->type()->isArray();
        canApplyRewrite = !(isArrayVector);
      }
    }

    if (canApplyRewrite) {
      const auto constantValueExpr =
          valueExpr->asUnchecked<core::ConstantTypedExpr>();
      const auto& inputs = expr->inputs();
      const auto numInputs = inputs.size();
      std::vector<core::TypedExprPtr> optimizedInputs;
      optimizedInputs.emplace_back(valueExpr);

      for (auto i = 1; i < numInputs; i++) {
        const auto& foldedLiteral = inputs.at(i);
        if (utils::isCall(foldedLiteral, kFail)) {
          return foldedLiteral;
        }
        if (foldedLiteral->isConstantKind()) {
          auto constantLiteral =
              foldedLiteral->asUnchecked<core::ConstantTypedExpr>();
          if (constantLiteral->equals(*constantValueExpr)) {
            return std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true);
          }
        } else {
          optimizedInputs.emplace_back(foldedLiteral);
        }
      }
      return std::make_shared<core::CallTypedExpr>(
          expr->type(), std::move(optimizedInputs), kIn);
    }
  }
  return expr;
}
} // namespace facebook::velox::expression
