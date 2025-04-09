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
#include "velox/expression/ExpressionOptimizer.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprCompiler.h"

namespace facebook::velox::expression {
namespace {

const std::string_view kAnd = "and";
const std::string_view kOr = "or";

bool isCall(const core::TypedExprPtr& expr, const std::string& name) {
  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    return call->name() == name;
  }
  return false;
}

/// Comparator for core::TypedExprPtr; used to deduplicate arguments to
/// COALESCE special form expression.
struct TypedExprComparator {
  bool operator()(const core::TypedExprPtr& a, const core::TypedExprPtr& b)
      const {
    return a->hash() < b->hash();
  }
};

core::TypedExprPtr tryConstantFold(
    const core::TypedExprPtr& expr,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  try {
    auto result = exec::evaluateConstantExpression(expr, config, pool);
    return std::make_shared<core::ConstantTypedExpr>(result);
  } catch (VeloxUserError& e) {
    const auto error = std::string(e.what());
    // References to variables will not be resolved so this error is expected.
    if (error.find("Field not found") != std::string::npos) {
      return expr;
    } else {
      return std::make_shared<core::CallTypedExpr>(
          VARCHAR(),
          std::vector<core::TypedExprPtr>(
              {std::make_shared<core::ConstantTypedExpr>(VARCHAR(), e.what())}),
          "fail");
    }
  }
}

// Input expression should be of form: IF(condition, valueIfTrue, valueIfFalse).
core::TypedExprPtr optimizeIfExpression(
    const core::TypedExprPtr& input,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  if (expr == nullptr || expr->name() != "if" || expr->inputs().size() != 3) {
    return nullptr;
  }

  auto condition = expr->inputs().at(0);
  auto folded = tryConstantFold(condition, config, pool);
  if (auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(folded)) {
    if (auto constVector = constantExpr->toConstantVector(pool)) {
      if (constVector->isNullAt(0) ||
          constVector->as<ConstantVector<bool>>()->valueAt(0)) {
        return expr->inputs().at(1);
      }
      return expr->inputs().at(2);
    }
  }
  return expr;
}

// Input expression should be of form: SWITCH(condition1, value1, condition2,
//   value2, ...., defaultValue).
core::TypedExprPtr optimizeSwitchExpression(
    const core::TypedExprPtr& input,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  if (expr == nullptr || expr->name() != "switch") {
    return nullptr;
  }

  const auto inputs = expr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;
  // If a case evaluates to true, it will be the new else clause.
  bool hasOptimizedElseValue = false;
  for (auto i = 0; i < numInputs - 1; i += 2) {
    const auto condition = inputs.at(i);
    const auto foldedCondition = tryConstantFold(condition, config, pool);
    const auto value = expr->inputs().at(i + 1);
    const auto foldedValue = tryConstantFold(value, config, pool);
    if (auto constantExpr =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(foldedCondition)) {
      if (auto constVector = constantExpr->toConstantVector(pool)) {
        if (!constVector->isNullAt(0) && constVector->as<ConstantVector<bool>>()->valueAt(0)) {
          if (optimizedInputs.empty()) {
            return foldedValue;
          } else {
            hasOptimizedElseValue = true;
            optimizedInputs.emplace_back(foldedValue);
            break;
          }
        }
      }
    } else {
      optimizedInputs.emplace_back(foldedCondition);
      optimizedInputs.emplace_back(foldedValue);
    }
  }

  if (!hasOptimizedElseValue) {
    const auto foldedElseValue = tryConstantFold(inputs.at(numInputs - 1), config, pool);
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
core::TypedExprPtr optimizeInExpression(
    const core::TypedExprPtr& input,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  if (expr == nullptr || expr->name() != "in" || expr->inputs().size() < 2) {
    return nullptr;
  }

  const auto value = expr->inputs().at(0);
  const auto foldedValue = tryConstantFold(value, config, pool);
  if (auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(foldedValue)) {
    const auto inList = expr->inputs().at(1);
    if (std::dynamic_pointer_cast<const core::ConstantTypedExpr>(inList) == nullptr) {
      const auto inputs = expr->inputs();
      const auto numInputs = inputs.size();
      std::vector<core::TypedExprPtr> optimizedInputs;
      optimizedInputs.emplace_back(foldedValue);

      for (auto i = 1; i < numInputs; i++) {
        const auto literal = inputs.at(i);
        const auto foldedLiteral = tryConstantFold(literal, config, pool);
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

core::TypedExprPtr optimizeConjunctExpression(
    const core::TypedExprPtr& input,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(input);
  if (expr == nullptr || (expr->name() != kAnd && expr->name() != kOr)) {
    return nullptr;
  }
  const bool isAnd = (expr->name() == kAnd) ? true : false;

  // If all inputs are AND or OR then we can flatten the inputs into a vector
  // before further optmizing.
  auto canFlatten = allInputTypesEquivalent(expr);
  std::vector<core::TypedExprPtr> flat;
  if (canFlatten) {
    flattenInput(input, expr->name(), flat);
  }

  auto inputsToOptimize = canFlatten ? flat : expr->inputs();
  bool allInputsConstant = true;
  bool hasNullInput = false;
  std::vector<core::TypedExprPtr> optimizedInputs;
  core::TypedExprPtr nullInput = nullptr;
  for (const auto& inputExpr : inputsToOptimize) {
    auto folded = tryConstantFold(inputExpr, config, pool);
    if (auto constantExpr =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(folded)) {
      auto constantVector = constantExpr->toConstantVector(pool);
      if (!constantVector->isNullAt(0)) {
        if (isAnd) {
          // AND (.., false, ..) -> false
          if (!constantVector->as<ConstantVector<bool>>()->valueAt(0)) {
            return constantExpr;
          }
        } else {
          // OR (.., true, ..) -> true
          if (constantVector->as<ConstantVector<bool>>()->valueAt(0)) {
            return constantExpr;
          }
        }
      } else if (!hasNullInput) {
        hasNullInput = true;
        nullInput = inputExpr;
      } // else do nothing because we encountered a NULL that we already dealt
        // with.
    } else {
      allInputsConstant = false;
      optimizedInputs.push_back(inputExpr);
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

core::TypedExprPtr addCoalesceArgument(
    const core::TypedExprPtr& input,
    std::set<core::TypedExprPtr, TypedExprComparator>& optimizedTypedExprs,
    std::vector<core::TypedExprPtr>& deduplicatedInputs,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  auto folded = tryConstantFold(input, config, pool);
  // First non-NULL constant input to COALESCE returns non-NULL value.
  if (auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(folded)) {
    auto constantVector = constantExpr->toConstantVector(pool);
    if (!constantVector->isNullAt(0)) {
      if (optimizedTypedExprs.find(folded) == optimizedTypedExprs.end()) {
        optimizedTypedExprs.insert(folded);
        deduplicatedInputs.push_back(input);
      }
      return input;
    }
  } else if (optimizedTypedExprs.find(folded) == optimizedTypedExprs.end()) {
    optimizedTypedExprs.insert(folded);
    deduplicatedInputs.push_back(input);
  }

  return nullptr;
}

core::TypedExprPtr optimizeCoalesceSpecialFormImpl(
    const core::CallTypedExprPtr& expr,
    std::set<core::TypedExprPtr, TypedExprComparator>& inputTypedExprSet,
    std::vector<core::TypedExprPtr>& deduplicatedInputs,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  // Once a constant input is seen, subsequent inputs to the COALESCE expression
  // can be ignored.
  for (const auto& input : expr->inputs()) {
    if (const auto call =
            std::dynamic_pointer_cast<const core::CallTypedExpr>(input)) {
      if (call->name() == "coalesce") {
        // If the argument is a COALESCE expression, the arguments of inner
        // COALESCE can be combined with the arguments of outer COALESCE
        // expression. If the inner COALESCE has a constant expression, return.
        if (auto optimizedCoalesceSubExpr = optimizeCoalesceSpecialFormImpl(
                call, inputTypedExprSet, deduplicatedInputs, config, pool)) {
          return optimizedCoalesceSubExpr;
        }
      } else if (
          auto optimized = addCoalesceArgument(
              input, inputTypedExprSet, deduplicatedInputs, config, pool)) {
        return optimized;
      }
    } else if (
        auto optimized = addCoalesceArgument(
            input, inputTypedExprSet, deduplicatedInputs, config, pool)) {
      return optimized;
    }
  }
  // Return null if COALESCE has no constant input.
  return nullptr;
}

core::TypedExprPtr optimizeCoalesceExpression(
    const core::TypedExprPtr& expr,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr);
  if (call == nullptr || call->name() != "coalesce") {
    return nullptr;
  }
  // Deduplicate inputs to COALESCE and remove NULL inputs, returning a list of
  // optimized inputs to COALESCE.
  std::set<core::TypedExprPtr, TypedExprComparator> inputTypedExprSet;
  std::vector<core::TypedExprPtr> deduplicatedInputs;
  optimizeCoalesceSpecialFormImpl(
      call, inputTypedExprSet, deduplicatedInputs, config, pool);

  // Return NULL if all inputs to COALESCE are NULL. If there is a single input
  // to COALESCE after optimization, return this expression. Otherwise, return
  // COALESCE expression with optimized inputs.
  if (deduplicatedInputs.empty()) {
    return call->inputs().front();
  } else if (deduplicatedInputs.size() == 1) {
    return deduplicatedInputs.front();
  }
  return std::make_shared<core::CallTypedExpr>(
      call->type(), std::move(deduplicatedInputs), call->name());
}

} // namespace

// Utility method to check eligibility for flattening.
bool allInputTypesEquivalent(const core::TypedExprPtr& expr) {
  const auto& inputs = expr->inputs();
  for (int i = 1; i < inputs.size(); i++) {
    if (!inputs[0]->type()->equivalent(*inputs[i]->type())) {
      return false;
    }
  }
  return true;
}

// Recursively flattens nested ANDs, ORs or eligible callable expressions into a
// vector of their inputs. Recursive flattening ceases exploring an input branch
// if it encounters either an expression different from 'flattenCall' or its
// inputs are not the same type.
// Examples:
// flattenCall: AND
// in: a AND (b AND (c AND d))
// out: [a, b, c, d]
//
// flattenCall: OR
// in: (a OR b) OR (c OR d)
// out: [a, b, c, d]
//
// flattenCall: concat
// in: (array1, concat(array2, concat(array2, intVal))
// out: [array1, array2, concat(array2, intVal)]
void flattenInput(
    const core::TypedExprPtr& input,
    const std::string& flattenCall,
    std::vector<core::TypedExprPtr>& flat) {
  if (isCall(input, flattenCall) && allInputTypesEquivalent(input)) {
    for (auto& child : input->inputs()) {
      flattenInput(child, flattenCall, flat);
    }
  } else {
    flat.emplace_back(input);
  }
}

void registerExpressionOptimizations() {
  exec::registerExpressionRewrite([&](const core::TypedExprPtr& expr,
                                      const core::QueryConfig& config,
                                      memory::MemoryPool* pool) {
    return optimizeCoalesceExpression(expr, config, pool);
  });
  exec::registerExpressionRewrite([&](const core::TypedExprPtr& expr,
                                      const core::QueryConfig& config,
                                      memory::MemoryPool* pool) {
    return optimizeIfExpression(expr, config, pool);
  });
  exec::registerExpressionRewrite([&](const core::TypedExprPtr& expr,
                                      const core::QueryConfig& config,
                                      memory::MemoryPool* pool) {
    return optimizeSwitchExpression(expr, config, pool);
  });
  exec::registerExpressionRewrite([&](const core::TypedExprPtr& expr,
                                      const core::QueryConfig& config,
                                      memory::MemoryPool* pool) {
    return optimizeInExpression(expr, config, pool);
  });
  exec::registerExpressionRewrite([&](const core::TypedExprPtr& expr,
                                      const core::QueryConfig& config,
                                      memory::MemoryPool* pool) {
    return optimizeConjunctExpression(expr, config, pool);
  });
}

core::TypedExprPtr constantFold(
    const core::TypedExprPtr& expr,
    const core::QueryConfig& config,
    memory::MemoryPool* pool) {
  core::TypedExprPtr result;
  std::vector<core::TypedExprPtr> foldedInputs;
  for (auto& input : expr->inputs()) {
    foldedInputs.push_back(constantFold(input, config, pool));
  }

  bool isField = false;
  if (auto callExpr =
          std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    result = std::make_shared<core::CallTypedExpr>(
        callExpr->type(), foldedInputs, callExpr->name());
  } else if (
      auto castExpr =
          std::dynamic_pointer_cast<const core::CastTypedExpr>(expr)) {
    VELOX_CHECK(!foldedInputs.empty());
    if (foldedInputs.at(0)->type() == expr->type()) {
      result = foldedInputs.at(0);
    } else {
      result = std::make_shared<core::CastTypedExpr>(
          expr->type(), foldedInputs, castExpr->nullOnFailure());
    }
  } else if (
      auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr)) {
    return constantExpr;
  } else if (
      auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    isField = true;
    result = field;
  } else if (
      auto concatExpr =
          std::dynamic_pointer_cast<const core::ConcatTypedExpr>(expr)) {
    result = concatExpr;
  } else {
    result = expr;
  }

  auto folded = !isField ? tryConstantFold(result, config, pool) : result;
  return folded;
}

} // namespace facebook::velox::expression
