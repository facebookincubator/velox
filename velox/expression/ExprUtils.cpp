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

#include "velox/expression/ExprUtils.h"

namespace facebook::velox::expression::utils {

using namespace facebook::velox;

bool isCall(const core::TypedExprPtr& expr, const std::string& name) {
  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    return call->name() == name;
  }
  return false;
}

core::TypedExprPtr tryConstantFold(
    const core::TypedExprPtr& expr,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  try {
    auto results =
        exec::tryEvaluateConstantExpression(expr, pool, queryCtx, false);
    if (nullptr != results) {
      return std::make_shared<core::ConstantTypedExpr>(results);
    } else {
      // Return the expression unevaluated.
      return expr;
    }

  } catch (VeloxUserError& e) {
    const auto error = std::string(e.what());

    return std::make_shared<core::CallTypedExpr>(
        UNKNOWN(),
        std::vector<core::TypedExprPtr>(
            {std::make_shared<core::ConstantTypedExpr>(VARCHAR(), e.what())}),
        kFail);
  }
}

core::TypedExprPtr constantFold(
    const core::TypedExprPtr& expr,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  core::TypedExprPtr result;
  std::vector<core::TypedExprPtr> foldedInputs;
  for (auto& input : expr->inputs()) {
    foldedInputs.push_back(constantFold(input, queryCtx, pool));
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

  auto folded = !isField ? tryConstantFold(result, queryCtx, pool) : result;
  return folded;
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

} // namespace facebook::velox::expression::utils
