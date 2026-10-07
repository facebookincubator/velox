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

#include "velox/functions/lib/SimpleComparisonMatcher.h"
#include "velox/expression/ExprConstants.h"
#include "velox/functions/FunctionRegistry.h"

namespace facebook::velox::functions {
namespace {

bool isDeterministicSpecialForm(std::string_view name) {
  return name == expression::kAnd || name == expression::kOr ||
      name == expression::kSwitch || name == expression::kIf ||
      name == expression::kCoalesce || name == expression::kCast ||
      name == expression::kTryCast || name == expression::kTry ||
      name == expression::kRowConstructor || name == expression::kNullIf ||
      name == expression::kCase || name == expression::kIn ||
      name == expression::kNot || name == "array_constructor";
}

bool isDeterministicTransform(const core::TypedExprPtr& expr) {
  if (auto lambda =
          std::dynamic_pointer_cast<const core::LambdaTypedExpr>(expr)) {
    return isDeterministicTransform(lambda->body());
  }

  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    const auto deterministic = facebook::velox::isDeterministic(call->name());
    if ((!deterministic.has_value() &&
         !isDeterministicSpecialForm(call->name())) ||
        (deterministic.has_value() && !deterministic.value())) {
      return false;
    }
  }

  for (const auto& input : expr->inputs()) {
    if (!isDeterministicTransform(input)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool Matcher::allMatch(
    const std::vector<core::TypedExprPtr>& exprs,
    std::vector<std::shared_ptr<Matcher>>& matchers) {
  if (exprs.size() != matchers.size()) {
    return false;
  }

  for (auto i = 0; i < exprs.size(); ++i) {
    if (!matchers[i]->match(exprs[i])) {
      return false;
    }
  }
  return true;
}

bool IfMatcher::match(const core::TypedExprPtr& expr) {
  if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    if (call->name() == expression::kIf &&
        allMatch(call->inputs(), inputMatchers_)) {
      return true;
    }
  }
  return false;
}

bool ComparisonMatcher::match(const core::TypedExprPtr& expr) {
  if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    const auto& name = call->name();
    if (exprNameMatch(name)) {
      if (allMatch(call->inputs(), inputMatchers_)) {
        *op_ = name;
        return true;
      }
    }
  }
  return false;
}

bool AnySingleInputMatcher::match(const core::TypedExprPtr& expr) {
  std::unordered_set<core::FieldAccessTypedExprPtr> inputs;
  collectInputs(expr, inputs);

  if (inputs.size() == 1) {
    *expr_ = expr;
    *input_ = *inputs.begin();
    return true;
  }

  return false;
}

void AnySingleInputMatcher::collectInputs(
    const core::TypedExprPtr& expr,
    std::unordered_set<core::FieldAccessTypedExprPtr>& inputs) {
  if (auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    if (field->isInputColumn()) {
      inputs.insert(field);
      return;
    }
  }

  for (const auto& input : expr->inputs()) {
    collectInputs(input, inputs);
  }
}

bool AnySingleLambdaInputMatcher::match(const core::TypedExprPtr& expr) {
  // Captured fields are allowed, but the expression must depend on exactly one
  // of the lambda arguments.
  std::unordered_set<core::FieldAccessTypedExprPtr> inputs;
  if (!collectInputs(expr, inputs)) {
    return false;
  }

  core::FieldAccessTypedExprPtr lambdaInput;
  for (const auto& input : inputs) {
    if (!lambdaInputs_.contains(input->name())) {
      continue;
    }
    if (lambdaInput != nullptr && lambdaInput->name() != input->name()) {
      return false;
    }
    lambdaInput = input;
  }

  if (lambdaInput == nullptr) {
    return false;
  }

  *expr_ = expr;
  *input_ = std::move(lambdaInput);
  return true;
}

bool AnySingleLambdaInputMatcher::collectInputs(
    const core::TypedExprPtr& expr,
    std::unordered_set<core::FieldAccessTypedExprPtr>& inputs) const {
  if (auto lambda =
          std::dynamic_pointer_cast<const core::LambdaTypedExpr>(expr)) {
    for (const auto& name : lambda->signature()->names()) {
      if (lambdaInputs_.contains(name)) {
        return false;
      }
    }
    return collectInputs(lambda->body(), inputs);
  }

  if (auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    if (field->isInputColumn()) {
      inputs.insert(field);
      return true;
    }
  }

  for (const auto& input : expr->inputs()) {
    if (!collectInputs(input, inputs)) {
      return false;
    }
  }
  return true;
}

bool ComparisonConstantMatcher::match(const core::TypedExprPtr& expr) {
  if (auto constant = asConstant(expr.get())) {
    *value_ = constant.value();
    return true;
  }
  return false;
}

std::optional<int64_t> ComparisonConstantMatcher::asConstant(
    const core::ITypedExpr* expr) {
  if (auto constant = dynamic_cast<const core::ConstantTypedExpr*>(expr)) {
    if (!constant->isNull()) {
      if (constant->hasValueVector()) {
        const auto& vector = constant->valueVector();
        if (constant->type()->isBigint()) {
          return vector->as<SimpleVector<int64_t>>()->valueAt(0);
        } else if (constant->type()->isInteger()) {
          return vector->as<SimpleVector<int32_t>>()->valueAt(0);
        }
      } else {
        if (constant->value().kind() == TypeKind::BIGINT) {
          return constant->value().value<int64_t>();
        }

        if (constant->value().kind() == TypeKind::INTEGER) {
          return constant->value().value<int32_t>();
        }
      }
    }
  }

  return std::nullopt;
}

bool SimpleComparisonChecker::isLessThen(
    const std::string& prefix,
    const std::string& operation,
    const core::FieldAccessTypedExprPtr& left,
    int64_t result,
    const std::string& inputLeft) {
  std::string op =
      (left->name() == inputLeft) ? operation : invert(prefix, operation);

  if (op == ltName(prefix)) {
    return result < 0;
  }

  return result > 0;
}

std::optional<SimpleComparison> SimpleComparisonChecker::isSimpleComparison(
    const std::string& prefix,
    const core::LambdaTypedExpr& expr) {
  return isSimpleComparison(prefix, expr, false);
}

std::optional<SimpleComparison> SimpleComparisonChecker::isSimpleComparison(
    const std::string& prefix,
    const core::LambdaTypedExpr& expr,
    bool supportsArbitraryComparatorResults) {
  // First, check the shape of the expression.
  // if (x(a) < y(b), c1, if (u(c) > v(d), c2, c3))
  core::FieldAccessTypedExprPtr a, b, c, d;
  core::TypedExprPtr x, y, u, v;
  std::string op1, op2;
  int64_t c1{0};
  int64_t c2{0};
  int64_t c3{0};

  const auto left = expr.signature()->nameOf(0);
  const auto right = expr.signature()->nameOf(1);
  const std::unordered_set<std::string> lambdaInputs{left, right};

  auto matcher = ifelse(
      comparison(
          prefix,
          anySingleInput(&x, &a, lambdaInputs),
          anySingleInput(&y, &b, lambdaInputs),
          &op1),
      comparisonConstant(&c1),
      ifelse(
          comparison(
              prefix,
              anySingleInput(&u, &c, lambdaInputs),
              anySingleInput(&v, &d, lambdaInputs),
              &op2),
          comparisonConstant(&c2),
          comparisonConstant(&c3)));

  if (!matcher->match(expr.body())) {
    return std::nullopt;
  }

  const auto usesBothLambdaArguments =
      [&](const core::FieldAccessTypedExprPtr& first,
          const core::FieldAccessTypedExprPtr& second) {
        return (first->name() == left && second->name() == right) ||
            (first->name() == right && second->name() == left);
      };

  if (left == right || !usesBothLambdaArguments(a, b) ||
      !usesBothLambdaArguments(c, d)) {
    return std::nullopt;
  }

  if (!supportsArbitraryComparatorResults) {
    const auto isNormalizedResult = [](int64_t result) {
      return result == -1 || result == 0 || result == 1;
    };
    if (!isNormalizedResult(c1) || !isNormalizedResult(c2) ||
        !isNormalizedResult(c3)) {
      return std::nullopt;
    }
  }

  // Verify that x, y, u, v are the same (except for input column).
  std::unordered_map<std::string, core::TypedExprPtr> inputMapping;
  inputMapping.emplace(
      a->name(),
      std::make_shared<core::FieldAccessTypedExpr>(a->type(), b->name()));
  const auto xRewritten = x->rewriteInputNames(inputMapping);
  if (!(*xRewritten == *y->rewriteInputNames(inputMapping) &&
        *xRewritten == *u->rewriteInputNames(inputMapping) &&
        *xRewritten == *v->rewriteInputNames(inputMapping))) {
    return std::nullopt;
  }

  const auto eq = eqName(prefix);
  const bool op1IsEquality = op1 == eq;
  const bool op2IsEquality = op2 == eq;
  const auto haveOppositeSigns = [](int64_t leftResult, int64_t rightResult) {
    return (leftResult < 0 && rightResult > 0) ||
        (leftResult > 0 && rightResult < 0);
  };

  if (op1IsEquality && op2IsEquality) {
    return std::nullopt;
  }

  const auto transform = a->name() == left ? x : y;
  if (!isDeterministicTransform(transform)) {
    return std::nullopt;
  }

  if (op1IsEquality) {
    // if (x(a) = y(b), 0,...)
    if (c1 != 0 || !haveOppositeSigns(c2, c3)) {
      return std::nullopt;
    }
    return {{transform, isLessThen(prefix, op2, c, c2, left)}};
  }

  if (op2IsEquality) {
    if (c2 != 0 || !haveOppositeSigns(c1, c3)) {
      return std::nullopt;
    }
    return {{transform, isLessThen(prefix, op1, a, c1, left)}};
  }

  if (c3 != 0 || !haveOppositeSigns(c1, c2)) {
    return std::nullopt;
  }

  // Make sure op1 and op2 are aligned.
  auto b1 = isLessThen(prefix, op1, a, c1, left);
  auto b2 = isLessThen(prefix, op2, c, c2, left);
  if (b1 != b2) {
    return std::nullopt;
  }

  return {{transform, b1}};
}

} // namespace facebook::velox::functions
