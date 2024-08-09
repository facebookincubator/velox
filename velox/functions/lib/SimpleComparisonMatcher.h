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
#pragma once

#include "velox/core/Expressions.h"
#include "velox/vector/ConstantVector.h"

namespace facebook::velox::functions {

struct SimpleComparison {
  core::TypedExprPtr expr;
  bool isLessThen;
};

class Matcher {
 public:
  virtual bool match(const core::TypedExprPtr& expr) = 0;

  virtual ~Matcher() = default;

  static bool allMatch(
      const std::vector<core::TypedExprPtr>& exprs,
      std::vector<std::shared_ptr<Matcher>>& matchers) {
    for (auto i = 0; i < exprs.size(); ++i) {
      if (!matchers[i]->match(exprs[i])) {
        return false;
      }
    }
    return true;
  }
};

using MatcherPtr = std::shared_ptr<Matcher>;

class IfMatcher : public Matcher {
 public:
  explicit IfMatcher(std::vector<MatcherPtr> inputMatchers)
      : inputMatchers_{std::move(inputMatchers)} {
    VELOX_CHECK_EQ(3, inputMatchers_.size());
  }

  bool match(const core::TypedExprPtr& expr) override {
    if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
      if (call->name() == "if" && allMatch(call->inputs(), inputMatchers_)) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<MatcherPtr> inputMatchers_;
};

using IfMatcherPtr = std::shared_ptr<IfMatcher>;

class ComparisonMatcher : public Matcher {
 public:
  ComparisonMatcher(
      const std::string& prefix,
      std::vector<MatcherPtr> inputMatchers,
      std::string* op)
      : prefix_{prefix}, inputMatchers_{std::move(inputMatchers)}, op_{op} {
    VELOX_CHECK_EQ(2, inputMatchers_.size());
  }

  // Checks if the given name specifies a comparison expression. Can be
  // overriden to use different function names for Spark.
  virtual bool exprNameMatch(const std::string& name) {
    return name == prefix_ + "eq" || name == prefix_ + "lt" ||
        name == prefix_ + "gt";
  }

  bool match(const core::TypedExprPtr& expr) override {
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

 protected:
  const std::string prefix_;

 private:
  std::vector<MatcherPtr> inputMatchers_;
  std::string* op_;
};

using ComparisonMatcherPtr = std::shared_ptr<ComparisonMatcher>;

class AnySingleInputMatcher : public Matcher {
 public:
  AnySingleInputMatcher(
      core::TypedExprPtr* expr,
      core::FieldAccessTypedExprPtr* input)
      : expr_{expr}, input_{input} {}

  bool match(const core::TypedExprPtr& expr) override {
    // Check if 'expr' depends on a single column.
    std::unordered_set<core::FieldAccessTypedExprPtr> inputs;
    collectInputs(expr, inputs);

    if (inputs.size() == 1) {
      *expr_ = expr;
      *input_ = *inputs.begin();
      return true;
    }

    return false;
  }

 private:
  static void collectInputs(
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

  core::TypedExprPtr* const expr_;
  core::FieldAccessTypedExprPtr* const input_;
};

/// Matches constant expression that represents values 1, 0, or -1 of type
/// BIGINT.
class ComparisonConstantMatcher : public Matcher {
 public:
  explicit ComparisonConstantMatcher(int64_t* value) : value_{value} {}

  bool match(const core::TypedExprPtr& expr) override {
    if (auto constant = asConstant(expr.get())) {
      auto v = constant.value();
      if (v == 0 || v == 1 || v == -1) {
        *value_ = v;
        return true;
      }
    }
    return false;
  }

 private:
  static std::optional<int64_t> asConstant(const core::ITypedExpr* expr) {
    if (auto constant = dynamic_cast<const core::ConstantTypedExpr*>(expr)) {
      if (constant->hasValueVector()) {
        auto constantVector =
            constant->valueVector()->as<SimpleVector<int64_t>>();
        if (!constantVector->isNullAt(0)) {
          return constantVector->valueAt(0);
        }
      } else {
        if (!constant->value().isNull()) {
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

  int64_t* const value_;
};

using ComparisonConstantMatcherPtr = std::shared_ptr<ComparisonConstantMatcher>;

class SimpleComparisonChecker {
 protected:
  MatcherPtr ifelse(
      const MatcherPtr& condition,
      const MatcherPtr& thenClause,
      const MatcherPtr& elseClause) {
    return std::make_shared<IfMatcher>(
        std::vector<MatcherPtr>{condition, thenClause, elseClause});
  }

  MatcherPtr anySingleInput(
      core::TypedExprPtr* expr,
      core::FieldAccessTypedExprPtr* input) {
    return std::make_shared<AnySingleInputMatcher>(expr, input);
  }

  MatcherPtr comparisonConstant(int64_t* value) {
    return std::make_shared<ComparisonConstantMatcher>(value);
  }

  std::string invert(const std::string& prefix, const std::string& op) {
    return op == ltName(prefix) ? gtName(prefix) : ltName(prefix);
  }

  // Returns true for a < b -> -1.
  bool isLessThen(
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

  virtual MatcherPtr comparison(
      const std::string& prefix,
      const MatcherPtr& left,
      const MatcherPtr& right,
      std::string* op) {
    return std::make_shared<ComparisonMatcher>(
        prefix, std::vector<MatcherPtr>{left, right}, op);
  }

  virtual std::string eqName(const std::string& prefix) {
    return prefix + "eq";
  }

  virtual std::string ltName(const std::string& prefix) {
    return prefix + "lt";
  }

  virtual std::string gtName(const std::string& prefix) {
    return prefix + "gt";
  }

 public:
  virtual ~SimpleComparisonChecker() = default;

  /// Given a lambda expression, checks it if represents a simple comparator and
  /// returns the summary of the same.
  ///
  /// For example, identifies
  ///     (x, y) -> if(length(x) < length(y), -1, if(length(x) > length(y), 1,
  ///     0))
  /// expression as a "less than" comparison over length(x).
  ///
  /// Recognizes different variations of this expression, e.g.
  ///
  ///     (x, y) -> if(expr(x) = expr(y), 0, if(expr(x) < expr(y), -1, 1))
  ///     (x, y) -> if(expr(x) = expr(y), 0, if(expr(y) > expr(x), -1, 1))
  ///
  /// Returns std::nullopt if expression is not recognized as a simple
  /// comparator.
  ///
  /// Can be used to re-write generic lambda expressions passed to array_sort
  /// into simpler ones that can be evaluated more efficiently.
  std::optional<SimpleComparison> isSimpleComparison(
      const std::string& prefix,
      const core::LambdaTypedExpr& expr);
};

} // namespace facebook::velox::functions
