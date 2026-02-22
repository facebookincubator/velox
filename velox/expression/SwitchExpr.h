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
#pragma once

#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/SpecialForm.h"

namespace facebook::velox::exec {

/// CASE expression:
///
/// case
///  when condition then result
///  [when ...]
///  [else result]
/// end
///
/// Evaluates each boolean condition from left to right until one is true and
/// returns the matching result. If no conditions are true, the result from the
/// ELSE clause is returned if it exists, otherwise null is returned.
///
/// IF expression can be represented as a CASE expression with a single
/// condition.
class SwitchExpr : public SpecialForm {
 public:
  /// Inputs are concatenated conditions and results with an optional "else" at
  /// the end, e.g. {condition1, result1, condition2, result2,..else}
  SwitchExpr(
      TypePtr type,
      const std::vector<ExprPtr>& inputs,
      bool inputsSupportFlatNoNullsFastPath);

  void evalSpecialForm(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override;

  bool isConditional() const override {
    return true;
  }

  void clearCache() override {
    Expr::clearCache();
    tempValues_.reset();
  }

 private:
  void computePropagatesNulls() override;

  const size_t numCases_;
  const bool hasElseClause_;
  BufferPtr tempValues_;
};

class SwitchCallToSpecialForm : public FunctionCallToSpecialForm {
 public:
  /// Returns the type of the 2nd argument. Throws if types of even arguments
  /// (conditions) are not BOOLEAN or if types of odd arguments (then and else
  /// branches) don't match.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Supports coercions for odd arguments (then and else branches).
  TypePtr resolveTypeWithCorsions(
      const std::vector<TypePtr>& argTypes,
      std::vector<TypePtr>& coercions) override;

  ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;
};

class IfCallToSpecialForm : public SwitchCallToSpecialForm {
 public:
  /// Returns the type of the 2nd argument. Throws if number of arguments is not
  /// 3 or types, type of 1st argument (condition) is not BOOLEAN, or types of
  /// 2nd (then) and 3rd (else) arguments don't match.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Supports coercions for 2nd and 3rd arguments (then and else branches).
  TypePtr resolveTypeWithCorsions(
      const std::vector<TypePtr>& argTypes,
      std::vector<TypePtr>& coercions) override;
};

} // namespace facebook::velox::exec
