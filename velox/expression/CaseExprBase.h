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

#include <optional>

#include "velox/expression/SpecialForm.h"

namespace facebook::velox::exec {

/// Base class of SwitchExpr and CaseExpr. Evaluates a list of cases from left
/// to right. Each row gets the THEN clause of the first case whose condition is
/// true for that row. Rows where no condition is true get the ELSE clause if
/// there is one, otherwise null.
///
/// Inputs are an optional subject, followed by the WHEN and THEN clauses of
/// each case and an optional ELSE clause. Subclasses define how the condition
/// of a case is computed from its WHEN clause and the subject.
///
/// A row where the subject or a condition fails still gets a value when all
/// values the row can get are equal, like 'x AND false' returns false even if
/// 'x' fails. If the condition of a case fails, the row can get the THEN value
/// of that case or the value of the following cases. If the subject fails, the
/// row can get the THEN value of any case or the ELSE value. NULL equals NULL.
/// For example, 'if(1 / 0 = 5, c0, c0)' returns 'c0'. Otherwise, the error is
/// raised.
class CaseExprBase : public SpecialForm {
 public:
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

 protected:
  /// @param hasSubject True if inputs[0] is a subject, which is evaluated once
  /// per row and passed to evalCondition().
  CaseExprBase(
      SpecialFormKind kind,
      TypePtr type,
      const std::vector<ExprPtr>& inputs,
      const std::string& name,
      bool hasSubject,
      bool inputsSupportFlatNoNullsFastPath);

  /// Evaluates the condition of case 'index' on 'rows' into a boolean 'result'.
  /// 'subject' is the value of the subject, or nullptr if there is no subject.
  /// Records errors in 'context'.
  virtual void evalCondition(
      size_t index,
      const SelectivityVector& rows,
      const VectorPtr& subject,
      EvalCtx& context,
      VectorPtr& result) = 0;

  const ExprPtr& subject() const {
    VELOX_DCHECK(hasSubject_);
    return inputs_[0];
  }

  const ExprPtr& whenClause(size_t index) const {
    return inputs_[firstCaseInput() + 2 * index];
  }

  const ExprPtr& thenClause(size_t index) const {
    return inputs_[firstCaseInput() + 2 * index + 1];
  }

  const ExprPtr& elseClause() const {
    VELOX_DCHECK(hasElseClause_);
    return inputs_.back();
  }

  const bool hasSubject_;
  const size_t numCases_;
  const bool hasElseClause_;

 private:
  size_t firstCaseInput() const {
    return hasSubject_ ? 1 : 0;
  }

  void computePropagatesNulls() override;

  // Handles the rows of 'rows' where 'errors' has an error and removes them
  // from 'rows'. 'failedCase' is the case whose condition failed, or
  // std::nullopt if the subject failed. If all values a row can get are equal,
  // writes the value to 'result' and drops the error. Otherwise, throws the
  // error or, if 'context' doesn't throw on errors, adds it to 'context'.
  void resolveErrors(
      const EvalErrorsPtr& errors,
      std::optional<size_t> failedCase,
      SelectivityVector& rows,
      const VectorPtr& subject,
      EvalCtx& context,
      VectorPtr& result);

  // Evaluates the values that the rows of 'rows' can get after 'failedCase'
  // failed and returns the first of these values. Removes from 'rows' the rows
  // where these values aren't all equal. Stops evaluating a row at its first
  // value that differs.
  VectorPtr evalCommonValue(
      std::optional<size_t> failedCase,
      SelectivityVector& rows,
      const VectorPtr& subject,
      EvalCtx& context);

  // Scratch buffer reused across calls by getFlatBool.
  BufferPtr tempValues_;
};

} // namespace facebook::velox::exec
