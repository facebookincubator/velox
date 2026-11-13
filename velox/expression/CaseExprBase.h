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

  // Scratch buffer reused across calls by getFlatBool.
  BufferPtr tempValues_;
};

} // namespace facebook::velox::exec
