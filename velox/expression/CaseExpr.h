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

#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::exec {

/// Simple CASE expression — evaluates the subject exactly once per row.
///
///   CASE subject
///     WHEN val  THEN result
///     [WHEN val THEN result ...]
///     [ELSE result]
///   END
///
/// Inputs layout:
///   inputs_[0]         = subject expression
///   inputs_[2*i + 1]   = WHEN value for case i  (i = 0..numCases_-1)
///   inputs_[2*i + 2]   = THEN result for case i
///   inputs_.back()     = ELSE result (present only when hasElseClause_)
///
/// Unlike SwitchExpr, which duplicates the subject into each eq() condition
/// (causing non-deterministic subjects like rand() to be re-evaluated per WHEN
/// branch), CaseExpr evaluates the subject once and applies a pre-built 'eq'
/// Expr to the cached subject vector and each WHEN value vector. Routing the
/// comparison through the Expr layer (rather than calling the VectorFunction
/// directly) preserves dictionary peeling, the registered eq variant's
/// semantic differences (e.g. Spark vs Presto null-handling), and per-row
/// error reporting via TRY.
class CaseExpr : public SpecialForm {
 public:
  /// @param eqExpr Pre-built 'eq' Expr whose children are FieldReferences
  /// over a synthetic two-column row (subject, when). Invoked at evaluation
  /// time against a RowVector populated with the cached subject and each
  /// per-iteration WHEN vector. Must not be null.
  /// @param eqInputType Two-column row type matching eqExpr's FieldReferences.
  CaseExpr(
      TypePtr type,
      const std::vector<ExprPtr>& inputs,
      ExprPtr eqExpr,
      RowTypePtr eqInputType,
      bool inputsSupportFlatNoNullsFastPath);

  /// Builds the eq Expr (eq(field("c0"), field("c1")) over a synthetic row of
  /// type ROW(comparisonType, comparisonType)) and constructs a CaseExpr.
  /// Mirrors NullIfExpr::create, which builds a CastExpr and stores it on
  /// the special form for use during evalSpecialForm.
  static ExprPtr create(
      TypePtr type,
      std::vector<ExprPtr> inputs,
      const TypePtr& comparisonType,
      bool trackCpuUsage,
      const core::QueryConfig& config);

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

  /// Generates standard SQL CASE syntax so that the round-trip in testToSql
  /// produces an identical expression tree.  The base Expr::toSql() would emit
  /// "case"(...) which DuckDB cannot re-parse as a CASE expression.
  std::string toSql(
      std::vector<VectorPtr>* complexConstants = nullptr) const override;

 private:
  void computePropagatesNulls() override;

  const size_t numCases_;
  const bool hasElseClause_;

  // Pre-built 'eq' Expr used to compare the cached subject vector against
  // each WHEN value vector. Holds two FieldReferences over eqInputType_.
  const ExprPtr eqExpr_;

  // Synthetic row type backing eqExpr_'s FieldReferences. Two columns of
  // comparisonType, named "c0" (subject) and "c1" (when).
  const RowTypePtr eqInputType_;

  // Scratch buffer reused across calls by getFlatBool.
  BufferPtr tempValues_;
};

class CaseCallToSpecialForm : public FunctionCallToSpecialForm {
 public:
  /// Returns the type of the first THEN clause.  Throws if:
  ///   - fewer than 3 arguments are provided,
  ///   - THEN (and optional ELSE) clause types are inconsistent, or
  ///   - no 'eq' function is registered for the subject / WHEN-value types.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Like resolveType but also fills in type coercions for the subject and WHEN
  /// values when their types differ.  The subject and all WHEN values are
  /// coerced to their least-common super-type so that the 'eq' VectorFunction
  /// can be looked up with a single uniform type.  THEN/ELSE coercions follow
  /// the same widening logic as SwitchCallToSpecialForm.
  TypePtr resolveTypeWithCoercions(
      const std::vector<TypePtr>& argTypes,
      std::vector<TypePtr>& coercions,
      const TypeCoercer& coercer) override;

  ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;
};

} // namespace facebook::velox::exec
