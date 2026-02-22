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

#include "velox/connectors/Connector.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive::iceberg {

/// Evaluates multiple expressions efficiently using batch evaluation.
/// Expressions are compiled once in the constructor and reused across multiple
/// input batches.
class TransformEvaluator {
 public:
  /// Creates an evaluator with the given expressions and connector query
  /// context. Compiles the expressions once for reuse across multiple
  /// evaluations.
  ///
  /// @param expressions Vector of typed expressions to evaluate. These are
  /// typically built using TransformExprBuilder::toExpressions() for Iceberg
  /// partition transforms, but can be any valid Velox expressions. The
  /// expressions are compiled once during construction.
  /// @param connectorQueryCtx Connector query context providing access to the
  /// expression evaluator (for compilation and evaluation) and memory pool.
  /// Must remain valid for the lifetime of this TransformEvaluator.
  TransformEvaluator(
      const std::vector<core::TypedExprPtr>& expressions,
      const ConnectorQueryCtx* connectorQueryCtx);

  /// Evaluates all expressions on the input data in a single pass.
  /// Uses the pre-compiled ExprSet from the constructor for efficiency.
  ///
  /// The input RowType must match the RowType used when building the
  /// expressions (passed to TransformExprBuilder::toExpressions). The column
  /// positions, names and types must align. Create new TransformEvaluator for
  /// input that has different RowType with the one when building the
  /// expressions.
  ///
  /// @param input Input row vector containing the source data. Must have the
  /// same RowType (column positions, names and types) as used when building the
  /// expressions in the constructor.
  /// @return Vector of result columns, one for each expression, in the same
  /// order as the expressions provided to the constructor.
  std::vector<VectorPtr> evaluate(const RowVectorPtr& input) const;

 private:
  const ConnectorQueryCtx* connectorQueryCtx_;
  std::unique_ptr<exec::ExprSet> exprSet_;
};

} // namespace facebook::velox::connector::hive::iceberg
