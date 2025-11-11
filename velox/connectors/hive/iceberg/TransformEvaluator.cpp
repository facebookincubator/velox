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

#include "velox/connectors/hive/iceberg/TransformEvaluator.h"

#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive::iceberg {

TransformEvaluator::TransformEvaluator(
    const std::vector<core::TypedExprPtr>& expressions,
    const ConnectorQueryCtx* connectorQueryCtx)
    : connectorQueryCtx_(connectorQueryCtx) {
  VELOX_CHECK_NOT_NULL(connectorQueryCtx_);
  exprSet_ = connectorQueryCtx_->expressionEvaluator()->compile(expressions);
  VELOX_CHECK_NOT_NULL(exprSet_);
}

std::vector<VectorPtr> TransformEvaluator::evaluate(
    const RowVectorPtr& input) const {
  const auto numRows = input->size();
  const auto numExpressions = exprSet_->exprs().size();

  std::vector<VectorPtr> results(numExpressions);
  SelectivityVector rows(numRows);

  // Evaluate all expressions in one pass.
  connectorQueryCtx_->expressionEvaluator()->evaluate(
      exprSet_.get(), rows, *input, results);

  return results;
}

} // namespace facebook::velox::connector::hive::iceberg
