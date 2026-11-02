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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox {

// TODO: Does not support Filter yet.
class CudfFilterProject : public CudfOperatorBase {
 public:
  CudfFilterProject(
      int32_t operatorId,
      velox::exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project);

  void initialize() override;

  bool needsInput() const override {
    return !input_;
  }

  void filter(
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
      cuda::stream_ref stream);

  /// Evaluates the projections over 'inputTableColumns'. 'numRows' is the
  /// row count of the batch, which sizes the constant projections of a
  /// zero-column input.
  std::vector<std::unique_ptr<cudf::column>> project(
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
      cudf::size_type numRows,
      cuda::stream_ref stream);

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

  void doClose() override {
    Operator::close();
    projectEvaluators_.clear();
    constantProjections_.clear();
    filterEvaluator_.reset();
  }

 private:
  bool allInputProcessed();

  // If true exprs_[0] is a filter and the other expressions are projections
  const bool hasFilter_{false};

  // Cached filter and project node for lazy initialization. After
  // initialization, they will be reset, and initialized_ will be set to true.
  std::shared_ptr<const core::ProjectNode> project_;
  std::shared_ptr<const core::FilterNode> filter_;

  std::vector<CudfExpressionPtr> projectEvaluators_;
  // Constant projections of a zero-column input, as (output channel, scalar)
  // with the scalar in the type the pipeline carries. project() fills each to
  // the row count of the batch.
  std::vector<std::pair<column_index_t, std::unique_ptr<cudf::scalar>>>
      constantProjections_;
  CudfExpressionPtr filterEvaluator_;

  std::vector<velox::exec::IdentityProjection> resultProjections_;
  std::vector<velox::exec::IdentityProjection> identityProjections_;
};

/// True when 'projection' is a plan constant over a zero-column 'inputType'.
/// Shared by FilterProjectAdapter::canRunOnGPU and
/// CudfFilterProject::initialize() so the two checks cannot drift.
bool isConstantProjectionOverZeroColumns(
    const RowTypePtr& inputType,
    const core::TypedExprPtr& projection);

} // namespace facebook::velox::cudf_velox
