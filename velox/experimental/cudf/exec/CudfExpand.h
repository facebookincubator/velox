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

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox {

/// GPU implementation of the Expand operator. Takes a single input batch and
/// produces one output batch per projection list. Each output column is either
/// an input column or a constant.
class CudfExpand : public CudfOperatorBase {
 public:
  CudfExpand(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::ExpandNode>& expandNode);

  /// Returns true if every constant in the projections has a type CudfExpand
  /// can materialize as a cuDF column. On failure, 'reason' is populated with
  /// an explanation when non-null.
  static bool canRunOnGPU(
      const core::ExpandNode& expandNode,
      std::string* reason);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doClose() override;

 private:
  // Input channel for each output column, one list per projection.
  // kConstantChannel marks columns taken from 'constantProjections_'.
  std::vector<std::vector<column_index_t>> fieldProjections_;

  // Constant expression for each output column, one list per projection.
  // Null where the column is an input field.
  std::vector<std::vector<core::ConstantTypedExprPtr>> constantProjections_;

  // cuDF scalars built from 'constantProjections_' on the first input batch,
  // reused for every batch.
  std::vector<std::vector<std::unique_ptr<cudf::scalar>>> constantScalars_;

  // Index into 'fieldProjections_' of the next projection to output for the
  // current input batch.
  size_t projectionIndex_{0};
};

} // namespace facebook::velox::cudf_velox
