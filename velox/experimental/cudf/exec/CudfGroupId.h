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
#include "velox/experimental/cudf/exec/CudfProjectionCycle.h"

#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

/// GPU implementation of the GroupId operator for GROUPING SETS, CUBE, and
/// ROLLUP operations. Takes a single input batch and produces N output batches
/// (one per grouping set).
class CudfGroupId : public CudfOperatorBase {
 public:
  CudfGroupId(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::GroupIdNode>& groupIdNode);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && !projectionCycle_.hasInput();
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

 private:
  // Produces one output batch per grouping set of each input batch.
  CudfProjectionCycle projectionCycle_;
};

} // namespace facebook::velox::cudf_velox
