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
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

/**
 * @brief GPU operator that validates input contains at most one row.
 *
 * This operator is used for scalar subqueries that must return exactly 0 or 1
 * row. If the input contains more than one row, it throws a user error. If the
 * input is empty, it returns a single row of nulls.
 *
 * This is a pass-through operator that performs validation on GPU metadata
 * (row count) without transferring data between host and device.
 */
class CudfEnforceSingleRow : public CudfOperatorBase {
 public:
  CudfEnforceSingleRow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::EnforceSingleRowNode>& planNode);

  bool isFilter() const override {
    return true;
  }

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override;

  bool isFinished() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
};

} // namespace facebook::velox::cudf_velox
