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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

#include <queue>

namespace facebook::velox::cudf_velox {

class CudfBatchConcat : public exec::Operator, public CudfOperator {
 public:
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode);

  bool needsInput() const override {
    return !noMoreInput_ && outputQueue_.empty() &&
        currentNumRows_ < targetRows_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  exec::DriverCtx* const driverCtx_;
  std::vector<CudfVectorPtr> buffer_;
  std::queue<std::unique_ptr<cudf::table>> outputQueue_;
  rmm::cuda_stream_view outputQueueStream_{rmm::cuda_stream_default};
  size_t currentNumRows_{0};
  const size_t targetRows_{0};
};

} // namespace facebook::velox::cudf_velox
