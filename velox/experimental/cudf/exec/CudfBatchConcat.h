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

#include <cstdint>
#include <optional>
#include <queue>

namespace facebook::velox::cudf_velox {

class CudfBatchConcat : public CudfOperatorBase {
 public:
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode);

  bool needsInput() const override {
    return !noMoreInput_ && outputQueue_.empty() && !targetReached();
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  // Returns true if 'numRows' rows occupying 'numBytes' estimated GPU bytes
  // meet the flush target: the byte target when set, the row target otherwise.
  bool meetsTarget(size_t numRows, uint64_t numBytes) const;

  bool targetReached() const {
    return meetsTarget(currentNumRows_, currentBytes_);
  }

  // Returns the estimated GPU bytes of 'vector', or 0 when no byte target is
  // set.
  uint64_t estimateBytes(const CudfVector& vector) const;

  // Input vectors awaiting concatenation.
  std::vector<CudfVectorPtr> buffer_;

  // Concatenated vectors ready for downstream consumption.
  std::queue<CudfVectorPtr> outputQueue_;

  // Rows held in buffer_.
  size_t currentNumRows_{0};

  // Estimated GPU bytes held in buffer_. Tracked only when targetBytes_ is set.
  uint64_t currentBytes_{0};

  // Byte target from batchSizeMinBytes. Unset for a zero-column output.
  const std::optional<uint64_t> targetBytes_;

  // Row target from batchSizeMinThreshold, used when targetBytes_ is unset.
  const size_t targetRows_;
};

} // namespace facebook::velox::cudf_velox
