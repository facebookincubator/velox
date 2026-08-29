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
  // Returns true when buffering is measured in logical rows, which happens
  // when no byte target is configured or the output has no GPU columns.
  bool usesRowFallback() const {
    return !targetBytes_.has_value();
  }

  // Returns true when the active byte or row target is met.
  bool targetReached() const {
    return usesRowFallback() ? currentNumRows_ >= targetRows_
                             : currentBytes_ >= targetBytes_.value();
  }

  // Driver context associated with this operator.
  exec::DriverCtx* const driverCtx_;

  // Input vectors awaiting concatenation.
  std::vector<CudfVectorPtr> buffer_;

  // Concatenated vectors ready for downstream consumption.
  std::queue<CudfVectorPtr> outputQueue_;

  // Estimated GPU bytes currently held in buffer_.
  uint64_t currentBytes_{0};

  // Logical rows currently buffered while the row fallback is active.
  size_t currentNumRows_{0};

  // Estimated GPU byte target. Empty when the row fallback is active, that is
  // when no byte target is configured or the output has no GPU columns. Both
  // targets are resolved from the config alone, so neither depends on the
  // other's position in this declaration list.
  const std::optional<uint64_t> targetBytes_;

  // Logical row target used while the row fallback is active.
  const size_t targetRows_{0};
};

} // namespace facebook::velox::cudf_velox
