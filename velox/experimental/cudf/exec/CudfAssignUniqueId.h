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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::cudf_velox {

class CudfAssignUniqueId : public exec::Operator, public NvtxHelper {
 public:
  CudfAssignUniqueId(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::AssignUniqueIdNode>& planNode,
      int32_t uniqueTaskId,
      std::shared_ptr<std::atomic_int64_t> rowIdPool);

  bool isFilter() const override {
    return true;
  }

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override {
    return input_ == nullptr;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool startDrain() override {
    // No need to drain for assignUniqueId operator.
    return false;
  }

  bool isFinished() override;

 private:
  std::unique_ptr<cudf::column> generateIdColumn(
      vector_size_t size,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  void requestRowIds();

  const int64_t kRowIdsPerRequest = 1L << 20;
  const int64_t kMaxRowId = 1L << 40;
  const int64_t kTaskUniqueIdLimit = 1L << 24;

  int64_t uniqueValueMask_;
  int64_t rowIdCounter_;
  int64_t maxRowIdCounterValue_;

  std::shared_ptr<std::atomic_int64_t> rowIdPool_;
  uint64_t queuedInputBytes_{0};
};
} // namespace facebook::velox::cudf_velox
