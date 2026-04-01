/*
 * Copyright (c) International Business Machines Corporation
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

#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/serializers/PrestoIterativePartitioningSerializer.h"

namespace facebook::velox::exec {

/// Partitioned output operator backed by PrestoIterativePartitioningSerializer.
///
/// Routes each input row to a partition via a hash function, buffers the
/// partitioned data, and flushes serialized Presto pages into the output
/// buffer manager when the buffer is full or the pipeline is draining.
class OptimizedPartitionedOutput : public Operator {
 public:
  /// Minimum flush size for non-final flush; 60 KB + overhead fits a 64 KB
  /// network MTU.
  static constexpr uint64_t kMinDestinationSize = 60 * 1024;

  OptimizedPartitionedOutput(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  void addInput(RowVectorPtr input) override;

  /// Returns true when the operator is not waiting for the output buffer to
  /// drain. The driver checks this before calling addInput() so a blocked
  /// state does not accumulate additional rows.
  bool needsInput() const override;

  /// Always returns nullptr; output is pushed into the buffer manager as a
  /// side-effect. Flushes the serializer when the buffer is full or the
  /// pipeline is draining, then signals noMoreData() once all rows are sent.
  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 private:
  /// Serializes all buffered rows into Presto pages and enqueues each page
  /// into the output buffer manager. All destinations are always enqueued;
  /// sets blockingReason_ and records a future if the output buffer is full.
  /// Increments numFlushes_ on each call.
  void flush();

  const std::string taskId_;
  /// Input row type; also used as output type (column reordering not yet
  /// applied).
  const RowTypePtr inputType_;
  const std::vector<column_index_t> keyChannels_;
  /// Non-empty when the output column order differs from the input.
  const std::vector<column_index_t> outputChannels_;
  const int32_t numDestinations_;

  const bool replicateNullsAndAny_;
  const std::weak_ptr<exec::OutputBufferManager> bufferManager_;
  /// Holds a reference to the owning task to prevent it from being destroyed
  /// while serialized pages are in flight inside the buffer manager.
  const std::function<void()> bufferReleaseFn_;
  const int64_t maxOutputBufferBytes_;

  velox::memory::MemoryPool* pool_;
  /// Computes per-row partition assignments. Null when numDestinations_ == 1.
  std::unique_ptr<core::PartitionFunction> partitionFunction_;
  /// Reusable buffer for per-row partition assignments.
  std::vector<uint32_t> partitions_;
  std::unique_ptr<serializer::presto::PrestoIterativePartitioningSerializer>
      serializer_;

  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  /// Counts addInput() calls that appended at least one row to the serializer.
  /// Exposed as the "numAppendTimes" runtime stat.
  uint64_t numAppends_{0};
  /// Counts non-empty flush() calls — flushes that serialized at least one
  /// row. Exposed as the "numFlushes" runtime stat for test verification.
  uint64_t numFlushes_{0};
  /// Counts flush() calls that caused the driver to block on a full output
  /// buffer. Exposed as the "numBlockedTimes" runtime stat.
  uint64_t numBlockedTimes_{0};
};

} // namespace facebook::velox::exec
