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

#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Spiller.h"

namespace facebook::velox::exec {

/// Shares completed build input with all probe drivers as either in-memory
/// batches or spill files. Both are empty for an empty build side.
struct NestedLoopJoinBuildData {
  /// Holds in-memory batches until all probes have finished.
  std::vector<RowVectorPtr> vectors;

  /// Preserves file order so every probe uses the same build-row identities.
  SpillFiles spillFiles;
};

/// Publishes build input once all build drivers have finished.
class NestedLoopJoinBridge : public JoinBridge {
 public:
  /// Makes the completed build input available and wakes waiting probes.
  void setData(NestedLoopJoinBuildData buildData);

  /// Returns build input, or sets 'future' if the build is not complete.
  std::shared_ptr<const NestedLoopJoinBuildData> dataOrFuture(
      ContinueFuture* future);

 private:
  // Shares immutable batch or file metadata across probe drivers.
  std::shared_ptr<const NestedLoopJoinBuildData> buildData_;
};

/// Buffers build input and spills it when memory arbitration requests reclaim.
class NestedLoopJoinBuild : public Operator {
 public:
  NestedLoopJoinBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  /// Spills buffered input while the task is paused for memory arbitration.
  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

  void close() override;

  /// Merges adjacent batches up to the configured maximum output batch size.
  std::vector<RowVectorPtr> mergeDataVectors() const;

 private:
  // Starts spilling on the first non-empty reclaim and releases buffered input.
  void spillBufferedVectors();

  // Describes the spill schema even when this driver receives no input.
  const RowTypePtr buildType_;

  // Owns buffered input in this operator's pool when spilling is enabled.
  std::vector<RowVectorPtr> dataVectors_;

  // Once created, receives all subsequent input until the build barrier.
  std::unique_ptr<NoRowContainerSpiller> spiller_;

  // Future for synchronizing with other Drivers of the same pipeline. All build
  // Drivers must be completed before making data available for the probe side.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::exec
