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

struct NestedLoopJoinBuildData {
  static std::shared_ptr<const NestedLoopJoinBuildData> createInMemory(
      std::vector<RowVectorPtr> vectors);

  static std::shared_ptr<const NestedLoopJoinBuildData> createSpilled(
      RowTypePtr type,
      SpillFiles files);

  bool spilled{false};
  std::vector<RowVectorPtr> vectors;
  RowTypePtr type;
  SpillFiles spillFiles;
};

class NestedLoopJoinBridge : public JoinBridge {
 public:
  void setData(std::shared_ptr<const NestedLoopJoinBuildData> buildData);

  std::shared_ptr<const NestedLoopJoinBuildData> dataOrFuture(
      ContinueFuture* future);

 private:
  std::shared_ptr<const NestedLoopJoinBuildData> buildData_;
};

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

  bool canReclaim() const override {
    return canSpill() && !dataVectors_.empty();
  }

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

  void close() override {
    dataVectors_.clear();
    spiller_.reset();
    Operator::close();
  }

  std::vector<RowVectorPtr> mergeDataVectors() const;

 private:
  void ensureSpiller();

  void spillVector(RowVectorPtr vector);

  void spillBufferedVectors();

  void finishSpill(SpillPartitionSet& spillPartitions);

  RowVectorPtr copyToOperatorPool(const RowVectorPtr& input);

  const RowTypePtr buildType_;

  std::vector<RowVectorPtr> dataVectors_;

  std::unique_ptr<NoRowContainerSpiller> spiller_;

  bool spilled_{false};

  // Future for synchronizing with other Drivers of the same pipeline. All build
  // Drivers must be completed before making data available for the probe side.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::exec
