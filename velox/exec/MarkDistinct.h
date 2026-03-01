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

#include "velox/exec/GroupingSet.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Spiller.h"

namespace facebook::velox::exec {

class MarkDistinct : public Operator {
 public:
  MarkDistinct(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::MarkDistinctNode>& planNode);

  bool preservesOrder() const override {
    return !spilled_;
  }

  bool needsInput() const override {
    return !noMoreInput_ && !input_;
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

 private:
  bool spillEnabled() const {
    return spillConfig_.has_value();
  }

  void ensureInputFits(const RowVectorPtr& input);

  void spill();

  void spillInput(const RowVectorPtr& input, memory::MemoryPool* pool);

  void setupInputSpiller(const SpillPartitionIdSet& spillPartitionIdSet);

  void setSpillPartitionBits(
      const SpillPartitionId* restoredPartitionId = nullptr);

  SpillPartitionIdSet spillHashTable();

  void restoreNextSpillPartition();

  void finishSpillInputAndRestoreNext();

  void processSpilledInput();

  void recursiveSpillInput();

  RowTypePtr inputType_;

  std::unique_ptr<GroupingSet> groupingSet_;

  HashBitRange spillPartitionBits_;

  std::unique_ptr<NoRowContainerSpiller> inputSpiller_;

  std::unique_ptr<UnorderedStreamReader<BatchStream>> spillInputReader_;

  std::optional<SpillPartitionId> restoringPartitionId_;

  SpillPartitionSet spillInputPartitionSet_;

  std::unique_ptr<HashPartitionFunction> spillHashFunction_;

  bool exceededMaxSpillLevelLimit_{false};

  bool spilled_{false};

  bool yield_{false};

  std::vector<column_index_t> distinctKeyChannels_;
};
} // namespace facebook::velox::exec
