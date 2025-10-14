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

#include <folly/CppAttributes.h>

#include "velox/exec/GroupingSet.h"
#include "velox/exec/Operator.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {

struct Partitioned {
  RowVectorPtr input;
  BufferPtr partitions;
  std::vector<vector_size_t> offsets;
  uint64_t size;
  std::atomic<int32_t> accessCount{0};

  Partitioned(
      const RowVectorPtr& input,
      const BufferPtr& partitions,
      const std::vector<vector_size_t>& offsets,
      uint64_t size)
      : input(input), partitions(partitions), offsets(offsets), size(size) {}

  Partitioned() = delete;
  Partitioned(const Partitioned&) = delete;
  Partitioned(Partitioned&&) = delete;

  ~Partitioned() {
    input = nullptr;
    partitions = nullptr;
  }
};

class PartitionBucket {
 public:
  PartitionBucket(size_t bucketSize, int32_t driverCount);

  bool tryAdd(Partitioned* partitioned);

  Partitioned* FOLLY_NULLABLE getSlotAt(int32_t index);

  void noMoreInput() {
    noMoreinputDriverCount_.fetch_add(1);
  }

  bool isFinished() {
    if (noMoreinputDriverCount_.load() == driverCount_) {
      if (partitionCount_.load() == 0) {
        return true;
      } else {
        needToConsume_.store(true);
      }
    }
    return false;
  }

  size_t bucketSize() const {
    return bucketSize_;
  }

  int32_t partitionCount() const {
    return partitionCount_.load();
  }

  uint64_t memoryUsage() const {
    return memoryUsage_.load();
  }

  inline void requestToConsume() {
    needToConsume_.store(true);
  }

  inline void resetNeedToConsume() {
    needToConsume_.store(false);
  }

  inline bool needToConsume() const {
    return needToConsume_.load();
  }

  ~PartitionBucket() = default;

  void testingValidate();

 private:
  size_t bucketSize_;
  std::vector<std::atomic<Partitioned*>> slots_;
  // Number of non-empty partitions in the bucket.
  std::atomic<int32_t> partitionCount_;
  // Total memory usage of non-empty partitions in the bucket.
  std::atomic<uint64_t> memoryUsage_;
  // Number of producers.
  int32_t driverCount_;
  // Number of producers that have no more input.
  std::atomic<int32_t> noMoreinputDriverCount_;
  std::atomic<bool> needToConsume_{false};
};

class ExchangeAggregation : public Operator {
 public:
  ExchangeAggregation(
      int32_t operatorId,
      DriverCtx* ctx,
      const core::PartitionFunctionSpecPtr& partitionFunctionSpec,
      const std::shared_ptr<const core::AggregationNode>& planNode);

  void initialize() override;

  std::string toString() const override {
    return fmt::format("ExchangeAggregation({})", numPartitions_);
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  void noMoreInput() override;

  bool isFinished() override;

  void close() override;

 private:
  void prepareForInput(RowVectorPtr& input);

  void setupGroupingKeyChannelProjections(
      std::vector<column_index_t>& groupingKeyInputChannels,
      std::vector<column_index_t>& groupingKeyOutputChannels) const;

  Partitioned* processPartition(
      RowVectorPtr input,
      const std::vector<uint32_t>& partitions,
      uint64_t inputSize);

  void addOneToBucket(int32_t bucketIndex, Partitioned* partitioned);

  void addWaitingListToBuckets();

  RowVectorPtr wrapInDictionary(
      const Partitioned* partitioned,
      int32_t dirverId);

  void doAggregationLocked(RowVectorPtr partitionVector);

  void doAggregation(RowVectorPtr partitionVector);

  bool tryDoAggregation(RowVectorPtr partitionVector);

  bool tryDoAggregation(int32_t driverId, RowVectorPtr partitionVector);

  void prepareOutput(vector_size_t size);

  void getDistinctOutput(bool newDistincts);

  RowVectorPtr fillOutput();

  void updateRuntimeStats();

  std::shared_ptr<const core::AggregationNode> planNode_;
  const int32_t driverId_;

  std::vector<Operator*> peers_;
  const std::vector<std::shared_ptr<PartitionBucket>>& buckets_;
  const size_t numPartitions_;
  std::unique_ptr<core::PartitionFunction> partitionFunction_;
  std::vector<uint32_t> partitions_;
  RowVectorPtr partitionedVector_{nullptr};
  std::queue<std::pair<int32_t, Partitioned*>> waitingList_;
  uint64_t memoryLimit_;

  const bool isDistinct_;
  std::unique_ptr<GroupingSet> groupingSet_;
  RowContainerIterator resultIterator_;
  // Possibly reusable output vector.
  RowVectorPtr output_;
  // During distinct aggregation, keep all output after processing all
  // partitions in the bucket and return them one by one. This queue is empty if
  // the current aggregation is not distinct aggregation.
  std::queue<RowVectorPtr> distinctOutputs_;
  std::optional<int64_t> estimatedOutputRowSize_;

  bool exchangeFinished_{false};
  bool aggregationFinished_{false};

  std::mutex mutex_;
};

} // namespace facebook::velox::exec
