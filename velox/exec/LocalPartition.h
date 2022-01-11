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

#include "velox/exec/Operator.h"
#include "velox/exec/VectorHasher.h"

namespace facebook::velox::exec {

/// Keeps track of the total size in bytes of the data buffered in all
/// LocalExchangeSources.
class LocalExchangeMemoryManager {
 public:
  explicit LocalExchangeMemoryManager(int64_t maxBufferSize)
      : maxBufferSize_{maxBufferSize} {}

  /// Returns 'true' if memory limit is reached or exceeded and sets future that
  /// will be complete when memory usage is update to be below the limit.
  bool increaseMemoryUsage(ContinueFuture* future, int64_t added);

  void decreaseMemoryUsage(int64_t removed);

 private:
  const int64_t maxBufferSize_;
  std::mutex mutex_;
  int64_t bufferedBytes_{0};
  std::vector<VeloxPromise<bool>> promises_;
};

/// Buffers data for a single partition produced by local exchange. Allows
/// multiple producers to enqueue data and multiple consumers fetch data. Each
/// producer must be registered with a call to 'addProducer'. 'noMoreProducers'
/// must be called after all producers have been registered. A producer calls
/// 'enqueue' multiple time to put the data and calls 'noMoreData' when done.
/// Consumers call 'next' repeatedly to fetch the data.
class LocalExchangeSource {
 public:
  LocalExchangeSource(LocalExchangeMemoryManager* memoryManager, int partition)
      : memoryManager_{memoryManager}, partition_{partition} {}

  std::string toString() const {
    return fmt::format("LocalExchangeSource({})", partition_);
  }

  void addProducer();

  void noMoreProducers();

  /// Used by a producer to add data. Returning kNotBlocked if can accept more
  /// data. Otherwise returns kWaitForConsumer and sets future that will be
  /// completed when ready to accept more data.
  BlockingReason enqueue(RowVectorPtr input, ContinueFuture* future);

  /// Called by a producer to indicate that no more data will be added.
  void noMoreData();

  /// Used by a consumer to fetch some data. Returns kNotBlocked and sets data
  /// to nullptr if all data has been fetched and all producers are done
  /// producing data. Returns kWaitForExchange if there is no data, but some
  /// producers are not done producing data. Sets future that will be completed
  /// once there is data to fetch or if all producers report completion.
  ///
  /// @param pool Memory pool used to copy the data before returning.
  BlockingReason
  next(ContinueFuture* future, memory::MemoryPool* pool, RowVectorPtr* data);

  /// Used by producers to get notified when all data has been fetched. Returns
  /// kNotBlocked if all data has been fetched. Otherwise, returns
  /// kWaitForConsumer and sets future that will be competed when all data is
  /// fetched. Producers must stay alive until all data has been fetched.
  /// Otherwise, the memory backing the data may get freed before the data was
  /// copied into the consumers memory pool.
  BlockingReason isFinished(ContinueFuture* future);

  bool isFinished();

  void close() {
    queue_.withWLock([](auto& queue) {
      while (!queue.empty()) {
        queue.pop();
      }
    });
  }

 private:
  LocalExchangeMemoryManager* memoryManager_;
  const int partition_;
  folly::Synchronized<std::queue<RowVectorPtr>> queue_;
  // Satisfied when data becomes available or all producers report that they
  // finished producing, e.g. queue_ is not empty or noMoreProducers_ is true
  // and pendingProducers_ is zero.
  std::vector<VeloxPromise<bool>> consumerPromises_;
  // Satisfied when all data has been fetched and no more data will be produced,
  // e.g. queue_ is empty, noMoreProducers_ is true and pendingProducers_ is
  // zero.
  std::vector<VeloxPromise<bool>> producerPromises_;
  int pendingProducers_{0};
  bool noMoreProducers_{false};
};

/// Fetches data for a single partition produced by local exchange from
/// LocalExchangeSource.
class LocalExchangeSourceOperator : public SourceOperator {
 public:
  LocalExchangeSourceOperator(
      int32_t operatorId,
      DriverCtx* ctx,
      RowTypePtr outputType,
      const std::string& planNodeId,
      int partition);

  std::string toString() override {
    return fmt::format("LocalExchangeSourceOperator({})", partition_);
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  RowVectorPtr getOutput() override;

  bool isFinished() override;

 private:
  const int partition_;
  const std::shared_ptr<LocalExchangeSource> source_{nullptr};
  ContinueFuture future_{false};
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
};

/// Hash partitions the data using specified keys. The number of partitions is
/// determined by the number of LocalExchangeSources(s) found in the task.
class LocalPartition : public Operator {
 public:
  LocalPartition(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::LocalPartitionNode>& planNode);

  std::string toString() override {
    return fmt::format("LocalPartition({})", numPartitions_);
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  // Always true but the caller will check isBlocked before adding input, hence
  // the blocked state does not accumulate input.
  bool needsInput() const override {
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  void noMoreInput() override;

  bool isFinished() override;

  void close() override {
    Operator::close();
    for (auto& source : localExchangeSources_) {
      source->close();
    }
  }

 private:
  BlockingReason
  enqueue(int32_t source, RowVectorPtr data, ContinueFuture* future);

  const std::vector<std::shared_ptr<LocalExchangeSource>> localExchangeSources_;
  const size_t numPartitions_;
  std::unique_ptr<core::PartitionFunction> partitionFunction_;
  // Empty if column order in the output is exactly the same as in input.
  const std::vector<ChannelIndex> outputChannels_;

  uint32_t numBlockedPartitions_{0};
  std::vector<BlockingReason> blockingReasons_;
  std::vector<ContinueFuture> futures_;

  /// Reusable memory for hash calculation.
  std::vector<uint32_t> partitions_;
};

} // namespace facebook::velox::exec
