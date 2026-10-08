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

#include <mutex>
#include <unordered_map>

#include "velox/exec/MaterializedOutputBuffer.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/OutputTransportRegistry.h"

namespace facebook::velox::exec {

/// Controls operator-local serialization batching before RowGroups are
/// emitted to individual destinations.
struct MaterializedOutputBatchConfig {
  int64_t minOutputBatchBytes{1L << 20};
  int64_t maxOutputBatchBytes{16L << 20};
  int64_t estimatedRowBytes{1'024};
};

/// Owns materialized output buffers for all tasks using one exchange transport.
class MaterializedOutputBufferManager final
    : public velox::exec::OutputBufferManager,
      public std::enable_shared_from_this<MaterializedOutputBufferManager> {
 public:
  MaterializedOutputBufferManager(
      ExchangeSink::Factory sinkFactory,
      int64_t maxBufferedBytes,
      MaterializedOutputBatchConfig outputBatchConfig = {});

  /// Returns the configured serialization batch size for a destination count.
  int64_t outputBatchSizeBytes(int32_t numDestinations) const;

  /// Returns the task buffer used by its materialized output operators.
  std::shared_ptr<MaterializedOutputBuffer> buffer(
      const std::string& taskId) const;

  /// Returns committed output locations for a completed task.
  folly::F14FastMap<int32_t, std::string> outputLocations(
      const std::string& taskId) const;

  /// Creates the Velox transport entry paired with this manager.
  std::shared_ptr<velox::exec::OutputTransportEntry> transportEntry();

  void initializeTask(
      std::shared_ptr<velox::exec::Task> task,
      velox::core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers,
      const std::string& transportOptions = {}) override;

  bool updateOutputBuffers(
      const std::string& taskId,
      int numDestinations,
      bool noMoreBuffers) override;

  bool updateNumDrivers(const std::string& taskId, uint32_t newNumDrivers)
      override;

  void removeTask(const std::string& taskId) override;

  std::optional<velox::exec::OutputBufferStats> stats(
      const std::string& taskId) override;

  std::optional<double> getUtilization(const std::string& taskId) override;

  /// Reported only in task statistics; not used by scaled TableWriter
  /// scheduling.
  std::optional<bool> isOverutilized(const std::string& taskId) override;

  std::string toString(const std::string& taskId) override;

 private:
  struct TaskBuffer {
    std::shared_ptr<MaterializedOutputBuffer> buffer;
    std::shared_ptr<velox::memory::MemoryPool> sinkPool;
    int64_t maxBufferedBytes;
  };

  mutable std::mutex mutex_;
  const ExchangeSink::Factory sinkFactory_;
  const int64_t maxBufferedBytes_;
  const MaterializedOutputBatchConfig outputBatchConfig_;
  std::unordered_map<std::string, TaskBuffer> buffers_;
  std::unordered_map<std::string, folly::F14FastMap<int32_t, std::string>>
      completedOutputs_;
};

} // namespace facebook::velox::exec
