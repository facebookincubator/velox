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

#include <memory>
#include <optional>
#include <string>

#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h"

namespace facebook::velox::exec {

class Task;

/// Abstract interface for managing output buffers used to deliver partitioned
/// output from a Task to downstream consumers. Concrete implementations (e.g.
/// OutputBufferManager for CPU, or a GPU-accelerated variant) are registered
/// in OutputBufferManagerRegistry and resolved per query via QueryCtx.
class IOutputBufferManager {
 public:
  virtual ~IOutputBufferManager() = default;

  /// Creates an output buffer for 'task' with the specified partitioning
  /// 'kind', number of destination partitions, and number of producing drivers.
  virtual void initializeTask(
      std::shared_ptr<Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers) = 0;

  /// Updates the number of output buffers for the task identified by 'taskId'.
  /// 'noMoreBuffers' signals that no additional buffers will be added. Returns
  /// true if a buffer exists for the given taskId, false otherwise.
  virtual bool updateOutputBuffers(
      const std::string& taskId,
      int numBuffers,
      bool noMoreBuffers) = 0;

  /// Removes the output buffer state associated with 'taskId'.
  virtual void removeTask(const std::string& taskId) = 0;

  /// Returns output buffer statistics for the task identified by 'taskId', or
  /// std::nullopt if no buffer exists for that task.
  virtual std::optional<OutputBuffer::Stats> stats(
      const std::string& taskId) = 0;

  /// Updates the number of producing drivers for grouped execution. Returns
  /// true if a buffer exists for the given taskId, false otherwise.
  virtual bool updateNumDrivers(
      const std::string& taskId,
      uint32_t newNumDrivers) = 0;

  /// Returns the memory utilization ratio (0.0 to 1.0) for the output buffer
  /// of the task identified by 'taskId'. Returns 0 if the task is not found.
  virtual double getUtilization(const std::string& taskId) = 0;

  /// Returns true if the output buffer for the task identified by 'taskId' is
  /// over-utilized and is blocking its producers. Returns false if the task is
  /// not found.
  virtual bool isOverutilized(const std::string& taskId) = 0;

  /// Returns a human-readable string representation of the output buffer state
  /// for the task identified by 'taskId'.
  virtual std::string toString(const std::string& taskId) = 0;
};

} // namespace facebook::velox::exec
