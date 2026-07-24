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
#include "velox/exec/OutputBufferStats.h"

namespace facebook::velox::exec {

class Task;

/// Abstract interface for the output buffer that delivers a Task's partitioned
/// output to downstream consumers. Implementations are registered in
/// OutputTransportRegistry and resolved per query.
///
/// Model: a task has a single output buffer holding one destination buffer per
/// downstream consumer -- 'numDestinations' of them. For kPartitioned output
/// the destination count is fixed at initializeTask(); for broadcast and
/// arbitrary output consumers register while the task runs, so the count grows
/// via updateOutputBuffers(). 'numDrivers' is how many producing drivers feed
/// the buffer.
///
/// Covers only the control plane (task lifecycle) and observability. The data
/// plane (enqueue / fetch / acknowledge / delete) stays on the concrete
/// managers because payloads are transport-specific -- serialized pages for the
/// in-memory transport, GPU buffers for UCX -- and is driven by the matching
/// output operator, not by Task.
///
/// Lifecycle: initializeTask() exactly once per task and before any other call
/// for it, then updateOutputBuffers() / updateNumDrivers() and the
/// observability methods while it runs, then removeTask() once at termination.
/// The update and observability methods tolerate an unknown task (returning
/// false / nullopt), so callers need not race teardown; only the producer must
/// have initialized before it enqueues. Skipping removeTask() leaks the buffer
/// -- and the Task it pins -- until the manager is destroyed.
///
/// Implementations must honor two contracts:
///
/// - Thread safety: a single instance serves many tasks at once, and each task
///   drives it from all of its driver threads, so every method must be safe
///   under concurrent calls, including concurrent calls for the same taskId.
///
/// - Lifetime: Task and the output operator hold only weak_ptrs and lock() per
///   use, which breaks the Task <-> OutputBuffer ownership cycle. What keeps an
///   instance alive is its OutputTransportRegistry entry -- or, for the
///   built-in default, a process-wide singleton -- so an implementation must
///   stay registered until every task it initialized has finished.
class OutputBufferManager {
 public:
  virtual ~OutputBufferManager() = default;

  // Lifecycle.

  /// Creates the task's output buffer. Must be called exactly once per task and
  /// before any other method for it; a second call for the same task is an
  /// error. 'kind' selects the buffer semantics; 'numDestinations' is the
  /// initial destination-buffer count (fixed for kPartitioned, a starting point
  /// updateOutputBuffers() grows for broadcast / arbitrary); 'numDrivers' is
  /// the producing-driver count whose completion marks the output done (see
  /// updateNumDrivers()).
  virtual void initializeTask(
      std::shared_ptr<Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers) = 0;

  /// Publishes the destination-buffer count as consumers register, finalizing
  /// it when 'noMoreBuffers' is true. Returns false if the task has no output
  /// buffer. Contract:
  ///  - kPartitioned: the count is fixed at initializeTask(); this only asserts
  ///    'numDestinations' matches and requires 'noMoreBuffers' == true.
  ///  - broadcast / arbitrary: 'numDestinations' is monotonically
  ///    non-decreasing; a value <= the current count is ignored, not an error.
  ///  - 'noMoreBuffers' is terminal: once set the count is frozen and adding
  ///    destinations afterwards is an error.
  virtual bool updateOutputBuffers(
      const std::string& taskId,
      int numDestinations,
      bool noMoreBuffers) = 0;

  /// Sets the absolute number of producing drivers feeding this task's output
  /// to 'newNumDrivers' (grouped execution learns the total only after all
  /// split groups are seen). Returns false if the task has no output buffer.
  /// This sets only the target count; how the manager learns a driver finished
  /// and decides the output is complete is a data-plane detail of the
  /// implementation, not part of this interface.
  virtual bool updateNumDrivers(
      const std::string& taskId,
      uint32_t newNumDrivers) = 0;

  /// Releases the task's output buffer and the Task reference it holds. Call
  /// once at termination; skipping it leaks the buffer until the manager is
  /// destroyed.
  virtual void removeTask(const std::string& taskId) = 0;

  // Observability.

  /// Stats for 'taskId', or nullopt if it has no output buffer.
  /// OutputBufferStats is transport-neutral (bytes / rows / pages);
  /// every transport maps its own accounting onto it.
  virtual std::optional<OutputBufferStats> stats(const std::string& taskId) = 0;

  /// Output-buffer memory utilization as buffered bytes / capacity, where
  /// capacity is the task's configured max output buffer size; nullopt if
  /// 'taskId' is unknown or the transport has no bounded capacity.
  /// Reported for observability; it does not gate producers by itself.
  virtual std::optional<double> getUtilization(const std::string& taskId) = 0;

  /// Whether the output buffer is over-utilized: filled enough to risk soon
  /// reaching capacity and back-pressuring its producers, though it is not
  /// blocking them yet. The threshold is implementation-defined. nullopt if
  /// 'taskId' is unknown. Consumed to drive dynamic consumer scaling, e.g.
  /// adding TableWriter tasks.
  virtual std::optional<bool> isOverutilized(const std::string& taskId) = 0;

  /// Human-readable dump of the task's output buffer state.
  virtual std::string toString(const std::string& taskId) = 0;
};

} // namespace facebook::velox::exec
