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

#include <cudf/contiguous_split.hpp>
#include <velox/exec/OutputBufferManager.h>
#include <velox/exec/Task.h>
#include <functional>
#include <string_view>
#include <unordered_map>
#include "velox/experimental/ucx-exchange/UcxQueues.h"

namespace facebook::velox::ucx_exchange {

class UcxOutputQueueManager : public exec::OutputBufferManager {
 public:
  /// Returns the process-wide output queue manager used by UCX producers and
  /// exchange servers.
  static const std::shared_ptr<UcxOutputQueueManager>& getInstanceRef();

  // Public to match DefaultOutputBufferManager and support direct unit tests.
  // Production UcxPartitionedOutput operators require getInstanceRef().
  UcxOutputQueueManager() = default;
  // no copy constructor.
  UcxOutputQueueManager(const UcxOutputQueueManager&) = delete;
  // no copy assignment.
  UcxOutputQueueManager& operator=(const UcxOutputQueueManager&) = delete;

  /// @brief Initializes a task and creates the corresponding output queues that
  /// are associated with this task. Initializes the task's Pending entry in
  /// place, or a new one if the task ID is absent or Removed. If the
  /// initialization throws, the entry stays Pending and, once removed, lives as
  /// long as 'task'.
  /// @param task The task.
  /// @param kind The output mode (partitioned, broadcast, etc.)
  /// @param numDestinations The number of queues (destinations or partitions)
  /// associated with this task.
  /// @param numDrivers The number of drivers that contribute data to these
  /// queues. Used to recognize when the queues are complete.
  /// @param transportOptions Opaque per-node transport configuration carried on
  /// the PartitionedOutputNode. Unused: UCX takes its settings from CudfConfig
  /// and the communicator, so there is nothing node-scoped to apply yet.
  void initializeTask(
      std::shared_ptr<exec::Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers,
      const std::string& transportOptions = {}) override;

  /// @brief Updates the number of destination buffers for a task.
  /// For broadcast mode, new destinations are backfilled with previously
  /// broadcast data.
  bool updateOutputBuffers(
      const std::string& taskId,
      int numBuffers,
      bool noMoreBuffers) override;

  /// @brief Enqueues a cudf packed column into the queue.
  /// @param taskId The unique task Id.
  /// @param destination The destination (partition, queue number) into which
  /// the data is queued.
  /// @param txData The data to enqueue.
  /// @param numRows The number of rows in the data. Supplied by the producer
  /// rather than read back from 'txData', which cannot report a row count once
  /// it has no columns.
  void enqueue(
      std::string_view taskId,
      int destination,
      std::unique_ptr<cudf::packed_columns> txData,
      vector_size_t numRows);

  /// @brief Checks if the queue for a task is over capacity.
  /// Should be called after enqueueing all partitions for a batch.
  /// @param taskId The unique task Id.
  /// @param future Output parameter - populated with a future if blocked.
  /// @return True if blocked (queue over capacity), false otherwise.
  bool checkBlocked(std::string_view taskId, ContinueFuture* future);

  /// @brief Indicates that no more data will be coming for this task.
  void noMoreData(std::string_view taskId);

  /// @returns true if noMoreData has been called and all the accumulated data
  /// have been fetched and acknowledged.
  bool isFinished(std::string_view taskId);

  /// @brief
  void deleteResults(std::string_view taskId, int destination);

  /// @brief For tests: bindReader() followed by UcxOutputQueue::getData().
  /// Never releases the reader it binds.
  /// @param taskId The unique taskId.
  /// @param destination The destination.
  /// @param notify The callback function.
  void getData(std::string_view taskId, int destination, ReaderCallback notify);

  /// Finds or creates the entry for 'taskId' and counts a bound reader. Never
  /// returns null: a Removed entry is returned as is. A server binds once and
  /// sends every later call to the entry, so that it never reaches a task that
  /// reuses the ID. Registers 'onReleased', if not null, with
  /// UcxOutputQueue::setOnReleased(). Does not throw, since it runs inside a
  /// UCX callback.
  std::shared_ptr<UcxOutputQueue> bindReader(
      std::string_view taskId,
      int destination,
      std::function<void()> onReleased);

  /// Releases 'destination' of task 'taskId' because its consumer sent the
  /// close message or its endpoint closed. Drops a close for an unknown task.
  void consumerGone(std::string_view taskId, int destination);

  /// Unbinds a reader. Erases a Pending entry that has no bound reader left and
  /// no Released destination. Does not throw.
  void releaseReader(
      const std::shared_ptr<UcxOutputQueue>& entry,
      int destination);

  /// Returns true if the given task can use intra-node transfer.
  /// Returns false if the task is not yet initialized (a Pending entry
  /// created by early sink connections) or if the task uses broadcast mode
  /// (broadcast shares packed_columns across destinations — the intra-node
  /// source's destructive move would corrupt data for other servers).
  bool canUseIntraNode(std::string_view taskId);

  /// Fails the producer task 'taskId' with 'error' if its entry is Active on
  /// this worker; otherwise only logs 'error'. Calls
  /// Task::setError() with no lock held. Task::setError() terminates the task
  /// synchronously, so this must not be called inside a UCX callback.
  void failTask(std::string_view taskId, std::string_view error);

  /// @brief Makes the task's entry Removed, creating it if absent, so that a
  /// later reader finds the removal. Reads the task's final state and error
  /// and passes them to UcxOutputQueue::remove(), which decides every
  /// destination's end. Erases the other Removed entries whose Task is gone and
  /// that have no bound reader.
  void removeTask(const std::string& taskId) override;

  /// @brief Returns the queue statistics of the queue associated with the given
  /// task. Returns nullopt when the specified output queue doesn't exist.
  std::optional<exec::OutputBuffer::Stats> stats(
      const std::string& taskId) override;

  bool updateNumDrivers(const std::string& taskId, uint32_t newNumDrivers)
      override;

  std::optional<double> getUtilization(const std::string& taskId) override;

  std::optional<bool> isOverutilized(const std::string& taskId) override;

  std::string toString(const std::string& taskId) override;

 private:
  // Returns the Pending or Active entry of a task, or null. Every method that
  // Velox calls looks up through it, so that a Removed task is absent to Velox.
  std::shared_ptr<UcxOutputQueue> getQueueIfExists(std::string_view taskId);

  // Same as getQueueIfExists(), but throws if there is no such entry.
  std::shared_ptr<UcxOutputQueue> getQueue(std::string_view taskId);

  // One entry per task ID, from the first contact to its erasure. The lock
  // only adds and erases entries; it is taken before an entry's lock.
  folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<UcxOutputQueue>>,
      std::mutex>
      entries_;
};

} // namespace facebook::velox::ucx_exchange
