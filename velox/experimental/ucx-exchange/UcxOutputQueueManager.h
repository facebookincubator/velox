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
#include <velox/exec/Task.h>
#include <functional>
#include <unordered_set>
#include "velox/experimental/ucx-exchange/UcxQueues.h"

namespace facebook::velox::ucx_exchange {

class UcxOutputQueueManager {
 public:
  /// Factory method to retrieve a reference to the output queue manager.
  static std::shared_ptr<UcxOutputQueueManager> getInstanceRef();

  // no constructor to prevent direct instantiation.
  UcxOutputQueueManager() = default;
  // no copy constructor.
  UcxOutputQueueManager(const UcxOutputQueueManager&) = delete;
  // no copy assignment.
  UcxOutputQueueManager& operator=(const UcxOutputQueueManager&) = delete;

  /// @brief Initializes a task and creates the corresponding output queues that
  /// are associated with this task.
  /// @param task The task.
  /// @param kind The output mode (partitioned, broadcast, etc.)
  /// @param numDestinations The number of queues (destinations or partitions)
  /// associated with this task.
  /// @param numDrivers The number of drivers that contribute data to these
  /// queues. Used to recognize when the queues are complete.
  void initializeTask(
      std::shared_ptr<exec::Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers);

  /// @brief Updates the number of destination buffers for a task.
  /// For broadcast mode, new destinations are backfilled with previously
  /// broadcast data.
  void updateOutputBuffers(
      const std::string& taskId,
      int numBuffers,
      bool noMoreBuffers);

  /// @brief Enqueues a cudf packed column into the queue.
  /// @param taskId The unique task Id.
  /// @param destination The destination (partition, queue number) into which
  /// the data is queued.
  /// @param txData The data to enqueue.
  /// @param numRows The number of rows in the data.
  void enqueue(
      const std::string& taskId,
      int destination,
      std::unique_ptr<cudf::packed_columns> txData,
      int32_t numRows);

  /// @brief Checks if the queue for a task is over capacity.
  /// Should be called after enqueueing all partitions for a batch.
  /// @param taskId The unique task Id.
  /// @param future Output parameter - populated with a future if blocked.
  /// @return True if blocked (queue over capacity), false otherwise.
  bool checkBlocked(const std::string& taskId, ContinueFuture* future);

  /// @brief Indicates that no more data will be coming for this task.
  void noMoreData(const std::string& taskId);

  /// @returns true if noMoreData has been called and all the accumulated data
  /// have been fetched and acknowledged.
  bool isFinished(const std::string& taskId);

  /// @brief
  void deleteResults(const std::string& taskId, int destination);

  /// @brief Asynchronously returns the head of the queue. If data is available,
  /// the callback function is triggered immediately and true is returned.
  /// Otherwise, the callback function is registered and called once data is
  /// available. If the queue is closed and no more data is available, then the
  /// callback function is called immediately with a nullptr. If the destination
  /// doesn't exist, then additional queues will be added for this task.
  /// @param taskId The unique taskId.
  /// @param destination The destination.
  /// @param notify The callback function.
  void getData(
      const std::string& taskId,
      int destination,
      UcxDataAvailableCallback notify);

  /// Returns true if the given task can use intra-node transfer.
  /// Returns false if the task is not yet initialized (placeholder queue
  /// from early sink connections) or if the task uses broadcast mode
  /// (broadcast shares packed_columns across destinations — the intra-node
  /// source's destructive move would corrupt data for other servers).
  bool canUseIntraNode(const std::string& taskId);

  /// @brief Removes the queue for the given task from the queue manager.
  /// Calls "terminate" on the queue to awake waiting producers.
  void removeTask(const std::string& taskId);

  /// @brief Returns the queue statistics of the queue associated with the given
  /// task. Returns nullopt when the specified output queue doesn't exist.
  std::optional<exec::OutputBuffer::Stats> stats(const std::string& taskId);

 private:
  // Retrieves the queue for a task if it exists.
  // Returns NULL if task not found.
  std::shared_ptr<UcxOutputQueue> getQueueIfExists(const std::string& taskId);

  // Throws an exception if queue doesn't exist.
  std::shared_ptr<UcxOutputQueue> getQueue(const std::string& taskId);

  folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<UcxOutputQueue>>,
      std::mutex>
      queues_;

  // Tasks that have been removed via removeTask(). Prevents getData() from
  // re-creating placeholder queues for tasks that are already dead, which
  // would cause crashes when deleteResults() is called with destinations
  // that exceed the placeholder's undersized queues_ vector.
  folly::Synchronized<std::unordered_set<std::string>, std::mutex>
      removedTasks_;
};

} // namespace facebook::velox::ucx_exchange
