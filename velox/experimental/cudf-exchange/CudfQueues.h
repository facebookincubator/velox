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
#include <deque>
#include <functional>
#include <memory>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h"

namespace facebook::velox::cudf_exchange {

/// @brief  Callback function for getting data from the queues.
/// A nullptr indicates that there is no more data.
/// The remainingBytes vector contains the sizes for the
/// packed_columns elements remaining in the queue.
using CudfDataAvailableCallback = std::function<void(
    std::unique_ptr<cudf::packed_columns> data,
    std::vector<int64_t> remainingBytes)>;

struct CudfDataAvailable {
  CudfDataAvailableCallback callback{nullptr};
  std::unique_ptr<cudf::packed_columns> data;
  std::vector<int64_t> remainingBytes;

  void notify() {
    if (callback) {
      callback(std::move(data), remainingBytes);
    }
  }
};

/// @brief The CudfDestinationQueue stores cudf::packed_columns for a single
/// downstream task. The data is enqueued by one or more parallel
/// CudfPartitionedOutput operators and dequeued again by the
/// CudfExchangeServer. The CudfDestinationQueue corresponds to the
/// DestinationBuffer of Velox. In Cudf, no serialization/deserialization is
/// needed, only packing of data nor is the data segmented and re-assembled.
class CudfDestinationQueue {
 public:
  struct Stats {
    void recordEnqueue(const cudf::packed_columns* data);

    void recordDequeue(const cudf::packed_columns* data);

    bool finished{false};

    // what has been queued
    int64_t bytesQueued{0};
    int64_t rowsQueued{0};
    int64_t packedColumnsQueued{0};

    // what has been dequeued
    int64_t bytesSent{0};
    int64_t rowsSent{0};
    int64_t packedColumnsSent{0};
  };

  /// @brief Enqueues the data to the back of the queue.
  /// @param data Corresponds to a RowVector
  void enqueueBack(std::unique_ptr<cudf::packed_columns> data);

  /// @brief Enqueues the data to the front of the queue. This is needed when
  /// a transfer fails.
  /// @param data
  void enqueueFront(std::unique_ptr<cudf::packed_columns> data);

  struct Data {
    std::unique_ptr<cudf::packed_columns> data;
    std::vector<int64_t> remainingBytes;
    /// Whether the result is returned immediately without invoking the `notify'
    /// callback.
    bool immediate{false};
  };

  /// @brief Removes the data from the front of the queue and transfers
  /// ownership to the caller. If there is no data, 'notify' is installed and it
  /// will be called when data becomes available. In this case, a nullptr is
  /// returned.
  [[nodiscard]] Data getData(CudfDataAvailableCallback notify);

  /// Removes all remaining data from the queue and returns the removed data.
  std::vector<std::unique_ptr<cudf::packed_columns>> deleteResults();

  /// Returns and clears the notify callback, if any, along with arguments for
  /// the callback.
  CudfDataAvailable getAndClearNotify();

  /// Finishes this destination buffer, set finished stats.
  void finish();

  /// Returns the stats of this buffer.
  Stats stats() const;

  std::string toString();

 private:
  void clearNotify();

  std::deque<std::unique_ptr<cudf::packed_columns>> queue_;
  CudfDataAvailableCallback notify_{nullptr};
  Stats stats_;
};

/// @brief The CudfOutputQueue manages all data coming from a single task that
/// are destined to one or more downstream sink tasks. The CudfOutputQueue uses
/// a vector of DestinationQueues, one for each destination. The CudfOutputQueue
/// is also responsible for tracking the number of drivers that produce data.
/// The number of drivers may change dynamically, so tracking this happens at
/// two levels:
/// - updateNumDrivers is used to track the number of drivers.
/// - noMoreData is called by each driver when the driver is done and has no
/// more data will be added.
class CudfOutputQueue {
 public:
  struct Stats {
    /// TODO: Fill in the stats and update the counters
  };

  /// @brief Creates a new output queue for a data-producing task.
  /// @param taskId The id of the source task that produces the data
  /// @param numDestinations The number of destinations, i.e. the partitions.
  /// @param numDrivers The initial number of drivers.
  CudfOutputQueue(
      std::shared_ptr<exec::Task> task,
      int numDestinations,
      uint32_t numDrivers);

  /// @brief initializes an unitialized queue. This is needed in order to
  /// support delayed construction, i.e. if a "getData" arrives before the queue
  /// exists, the queue manager can create an unitialized queue just for the
  /// sake of storing the callback notification. The queue is then initialized
  /// later properly, and eventually the callback fires.
  /// @return True, if initialization was successful, i.e. the queue wasn't
  /// already initialized.
  bool initialize(
      std::shared_ptr<exec::Task> task,
      int numDestinations,
      uint32_t numDrivers);

  core::PartitionedOutputNode::Kind kind() const {
    // TODO: Need to support the other modes as well
    return core::PartitionedOutputNode::Kind::kPartitioned;
  }

  /// @brief When we understand the final number of split groups (for grouped
  /// execution only), we need to update the number of producing drivers here.
  void updateNumDrivers(uint32_t newNumDrivers);

  /// @brief Enqueues the data for the given destination. Currently, only
  /// partitioned output mode is supported where the number of destinations is
  /// fixed. Is is an error to provide a destination larger than the initial
  /// number of destinations. This will change in the future and if destination
  /// > numDestinations, then this will be dynamically adapted like it is done
  /// in OutputBuffer.
  /// @param destination The destination, must be < numDestinations.
  /// @param data The data.
  /// @param future Currently not used since no max queue size is enforced.
  /// @return Currently always false, meaning "not blocked".
  bool enqueue(
      int destination,
      std::unique_ptr<cudf::packed_columns> data,
      ContinueFuture* future);

  /// @brief Returns the data for the given destination through the callback
  /// function. If data is available, notify will be called immediately. If
  /// there is no data, 'notify' is installed and it will be called when data
  /// becomes available.
  void getData(int destination, CudfDataAvailableCallback notify);

  /// @brief Indicates that a driver is done and won't enqueue any more data.
  void noMoreData();

  /// @brief Returns true if the OutputBuffer is finished. Thread-safe.
  bool isFinished();

  /// @brief Same as isFinished but must only be called when owning the lock.
  bool isFinishedLocked();

  /// @brief Deletes all buffered data and makes all subsequent getData requests
  /// for 'destination' return empty results. Returns true if all destinations
  /// are deleted, meaning that the buffer is fully consumed and the producer
  /// can be marked finished and the buffers freed.
  bool deleteResults(int destination);

  /// Continues any possibly waiting producers. Called when the producer task
  /// has an error or is cancelled.
  void terminate();

  std::string toString();

 private:
  // internal function that is called when all drivers are done.
  void noMoreDrivers();

  // If this is called due to a driver processed all its data (no more data),
  // we increment the number of finished drivers. If it is called due to us
  // updating the total number of drivers, we don't.
  void checkIfDone(bool oneDriverFinished);

  void enqueuePartitionedOutputLocked(
      int destination,
      std::unique_ptr<cudf::packed_columns> data,
      std::vector<CudfDataAvailable>& dataAvailableCbs);

  // Reference to the task that owns this CudfQueue.
  std::shared_ptr<exec::Task> task_{nullptr};

  // Total number of drivers expected to produce results. This number will
  // decrease in the end of grouped execution, when we understand the real
  // number of producer drivers (depending on the number of split groups).
  uint32_t numDrivers_{0};

  // If true, then we don't allow to add new destination buffers. This only
  // applies for non-partitioned output buffer type.
  bool noMoreBuffers_{false};

  // For governing multi-threaded access.
  std::mutex mutex_;

  // One buffer per destination.
  std::vector<std::unique_ptr<CudfDestinationQueue>> queues_;

  // keep track of the number of drivers that have finished.
  uint32_t numFinished_{0};

  bool atEnd_ = false;
};

} // namespace facebook::velox::cudf_exchange
