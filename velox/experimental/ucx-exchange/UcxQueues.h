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
#include <atomic>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h" // for the Stats structure
#include "velox/exec/Task.h"

namespace facebook::velox::ucx_exchange {

/// @brief  Callback function for getting data from the queues.
/// A nullptr indicates that there is no more data.
/// The remainingBytes vector contains the sizes for the
/// packed_columns elements remaining in the queue.
/// Uses shared_ptr to support broadcast mode where the same GPU data
/// is shared across multiple destination queues without copying.
using UcxDataAvailableCallback = std::function<void(
    std::shared_ptr<cudf::packed_columns> data,
    std::vector<int64_t> remainingBytes)>;

struct UcxDestinationTransferStats {
  int64_t bytesQueued{0};
  int64_t bytesInFlight{0};
  int64_t bytesReserved{0};
  int64_t retainedBytes{0};
  int64_t maxBytes{0};
  bool waitingForData{false};
};

struct UcxDataAvailable {
  UcxDataAvailableCallback callback{nullptr};
  std::shared_ptr<cudf::packed_columns> data;
  std::vector<int64_t> remainingBytes;

  void notify() {
    if (callback) {
      callback(std::move(data), std::move(remainingBytes));
    }
  }
};

/// @brief The UcxDestinationQueue stores cudf::packed_columns for a single
/// downstream task. The data is enqueued by one or more parallel
/// UcxPartitionedOutput operators and dequeued again by the
/// UcxExchangeServer. The UcxDestinationQueue corresponds to the
/// DestinationBuffer of Velox. In Ucx, no serialization/deserialization is
/// needed, only packing of data nor is the data segmented and re-assembled.
class UcxDestinationQueue {
 public:
  struct Stats {
    void recordEnqueue(const cudf::packed_columns* data);

    void recordDequeue(const cudf::packed_columns* data);

    // what has been queued
    int64_t bytesQueued{0};
    int64_t packedColumnsQueued{0};

    // What has left this destination queue but is still retained by a server
    // send or intra-node handoff.
    int64_t bytesInFlight{0};
    int64_t packedColumnsInFlight{0};

    // what has been dequeued
    int64_t bytesSent{0};
    int64_t packedColumnsSent{0};
  };

  /// @brief Enqueues the data to the back of the queue.
  /// @param data Corresponds to a RowVector
  void enqueueBack(std::shared_ptr<cudf::packed_columns> data);

  /// @brief Enqueues the data to the front of the queue. This is needed when
  /// a transfer fails.
  /// @param data
  void enqueueFront(std::shared_ptr<cudf::packed_columns> data);

  struct Data {
    std::shared_ptr<cudf::packed_columns> data;
    std::vector<int64_t> remainingBytes;
    /// Whether the result is returned immediately without invoking the `notify'
    /// callback.
    bool immediate{false};
  };

  /// @brief Removes the data from the front of the queue and transfers
  /// ownership to the caller. If there is no data, 'notify' is installed and it
  /// will be called when data becomes available. In this case, a nullptr is
  /// returned.
  [[nodiscard]] Data getData(UcxDataAvailableCallback notify);

  /// Removes all remaining data from the queue.
  void deleteResults();

  /// Returns and clears the notify callback, if any, along with arguments for
  /// the callback.
  UcxDataAvailable getAndClearNotify();

  /// Finishes this destination buffer, set finished stats.
  void finish();

  /// Returns the stats of this buffer.
  Stats stats() const;

  /// Returns bytes queued or in-flight for this destination.
  int64_t transferBytes() const;

  /// Returns true when a server has asked for data and is waiting for the next
  /// enqueue to satisfy that request.
  bool waitingForData() const;

  /// Marks bytes for this destination as no longer in-flight.
  void releaseInFlight(int64_t bytes, int64_t numPackedCols);

  std::string toString();

 private:
  void clearNotify();

  std::deque<std::shared_ptr<cudf::packed_columns>> queue_;
  UcxDataAvailableCallback notify_{nullptr};
  Stats stats_;
};

/// @brief The UcxOutputQueue manages all data coming from a single task that
/// are destined to one or more downstream sink tasks. The UcxOutputQueue uses
/// a vector of DestinationQueues, one for each destination. The UcxOutputQueue
/// is also responsible for tracking the number of drivers that produce data.
/// The number of drivers may change dynamically, so tracking this happens at
/// two levels:
/// - updateNumDrivers is used to track the number of drivers.
/// - noMoreData is called by each driver when the driver is done and has no
/// more data will be added.
class UcxOutputQueue : public std::enable_shared_from_this<UcxOutputQueue> {
 public:
  /// @brief Creates a new output queue for a data-producing task.
  /// @param taskId The id of the source task that produces the data
  /// @param numDestinations The number of destinations, i.e. the partitions.
  /// @param numDrivers The initial number of drivers.
  /// @param kind The output mode (partitioned, broadcast, etc.)
  UcxOutputQueue(
      std::shared_ptr<exec::Task> task,
      uint32_t numDestinations,
      uint32_t numDrivers,
      core::PartitionedOutputNode::Kind kind =
          core::PartitionedOutputNode::Kind::kPartitioned);

  /// @brief initializes an unitialized queue. This is needed in order to
  /// support delayed construction, i.e. if a "getData" arrives before the queue
  /// exists, the queue manager can create an unitialized queue just for the
  /// sake of storing the callback notification. The queue is then initialized
  /// later properly, and eventually the callback fires.
  /// @return True, if initialization was successful, i.e. the queue wasn't
  /// already initialized.
  bool initialize(
      std::shared_ptr<exec::Task> task,
      uint32_t numDestinations,
      uint32_t numDrivers,
      core::PartitionedOutputNode::Kind kind =
          core::PartitionedOutputNode::Kind::kPartitioned);

  core::PartitionedOutputNode::Kind kind() const {
    return kind_;
  }

  /// Returns true once task metadata has been published via initializeTask()
  /// (not just a placeholder created by early getData() calls).
  /// This is published before destination queue expansion completes so local
  /// UCX handshakes do not permanently fall back to the remote path.
  bool isInitialized() const {
    return initialized_.load(std::memory_order_acquire);
  }

  /// @brief When we understand the final number of split groups (for grouped
  /// execution only), we need to update the number of producing drivers here.
  void updateNumDrivers(uint32_t newNumDrivers);

  /// @brief Enqueues the data for the given destination. Currently, only
  /// partitioned output mode is supported where the number of destinations is
  /// fixed. Is is an error to provide a destination larger than the initial
  /// number of destinations. This will change in the future and if destination
  /// > numDestinations, then this will be dynamically adapted like it is done
  /// in OutputQueue.
  /// @param destination The destination, must be < numDestinations.
  /// @param data The data.
  /// @param numRows The number of rows in the data.
  void enqueue(
      int destination,
      std::unique_ptr<cudf::packed_columns> data,
      int32_t numRows,
      int64_t transferReservationBytes = 0);

  /// @brief Checks if the queue is over capacity and returns a future if so.
  /// Producers call this before accepting more input and after enqueueing a
  /// batch.
  /// @param future Output parameter - populated with a future if blocked.
  /// @return True if blocked (queue over capacity), false otherwise.
  bool checkBlocked(ContinueFuture* future);

  /// @brief Checks if queued/in-flight transfer bytes exceed the active
  /// producer's drain window. Unlike checkBlocked(), this intentionally ignores
  /// producer-side reserved bytes so the active producer can pause while
  /// holding a materialized partitioned table without deadlocking on its own
  /// reservation.
  bool checkTransferCapacity(
      int destination,
      int64_t maxBytes,
      ContinueFuture* future);

  /// @brief Reserves destination-local transfer capacity before the producer
  /// allocates a GPU payload for that destination.
  bool reserveTransferBytes(
      int destination,
      int64_t bytes,
      int64_t maxBytes,
      ContinueFuture* future);

  /// @brief Reserves capacity for a full contiguous_split payload. This uses a
  /// task-wide retained-byte budget plus a destination fairness budget, so
  /// receivers can build backlog for throughput without allowing unbounded GPU
  /// retention.
  bool reserveFullTransferBytes(
      int destination,
      int64_t bytes,
      ContinueFuture* future);

  /// @brief Blocks until the learned full-transfer retained-byte window has
  /// room. Does not reserve bytes.
  bool waitForFullTransferCapacity(int64_t bytes, ContinueFuture* future);

  /// @brief Releases destination-local transfer capacity.
  void releaseTransferReservation(int destination, int64_t bytes);

  /// Returns the destination-local byte admission window for the next GPU
  /// payload. The queue owns this feedback state so parallel producers share
  /// the same congestion signal for a destination.
  int64_t transferWindowBytes(
      int destination,
      int64_t baseBytes,
      int64_t normalBytes,
      int64_t maxBytes);

  /// Records a failed allocation/admission probe and reduces the destination
  /// admission window.
  void recordTransferCongestion(int destination, int64_t baseBytes);

  /// Records a larger payload demand that did not fit the current window. This
  /// is not congestion: it is a signal to probe the destination window upward.
  void recordTransferDemand(
      int destination,
      int64_t targetBytes,
      int64_t baseBytes,
      int64_t maxBytes);

  /// Records GPU allocation/admission pressure from the full contiguous_split
  /// path and lowers the learned retained-byte high watermark.
  void recordFullTransferCongestion();

  /// Returns transfer pressure for one destination. The snapshot is used by
  /// producers to tune their admission window without exposing the queue lock.
  UcxDestinationTransferStats transferStats(int destination);

  /// @brief Reserves producer-side bytes before GPU output materialization.
  /// Returns true and populates 'future' if accepting this reservation would
  /// exceed the retained-byte budget.
  bool reserveOutputBytes(int64_t bytes, ContinueFuture* future);

  /// @brief Releases a producer-side byte reservation.
  void releaseOutputReservation(int64_t bytes);

  /// @brief Releases bytes retained by an in-flight exchange transfer.
  void releaseInFlightBytes(
      int destination,
      int64_t bytes,
      int64_t numPackedCols);

  /// @brief Returns the data for the given destination through the callback
  /// function. If data is available, notify will be called immediately. If
  /// there is no data, 'notify' is installed and it will be called when data
  /// becomes available.
  void getData(int destination, UcxDataAvailableCallback notify);

  /// @brief Indicates that a driver is done and won't enqueue any more data.
  void noMoreData();

  /// @brief Updates the number of destination buffers. For broadcast mode,
  /// new destinations are backfilled with previously broadcast data.
  /// Modeled on OutputBuffer::updateOutputBuffers().
  void updateOutputBuffers(int numBuffers, bool noMoreBuffers);

  /// @brief Returns true if the OutputQueue is finished. Thread-safe.
  bool isFinished();

  /// @brief Same as isFinished but must only be called when owning the lock.
  bool isFinishedLocked();

  /// @brief Deletes all queued data and makes all subsequent getData requests
  /// for 'destination' return empty results.
  void deleteResults(int destination);

  /// Continues any possibly waiting producers. Called when the producer task
  /// has an error or is cancelled.
  void terminate();

  std::string toString();

  /// @brief The stats of this output queue are shoe-horned into the stats
  /// object of OutputBuffer. Since the OutputBuffer's stat object is part of
  /// the Task stats and eventually processed at the Presto layer, this is the
  /// least intrusive way to convey stats information. The stats info from the
  /// UcxDestinationQueue are omitted since also the DestinationBuffer's stats
  /// are never processed by Presto.
  exec::OutputBuffer::Stats stats();

 private:
  // Percentage of maxSize below which a blocked producer should
  // be unblocked.
  static constexpr int32_t kContinuePct = 90;

  // Methods that update the statistics.
  void updateStatsWithEnqueuedLocked(int64_t bytes, int64_t rows);

  // Moves queued bytes into the in-flight transfer accounting. The data has
  // left a destination queue, but UCX or the intra-node transfer registry still
  // owns a reference to the packed columns.
  void updateStatsWithDequeuedLocked(
      int64_t bytes,
      int64_t numPackedCols,
      std::vector<ContinuePromise>& promises);

  // updates the counters and returns promises if retained output bytes fall
  // below the continueSize_ low water mark. These promises then need to be
  // realized outside the lock.
  void updateStatsWithFreedLocked(
      int64_t bytes,
      int64_t numPackedCols,
      std::vector<ContinuePromise>& promises);

  void updateStatsWithSendCompleteLocked(
      int64_t bytes,
      int64_t numPackedCols,
      std::vector<ContinuePromise>& promises);

  void updateTotalQueuedBytesMsLocked();

  int64_t getAverageQueueTimeMsLocked() const;

  void maybeContinueProducersLocked(std::vector<ContinuePromise>& promises);

  int64_t retainedBytesLocked() const;

  int64_t producerBlockedBytesLocked() const;

  int64_t retainedPackedColumnsLocked() const;

  int64_t transferBytesLocked(int destination) const;

  int64_t transferReservedBytesLocked() const;

  int64_t retainedBytesWithTransferReservationsLocked() const;

  int64_t activeDestinationCountLocked() const;

  int64_t defaultFullTransferRetainedLimitLocked() const;

  int64_t fullTransferRetainedLimitLocked() const;

  void maybeGrowFullTransferRetainedLimitLocked(int64_t retainedBytes);

  // internal function that is called when all drivers are done.
  void noMoreDrivers();

  // If this is called due to a driver processed all its data (no more data),
  // we increment the number of finished drivers. If it is called due to us
  // updating the total number of drivers, we don't.
  void checkIfDone(bool oneDriverFinished);

  bool enqueuePartitionedOutputLocked(
      int destination,
      std::shared_ptr<cudf::packed_columns> data,
      std::vector<UcxDataAvailable>& dataAvailableCbs,
      int64_t transferReservationBytes);

  void releaseTransferReservationLocked(int destination, int64_t bytes);

  int64_t transferWindowBytesLocked(
      int destination,
      int64_t baseBytes,
      int64_t normalBytes,
      int64_t maxBytes);

  void collectTransferPromisesLocked(
      int destination,
      std::vector<ContinuePromise>& promises);

  void collectAllTransferPromisesLocked(std::vector<ContinuePromise>& promises);

  void enqueueBroadcastOutputLocked(
      std::shared_ptr<cudf::packed_columns> data,
      std::vector<UcxDataAvailable>& dataAvailableCbs);

  void enqueueArbitraryOutputLocked(
      std::shared_ptr<cudf::packed_columns> data,
      std::vector<UcxDataAvailable>& dataAvailableCbs);

  // Reference to the task that owns this UcxQueue.
  std::shared_ptr<exec::Task> task_{nullptr};

  // The output mode (partitioned, broadcast, etc.)
  core::PartitionedOutputNode::Kind kind_{
      core::PartitionedOutputNode::Kind::kPartitioned};

  // Set to true once task metadata is available. Lock-free readers
  // (canUseIntraNode) load with memory_order_acquire so kind_ and task_
  // written before the store are visible.
  std::atomic<bool> initialized_{false};

  // For broadcast: stores data for late-arriving destinations that need
  // backfill. Cleared once noMoreQueues_ is set.
  std::vector<std::shared_ptr<cudf::packed_columns>> dataToBroadcast_;

  // For arbitrary: shared pool of data that any consumer can pull from.
  std::deque<std::shared_ptr<cudf::packed_columns>> arbitraryBuffer_;

  // For arbitrary: round-robin index for distributing data to waiting
  // consumers.
  int32_t nextArbitraryLoadIndex_{0};

  /// If 'queuedBytes_' > 'maxSize_', each producer is blocked after adding
  /// data.
  uint64_t maxSize_{0};
  // When 'queuedBytes_' goes below 'continueSize_', blocked producers are
  // resumed.
  uint64_t continueSize_{0};

  // Total number of drivers expected to produce results. This number will
  // decrease in the end of grouped execution, when we understand the real
  // number of producer drivers (depending on the number of split groups).
  uint32_t numDrivers_{0};

  // If true, then we don't allow to add new destination buffers. This only
  // applies for non-partitioned output buffer type.
  bool noMoreQueues_{false};

  // For governing multi-threaded access.
  std::mutex mutex_;

  // One buffer per destination.
  std::vector<std::unique_ptr<UcxDestinationQueue>> queues_;

  // keep track of the number of drivers that have finished.
  uint32_t numFinished_{0};

  bool atEnd_ = false;

  // promises when buffer reached capacity and blocked further enqueueing.
  std::vector<ContinuePromise> promises_;

  // Promises for active producers waiting for queued/in-flight UCX transfer
  // bytes to drop on a specific destination. Kept separate from promises_
  // because producer-side reservations remain held while a partitioned batch is
  // partially drained.
  std::vector<std::vector<ContinuePromise>> transferPromises_;

  // Bytes admitted for destination-local transfer but not yet enqueued. This
  // prevents multiple producers from simultaneously allocating fast-path
  // payloads against the same apparent destination capacity.
  std::vector<int64_t> transferReservedBytes_;

  // Destination-local adaptive payload windows. These are logical admission
  // windows, not GPU-size constants: they grow when consumers are starved and
  // decay when retained/queued/in-flight pressure appears.
  std::vector<int64_t> transferWindowBytes_;

  // Learned task-wide retained-byte high watermark for full contiguous_split
  // payloads. Zero means use the initial window derived from the output buffer
  // size and number of active destinations. Before the first allocation or
  // admission pressure signal, the full split path may grow beyond this initial
  // window to probe available hardware headroom. After pressure, this becomes a
  // congestion window that gates later full-split reservations.
  int64_t fullTransferRetainedLimit_{0};
  bool fullTransferCongested_{false};

  // Bytes reserved by producers before GPU materialization is visible in the
  // destination queues.
  int64_t reservedBytes_{0};

  // Bytes retained by destination queues.
  int64_t queuedBytes_{0};
  int64_t queuedPackedColumns_{0};

  // Bytes already dequeued by servers but still retained by UCX sends or the
  // intra-node transfer registry.
  int64_t inFlightBytes_{0};
  int64_t inFlightPackedColumns_{0};

  // The total number of bytes/rows/packedColumns sent via this output queue.
  int64_t totalBytesSent_{0};
  int64_t totalRowsSent_{0};
  int64_t totalPackedColumnsSent_{0};

  // Time since last change in queuedBytes_. Used to compute total time data
  // is queued. Ignored if queuedBytes_ is zero.
  uint64_t queueStartMs_{0};

  // Total time data is queued as bytes * time.
  double totalQueuedBytesMs_{0};
};

} // namespace facebook::velox::ucx_exchange
