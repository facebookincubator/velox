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
#include <folly/stop_watch.h>
#include <atomic>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h" // for the Stats structure
#include "velox/exec/Task.h"

namespace facebook::velox::ucx_exchange {

/// @brief  Callback function for getting data from the queues.
/// A nullptr indicates that there is no more data.
/// 'numRows' is the logical row count of 'data', supplied by the producer
/// because a packed table with no columns cannot report one: cuDF derives
/// num_rows() from the columns. It is 0 for the end-of-stream marker.
/// The remainingBytes vector contains the sizes for the
/// packed_columns elements remaining in the queue.
/// Uses shared_ptr to support broadcast mode where the same GPU data
/// is shared across multiple destination queues without copying.
using UcxDataAvailableCallback = std::function<void(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    std::vector<int64_t> remainingBytes)>;

struct UcxDataAvailable {
  UcxDataAvailableCallback callback{nullptr};
  std::shared_ptr<cudf::packed_columns> data;
  vector_size_t numRows{0};
  std::vector<int64_t> remainingBytes;

  void notify() {
    if (callback) {
      callback(std::move(data), numRows, std::move(remainingBytes));
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
  /// @param numRows The logical row count of 'data'
  void enqueueBack(
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows);

  /// @brief Enqueues the data to the front of the queue. This is needed when
  /// a transfer fails.
  /// @param data
  /// @param numRows The logical row count of 'data'
  void enqueueFront(
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows);

  struct Data {
    std::shared_ptr<cudf::packed_columns> data;
    vector_size_t numRows{0};
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

  /// Returns true when a server is waiting for the next payload.
  bool waitingForData() const;

  /// Marks bytes as no longer retained by an exchange transfer.
  void releaseInFlight(int64_t bytes, int64_t numPackedColumns);

  std::string toString();

 private:
  friend class UcxOutputQueue;

  /// Rolls back the most recent non-null enqueue. Used only by broadcast's
  /// multi-destination commit before any callback is exposed.
  void rollbackEnqueueBack();

  void clearNotify();

  // A queued page and the logical row count the producer gave it. Paired here
  // rather than recovered from the page, which reports no rows when it has no
  // columns.
  struct QueuedPage {
    std::shared_ptr<cudf::packed_columns> data;
    vector_size_t numRows{0};
  };

  std::deque<QueuedPage> queue_;
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
  /// @param numRows The number of rows in the data. Supplied by the producer
  /// rather than read back from 'data', which cannot report a row count once it
  /// has no columns.
  void enqueue(
      int destination,
      std::unique_ptr<cudf::packed_columns> data,
      vector_size_t numRows,
      int64_t transferReservationBytes = 0);

  /// @brief Checks if the queue is over capacity and returns a future if so.
  /// This should be called after enqueueing all partitions for a batch.
  /// @param future Output parameter - populated with a future if blocked.
  /// @return True if blocked (queue over capacity), false otherwise.
  bool checkBlocked(ContinueFuture* future);

  /// Checks whether queued, in-flight and reserved transfer bytes for a
  /// destination have filled the producer's active drain window.
  bool checkTransferCapacity(
      int destination,
      int64_t maxBytes,
      ContinueFuture* future);

  /// Reserves destination-local capacity before materializing a GPU payload.
  /// Returns true when blocked and, when supplied, installs a future that is
  /// fulfilled when that destination makes progress.
  bool reserveTransferBytes(
      int destination,
      int64_t bytes,
      int64_t maxBytes,
      ContinueFuture* future);

  /// Reserves a full contiguous-split payload against the learned task-wide
  /// retained-byte limit and the destination's fair share.
  bool reserveFullTransferBytes(
      int destination,
      int64_t bytes,
      ContinueFuture* future);

  /// Waits for room below the learned full-transfer retained-byte limit. This
  /// does not reserve bytes.
  bool waitForFullTransferCapacity(int64_t bytes, ContinueFuture* future);

  /// Releases a destination-local materialization reservation.
  void releaseTransferReservation(int destination, int64_t bytes);

  /// Returns the shared adaptive transfer window for a destination.
  int64_t transferWindowBytes(
      int destination,
      int64_t baseBytes,
      int64_t normalBytes,
      int64_t maxBytes);

  /// Applies multiplicative decrease after allocation or admission pressure.
  void recordTransferCongestion(int destination, int64_t baseBytes);

  /// Applies additive/probe growth when a payload needs a larger window.
  void recordTransferDemand(
      int destination,
      int64_t targetBytes,
      int64_t baseBytes,
      int64_t maxBytes);

  /// Lowers the task-wide retained-byte limit after full-split pressure.
  void recordFullTransferCongestion();

  /// Releases bytes after UCXX completion or intra-node retrieval.
  void releaseInFlightBytes(
      int destination,
      int64_t bytes,
      int64_t numPackedColumns);

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

  /// @brief Retained queued and in-flight bytes over this queue's byte
  /// capacity, the task's configured maxOutputBufferSize. Producers are only
  /// blocked after adding data (see checkBlocked), so the ratio can exceed
  /// 1.0. Returns nullopt while the
  /// capacity is still unknown, which is the case for a placeholder queue
  /// created by a getData() that arrived before initializeTask() supplied the
  /// task's query config.
  std::optional<double> getUtilization();

  /// @brief Whether enough data is retained to risk back-pressuring producers
  /// soon, or the last data has been seen. Half the capacity is the threshold,
  /// matching exec::OutputBuffer. Returns nullopt while capacity is unknown,
  /// same reason as getUtilization(): without a capacity there is no ratio to
  /// compare against, and reporting `false` there would let a placeholder queue
  /// masquerade as having spare room.
  std::optional<bool> isOverutilized();

 private:
  // Percentage of maxSize below which a blocked producer should
  // be unblocked.
  static constexpr int32_t kContinuePct = 90;

  /// Grows all per-destination vectors together. For broadcast, a new
  /// destination is backfilled before a possible end marker is appended.
  void addOutputBuffersLocked(int numBuffers);

  // Methods that update the statistics.
  void updateStatsWithEnqueuedLocked(int64_t bytes, int64_t rows);

  /// Moves queue ownership into the server's in-flight accounting.
  void updateStatsWithDequeuedLocked(
      int64_t bytes,
      int64_t numPackedColumns,
      std::vector<ContinuePromise>& promises);

  // updates the counters and returns promises if the queuedBytes_ counter falls
  // below the continueSize_ low water mark. These promises then need to be
  // realized outside the lock.
  void updateStatsWithFreedLocked(
      int64_t bytes,
      int64_t numPackedCols,
      std::vector<ContinuePromise>& promises);

  void updateStatsWithSendCompleteLocked(
      int64_t bytes,
      int64_t numPackedColumns,
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
      vector_size_t numRows,
      std::vector<UcxDataAvailable>& dataAvailableCbs);

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
      vector_size_t numRows,
      std::vector<UcxDataAvailable>& dataAvailableCbs);

  /// Releases the extra retained reference used to backfill future broadcast
  /// destinations and wakes producers if it was their limiting retention.
  void clearBroadcastHistoryLocked(std::vector<ContinuePromise>& promises);

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
  // Paired with the row count for the same reason QueuedPage is.
  std::vector<std::pair<std::shared_ptr<cudf::packed_columns>, vector_size_t>>
      dataToBroadcast_;
  // Conservative one-copy accounting for dataToBroadcast_. Destination queue
  // accounting alone reaches zero after current sends complete even though
  // these GPU buffers remain resident for late broadcast destinations.
  int64_t broadcastHistoryBytes_{0};
  int64_t broadcastHistoryPackedColumns_{0};

  /// If retained queued and in-flight bytes reach 'maxSize_', each producer is
  /// blocked after adding data.
  uint64_t maxSize_{0};
  // When retained bytes go below 'continueSize_', blocked producers resume.
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

  // Destination-local waiters, materialization reservations, and adaptive
  // windows. These vectors always grow in lockstep with queues_.
  std::vector<std::vector<ContinuePromise>> transferPromises_;
  std::vector<int64_t> transferReservedBytes_;
  std::vector<int64_t> transferWindowBytes_;

  // Learned retained-byte congestion window for full contiguous-split
  // materialization. Zero means use the default derived from maxSize_.
  int64_t fullTransferRetainedLimit_{0};
  bool fullTransferCongested_{false};

  // Set when terminate() has cancelled all outstanding reservations. Late
  // cancellation cleanup becomes a no-op instead of looking like underflow.
  bool terminated_{false};

  // Payloads still resident in destination queues.
  int64_t queuedBytes_{0};
  int64_t queuedPackedColumns_{0};

  // Payloads dequeued by servers but retained by UCXX or the intra-node
  // registry until exact completion.
  int64_t inFlightBytes_{0};
  int64_t inFlightPackedColumns_{0};

  // The total number of bytes/rows/packedColumns sent via this output queue.
  int64_t totalBytesSent_{0};
  int64_t totalRowsSent_{0};
  int64_t totalPackedColumnsSent_{0};

  // Restarted at each change in queuedBytes_. Used to compute total time data
  // is queued. Ignored if queuedBytes_ is zero.
  folly::stop_watch<std::chrono::milliseconds> queueWatch_;

  // Total time data is queued as bytes * time.
  double totalQueuedBytesMs_{0};
};

} // namespace facebook::velox::ucx_exchange
