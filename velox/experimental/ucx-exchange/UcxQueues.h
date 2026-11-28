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
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h" // for the Stats structure
#include "velox/exec/Task.h"
#include "velox/experimental/ucx-exchange/UcxChannelState.h"

namespace facebook::velox::ucx_exchange {

/// @brief One step that a reader takes from the queues: a chunk, the end of the
/// channel, or a stop.
/// The chunk is a shared_ptr to support broadcast mode, where the same GPU data
/// is shared across multiple destination queues without copying.
struct ReaderStep {
  /// What a step carries.
  enum class Kind : uint8_t {
    /// 'data', 'numRows' and 'remainingBytes' are set.
    kChunk,
    /// No more chunks follow. 'outcome' says how the channel ended.
    kEnd,
    /// The reader must finish without sending anything.
    kStop,
  };

  /// Which of the fields below are set. Build steps through the factories.
  Kind kind;
  /// The chunk. kChunk only. It is not named 'chunk', which the factory takes.
  std::shared_ptr<cudf::packed_columns> data;
  /// The logical row count of 'data', supplied by the producer because a
  /// packed table with no columns cannot report one: cuDF derives num_rows()
  /// from the columns. kChunk only.
  vector_size_t numRows{0};
  /// The sizes of the chunks remaining in the queue. kChunk only.
  std::vector<int64_t> remainingBytes;
  /// How the channel ended. kEnd only.
  ChannelOutcome outcome;

  /// Returns a kChunk step. 'numRows' is the logical row count of 'chunk', and
  /// 'remainingBytes' the sizes of the chunks still queued.
  static ReaderStep chunk(
      std::shared_ptr<cudf::packed_columns> chunk,
      vector_size_t numRows,
      std::vector<int64_t> remainingBytes);

  /// Returns a kEnd step that carries 'outcome'.
  static ReaderStep end(ChannelOutcome outcome);

  /// Returns a kStop step.
  static ReaderStep stop();

 private:
  explicit ReaderStep(Kind kind) : kind(kind) {}
};

/// @brief Callback function for getting data from the queues.
using ReaderCallback = std::function<void(ReaderStep step)>;

/// A reader callback together with the step to hand it. A queue returns these
/// for the caller to notify after it has released its locks.
struct UcxDataAvailable {
  /// The reader to notify. Null if no reader waits.
  ReaderCallback callback{nullptr};
  /// The step for 'callback'. Set whenever 'callback' is.
  std::optional<ReaderStep> step;

  /// Invokes 'callback' with 'step', if there is a callback. Call it without
  /// holding any lock.
  void notify();
};

/// The text of the producer error that ends the readers of a task that was
/// removed without a successful initializeTask().
inline constexpr std::string_view kProducerNotInitializedError{
    "Producer task ended before initializing its output"};

/// What UcxOutputQueue::remove() needs to know about the removed task. The
/// caller reads it from the task before it takes any output queue lock: no Task
/// method may run under one.
struct RemovalFacts {
  /// The ID of the removed task.
  std::string taskId;
  /// The task's final state.
  exec::TaskState state{exec::TaskState::kRunning};
  /// The task's error message, without its context and stack trace. Empty if
  /// the task has no error.
  std::string error;
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

  /// Removes the step at the front of the queue, a chunk or the end marker,
  /// and returns it. If the queue is empty, installs 'notify' until a step is
  /// queued and returns nullopt.
  [[nodiscard]] std::optional<ReaderStep> getData(ReaderCallback notify);

  /// Removes all queued chunks and the end marker from the queue and its queued
  /// stats, and returns the chunks for the caller to destroy outside its locks.
  std::vector<std::shared_ptr<cudf::packed_columns>> takeQueuedData();

  /// Returns the waiting reader with the step at the front of the queue, and
  /// removes both. Returns no reader if none waits or the queue is empty.
  UcxDataAvailable takeReadyReader();

  /// Returns and clears the waiting reader, if any.
  ReaderCallback takeReader();

  /// Returns the stats of this buffer.
  Stats stats() const;

  std::string toString();

 private:
  // A queued page and the logical row count the producer gave it. Paired here
  // rather than recovered from the page, which reports no rows when it has no
  // columns.
  struct QueuedPage {
    std::shared_ptr<cudf::packed_columns> data;
    vector_size_t numRows{0};
  };

  std::deque<QueuedPage> queue_;
  ReaderCallback notify_{nullptr};
  Stats stats_;
};

/// The state of a task's output queue in UcxOutputQueueManager.
enum class OutputQueueState : uint8_t {
  /// A server asked for the task before initializeTask().
  kPending,
  /// initializeTask() succeeded.
  kActive,
  /// removeTask() ended the task's output.
  kRemoved,
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
///
/// UcxOutputQueueManager keeps one output queue per task, from the first
/// contact for the task ID to its erasure. An output queue mirrors the per-task
/// exec::OutputBuffer of Velox. It starts Pending, becomes Active at
/// initialize() and Removed at remove(), and stays the same object throughout,
/// so that a server bound to it reads every change.
///
/// Each destination has a DestinationState, and every event that can end it
/// goes through decideDestination() under the output queue's lock. The first
/// decision for a destination wins. A reader of a destination that has ended
/// gets the decided outcome; a reader of a Released destination gets that
/// outcome if it is an error, and ReaderStep::stop() otherwise.
class UcxOutputQueue : public std::enable_shared_from_this<UcxOutputQueue> {
 public:
  /// @brief Creates the Pending output queue of task 'taskId'. initialize()
  /// supplies the task, its destinations and its drivers.
  explicit UcxOutputQueue(std::string taskId);

  /// @brief Initializes a Pending output queue, which a server may have created
  /// for its reader before the producer task initialized its output. The
  /// readers waiting on it are kept, and get their data once it is enqueued.
  /// Appends ready notifications for the caller to invoke outside manager
  /// locks. Throws, and leaves the output queue Pending and unchanged, if
  /// 'numDestinations' is negative.
  /// @return True, if initialization was successful, i.e. the output queue was
  /// Pending.
  bool initialize(
      std::shared_ptr<exec::Task> task,
      int32_t numDestinations,
      uint32_t numDrivers,
      core::PartitionedOutputNode::Kind kind,
      std::vector<UcxDataAvailable>& notifications);

  /// Returns the ID of the task this output queue belongs to.
  const std::string& taskId() const {
    return taskId_;
  }

  /// Returns the state of this output queue.
  OutputQueueState state() const {
    return state_.load(std::memory_order_acquire);
  }

  /// Counts a server bound to this output queue. Called under the manager's
  /// lock.
  void addBoundReader();

  /// Unbinds the server of 'destination' and drops its release callback.
  /// Called under the manager's lock. Returns true if the output queue is
  /// Pending, has no bound reader, and no destination is Released, so that the
  /// manager can erase it.
  bool removeBoundReader(int destination);

  /// Records the task of a failed initializeTask(). The output queue stays
  /// Pending, and once Removed lives as long as 'task'.
  void recordFailedInitialization(std::weak_ptr<exec::Task> task);

  /// Makes the output queue Removed and decides every destination at once:
  /// clean if every driver called noMoreData() and no chunk is undelivered, a
  /// producer error otherwise. A chunk is undelivered while it is queued, kept
  /// for a broadcast destination not added yet, or in flight on the intra-node
  /// path to a destination that is not Released. Frees the chunks, wakes
  /// blocked producers, and keeps only a weak pointer to the task and the
  /// outcome, which later readers get. The caller reads 'facts' before taking
  /// any output queue lock. A second call decides nothing.
  /// @return The outcome of the removal, also for a second call.
  ChannelOutcome remove(const RemovalFacts& facts);

  /// Releases 'destination' because its consumer sent the close message or its
  /// endpoint closed, and frees its chunks. A destination that is not published
  /// yet is kept Released, and publishing it later keeps it Released.
  void consumerGone(int destination);

  /// Releases 'destination' because a send to its consumer failed while the
  /// endpoint stayed open, or its server failed, and frees its chunks. Returns
  /// the error for the caller to pass to Task::setError() outside every lock,
  /// or nullopt if the destination was released already or is unpublished. A
  /// later remove() ends the remaining consumers with the first such error.
  std::optional<std::string> sendFailed(
      int destination,
      std::string_view status);

  /// Registers the bound reader's release callback for 'destination'. It is
  /// called once, outside the lock, when the destination becomes Released.
  void setOnReleased(int destination, std::function<void()> onReleased);

  /// Returns the state of 'destination'. A destination the output queue has not
  /// seen is Unpublished. For tests.
  DestinationState destinationState(int destination);

  /// True once Removed, its Task is gone (the weak pointer expired), and no
  /// reader is bound.
  bool isDroppable();

  core::PartitionedOutputNode::Kind kind() const {
    return kind_;
  }

  /// Returns true once task metadata has been published via initializeTask(),
  /// that is, the output queue is no longer a Pending one created by an early
  /// server. Stays true once Removed.
  /// This is published before destination expansion completes so local
  /// UCX handshakes do not permanently fall back to the remote path.
  bool isInitialized() const {
    return initialized_.load(std::memory_order_acquire);
  }

  /// Returns the task that initialize() supplied while the output queue is
  /// Active, or null.
  std::shared_ptr<exec::Task> task();

  /// @brief When we understand the final number of split groups (for grouped
  /// execution only), we need to update the number of producing drivers here.
  void updateNumDrivers(uint32_t newNumDrivers);

  /// @brief Enqueues the data for the given destination. Currently, only
  /// partitioned output mode is supported where the number of destinations is
  /// fixed. Is is an error to provide a destination larger than the initial
  /// number of destinations. This will change in the future and if destination
  /// > numDestinations, then this will be dynamically adapted like it is done
  /// in Velox's OutputBuffer.
  /// @param destination The destination, must be < numDestinations.
  /// @param data The data.
  /// @param numRows The number of rows in the data. Supplied by the producer
  /// rather than read back from 'data', which cannot report a row count once it
  /// has no columns.
  void enqueue(
      int destination,
      std::unique_ptr<cudf::packed_columns> data,
      vector_size_t numRows);

  /// @brief Checks if the queue is over capacity and returns a future if so.
  /// This should be called after enqueueing all partitions for a batch.
  /// @param future Output parameter - populated with a future if blocked.
  /// @return True if blocked (queue over capacity), false otherwise.
  bool checkBlocked(ContinueFuture* future);

  /// @brief Returns data for the destination through 'notify', immediately if
  /// available. Otherwise, installs 'notify' until data becomes available.
  /// Unpublished readers wait for initialize() or updateOutputBuffers() to
  /// publish their destination and backfill any broadcast data and end marker.
  /// Invokes 'notify' with a clean kEnd step when the final count excludes the
  /// reader, and for negative destinations. A destination that has ended or
  /// was released gets its decided outcome, and a destination of a Removed
  /// output queue that was never published gets the removal's outcome.
  /// @param intraNode True if the reader hands the chunk over through
  /// IntraNodeTransferRegistry. The chunk it takes is then in flight until the
  /// reader calls intraNodeChunkRetrieved(): a removal meanwhile counts it as
  /// undelivered.
  void getData(int destination, ReaderCallback notify, bool intraNode);

  /// Reports that the source of 'destination' read the chunk that an
  /// intra-node reader took with getData(), or that the chunk's transfer was
  /// cancelled.
  void intraNodeChunkRetrieved(int destination);

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

  /// @brief Deletes all queued data of 'destination'. While the destination is
  /// open or not published yet, as when the producer runs out of GPU memory,
  /// the outcome is an error that says the producer discarded its output.
  /// After the destination ended, as when its server sent the end, the
  /// destination becomes Released. A consumer that is done calls
  /// consumerGone() instead.
  void deleteResults(int destination);

  std::string toString();

  /// @brief The stats of this output queue are shoe-horned into the stats
  /// object of OutputBuffer. Since the OutputBuffer's stat object is part of
  /// the Task stats and eventually processed at the Presto layer, this is the
  /// least intrusive way to convey stats information. The stats info from the
  /// UcxDestinationQueue are omitted since also the DestinationBuffer's stats
  /// are never processed by Presto.
  exec::OutputBuffer::Stats stats();

  /// @brief Queued bytes over this queue's byte capacity, the task's configured
  /// maxOutputBufferSize. Producers are only blocked after adding data (see
  /// checkBlocked), so the ratio can exceed 1.0. Returns nullopt while the
  /// capacity is still unknown, which is the case for a Pending output queue
  /// created by a server that arrived before initializeTask() supplied the
  /// task's query config.
  std::optional<double> getUtilization();

  /// @brief Whether enough is queued to risk back-pressuring producers soon, or
  /// the last data has been seen. Half the capacity is the threshold, matching
  /// exec::OutputBuffer. Returns nullopt while the capacity is unknown, for the
  /// same reason as getUtilization(): without a capacity there is no ratio to
  /// compare against, and reporting `false` there would let a Pending output
  /// queue masquerade as having spare room.
  std::optional<bool> isOverutilized();

 private:
  // Percentage of maxSize below which a blocked producer should
  // be unblocked.
  static constexpr int32_t kContinuePct = 90;

  // Methods that update the statistics.
  void updateStatsWithEnqueuedLocked(int64_t bytes, int64_t rows);

  // Updates the counters and appends the blocked producers' promises to
  // 'promises' if queuedBytes_ falls below the continueSize_ low water mark.
  // The caller fulfills them outside the lock.
  void updateStatsWithFreedLocked(
      int64_t bytes,
      int64_t numPackedCols,
      std::vector<ContinuePromise>& promises);

  void updateTotalQueuedBytesMsLocked();

  int64_t getAverageQueueTimeMsLocked() const;

  // internal function that is called when all drivers are done.
  void noMoreDrivers();

  // If this is called due to a driver processed all its data (no more data),
  // we increment the number of finished drivers. If it is called due to us
  // updating the total number of drivers, we don't.
  void checkIfDone(bool oneDriverFinished);

  // One destination: its state, its queue while it can still take chunks,
  // and the outcome decided for it.
  struct Destination {
    // Moved only by applyLocked().
    DestinationState state{DestinationState::kUnpublished};
    // Null once the destination ended or was released.
    std::unique_ptr<UcxDestinationQueue> queue;
    // The decided outcome, if any. Shared, since a removal decides one outcome
    // for every destination.
    std::shared_ptr<const ChannelOutcome> outcome;
    // True if the last reader that asked for a chunk hands chunks over through
    // IntraNodeTransferRegistry.
    bool intraNodeReader{false};
    // Chunks an intra-node reader took and its source has not read yet.
    int64_t numIntraNodeChunksInFlight{0};
  };

  // The work that decisions leave for after mutex_ is released. Declare it
  // before taking the lock, so that the chunks it holds are destroyed after the
  // lock is released.
  struct Deferred {
    // Readers to hand a step.
    std::vector<UcxDataAvailable> notifications;
    // Release callbacks of destinations that became Released.
    std::vector<std::function<void()>> onReleased;
    // Blocked producers to wake.
    std::vector<ContinuePromise> promises;
    // Set if the output is finished, for Task::setAllOutputConsumed().
    std::shared_ptr<exec::Task> finishedTask;
    // Chunks that no reader will fetch.
    std::vector<std::shared_ptr<cudf::packed_columns>> freedChunks;

    // Notifies readers, calls release callbacks, wakes producers and reports a
    // finished output, in this order.
    void run();
  };

  // Applies 'event' to 'destination' through decideDestination(). Frees its
  // chunks if the decision says so, ends its waiting reader once it ended or
  // was released, and queues its release callback.
  DestinationDecision applyLocked(
      Destination& destination,
      int index,
      DestinationEvent event,
      const std::shared_ptr<const ChannelOutcome>& candidate,
      Deferred& deferred);

  // Returns the step for a reader of a destination that ended or was released.
  static ReaderStep endedStep(const Destination& destination);

  // Hands the step at the front of 'destination' to its waiting reader, if
  // both exist. Taking the end marker ends the destination cleanly.
  void notifyReadyReaderLocked(
      Destination& destination,
      int index,
      Deferred& deferred);

  // Returns the published or Unpublished destination 'index', or null.
  Destination* findDestinationLocked(int index);

  // The outcome that deleteResults() proposes: the producer discarded its
  // output.
  std::shared_ptr<const ChannelOutcome> discardedOutcome() const;

  // The outcome that a removal decides for every destination.
  ChannelOutcome removalOutcomeLocked(const RemovalFacts& facts) const;

  // Returns true when no more destinations can be published. Requires mutex_.
  bool isDestinationCountFinalLocked() const;

  // Adds destinations and restores broadcast data and end markers.
  void addDestinationsLocked(int32_t numDestinations, Deferred& deferred);

  // Ends unpublished readers once the destination count is final: their
  // destination will never be published, so no rows are lost. Drops the
  // destinations that are not published.
  void clearPendingReadersLocked(Deferred& deferred);

  // True if a destination that is not published yet is Released. Requires
  // mutex_.
  bool hasReleasedDestinationLocked() const;

  const std::string taskId_;

  // Written under mutex_. Read without it by the manager, which must not see
  // a Removed output queue.
  std::atomic<OutputQueueState> state_{OutputQueueState::kPending};

  // Reference to the task that owns this UcxOutputQueue while it is Active.
  std::shared_ptr<exec::Task> task_{nullptr};

  // The task of a Removed output queue, or of a failed initialize(). Expires
  // when the host destroys the task, after which no server is expected.
  std::weak_ptr<exec::Task> removedTask_;

  // The servers bound to this output queue.
  int32_t numBoundReaders_{0};

  // Set by recordFailedInitialization(). Keeps the Pending output queue when
  // its last server is released, so that its removal can tie the Removed output
  // queue to the Task.
  bool failedInitialization_{false};

  // The output mode (partitioned, broadcast, etc.)
  core::PartitionedOutputNode::Kind kind_{
      core::PartitionedOutputNode::Kind::kPartitioned};

  // Set to true once task metadata is available. Lock-free readers
  // (canUseIntraNode) load with memory_order_acquire so that kind_, written
  // before the store, is visible. They must not read task_, which remove()
  // resets under mutex_.
  std::atomic<bool> initialized_{false};

  // For broadcast: stores data for late-arriving destinations that need
  // backfill. Cleared once noMoreQueues_ is set.
  // Paired with the row count for the same reason QueuedPage is.
  std::vector<std::pair<std::shared_ptr<cudf::packed_columns>, vector_size_t>>
      dataToBroadcast_;

  // If 'queuedBytes_' > 'maxSize_', each producer is blocked after adding
  // data.
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

  // The published destinations. Only control-plane updates grow it.
  std::vector<Destination> destinations_;

  // The destinations that are not published yet: Unpublished ones, whose
  // early reader waits in their queue, and Released ones, which publication
  // keeps Released. Cleared once the destination count is final; a removal
  // ends them instead.
  std::unordered_map<int, Destination> unpublishedDestinations_;

  // The outcome remove() decided. Readers of a Removed output queue's
  // destinations that were never published get it.
  std::shared_ptr<const ChannelOutcome> removedOutcome_;

  // The error of the first send that failed the task. A failed send frees its
  // destination's chunks, so remove() reports this error instead of counting
  // the output left.
  std::optional<std::string> sendFailure_;

  // The release callbacks of the bound readers, by destination.
  std::unordered_map<int, std::function<void()>> onReleased_;

  // keep track of the number of drivers that have finished.
  uint32_t numFinished_{0};

  bool atEnd_ = false;

  // promises when buffer reached capacity and blocked further enqueueing.
  std::vector<ContinuePromise> promises_;

  // actual data in 'destinations_'
  int64_t queuedBytes_{0};
  int64_t queuedPackedColumns_{0};

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
