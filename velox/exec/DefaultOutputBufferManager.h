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

#include <unordered_set>

#include "velox/exec/OutputBuffer.h"
#include "velox/exec/OutputBufferManager.h"

namespace facebook::velox::exec {

struct OutputTransportEntry;

class DefaultOutputBufferManager : public OutputBufferManager {
 public:
  /// Options for shuffle. This is initialized once and affects both
  /// PartitionedOutput and Exchange. This can be used for controlling
  /// protocol version and other matters where shuffle sides should
  /// agree.
  struct Options {};

  explicit DefaultOutputBufferManager(Options) {}

  static const std::shared_ptr<DefaultOutputBufferManager>& getInstanceRef();

  static const std::shared_ptr<DefaultOutputBufferManager>& getInstanceRef(
      const Options& options);

  /// Builds the registry entry for the built-in in-memory transport, pairing
  /// the default manager singleton with the PartitionedOutput operator.
  /// OutputTransportRegistry::global() registers it under kInMemory.
  static std::shared_ptr<OutputTransportEntry> makeDefaultTransportEntry();

  // Lifecycle.

  void initializeTask(
      std::shared_ptr<Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers,
      const std::string& transportOptions = {}) override;

  /// Updates the number of destination buffers. Returns true if the buffer
  /// exists for a given taskId, else returns false.
  bool updateOutputBuffers(
      const std::string& taskId,
      int numDestinations,
      bool noMoreBuffers) override;

  /// When we understand the final number of split groups (for grouped
  /// execution only), we need to update the number of producing drivers here.
  /// Returns true if the buffer exists for a given taskId, else returns false.
  bool updateNumDrivers(const std::string& taskId, uint32_t newNumDrivers)
      override;

  void removeTask(const std::string& taskId) override;

  // Data plane.

  /// Adds data to the outgoing queue for 'destination'. 'data' must not be
  /// nullptr. 'data' is always added but if the buffers are full the future is
  /// set to a ContinueFuture that will be realized when there is space.
  bool enqueue(
      const std::string& taskId,
      int destination,
      std::unique_ptr<SerializedPageBase> data,
      ContinueFuture* future);

  void noMoreData(const std::string& taskId);

  /// Returns true if noMoreData has been called and all the accumulated data
  /// have been fetched and acknowledged.
  bool isFinished(const std::string& taskId);

  /// Removes data with sequence number < 'sequence' from the queue for
  /// 'destination_'.
  void
  acknowledge(const std::string& taskId, int destination, int64_t sequence);

  void deleteResults(const std::string& taskId, int destination);

  /// Adds up to 'maxBytes' bytes worth of data for 'destination' from 'taskId'.
  /// The sequence number of the data must be >= 'sequence'. If there is no
  /// buffer associated with the given taskId, returns false. If there is no
  /// data, 'notify' will be registered and called when there is data or the
  /// source is at end, the function returns true. If deleteResults was
  /// previously called for the destination, 'notify' will be called immediately
  /// with a list of pages containing a single "end of data" marker. Existing
  /// data with a sequence number < sequence is deleted. The caller is expected
  /// to increment the sequence number between calls by the number of items
  /// received. In this way the next call implicitly acknowledges receipt of the
  /// results from the previous. The acknowledge method is offered for an early
  /// ack, so that the producer can continue before the consumer is done
  /// processing the received data. If not null, 'activeCheck' is used to check
  /// if data consumer is currently active or not. This only applies for
  /// arbitrary output buffer for now.
  bool getData(
      const std::string& taskId,
      int destination,
      uint64_t maxBytes,
      int64_t sequence,
      DataAvailableCallback notify,
      DataConsumerActiveCheckCallback activeCheck = nullptr);

  /// Like getData(), but for a sub-task 'taskId' that the task 'parentTaskId'
  /// may not have created yet, such as a fixed point's producer that a peer
  /// worker reads in the same iteration. 'parentTaskId' has an output buffer
  /// in this manager for as long as it may create 'taskId'. When the buffer
  /// for 'taskId' does not exist yet and the one for 'parentTaskId' does,
  /// holds the request and serves it once initializeTask() creates the buffer.
  /// When neither exists, the sub-task is never created and the request is
  /// dropped. 'activeCheck' is required: a held request is dropped once it
  /// returns false, e.g. after the consumer's own wait ran out, by
  /// deleteResultsWhenCreated() for its destination, and by
  /// removeTask('parentTaskId'). 'activeCheck' may run under this manager's
  /// lock, so it must be cheap and must not call this manager.
  void getDataWhenCreated(
      const std::string& parentTaskId,
      const std::string& taskId,
      int destination,
      uint64_t maxBytes,
      int64_t sequence,
      DataAvailableCallback notify,
      DataConsumerActiveCheckCallback activeCheck);

  /// Like deleteResults(), but for a sub-task 'taskId' that the task
  /// 'parentTaskId' may not have created yet, as in getDataWhenCreated(). When
  /// the buffer for 'taskId' does not exist yet and the one for 'parentTaskId'
  /// does, initializeTask() deletes the results of 'destination' as it creates
  /// the buffer, so the producer does not wait for a consumer that is gone. A
  /// delete still pending is dropped by removeTask('parentTaskId').
  void deleteResultsWhenCreated(
      const std::string& parentTaskId,
      const std::string& taskId,
      int destination);

  // Returns a new stream listener if a listener factory has been set.
  std::unique_ptr<OutputStreamListener> newListener() const {
    return listenerFactory_ ? listenerFactory_() : nullptr;
  }

  // Sets the stream listener factory. This allows custom processing of data
  // for repartitioning, e.g. computing checksums.
  void setListenerFactory(
      std::function<std::unique_ptr<OutputStreamListener>()> factory) {
    listenerFactory_ = std::move(factory);
  }

  // Retrieves the set of buffers for a query if exists.
  // Returns NULL if task not found.
  std::shared_ptr<OutputBuffer> getBufferIfExists(const std::string& taskId);

  // Observability.

  uint64_t numBuffers() const;

  std::string toString();

  /// Returns a human-readable representation of the output buffer state for
  /// the task identified by 'taskId', or an empty string if no buffer exists.
  std::string toString(const std::string& taskId) override;

  std::optional<double> getUtilization(const std::string& taskId) override;

  // Over-utilized when buffered bytes exceed half of the configured max output
  // buffer size, or the buffer is at end.
  std::optional<bool> isOverutilized(const std::string& taskId) override;

  std::optional<OutputBufferStats> stats(const std::string& taskId) override;

 private:
  // Retrieves the set of buffers for a query.
  // Throws an exception if buffer doesn't exist.
  std::shared_ptr<OutputBuffer> getBuffer(const std::string& taskId);

  folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<OutputBuffer>>,
      std::mutex>
      buffers_;

  // A getDataWhenCreated() request held until its buffer exists.
  struct HeldRequest {
    int destination;
    uint64_t maxBytes;
    int64_t sequence;
    DataAvailableCallback notify;
    DataConsumerActiveCheckCallback activeCheck;
  };

  // What consumers asked of a sub-task's buffer before it existed.
  struct PendingBuffer {
    // The task that creates the sub-task.
    std::string parentTaskId;
    // Reads waiting for the buffer.
    std::vector<HeldRequest> requests;
    // Destinations whose consumers deleted their results.
    std::vector<int> deletedDestinations;
  };

  // The pending buffers by sub-task id. Guarded by the mutex of 'buffers_', so
  // nothing is added for a sub-task after initializeTask() looked for it.
  std::unordered_map<std::string, PendingBuffer> pendingBuffers_;

  // The ids of the sub-tasks in 'pendingBuffers_' by parent task id, so
  // removeTask() of a parent finds them. Guarded by the mutex of 'buffers_'.
  std::unordered_map<std::string, std::unordered_set<std::string>>
      pendingSubTaskIds_;

  std::function<std::unique_ptr<OutputStreamListener>()> listenerFactory_{
      nullptr};
};
} // namespace facebook::velox::exec
