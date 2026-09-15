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

#include <algorithm>
#include <cinttypes>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <atomic>
#include <cuda/stream>

#include "velox/common/base/Exceptions.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::ucx_exchange {

/// Struct that bundles a packed_table with the CUDA stream that was used
/// to allocate its memory. This allows the receiver to reuse the same stream
/// for subsequent operations on the data.
struct PackedTableWithStream {
  std::unique_ptr<cudf::packed_table> packedTable;
  cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};

  /// Logical rows in 'packedTable', as reported by the producer. Authoritative:
  /// cudf::table_view::num_rows() derives the count from the columns and so
  /// returns 0 for a table with none, which is what an exchange fragment with
  /// an empty output layout sends. 32-bit because a cuDF table cannot hold more
  /// rows than a cudf::size_type can index.
  vector_size_t numRows{0};

  PackedTableWithStream() = default;
  PackedTableWithStream(
      std::unique_ptr<cudf::packed_table>&& table,
      cuda::stream_ref s,
      vector_size_t numRows)
      : packedTable(std::move(table)), stream(s), numRows(numRows) {
    // Intra-node payloads were allocated on the producer's stream. Release
    // them on the receive stream so synchronizing it also completes the free
    // before returning receive credit, including when a page is dropped.
    if (packedTable && packedTable->data.gpu_data) {
      packedTable->data.gpu_data->set_stream(stream);
    }
  }

  /// Returns the size of the GPU data buffer, or 0 if packedTable is null.
  size_t gpuDataSize() const {
    return packedTable ? packedTable->data.gpu_data->size() : 0;
  }
};

using PackedTableWithStreamPtr = std::unique_ptr<PackedTableWithStream>;

class UcxExchangeQueue {
 public:
  explicit UcxExchangeQueue(
      int32_t numberOfConsumers,
      uint64_t receiveHighWaterBytes = 0)
      : numberOfConsumers_{numberOfConsumers},
        receiveHighWaterBytes_{receiveHighWaterBytes} {
    VELOX_CHECK_GE(numberOfConsumers, 1);
  }

  ~UcxExchangeQueue() {
    clearAllPromises();
  }

  std::mutex& mutex() {
    return mutex_;
  }

  /// Returns true if no packed table is queued. Safe to call without
  /// 'mutex_'.
  bool empty() const {
    return size() == 0;
  }

  /// Enqueues 'data' to the queue. One random promise(top of promise queue)
  /// associated with the future that is waiting for the data from the queue
  /// is returned in 'promises' if 'data' is not nullptr. When 'data' is
  /// nullptr and the queue is completed serving data, all left over promises
  /// will be returned in 'promises'. When 'data' is nullptr and the queue is
  /// not completed serving data, no 'promises' will be added and returned.
  void enqueueLocked(
      PackedTableWithStreamPtr&& data,
      std::vector<ContinuePromise>& promises,
      uint64_t reservedReceiveBytes = 0,
      bool* receiveReservationActive = nullptr);

  /// If data is permanently not available, e.g. the source cannot be
  /// contacted, this registers an error message and causes the reading
  /// Exchanges to throw with the message.
  void setError(std::string_view error);

  bool isInError() {
    return !error_.empty();
  }
  /// Returns a PackedTableWithStream object.
  ///
  /// Returns a nullptr if no data is available. If data is still expected,
  /// sets 'atEnd' to false and 'future' to a Future that will complete when
  /// data arrives. If no more data is expected, sets 'atEnd' to true. Returns
  /// one PackedTableWithStream if data is available.
  /// It's possible that the same consumer is already waiting for data. In this
  /// case, a stalePromise is returned which needs to be cleaned up.
  PackedTableWithStreamPtr dequeueLocked(
      int consumerId,
      bool* atEnd,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  /// Returns the number of queued packed tables. Safe to call without
  /// 'mutex_'.
  int32_t size() const {
    return size_.load(std::memory_order_relaxed);
  }

  /// Returns the total bytes held by packed tables in 'this'.
  uint64_t totalBytes() const {
    return totalBytes_;
  }

  /// Bytes still owned by the receive path: queued tables, allocations for
  /// posted receives, and dequeued tables retained by downstream CudfVectors.
  /// Caller must hold mutex().
  uint64_t retainedReceiveBytesLocked() const {
    const auto max = std::numeric_limits<uint64_t>::max();
    VELOX_CHECK_LE(
        reservedBytes_, max - totalBytes_, "UCX retained byte overflow");
    const auto queuedBytes = totalBytes_ + reservedBytes_;
    VELOX_CHECK_LE(
        inFlightBytes_, max - queuedBytes, "UCX retained byte overflow");
    return queuedBytes + inFlightBytes_;
  }

  /// Bytes that can block another receive. Dequeued tables are deliberately
  /// excluded so stateful downstream operators cannot prevent metadata/EOS
  /// progress by retaining input until end-of-stream.
  uint64_t queuedReceiveBytesLocked() const {
    VELOX_CHECK_LE(
        reservedBytes_,
        std::numeric_limits<uint64_t>::max() - totalBytes_,
        "UCX queued receive byte overflow");
    return totalBytes_ + reservedBytes_;
  }

  uint64_t reservedReceiveBytesLocked() const {
    return reservedBytes_;
  }

  uint64_t inFlightReceiveBytesLocked() const {
    return inFlightBytes_;
  }

  bool tracksInFlightReceiveBytes() const {
    return receiveHighWaterBytes_ > 0;
  }

  uint64_t receiveHighWaterBytes() const {
    return receiveHighWaterBytes_;
  }

  uint64_t receivePipelineTableLimitLocked() const {
    // Keep one large payload available to each consumer without allowing the
    // receive queue to scale with the number of sources.
    return std::max<uint64_t>(2, static_cast<uint64_t>(numberOfConsumers_));
  }

  uint64_t estimatedReceivePayloadBytesLocked() const {
    return std::max<uint64_t>(
        averageReceivedTablesBytes(), receiveHighWaterBytes_);
  }

  uint64_t defaultReceivePrefetchByteLimitLocked() const {
    VELOX_CHECK_GT(receiveHighWaterBytes_, 0);

    const auto payloadBytes = estimatedReceivePayloadBytesLocked();
    const auto tableLimit = receivePipelineTableLimitLocked();
    const auto payloadWithSlack =
        saturatingAdd(payloadBytes, receiveHighWaterBytes_);
    if (payloadWithSlack > std::numeric_limits<uint64_t>::max() / tableLimit) {
      return std::numeric_limits<uint64_t>::max();
    }
    return payloadWithSlack * tableLimit;
  }

  uint64_t receivePrefetchByteLimitLocked() const {
    if (receiveHighWaterBytes_ == 0) {
      return std::numeric_limits<uint64_t>::max();
    }

    const auto defaultLimit = defaultReceivePrefetchByteLimitLocked();
    return receivePrefetchLimitActive_
        ? std::min(receivePrefetchByteLimit_, defaultLimit)
        : defaultLimit;
  }

  bool receiveBytesBelowPrefetchLimitLocked() const {
    const auto limit = receivePrefetchByteLimitLocked();
    if (limit == std::numeric_limits<uint64_t>::max()) {
      return true;
    }

    const auto queuedBytes = queuedReceiveBytesLocked();
    if (queuedBytes == 0) {
      // Always admit one oversized payload so a source cannot deadlock solely
      // because its next table is larger than the configured target.
      return true;
    }
    const auto payloadBytes = estimatedReceivePayloadBytesLocked();
    return queuedBytes <= limit && payloadBytes <= limit - queuedBytes;
  }

  bool receiveCanPrefetchLocked() const {
    return receiveBytesBelowPrefetchLimitLocked();
  }

  /// Attempts to reserve bytes before allocating or posting a UCX receive.
  /// Caller must hold mutex().
  bool tryReserveReceiveBytesLocked(uint64_t bytes);

  /// Releases a posted/pending receive reservation. Caller must hold mutex().
  void releaseReceiveBytesLocked(uint64_t bytes);

  /// Consumes a reservation without growing the adaptive window. Use this when
  /// the reservation is transferred to queued ownership, or immediately before
  /// recording allocation pressure: neither case represents memory becoming
  /// available to the process. Caller must hold mutex().
  void consumeReceiveReservationLocked(uint64_t bytes);

  /// Releases bytes retained by a downstream CudfVector. Caller must hold
  /// mutex().
  void releaseInFlightReceiveBytesLocked(uint64_t bytes);

  /// Records allocation pressure and shrinks the adaptive receive window.
  /// Returns true only when queued or reserved exchange memory can be drained
  /// before retry. Downstream in-flight memory is not independently drainable
  /// because a stateful operator may retain it until end-of-stream.
  /// Caller must hold mutex().
  bool recordReceiveAllocationPressureLocked(uint64_t attemptedBytes);

  /// Returns the maximum value of total bytes.
  uint64_t peakBytes() const {
    return peakBytes_;
  }

  /// Returns total number of packed tables received from all sources.
  uint64_t receivedTables() const {
    return receivedTables_;
  }

  /// Returns an average size of tables. Returns 0 if hasn't received
  /// any tables yet.
  uint64_t averageReceivedTablesBytes() const {
    return receivedTables_ > 0 ? receivedBytes_ / receivedTables_ : 0;
  }

  void addSourceLocked() {
    VELOX_CHECK(!noMoreSources_, "addSource called after noMoreSources");
    numSources_++;
  }

  void noMoreSources();

  void close();

 private:
  // Publishes the depth of 'queue_' to 'size_'. Call with 'mutex_' held after
  // every mutation of 'queue_'.
  void publishSizeLocked() {
    size_.store(static_cast<int32_t>(queue_.size()), std::memory_order_relaxed);
  }

  static constexpr uint64_t saturatingAdd(uint64_t left, uint64_t right) {
    const auto max = std::numeric_limits<uint64_t>::max();
    return left > max - right ? max : left + right;
  }

  void maybeGrowReceivePrefetchLimitLocked() {
    if (!receivePrefetchLimitActive_ || receiveHighWaterBytes_ == 0) {
      return;
    }

    const auto defaultLimit = defaultReceivePrefetchByteLimitLocked();
    if (receivePrefetchByteLimit_ >= defaultLimit) {
      receivePrefetchByteLimit_ = 0;
      receivePrefetchLimitActive_ = false;
      return;
    }

    if (retainedReceiveBytesLocked() > receivePrefetchByteLimit_ / 2) {
      return;
    }

    const auto increment = std::max<uint64_t>(
        receiveHighWaterBytes_, estimatedReceivePayloadBytesLocked() / 2);
    receivePrefetchByteLimit_ =
        saturatingAdd(receivePrefetchByteLimit_, increment);
    if (receivePrefetchByteLimit_ >= defaultLimit) {
      receivePrefetchByteLimit_ = 0;
      receivePrefetchLimitActive_ = false;
    }
  }

  std::vector<ContinuePromise> closeLocked() {
    queue_.clear();
    publishSizeLocked();
    totalBytes_ = 0;
    reservedBytes_ = 0;
    inFlightBytes_ = 0;
    receiveAccountingClosed_ = true;
    atEnd_ = true;
    return clearAllPromisesLocked();
  }

  std::vector<ContinuePromise> checkCompleteLocked() {
    if (noMoreSources_ && numCompleted_ == numSources_) {
      atEnd_ = true;
      return clearAllPromisesLocked();
    }
    return {};
  }

  void addPromiseLocked(
      int consumerId,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  void clearAllPromises() {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      promises = clearAllPromisesLocked();
    }
    clearPromises(promises);
  }

  std::vector<ContinuePromise> clearAllPromisesLocked() {
    std::vector<ContinuePromise> promises;
    promises.reserve(promises_.size());
    auto it = promises_.begin();
    while (it != promises_.end()) {
      promises.push_back(std::move(it->second));
      it = promises_.erase(it);
    }
    VELOX_CHECK(promises_.empty());
    return promises;
  }

  static void clearPromises(std::vector<ContinuePromise>& promises) {
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  const int32_t numberOfConsumers_;

  int numCompleted_{0};
  int numSources_{0};
  bool noMoreSources_{false};
  bool atEnd_{false};

  std::mutex mutex_;
  std::deque<PackedTableWithStreamPtr> queue_;
  // Depth of 'queue_', maintained under 'mutex_' so that size() and empty() can
  // be read without it. The Communicator progress thread polls the depth on
  // every UcxExchangeSource::process() call to decide backpressure, and taking
  // 'mutex_' there would contend with dequeueLocked() on the driver threads.
  // Reading queue_.size() unlocked is not an option: std::deque computes it by
  // subtracting iterators over the node map, so a concurrent push_back or
  // pop_front that crosses a node boundary can yield an arbitrary value.
  std::atomic<int32_t> size_{0};
  // The map from consumer id to the waiting promise
  folly::F14FastMap<int, ContinuePromise> promises_;

  // When set, all promises will be realized and the next dequeue will
  // throw an exception with this message.
  std::string error_;
  // Total size of packed tables in queue.
  uint64_t totalBytes_{0};
  // Bytes reserved for pending or posted receives not yet enqueued.
  uint64_t reservedBytes_{0};
  // Bytes dequeued from this queue but retained by downstream CudfVectors.
  uint64_t inFlightBytes_{0};
  const uint64_t receiveHighWaterBytes_{0};
  bool receivePrefetchLimitActive_{false};
  uint64_t receivePrefetchByteLimit_{0};
  // close() and setError() invalidate all outstanding receive accounting at
  // once. Late UCX callbacks are expected and release as explicit no-ops.
  bool receiveAccountingClosed_{false};
  // Number of packed tables received.
  uint64_t receivedTables_{0};
  // Total size of packed tables received. Used to calculate an average
  // expected size.
  uint64_t receivedBytes_{0};
  // Maximum value of totalBytes_.
  uint64_t peakBytes_{0};
  // Peak queue size (number of items). Used for high-water-mark alerts.
  int64_t peakSize_{0};
};

} // namespace facebook::velox::ucx_exchange
