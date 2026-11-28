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
#include <cinttypes>
#include <cuda/stream>
#include <deque>
#include <memory>
#include <optional>
#include <vector>
#include "velox/common/base/Exceptions.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/experimental/ucx-exchange/UcxChannelState.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::ucx_exchange {

/// Struct that bundles a packed_table with the CUDA stream that was used
/// to allocate its memory. UcxExchange hands the table downstream on a stream
/// of its own, so that operator kernels never land on the receive stream.
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
      : packedTable(std::move(table)), stream(s), numRows(numRows) {}

  /// Returns the size of the GPU data buffer, or 0 if packedTable is null.
  size_t gpuDataSize() const {
    return packedTable ? packedTable->data.gpu_data->size() : 0;
  }
};

using PackedTableWithStreamPtr = std::unique_ptr<PackedTableWithStream>;

/// Buffers the packed tables that the sources of one exchange client receive,
/// and keeps one entry per source. Whether the consumer has finished or failed
/// is a view of those entries: see decideSource() for how each entry moves.
class UcxExchangeQueue {
 public:
  /// Identifies a source. Ids are dense and start at 0.
  using SourceId = int32_t;

  explicit UcxExchangeQueue(int32_t numberOfConsumers)
      : numberOfConsumers_{numberOfConsumers} {
    VELOX_CHECK_GE(numberOfConsumers, 1);
  }

  ~UcxExchangeQueue();

  std::mutex& mutex() {
    return mutex_;
  }

  /// Returns true if no packed table is queued. Safe to call without
  /// 'mutex_'.
  bool empty() const {
    return size() == 0;
  }

  /// Adds a source in kRegistered and returns its id. Call with 'mutex_' held.
  SourceId addSourceLocked();

  /// Adds a source in kReleased and returns its id, for a source the client
  /// adds after close(). The consumer does not wait for it. Call with 'mutex_'
  /// held.
  SourceId addReleasedSourceLocked();

  /// Records that the handshake response of source 'id' arrived and was
  /// accepted.
  void sourceOpened(SourceId id);

  /// Takes 'table' from source 'id' if the source is kOpen and the consumer has
  /// neither failed nor closed. Adds the promises of the consumers to wake to
  /// 'promises'; the caller fulfills them with no lock held. Returns false if
  /// the table was not taken: it then stays in 'table', and the caller frees it
  /// outside every lock.
  [[nodiscard]] bool enqueueTable(
      SourceId id,
      PackedTableWithStreamPtr& table,
      std::vector<ContinuePromise>& promises);

  /// Decides the end of source 'id' with 'outcome'. The first decision for a
  /// source wins. An error fails the consumer, unless it failed or closed
  /// already: the queued tables are freed and the waiting consumers woken.
  void end(SourceId id, ChannelOutcome outcome);

  /// Returns a PackedTableWithStream object.
  ///
  /// Throws the error of the source that failed the consumer, also after
  /// close(). Otherwise, after close(), returns nullptr and sets 'atEnd' to
  /// true. Returns a nullptr if no data is available. If data is still
  /// expected, sets 'atEnd' to false and 'future' to a Future that will
  /// complete when data arrives. If no more data is expected, sets 'atEnd' to
  /// true. Returns one PackedTableWithStream if data is available.
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
  int64_t totalBytes() const {
    return totalBytes_;
  }

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

  /// Records that the client adds no more sources. The consumer has finished
  /// once every source is kEnded or kReleased.
  void noMoreSources();

  /// Releases every source, frees the queued tables and wakes the waiting
  /// consumers. Returns the ids of the sources whose producer's server must be
  /// told that the consumer is gone: those that had not ended. Returns no ids
  /// when called again.
  std::vector<SourceId> close();

  /// Returns true after close().
  bool isClosed();

  /// Returns true after close(). Call with 'mutex_' held.
  bool isClosedLocked() const {
    return closed_;
  }

  /// Returns the state of source 'id'.
  SourceState sourceState(SourceId id);

 private:
  // The consumer's view of one source. decideSource() moves it.
  struct SourceEntry {
    SourceState state{SourceState::kRegistered};
    // How the source ended. Set when it moves to kEnded.
    std::optional<ChannelOutcome> outcome;
  };

  // Publishes the depth of 'queue_' to 'size_'. Call with 'mutex_' held after
  // every mutation of 'queue_'.
  void publishSizeLocked() {
    size_.store(static_cast<int32_t>(queue_.size()), std::memory_order_relaxed);
  }

  // Returns true if the client adds no more sources and every source is kEnded
  // or kReleased.
  bool isFinishedLocked() const;

  // Moves the queued tables out, so that the caller frees them after releasing
  // 'mutex_'.
  std::deque<PackedTableWithStreamPtr> takeTablesLocked();

  void addPromiseLocked(
      int consumerId,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  std::vector<ContinuePromise> clearAllPromisesLocked();

  static void clearPromises(std::vector<ContinuePromise>& promises) {
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  const int32_t numberOfConsumers_;

  bool noMoreSources_{false};
  bool closed_{false};

  // Indexed by SourceId.
  std::vector<SourceEntry> sources_;

  // The first source that ended with an error before close(). Its error fails
  // every later read, also after close().
  std::optional<SourceId> failedSource_;

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

  // Total size of packed tables in queue.
  int64_t totalBytes_{0};
  // Number of packed tables received.
  int64_t receivedTables_{0};
  // Total size of packed tables received. Used to calculate an average
  // expected size.
  int64_t receivedBytes_{0};
  // Maximum value of totalBytes_.
  int64_t peakBytes_{0};
  // Peak queue size (number of items). Used for high-water-mark alerts.
  int64_t peakSize_{0};
};

} // namespace facebook::velox::ucx_exchange
