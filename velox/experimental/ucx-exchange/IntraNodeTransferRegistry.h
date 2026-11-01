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
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>

#include "velox/vector/TypeAliases.h"

namespace facebook::velox::ucx_exchange {

/// @brief Key for identifying intra-node transfer entries in the registry.
/// Used when UcxExchangeServer and UcxExchangeSource are on the same node.
struct IntraNodeTransferKey {
  std::string taskId;
  uint32_t destination;
  uint32_t sequenceNumber;

  bool operator<(const IntraNodeTransferKey& other) const {
    if (taskId != other.taskId) {
      return taskId < other.taskId;
    }
    if (destination != other.destination) {
      return destination < other.destination;
    }
    return sequenceNumber < other.sequenceNumber;
  }
};

/// @brief Result from intra-node transfer containing data and end marker.
struct IntraNodeTransferResult {
  std::shared_ptr<cudf::packed_columns> data;
  /// Logical rows in 'data'. Carried explicitly for the same reason the remote
  /// path puts a row count on the wire: a packed table with no columns cannot
  /// report its own row count.
  vector_size_t numRows{0};
  bool atEnd{false}; // True if this is the end-of-stream marker
  /// Why the producer ended before sending all its output; set only with
  /// 'atEnd'.
  std::optional<std::string> producerError;
};

/// @brief Entry in the intra-node transfer registry containing the data and
/// synchronization primitives. The server publishes data via publish() and
/// waits on retrievedPromise; the source polls via poll() and fulfils the
/// promise on retrieval.
struct IntraNodeTransferEntry {
  std::shared_ptr<cudf::packed_columns> data;
  vector_size_t numRows{
      0}; // Logical rows in 'data'; see IntraNodeTransferResult.
  bool atEnd{false}; // True if this is the end-of-stream marker
  /// See IntraNodeTransferResult.
  std::optional<std::string> producerError;
  /// Runs when the source reads 'data'; see publish().
  std::function<void()> onRetrieved;
  std::promise<void> retrievedPromise; // Server waits on this after publishing
  std::mutex entryMutex;
  bool ready{false}; // True when data is ready to retrieve
};

/// @brief Singleton registry for intra-node data transfers.
///
/// When UcxExchangeServer and UcxExchangeSource are on the same node
/// (same Communicator instance), this registry enables direct data sharing
/// without going through UCXX network transfers.
///
/// The server publishes packed_columns data to the registry, and the source
/// polls for it. Synchronization is handled via futures/promises.
class IntraNodeTransferRegistry {
 public:
  /// @brief Get the singleton instance of the registry.
  static std::shared_ptr<IntraNodeTransferRegistry> getInstance();

  // Prevent copying
  IntraNodeTransferRegistry(const IntraNodeTransferRegistry&) = delete;
  IntraNodeTransferRegistry& operator=(const IntraNodeTransferRegistry&) =
      delete;

  /// @brief Publish data for intra-node transfer with condition variable
  /// signaling. Server calls this to make data available to the source.
  /// The producer must synchronize its CUDA stream before calling publish()
  /// so that the GPU data is ready when the consumer reads it.
  /// @param key The unique key identifying this transfer (taskId, dest, seq)
  /// @param data The packed_columns data to share (nullptr for atEnd)
  /// @param numRows Logical rows in 'data'; 0 for atEnd
  /// @param atEnd True if this is the end-of-stream marker
  /// @param producerError Why the producer ended before sending all its
  /// output; set only with 'atEnd'
  /// @param onRetrieved Runs inside the poll() that hands 'data' to the
  /// source, under the registry lock, so that a removal of the producer sees
  /// the data either read or still here. It must not throw or call into the
  /// registry. It does not run if the task is cancelled first. May be empty.
  /// @return A future that completes when source has retrieved the data
  [[nodiscard]] std::future<void> publish(
      const IntraNodeTransferKey& key,
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows,
      bool atEnd,
      std::optional<std::string> producerError,
      std::function<void()> onRetrieved);

  /// @brief Non-blocking poll for intra-node transfer data.
  /// Returns immediately whether data is available or not.
  /// @param key The unique key identifying this transfer
  /// @return nullopt if not ready or the task is frozen, or
  ///         IntraNodeTransferResult if ready. Data is nullptr when atEnd is
  ///         true. For a cancelled task the result is atEnd with the
  ///         producerError given to cancelTask().
  [[nodiscard]] std::optional<IntraNodeTransferResult> poll(
      const IntraNodeTransferKey& key);

  /// @brief Cancel all pending transfers for a task.
  /// Called when a producing task is removed. Subsequent poll() calls for
  /// this taskId will return an atEnd result instead of nullopt.
  /// @param taskId The task to cancel
  /// @param producerError Why the producer ended before sending all its
  /// output, or nullopt if it sent everything. The first reason recorded for
  /// a task wins.
  void cancelTask(
      std::string_view taskId,
      std::optional<std::string> producerError);

  /// @brief Holds back poll() for a task until unfreezeTask(): poll() returns
  /// nullopt meanwhile, unless the task is cancelled. Called while the
  /// producer's removal decides whether its consumers read all its output, so
  /// that no page is read in the middle of that decision. publish() is
  /// unaffected.
  /// @param taskId The task to freeze
  void freezeTask(std::string_view taskId);

  /// @brief Ends a freezeTask(). Does not allocate.
  /// @param taskId The task to unfreeze
  void unfreezeTask(std::string_view taskId);

  /// @brief Remove a task from the cancelled tasks.
  /// Called when a task ID is (re)initialized, so that the cancelled tasks
  /// do not grow unboundedly across queries.
  /// @param taskId The task to clear from the cancelled tasks
  void clearCancelledTask(std::string_view taskId);

 private:
  IntraNodeTransferRegistry() = default;

  std::map<IntraNodeTransferKey, std::shared_ptr<IntraNodeTransferEntry>>
      registry_;
  // Cancelled tasks and the producerError each was cancelled with.
  std::unordered_map<std::string, std::optional<std::string>> cancelledTasks_;
  // One element per freezeTask() not yet ended, so that overlapping removals of
  // a task each keep it frozen. The transparent comparator lets unfreezeTask()
  // find a task without allocating: it runs from a scope guard.
  std::multiset<std::string, std::less<>> frozenTasks_;
  std::mutex mutex_;
};

} // namespace facebook::velox::ucx_exchange
