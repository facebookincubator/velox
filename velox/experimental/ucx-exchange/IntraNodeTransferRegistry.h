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
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "velox/experimental/ucx-exchange/UcxChannelState.h"
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
  /// How the channel ended. Meaningful only with 'atEnd'.
  ChannelOutcome outcome;
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
  ChannelOutcome outcome;
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
  /// @param outcome How the channel ended; meaningful only with 'atEnd'
  /// @return A future that completes when source has retrieved the data, or
  /// when the transfer or its task is cancelled
  [[nodiscard]] std::future<void> publish(
      const IntraNodeTransferKey& key,
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows,
      bool atEnd,
      ChannelOutcome outcome);

  /// @brief Non-blocking poll for intra-node transfer data.
  /// Returns immediately whether data is available or not.
  /// @param key The unique key identifying this transfer
  /// @return nullopt if not ready, or IntraNodeTransferResult if ready.
  ///         Data is nullptr when atEnd is true. For a cancelled task the
  ///         result is atEnd with the outcome given to cancelTask().
  [[nodiscard]] std::optional<IntraNodeTransferResult> poll(
      const IntraNodeTransferKey& key);

  /// @brief Cancel all pending transfers for a task.
  /// Called when a producing task is removed. Drops the published transfers,
  /// completes their retrieval promises, and makes subsequent poll() calls for
  /// this taskId return an atEnd result with 'outcome' instead of nullopt.
  /// @param taskId The task to cancel
  /// @param outcome The outcome the removal decided. The first outcome recorded
  /// for a task wins, so that a later call cannot turn an error into a clean
  /// end.
  void cancelTask(std::string_view taskId, ChannelOutcome outcome);

  /// Drops one published transfer, frees its data and completes its retrieval
  /// promise. Called by a server whose destination was Released, since no
  /// source reads the transfer then. Other destinations and later transfers of
  /// the task are unchanged. Does nothing if the transfer was read or
  /// cancelled.
  void cancelTransfer(const IntraNodeTransferKey& key);

  /// @brief Remove a task from the cancelled tasks.
  /// UcxOutputQueueManager calls it when it drops the Removed entry of a task,
  /// once the host destroyed that Task and no server is bound to it, and when
  /// a task ID is initialized again.
  /// @param taskId The task to clear from the cancelled tasks
  void clearCancelledTask(std::string_view taskId);

 private:
  IntraNodeTransferRegistry() = default;

  std::map<IntraNodeTransferKey, std::shared_ptr<IntraNodeTransferEntry>>
      registry_;
  // Cancelled tasks and the outcome each was cancelled with.
  std::unordered_map<std::string, ChannelOutcome> cancelledTasks_;
  // Taken before an entry's entryMutex. Never held together with a lock of
  // UcxOutputQueueManager or UcxOutputQueue.
  std::mutex mutex_;
};

} // namespace facebook::velox::ucx_exchange
