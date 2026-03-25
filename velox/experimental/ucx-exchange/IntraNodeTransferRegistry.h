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
#include <rmm/cuda_stream_view.hpp>
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>

namespace facebook::velox::ucx_exchange {

/// @brief Key for identifying intra-node transfer entries in the registry.
/// Used when UcxExchangeServer and UcxExchangeSource are on the same node.
struct IntraNodeTransferKey {
  std::string taskId;
  uint32_t destination;
  uint32_t sequenceNumber;

  bool operator<(const IntraNodeTransferKey& other) const {
    if (taskId != other.taskId)
      return taskId < other.taskId;
    if (destination != other.destination)
      return destination < other.destination;
    return sequenceNumber < other.sequenceNumber;
  }
};

/// @brief Result from intra-node transfer containing data, stream, and end
/// marker.
struct IntraNodeTransferResult {
  std::shared_ptr<cudf::packed_columns> data;
  rmm::cuda_stream_view stream; // Stream on which data was produced
  bool atEnd{false}; // True if this is the end-of-stream marker
};

/// @brief Entry in the intra-node transfer registry containing the data and
/// synchronization primitives. The server publishes data and waits for
/// retrieval; the source waits for data availability and retrieves it.
struct IntraNodeTransferEntry {
  std::shared_ptr<cudf::packed_columns> data;
  rmm::cuda_stream_view stream{
      rmm::cuda_stream_default}; // Stream data was produced on
  bool atEnd{false}; // True if this is the end-of-stream marker
  std::promise<void> retrievedPromise; // Server waits on this after publishing
  std::condition_variable dataAvailable; // Source waits on this for data
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
/// retrieves it directly. Synchronization is handled via futures/promises
/// and condition variables.
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
  /// @param key The unique key identifying this transfer (taskId, dest, seq)
  /// @param data The packed_columns data to share (nullptr for atEnd)
  /// @param stream The CUDA stream on which the data was produced. Consumer
  ///        must synchronize with this stream before using the data.
  /// @param atEnd True if this is the end-of-stream marker
  /// @return A future that completes when source has retrieved the data
  [[nodiscard]] std::future<void> publish(
      const IntraNodeTransferKey& key,
      std::shared_ptr<cudf::packed_columns> data,
      rmm::cuda_stream_view stream,
      bool atEnd);

  /// @brief Non-blocking poll for intra-node transfer data.
  /// Returns immediately whether data is available or not.
  /// @param key The unique key identifying this transfer
  /// @return nullopt if not ready, or IntraNodeTransferResult if ready.
  ///         Data is nullptr when atEnd is true.
  [[nodiscard]] std::optional<IntraNodeTransferResult> poll(
      const IntraNodeTransferKey& key);

  /// @brief Wait for and retrieve data from intra-node transfer registry.
  /// Source calls this - blocks until data is available or timeout.
  /// WARNING: This can block, use with caution on single-threaded contexts.
  /// @param key The unique key identifying this transfer
  /// @param timeout Maximum time to wait for data
  /// @return IntraNodeTransferResult. Data is nullptr if atEnd or timeout.
  [[nodiscard]] IntraNodeTransferResult waitFor(
      const IntraNodeTransferKey& key,
      std::chrono::milliseconds timeout);

  /// @brief Retrieve data from intra-node transfer registry (legacy method).
  /// Called by UcxExchangeSource when isIntraNodeTransfer is true.
  /// The entry is removed from the registry and the promise is fulfilled.
  /// @param key The unique key identifying this transfer
  /// @return The shared data, or nullptr if not found
  std::shared_ptr<cudf::packed_columns> retrieve(
      const IntraNodeTransferKey& key);

  /// @brief Cancel all pending transfers for a task.
  /// Called when a producing task is removed. Subsequent poll() calls for
  /// this taskId will return an atEnd result instead of nullopt.
  /// @param taskId The task to cancel
  void cancelTask(const std::string& taskId);

  /// @brief Remove a task from the cancelled set.
  /// Called when a task ID is (re)initialized, so that the cancelledTasks_
  /// set does not grow unboundedly across queries.
  /// @param taskId The task to clear from the cancelled set
  void clearCancelledTask(const std::string& taskId);

 private:
  IntraNodeTransferRegistry() = default;

  std::map<IntraNodeTransferKey, std::shared_ptr<IntraNodeTransferEntry>>
      registry_;
  std::unordered_set<std::string> cancelledTasks_;
  std::mutex mutex_;
};

} // namespace facebook::velox::ucx_exchange
