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
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace facebook::velox::cudf_exchange {

/// @brief Key for identifying intra-node transfer entries in the registry.
/// Used when CudfExchangeServer and CudfExchangeSource are on the same node.
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

/// @brief Entry in the intra-node transfer registry containing the data and
/// synchronization primitives. The server publishes data and waits for
/// retrieval; the source waits for data availability and retrieves it.
struct IntraNodeTransferEntry {
  std::shared_ptr<cudf::packed_columns> data;
  bool atEnd{false}; // True if this is the end-of-stream marker
  std::promise<void> retrievedPromise; // Server waits on this after publishing
  std::condition_variable dataAvailable; // Source waits on this for data
  std::mutex entryMutex;
  bool ready{false}; // True when data is ready to retrieve
};

/// @brief Singleton registry for intra-node data transfers.
///
/// When CudfExchangeServer and CudfExchangeSource are on the same node
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
  /// @param atEnd True if this is the end-of-stream marker
  /// @return A future that completes when source has retrieved the data
  std::future<void> publish(
      const IntraNodeTransferKey& key,
      std::shared_ptr<cudf::packed_columns> data,
      bool atEnd);

  /// @brief Non-blocking poll for intra-node transfer data.
  /// Returns immediately whether data is available or not.
  /// @param key The unique key identifying this transfer
  /// @return nullopt if not ready, or pair of (data, atEnd) if ready.
  ///         Data is nullptr when atEnd is true.
  std::optional<std::pair<std::shared_ptr<cudf::packed_columns>, bool>> poll(
      const IntraNodeTransferKey& key);

  /// @brief Wait for and retrieve data from intra-node transfer registry.
  /// Source calls this - blocks until data is available or timeout.
  /// WARNING: This can block, use with caution on single-threaded contexts.
  /// @param key The unique key identifying this transfer
  /// @param timeout Maximum time to wait for data
  /// @return Pair of (data, atEnd). Data is nullptr if atEnd or timeout.
  std::pair<std::shared_ptr<cudf::packed_columns>, bool> waitFor(
      const IntraNodeTransferKey& key,
      std::chrono::milliseconds timeout);

  /// @brief Retrieve data from intra-node transfer registry (legacy method).
  /// Called by CudfExchangeSource when isIntraNodeTransfer is true.
  /// The entry is removed from the registry and the promise is fulfilled.
  /// @param key The unique key identifying this transfer
  /// @return The shared data, or nullptr if not found
  std::shared_ptr<cudf::packed_columns> retrieve(
      const IntraNodeTransferKey& key);

 private:
  IntraNodeTransferRegistry() = default;

  std::map<IntraNodeTransferKey, std::shared_ptr<IntraNodeTransferEntry>>
      registry_;
  std::mutex mutex_;
};

} // namespace facebook::velox::cudf_exchange
