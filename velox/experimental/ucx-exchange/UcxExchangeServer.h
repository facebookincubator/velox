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
#include <folly/Synchronized.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <velox/exec/Task.h>
#include <velox/experimental/ucx-exchange/UcxOutputQueueManager.h>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include "velox/common/EnumDeclare.h"
#include "velox/common/EnumDefine.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"

namespace facebook::velox::ucx_exchange {

class UcxExchangeServer
    : public CommElement,
      public std::enable_shared_from_this<UcxExchangeServer> {
 public:
  // Public for logging and the VELOX_DEFINE_EMBEDDED_ENUM_NAME names map.
  enum class ServerState : uint32_t {
    Created,
    ReadyToTransfer,
    WaitingForDataFromQueue,
    DataReady,
    WaitingForSendComplete,
    WaitingForIntraNodeRetrieve,
    Done,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(ServerState);

  /// @brief Factory method to create a UcxExchangeServer.
  /// @param communicator The Communicator instance.
  /// @param endpointRef The endpoint reference for UCXX communication.
  /// @param key The partition key identifying the data to serve.
  /// @param isIntraNodeTransfer True if the source is on the same node,
  ///        determined by checking if the peer's IP is in the local IP set.
  static std::shared_ptr<UcxExchangeServer> create(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      bool isIntraNodeTransfer);

  void process() override;

  void close() noexcept override;

  std::string toString();

  const PartitionKey& getPartitionKey() const {
    return partitionKey_;
  }

  /// @brief Returns true if this server detected same-node with the source.
  bool isIntraNodeTransfer() const {
    return isIntraNodeTransfer_;
  }

 private:
  explicit UcxExchangeServer(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      bool isIntraNodeTransfer);

  /// @return A shared pointer to itself.
  std::shared_ptr<UcxExchangeServer> getSelfPtr();

  /// @brief Sends metadata and data to the connected receiver.
  void sendData();

  /// @brief Completion handler after data has been sent.
  void sendComplete(ucs_status_t status);

  /// Handles metadata completion on the serialized state-machine thread.
  void metadataSendComplete(ucs_status_t status, uint64_t metadataTag);

  /// Contains failures while handing a C/UCXX callback to process().
  void handleCallbackDispatchFailure(const char* callback) noexcept;

  /// Contains all other state-machine failures so Communicator::run() cannot
  /// be unwound by an exchange-specific allocation or lifecycle error.
  void handleProcessFailure(const char* error) noexcept;

  /// @brief Completion handler for intra-node transfer after source retrieves
  /// data.
  void onIntraNodeRetrieveComplete();

  /// Releases a committed intra-node payload exactly once.
  void releaseIntraNodeInFlightBytes();

  /// Releases a dequeued payload that was never committed to a transport.
  void releasePendingData();

  /// Releases a completed phase request immediately, or hands a genuinely
  /// in-flight request to Communicator until UCXX marks it complete.
  void retireRequest(std::shared_ptr<ucxx::Request>& request);

  /// @brief Sets the new state of this exchange server using
  /// sequential consistency. Logs transitions at VLOG(2).
  /// @param newState the new state of the UcxExchangeServer.
  void setState(ServerState newState);

  /// @brief Returns the state.
  ServerState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  const PartitionKey partitionKey_;
  const uint32_t
      partitionKeyHash_; // A hash of above, used to create unique tags.

  /// True if server and source are on the same node (determined by checking
  /// if peer's actual IP is in the local IP set). When true, data is passed
  /// via IntraNodeTransferRegistry instead of UCXX transfer.
  bool isIntraNodeTransfer_{false};

  std::atomic<ServerState> state_{ServerState::Created};
  std::shared_ptr<cudf::packed_columns> dataPtr_{nullptr};
  /// Logical rows in 'dataPtr_', taken from the output queue rather than from
  /// the packed table, which reports zero rows when it has no columns.
  vector_size_t dataNumRows_{0};
  /// Protects dataPtr_. Must be recursive because sendData() holds the lock
  /// when calling tagSend(), and for small messages UCX completes inline via
  /// its fast-completion path, firing the sendComplete() callback on the same
  /// thread while the lock is still held.
  std::recursive_mutex dataMutex_;
  /// Excludes process() from close() and serializes callback event dispatch.
  /// Recursive because process() reaches close() in the Done state.
  std::recursive_mutex processMutex_;
  /// True while the communicator thread is inside process(). Inline UCXX
  /// callbacks defer emergency close until the active send call has returned
  /// and published its Request member.
  std::atomic<bool> processing_{false};
  std::atomic<bool> closed_{false};
  /// Protected by processMutex_. A separate guard makes cleanup retryable
  /// after closed_ has already stopped new work.
  bool cleanupComplete_{false};

  /// Future for intra-node transfer - signaled when source retrieves data.
  std::future<void> intraNodeRetrieveFuture_;

  /// For intra-node transfer: true if the last published entry was atEnd.
  bool intraNodeAtEndPublished_{false};

  uint32_t sequenceNumber_{0};
  uint32_t intraNodePollCount_{0};

  // The outstanding requests - there can only be one outstanding request
  // of each type at any point in time.
  // NOTE: The request owns/holds references to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> metaRequest_{nullptr};
  std::shared_ptr<ucxx::Request> dataRequest_{nullptr};

  std::chrono::time_point<std::chrono::high_resolution_clock> sendStart_;
  std::size_t bytes_{0};

  // Keeps the output queue alive across manager removal and async callbacks so
  // committed byte accounting can always be released exactly once.
  std::shared_ptr<UcxOutputQueue> outputQueue_;
  std::atomic<bool> intraNodeBytesInFlight_{false};
  bool outputResultsDeleted_{false};

  std::shared_ptr<UcxOutputQueueManager> queueMgr_;
};

} // namespace facebook::velox::ucx_exchange
