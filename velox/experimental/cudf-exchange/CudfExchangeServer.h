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
#include <velox/experimental/cudf-exchange/CudfOutputQueueManager.h>
#include <chrono>
#include <future>
#include <memory>
#include <tuple>
#include "velox/experimental/cudf-exchange/CommElement.h"
#include "velox/experimental/cudf-exchange/EndpointRef.h"
#include "velox/experimental/cudf-exchange/PartitionKey.h"

namespace facebook::velox::cudf_exchange {

class CudfExchangeServer
    : public CommElement,
      public std::enable_shared_from_this<CudfExchangeServer> {
 public:
  /// @brief Factory method to create a CudfExchangeServer.
  /// @param communicator The Communicator instance.
  /// @param endpointRef The endpoint reference for UCXX communication.
  /// @param key The partition key identifying the data to serve.
  /// @param sourceListenerIp The source's Communicator listener IP address
  ///        (from HandshakeMsg) for same-node detection.
  /// @param sourceListenerPort The source's Communicator listener port
  ///        (from HandshakeMsg) for same-node detection.
  static std::shared_ptr<CudfExchangeServer> create(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      const std::string& sourceListenerIp,
      uint16_t sourceListenerPort);

  void process() override;

  void close() override;

  std::string toString();

  const PartitionKey& getPartitionKey() const {
    return partitionKey_;
  }

  /// @brief Returns true if this server detected same-node with the source.
  bool isIntraNodeTransfer() const {
    return isIntraNodeTransfer_;
  }

 private:
  enum class ServerState : uint32_t {
    Created,
    ReadyToTransfer,
    WaitingForDataFromQueue,
    DataReady,
    WaitingForSendComplete,
    WaitingForIntraNodeRetrieve, // Intra-node transfer: waiting for source to
                                 // retrieve
    Done
  };

  explicit CudfExchangeServer(
      const std::shared_ptr<Communicator> communicator,
      std::shared_ptr<EndpointRef> endpointRef,
      const PartitionKey& key,
      const std::string& sourceListenerIp,
      uint16_t sourceListenerPort);

  /// @return A shared pointer to itself.
  std::shared_ptr<CudfExchangeServer> getSelfPtr();

  /// @brief Sends metadata and data to the connected receiver.
  void sendData();

  /// @brief Completion handler after data has been sent.
  void sendComplete(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Completion handler for intra-node transfer after source retrieves
  /// data.
  void onIntraNodeRetrieveComplete();

  /// @brief Sets the new state of this exchange server using
  /// sequential consistency.
  /// @param newState the new state of the CudfExchangeServer.
  void setState(ServerState newState) {
    state_.store(newState, std::memory_order_seq_cst);
  }

  /// @brief Returns the state.
  ServerState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  std::string getStateAsString() {
    const std::string stateMap[] = {
        "Created",
        "ReadyToTransfer",
        "WaitingForDataFromQueue",
        "DataReady",
        "WaitingForSendComplete",
        "WaitingForIntraNodeRetrieve",
        "Done"};
    return stateMap[static_cast<uint32_t>(state_.load())];
  }

  const PartitionKey partitionKey_;
  const uint32_t
      partitionKeyHash_; // A hash of above, used to create unique tags.

  /// Source's Communicator listener IP address (from HandshakeMsg).
  const std::string sourceListenerIp_;
  /// Source's Communicator listener port (from HandshakeMsg).
  const uint16_t sourceListenerPort_;
  /// True if server and source are on the same node (same Communicator
  /// listener address). When true, data is passed via IntraNodeTransferRegistry
  /// instead of UCXX transfer.
  bool isIntraNodeTransfer_{false};

  std::atomic<ServerState> state_;
  std::unique_ptr<cudf::packed_columns> dataPtr_{nullptr};
  std::recursive_mutex dataMutex_; // mutex for above ptr.
  std::atomic<bool> closed_{false};

  /// Future for intra-node transfer - signaled when source retrieves data.
  std::future<void> intraNodeRetrieveFuture_;

  /// For intra-node transfer: true if the last published entry was atEnd.
  bool intraNodeAtEndPublished_{false};

  uint32_t sequenceNumber_{0};

  // The outstanding requests - there can only be one outstanding request
  // of each type at any point in time.
  // NOTE: The request owns/holds references to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> metaRequest_{nullptr};
  std::shared_ptr<ucxx::Request> dataRequest_{nullptr};

  std::chrono::time_point<std::chrono::high_resolution_clock> sendStart_;
  std::size_t bytes_;

  std::shared_ptr<CudfOutputQueueManager> queueMgr_;
};

} // namespace facebook::velox::cudf_exchange
