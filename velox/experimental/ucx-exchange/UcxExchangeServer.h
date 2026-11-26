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
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string_view>
#include <tuple>
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

  /// Releases the destination, since its consumer is gone, and closes. Called
  /// on the Communicator thread.
  void onEndpointClosed(std::string_view reason) override;

  /// Cancels the outstanding requests and the intra-node transfer that no
  /// source read, unbinds from the producer task's entry
  /// and unregisters from the Communicator and the endpoint. Decides nothing
  /// about the destination. Called on the Communicator thread. A second call
  /// does nothing.
  void close();

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

  // Holds a send's buffer until the send completes, and its status.
  struct MetaSendContext;
  struct DataSendContext;

  /// @brief Sends metadata and data to the connected receiver.
  void sendData();

  // Acts on the statuses that the completions of the current sends recorded.
  // Called by process() in WaitingForSendComplete, with no lock held.
  void onSendsCompleted();

  // Reports a send that did not complete with UCS_OK to the entry, by its
  // cause, and fails the producer task if the entry says so. Called by
  // process() with no lock held.
  void onSendFailed(ucs_status_t status);

  /// @brief Completion handler for intra-node transfer after source retrieves
  /// data.
  void onIntraNodeRetrieveComplete();

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

  std::atomic<ServerState> state_;
  std::shared_ptr<cudf::packed_columns> dataPtr_{nullptr};
  /// Logical rows in 'dataPtr_', taken from the output queue rather than from
  /// the packed table, which reports zero rows when it has no columns.
  vector_size_t dataNumRows_{0};
  /// How the output queue ended the channel. Sent in the end message once
  /// 'dataPtr_' is null. Protected by dataMutex_.
  ChannelOutcome endOutcome_;
  // Protects dataPtr_, which the getData() callback sets on another thread.
  // UCX can complete a send inline, inside the tagSend() that sendData()
  // calls with this lock held, so send completions never take it.
  std::mutex dataMutex_;
  std::atomic<bool> closed_{false};

  // Set when the destination was Released, by a kStop step or the entry's
  // onReleased callback, either of which may run on any thread. process()
  // then cancels the outstanding requests and stops.
  std::atomic<bool> released_{false};

  // The current sends: the metadata or end message, and the chunk's data. A
  // null 'dataContext_' while WaitingForSendComplete means the end message is
  // the only send. Touched on the Communicator thread only.
  std::shared_ptr<MetaSendContext> metaContext_;
  std::shared_ptr<DataSendContext> dataContext_;

  /// Future for intra-node transfer - signaled when source retrieves data, or
  /// when the transfer or the producer task is cancelled. Valid while the
  /// transfer is published and not processed.
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

  // Completed UCXX requests are kept alive here to prevent use-after-free.
  // UCP's ucp_wireup_replay_pending_requests can fire callbacks on already-
  // completed requests; if the ucxx::Request has been freed, the callback
  // lambda is in freed memory and crashes. Retaining them here ensures the
  // Request (and its callback lambda) stays valid for the lifetime of this
  // server.
  std::vector<std::shared_ptr<ucxx::Request>> completedRequests_;

  std::chrono::time_point<std::chrono::high_resolution_clock> sendStart_;
  std::size_t bytes_;

  std::shared_ptr<UcxOutputQueueManager> queueMgr_;

  // The entry of the producer task, bound once in create(). Every call for
  // the task's data goes to it, so that a task that reuses the ID is never
  // reached.
  std::shared_ptr<UcxOutputQueue> entry_;
};

} // namespace facebook::velox::ucx_exchange
