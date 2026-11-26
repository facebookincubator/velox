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

#include "velox/common/EnumDeclare.h"
#include "velox/common/EnumDefine.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/exec/Exchange.h"
#include "velox/experimental/ucx-exchange/CommElement.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <chrono>
#include <cuda/stream>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace facebook::velox::ucx_exchange {

struct UcxExchangeMetrics {
  UcxExchangeMetrics()
      : numPackedColumns_(RuntimeMetric(RuntimeCounter::Unit::kNone)),
        totalBytes_(RuntimeCounter::Unit::kBytes),
        rttPerRequest_(RuntimeMetric(RuntimeCounter::Unit::kNanos)) {}
  RuntimeMetric numPackedColumns_; // total number of packed columns received.
  RuntimeMetric totalBytes_; // total number of bytes received
  RuntimeMetric rttPerRequest_;
};

/// The UcxExchangeSource is the client that communicates with the remote
/// UcxExchangeServer. There is a distinct instance for every remoteTask. It
/// encapsulates the client side of the transport layer initiating communication
/// and handling callbacks.
/// The higher layer  initiates the transfer of a single via calling request
/// CudfVector by calling UcxExchangeSource::request. The transfer of one
/// CudfVector involves two transport exchanges
/// - tx the metadata
/// - tx the data
/// The higher layer needs to call for each transfer

class UcxExchangeSource
    : public CommElement,
      public std::enable_shared_from_this<UcxExchangeSource> {
 public:
  // Public for logging and the VELOX_DEFINE_EMBEDDED_ENUM_NAME names map.
  enum class ReceiverState : uint32_t {
    Created,
    WaitingForHandshakeComplete,
    WaitingForHandshakeResponse,
    ReadyToReceive,
    WaitingForMetadata,
    WaitingForData,
    WaitingForIntraNodeData,
    Done,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(ReceiverState);

  virtual ~UcxExchangeSource() = default;

  /// Creates a UCX exchange source and registers it with the Communicator,
  /// which may run it at once.
  /// @param sourceId The id of the source's entry in 'queue', which the caller
  /// added before.
  static std::shared_ptr<UcxExchangeSource> create(
      std::string_view taskId,
      std::string_view url,
      const std::shared_ptr<UcxExchangeQueue>& queue,
      UcxExchangeQueue::SourceId sourceId);

  bool supportsMetrics() const {
    return true;
  }

  /// @brief Advances the UCXX communication that was started by issuing
  /// "request"
  void process() override;

  /// Closes the source before its stream ended, as when a LIMIT has all the
  /// rows it needs. UcxExchangeClient calls it only for a source whose server
  /// the exchange queue says to tell. Marks the source closed and requeues it;
  /// on the Communicator thread, process() then sends the close message, once
  /// the handshake response has arrived, and cleans up. Safe to call from any
  /// thread.
  void close();

  /// Ends the source with a transport error that names the endpoint's close
  /// 'reason'. The exchange queue ignores the error if the source ended or was
  /// released already. Called on the Communicator thread.
  void onEndpointClosed(std::string_view reason) override;

  /// @brief Called by UcxExchangeClient::next() on the consumer (driver)
  /// thread to wake up this source after it went dormant due to backpressure.
  /// Uses CAS to ensure exactly one wake-up per dormant period.
  void resumeFromBackpressure();

  /// How long a source that closes during the handshake waits for the
  /// handshake response, which tells it whether its server exists. A source
  /// that gets no response in time ends without sending the close message.
  static constexpr std::chrono::seconds kCloseTimeout{1};

  // Backpressure thresholds. Public so UcxExchangeClient can use them.
  static constexpr int32_t kBackpressureHighWaterMark = 32;
  static constexpr int32_t kBackpressureLowWaterMark = 16;

  // Returns runtime statistics. ExchangeSource is expected to report
  // background CPU time by including a runtime metric named
  // ExchangeClient::kBackgroundCpuTimeMs.
  folly::F14FastMap<std::string, int64_t> stats() const;

  /// Returns runtime statistics. ExchangeSource is expected to report
  /// Specify units of individual counters in ExchangeSource.
  /// for an example: 'totalBytes ：count: 9, sum: 11.17GB, max: 1.39GB,
  /// min:  1.16GB'
  folly::F14FastMap<std::string, RuntimeMetric> metrics() const;

  std::string toString() const {
    std::stringstream out;
    out << "@" << taskId_ << " - @" << partitionKey_.toString();
    return out.str();
  }

  folly::dynamic toJson() const {
    folly::dynamic obj = folly::dynamic::object;
    obj["remoteTaskId"] = partitionKey_.taskId;
    obj["destination"] = partitionKey_.destination;
    obj["closed"] = std::to_string(closed_);
    return obj;
  }

 private:
  struct DataAndMetadata {
    MetadataMsg metadata;
    std::unique_ptr<rmm::device_buffer> dataBuf;
    cuda::stream_ref stream{
        cudaStream_t{cudaStreamDefault}}; // The stream used to allocate dataBuf
  };

  // @brief The constructor is private in order to ensure that exchange sources
  // are always generated through a shared pointer. This ensures that
  // shared_from_this works properly.
  // @param taskId The local task identifier
  // @param host The hostname of the upstream UcxExchangeServer
  // @param port The portnumber of the upstream UcxExchangeServer
  // @param partitionKey the partition identifier that the remote task is
  // writing to
  // @param queue the queue of packed tables that we write to when data is
  // available
  explicit UcxExchangeSource(
      const std::shared_ptr<Communicator> communicator,
      std::string_view taskId,
      std::string_view host,
      uint16_t port,
      const PartitionKey& partitionKey,
      const std::shared_ptr<UcxExchangeQueue> queue,
      UcxExchangeQueue::SourceId sourceId);

  // Extracts taskId and destinationId from the path part of the task URL
  static PartitionKey extractTaskAndDestinationId(std::string_view path);

  // @return A shared pointer to itself.
  std::shared_ptr<UcxExchangeSource> getSelfPtr();

  // Put the received data into the exchange queue.
  void enqueue(PackedTableWithStreamPtr data);

  // @brief Sets the endpoint for this receiver.
  void setEndpoint(std::shared_ptr<EndpointRef> endpointRef);

  // @brief Sends a handshake request to the server. The endpoint must exist.
  void sendHandshake();

  // @brief Called by the transport layer when handshake is completed
  // @param status indication by transport layer of transfer status
  // @param arg NOT USED at the moment
  void onHandshake(ucs_status_t status, std::shared_ptr<void> arg);

  // @brief Waits for metadata and installs the onMetadata callback.
  void getMetadata();

  // @brief Called by the transport layer when data is available
  // @param status indication by transport layer of transfer status
  // @param arg the serialized form of the metadata
  void onMetadata(ucs_status_t status, std::shared_ptr<void> arg);

  // @brief Called by the transport layer when data is available
  // @param status indication by transport layer of transfer status
  // @param arg
  void onData(ucs_status_t status, std::shared_ptr<void> arg);

  // @brief Initiates receiving the HandshakeResponse from server. Posts the
  // receive once, from process(); a post that fails ends the source.
  void receiveHandshakeResponse();

  // @brief Called when HandshakeResponse is received from server.
  // @param status indication by transport layer of transfer status
  // @param arg The HandshakeResponse data
  void onHandshakeResponse(ucs_status_t status, std::shared_ptr<void> arg);

  // @brief For intra-node transfer: initiates waiting for data from registry.
  void waitForIntraNodeData();

  // @brief For intra-node transfer: handles data retrieved from registry.
  // @param data The packed_columns from registry (nullptr if atEnd or error)
  // @param numRows Logical rows in 'data'
  // @param atEnd True if this is end-of-stream
  // @param outcome How the producer ended the channel; meaningful only with
  // 'atEnd'
  void onIntraNodeData(
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows,
      bool atEnd,
      const ChannelOutcome& outcome);

  // @brief Sets the new state of this exchange source using
  // sequential consistency. Logs transitions at VLOG(2).
  // @param newState the new state of the UcxExchangeSource.
  void setState(ReceiverState newState);

  // @brief Returns the state.
  ReceiverState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  // Removes the state associated with the source, called by the state-machine.
  void cleanUp();

  // Runs process() for a closed source. Returns false if the source is in
  // Created, which connects as usual so that it can send the close message.
  bool processClosed();

  // Returns the completion callback of a request, which runs 'handler' on this
  // source if the source still exists. No exception may leave a UCX callback,
  // so one that 'handler' throws ends the source with a transport error.
  ucxx::RequestCallbackUserFunction completionCallback(
      void (UcxExchangeSource::*handler)(ucs_status_t, std::shared_ptr<void>));

  // Ends the source with a transport error that names 'error', which a
  // completion handler threw. Does not throw.
  void failCompletion(std::string_view error) noexcept;

  // Sends the close message to the server. Does not wait for the send to
  // complete; the Communicator keeps the request, and with it the message,
  // until it completes.
  void sendClose();

  // Reports the end of the source to the exchange queue, which keeps the first
  // end it is told, and moves to Done. Every end of the source goes through
  // here.
  void end(ChannelOutcome outcome);

  // Returns '<prefix> host <host>:<port>, task <partition key>: <detail>', the
  // form of every error that names this source's producer.
  std::string formatError(std::string_view prefix, std::string_view detail)
      const;

  // Ends the source with the outcome the producer gave the channel's end, in
  // an end message or an intra-node end: the producer's error names this
  // source's producer, and anything else is a clean end.
  void endWithProducerOutcome(const ChannelOutcome& outcome);

  // @brief Sets the state to "desired" if and only if the current
  // state is "expected".
  // @param expected The expected state
  // @param desired The desired state
  // @return Returns true if state was changed, false otherwise.
  bool setStateIf(ReceiverState expected, ReceiverState desired);

  // The connection parameters
  const std::string host_;
  uint16_t port_;
  const std::string taskId_;

  const PartitionKey partitionKey_;
  const uint32_t
      partitionKeyHash_; // A hash of above, used to create unique tags.

  std::atomic<ReceiverState> state_;

  uint32_t sequenceNumber_{0};
  uint32_t intraNodePollCount_{0};

  // The shared queue of packed tables that all UcxExchangeSources write to
  const std::shared_ptr<UcxExchangeQueue> queue_{nullptr};
  // The id of this source's entry in 'queue_'.
  const UcxExchangeQueue::SourceId sourceId_;
  std::atomic<bool> closed_{false};
  // Set kCloseTimeout after close(): a source still waiting for the handshake
  // response then ends without the close message.
  std::atomic<bool> closeDeadlineExpired_{false};

  // True if the server detected that this source is on the same node.
  // Set from the isIntraNodeTransfer flag in HandshakeResponse, which the
  // server determines by comparing this source's listener address (sent in
  // HandshakeMsg) with its own Communicator's listener address.
  // When true, intra-node transfer optimizations bypass UCXX transfers.
  bool isIntraNodeTransfer_{false};

  // Backpressure: when queue exceeds kBackpressureHighWaterMark, the source
  // goes dormant. The consumer thread wakes it via resumeFromBackpressure()
  // when the queue drains to kBackpressureLowWaterMark.
  std::atomic<bool> backpressureActive_{false};

  // Some metrics/counters:
  UcxExchangeMetrics metrics_;

  // The handshake send. Kept apart from 'request_', which holds the response
  // receive that process() posts once the send completed: UCX may complete
  // the send inside the call that posts it.
  std::shared_ptr<ucxx::Request> handshakeRequest_;

  // True once receiveHandshakeResponse() posted the receive of the
  // HandshakeResponse. Used only on the Communicator thread.
  bool responseReceivePosted_{false};

  // The outstanding receive - there can only be one outstanding receive at any
  // point in time. Used for the handshake response, metadata and data.
  // NOTE: The request owns/holds a reference to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> request_{nullptr};

  // Completed UCXX requests are kept alive here to prevent use-after-free.
  // UCP's ucp_wireup_replay_pending_requests can fire callbacks on already-
  // completed requests; if the ucxx::Request has been freed, the callback
  // lambda is in freed memory and crashes. Retaining them here ensures the
  // Request (and its callback lambda) stays valid for the lifetime of this
  // source.
  std::vector<std::shared_ptr<ucxx::Request>> completedRequests_;
};

} // namespace facebook::velox::ucx_exchange
