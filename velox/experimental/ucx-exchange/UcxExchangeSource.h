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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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
    WaitingForReceiveCredit,
    WaitingForData,
    WaitingForIntraNodeData,
    Done,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(ReceiverState);

  virtual ~UcxExchangeSource() = default;

  // factory method to create a UCX exchange source.
  static std::shared_ptr<UcxExchangeSource> create(
      std::string_view taskId,
      std::string_view url,
      const std::shared_ptr<UcxExchangeQueue>& queue);

  /// Publishes this source to the Communicator. The client must first account
  /// for the source in its exchange queue and call setRegistered(), so even an
  /// immediately completed handshake cannot deliver an unbalanced end marker.
  /// Idempotent, and a no-op if close() won the lifecycle race.
  void start();

  bool supportsMetrics() const {
    return true;
  }

  /// @brief Advances the UCXX communication that was started by issuing
  /// "request"
  void process() override;

  /// Close the exchange source. May be called before all data
  /// has been received and processed. This can happen in case
  /// of an error or an operator like Limit aborting the query
  /// once it received enough data.
  void close();

  /// @brief Marks this source as registered with the exchange queue.
  /// Must be called after addSourceLocked() increments numSources_ for this
  /// source. Without this, deliverEndMarker() will not enqueue the nullptr
  /// (preventing spurious numCompleted_ increments for unregistered sources).
  void setRegistered() {
    registered_.store(true, std::memory_order_release);
  }

  /// @brief Called by UcxExchangeClient::next() on the consumer (driver)
  /// thread to wake up this source after it went dormant due to backpressure.
  /// Uses CAS to ensure exactly one wake-up per dormant period.
  void resumeFromBackpressure() noexcept;

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
  enum class CallbackPhase {
    Handshake,
    Metadata,
    Data,
    HandshakeResponse,
  };

  /// Runs one state-machine turn. process() contains any exception from this
  /// method so failures cannot terminate the Communicator loop.
  void processStateMachine();

  struct DataAndMetadata {
    MetadataMsg metadata;
    std::unique_ptr<rmm::device_buffer> dataBuf;
    cuda::stream_ref stream{
        cudaStream_t{cudaStreamDefault}}; // The stream used to allocate dataBuf
    bool receiveBytesReserved{false};
  };

  /// @brief The constructor is private in order to ensure that exchange sources
  /// are always generated through a shared pointer. This ensures that
  /// shared_from_this works properly.
  /// @param taskId The local task identifier
  /// @param host The hostname of the upstream UcxExchangeServer
  /// @param port The portnumber of the upstream UcxExchangeServer
  /// @param partitionKey the partition identifier that the remote task is
  /// writing to
  /// @param queue the queue of packed tables that we write to when data is
  /// available
  explicit UcxExchangeSource(
      const std::shared_ptr<Communicator> communicator,
      std::string_view taskId,
      std::string_view host,
      uint16_t port,
      const PartitionKey& partitionKey,
      const std::shared_ptr<UcxExchangeQueue> queue);

  // Extracts taskId and destinationId from the path part of the task URL
  static PartitionKey extractTaskAndDestinationId(std::string_view path);

  /// @return A shared pointer to itself.
  std::shared_ptr<UcxExchangeSource> getSelfPtr();

  // Put the received data into the exchange queue.
  void enqueue(
      PackedTableWithStreamPtr data,
      uint64_t reservedReceiveBytes = 0,
      bool* receiveReservationActive = nullptr);

  /// @brief Sets the endpoint for this receiver.
  void setEndpoint(std::shared_ptr<EndpointRef> endpointRef);

  /// @brief Sends a handshake request to the server. The endpoint must exist.
  void sendHandshake();

  /// @brief Called by the transport layer when handshake is completed
  /// @param status indication by transport layer of transfer status
  /// @param arg NOT USED at the moment
  void onHandshake(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Waits for metadata and installs the onMetadata callback.
  void getMetadata();

  /// @brief Called by the transport layer when data is available
  /// @param status indication by transport layer of transfer status
  /// @param arg the serialized form of the metadata
  void onMetadata(ucs_status_t status, std::shared_ptr<void> arg);

  /// Reserves receive bytes, allocates the GPU buffer, and posts the data
  /// receive for metadata already accepted from the producer.
  void startDataReceive(std::shared_ptr<DataAndMetadata> dataAndMetadata);

  bool tryReserveReceiveBytes(
      const std::shared_ptr<DataAndMetadata>& dataAndMetadata);

  void releaseReceiveBytes(
      const std::shared_ptr<DataAndMetadata>& dataAndMetadata);

  /// Releases a completed phase request immediately, or hands a genuinely
  /// in-flight request to Communicator until UCXX marks it complete.
  void retireRequest(std::shared_ptr<ucxx::Request>& request);

  /// @brief Called by the transport layer when data is available
  /// @param status indication by transport layer of transfer status
  /// @param arg
  void onData(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Initiates receiving the HandshakeResponse from server.
  void receiveHandshakeResponse();

  /// @brief Called when HandshakeResponse is received from server.
  /// @param status indication by transport layer of transfer status
  /// @param arg The HandshakeResponse data
  void onHandshakeResponse(ucs_status_t status, std::shared_ptr<void> arg);

  /// Runs a UCXX completion after it has been serialized onto process().
  /// Contains all exceptions so none can unwind through a C callback or tear
  /// down the communicator loop.
  void runSerializedCallback(
      CallbackPhase phase,
      ucs_status_t status,
      std::shared_ptr<void> arg) noexcept;

  /// Safely hands a UCXX callback to the Communicator thread. This contains
  /// allocation failures from std::function and the work queue.
  void enqueueSerializedCallback(
      CallbackPhase phase,
      ucs_status_t status,
      std::shared_ptr<void> arg) noexcept;

  /// Records a terminal source failure and schedules state-machine cleanup.
  void failSource(
      std::string_view operation,
      const char* detail = nullptr) noexcept;

  /// Minimal thread-safe terminal path used if callback handoff itself fails.
  /// It deliberately leaves request/reservation cleanup to process().
  void failCallbackDispatch() noexcept;

  /// @brief For intra-node transfer: initiates waiting for data from registry.
  void waitForIntraNodeData();

  /// @brief For intra-node transfer: handles data retrieved from registry.
  /// @param data The packed_columns from registry (nullptr if atEnd or error)
  /// @param numRows Logical rows in 'data'
  /// @param atEnd True if this is end-of-stream
  void onIntraNodeData(
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows,
      bool atEnd,
      uint32_t completedSequence);

  /// @brief Sets the new state of this exchange source using
  /// sequential consistency. Logs transitions at VLOG(2).
  /// @param newState the new state of the UcxExchangeSource.
  void setState(ReceiverState newState);

  /// @brief Returns the state.
  ReceiverState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  // Removes the state associated with the source, called by the state-machine.
  void cleanUp();

  /// @brief Delivers the nullptr end-of-stream marker to the queue exactly
  /// once. Safe to call from any thread. Uses atomic CAS on
  /// endMarkerDelivered_ to guarantee at-most-once delivery.
  /// Returns early without enqueuing if the source was never registered
  /// with the queue (i.e., registered_ is false).
  void deliverEndMarker();

  /// @brief Sets the state to "desired" if and only if the current
  /// state is "expected".
  /// @param expected The expected state
  /// @param desired The desired state
  /// @return Returns true if state was changed, false otherwise.
  bool setStateIf(ReceiverState expected, ReceiverState desired);

  /// Checks queue pressure and publishes the dormant flag while holding the
  /// same mutex consumers use to decide whether to wake sources.
  bool pauseForBackpressureIfNeeded();

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
  // Serializes initial publication with close(). The Communicator may process
  // the source as soon as registerCommElement() is called.
  std::mutex lifecycleMutex_;
  bool started_{false};
  bool cleanupComplete_{false};
  std::atomic<bool> closed_{false};
  bool atEnd_{false}; // set when "atEnd" is being received.

  /// @brief Guards exactly-once delivery of the nullptr end-of-stream marker.
  /// Only one thread can win the CAS and call enqueue(nullptr).
  std::atomic<bool> endMarkerDelivered_{false};

  /// @brief True only after addSourceLocked() has been called for this source.
  /// Prevents deliverEndMarker() from incrementing numCompleted_ for sources
  /// that were never registered with the queue (e.g., created after client
  /// close). Atomic because setRegistered() is called on the driver thread
  /// while deliverEndMarker() reads it on the Communicator thread.
  std::atomic<bool> registered_{false};

  /// True if the server detected that this source is on the same node.
  /// Set from the isIntraNodeTransfer flag in HandshakeResponse, which the
  /// server determines by comparing this source's listener address (sent in
  /// HandshakeMsg) with its own Communicator's listener address.
  /// When true, intra-node transfer optimizations bypass UCXX transfers.
  bool isIntraNodeTransfer_{false};

  // Backpressure: when queue exceeds kBackpressureHighWaterMark, the source
  // goes dormant. The consumer thread wakes it via resumeFromBackpressure()
  // when the queue drains to kBackpressureLowWaterMark.
  std::atomic<bool> backpressureActive_{false};
  std::shared_ptr<DataAndMetadata> pendingDataReceive_;
  std::shared_ptr<DataAndMetadata> activeDataReceive_;

  // Some metrics/counters:
  UcxExchangeMetrics metrics_;

  // Keep request handles classified by phase. Some UCXX operations may invoke
  // their completion inline: assigning every phase into one shared slot can
  // overwrite a data receive posted by a nested metadata callback.
  std::shared_ptr<ucxx::Request> handshakeRequest_{nullptr};
  std::shared_ptr<ucxx::Request> handshakeResponseRequest_{nullptr};
  std::shared_ptr<ucxx::Request> metadataRequest_{nullptr};
  std::shared_ptr<ucxx::Request> dataRequest_{nullptr};
};

} // namespace facebook::velox::ucx_exchange
