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

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/exec/Exchange.h"
#include "velox/experimental/cudf-exchange/CommElement.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/CudfExchangeProtocol.h"
#include "velox/experimental/cudf-exchange/CudfExchangeQueue.h"
#include "velox/experimental/cudf-exchange/EndpointRef.h"
#include "velox/experimental/cudf-exchange/PartitionKey.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

using namespace facebook::velox::exec;
namespace facebook::velox::cudf_exchange {

struct CudfExchangeMetrics {
  CudfExchangeMetrics()
      : numPackedColumns_(RuntimeMetric(RuntimeCounter::Unit::kNone)),
        totalBytes_(RuntimeCounter::Unit::kBytes),
        rttPerRequest_(RuntimeMetric(RuntimeCounter::Unit::kNanos)) {}
  RuntimeMetric numPackedColumns_; // total number of packed columns received.
  RuntimeMetric totalBytes_; // total number of bytes received
  RuntimeMetric rttPerRequest_;
};

/// The CudfExchangeSource is the client that communicates with the remote
/// CudfExchangeServer. There is a distinct instance for every remoteTask. It
/// encapsulates the client side of the transport layer initiating communication
/// and handling callbacks.
/// The higher layer  initiates the transfer of a single via calling request
/// CudfVector by calling CudfExchangeSource::request. The transfer of one
/// CudfVector involves two transport exchanges
/// - tx the metadata
/// - tx the data
/// The higher layer needs to call for each transfer

class CudfExchangeSource
    : public CommElement,
      public std::enable_shared_from_this<CudfExchangeSource> {
 public:
  virtual ~CudfExchangeSource() = default;

  // factory method to create a cudf exchange resource.
  static std::shared_ptr<CudfExchangeSource> create(
      const std::string& url,
      const std::shared_ptr<CudfExchangeQueue>& queue);

  bool supportsMetrics() const {
    return true;
  }

  /// Returns true if there is no request to the source pending or if
  /// this should be retried. If true, the caller is expected to call
  /// request(). This is expected to be called while holding lock over
  /// queue_.mutex(). This sets the status of 'this' to be pending. The
  /// caller is thus expected to call request() without holding a lock over
  /// queue_.mutex(). This pattern prevents multiple exchange consumer
  /// threads from issuing the same request.
  bool shouldRequestLocked();

  /// @brief Called by the upperlayer, e.g CudfExchange to request a transfer
  /// from remote task
  /// @param taskId the timeout we are prepared to wait on the server before
  /// failing the transfer
  /// @return The future that will indicate when the transfer is done or failed.
  folly::SemiFuture<ExchangeSource::Response> request(
      std::chrono::microseconds maxWait);

  /// @brief Advances the UCXX communication that was started by issuing
  /// "request"
  void process() override;

  /// Close the exchange source. May be called before all data
  /// has been received and processed. This can happen in case
  /// of an error or an operator like Limit aborting the query
  /// once it received enough data.
  void close();

  // Returns runtime statistics. ExchangeSource is expected to report
  // background CPU time by including a runtime metric named
  // ExchangeClient::kBackgroundCpuTimeMs.
  folly::F14FastMap<std::string, int64_t> stats() const;

  /// Returns runtime statistics. ExchangeSource is expected to report
  /// Specify units of individual counters in ExchangeSource.
  /// for an example: 'totalBytes ：count: 9, sum: 11.17GB, max: 1.39GB,
  /// min:  1.16GB'
  folly::F14FastMap<std::string, RuntimeMetric> metrics() const;

  std::string toString() {
    std::stringstream out;
    out << "[ExchangeSource " << partitionKey_.toString() << "]";
    return out.str();
  }

  folly::dynamic toJson() {
    folly::dynamic obj = folly::dynamic::object;
    obj["remoteTaskId"] = partitionKey_.taskId;
    obj["destination"] = partitionKey_.destination;
    obj["closed"] = std::to_string(closed_);
    return obj;
  }

 private:
  enum class ReceiverState : uint32_t {
    Created,
    WaitingForHandshakeComplete,
    HandshakeComplete,
    ReadyToReceive,
    WaitingForMetadata,
    WaitingForData,
    Done
  };

  struct DataAndMetadata {
    MetadataMsg metadata;
    std::unique_ptr<rmm::device_buffer> dataBuf;
  };

  /// @brief The constructor is private in order to ensure that exchange sources
  /// are always generated through a shared pointer. This ensures that
  /// shared_from_this works properly.
  /// @param host The hostname of the upstream CudfExchangeServer
  /// @param port The portnumber of the upstream CudfExchangeServer
  /// @param partitionKey the partition identifier that the remote task is
  /// writing to
  /// @param queue the queue of packed_columns that we write to when data is
  /// available
  explicit CudfExchangeSource(
      const std::shared_ptr<Communicator> communicator,
      const std::string& host,
      uint16_t port,
      const PartitionKey& partitionKey,
      const std::shared_ptr<CudfExchangeQueue> queue);

  // Extracts taskId and destinationId from the path part of the task URL
  static PartitionKey extractTaskAndDestinationId(const std::string& path);

  /// @return A shared pointer to itself.
  std::shared_ptr<CudfExchangeSource> getSelfPtr();

  // Put the received data into the queue and keep track of the promises.
  void enqueueAndFulfillPromise(
      std::unique_ptr<cudf::packed_columns> columns,
      MetadataMsg& metadata);

  /// Completes the future returned from 'request()' if it hasn't completed
  /// already.
  bool checkSetRequestPromise();

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

  /// @brief Called by the transport layer when data is availab;e
  /// @param status indication by transport layer of transfer status
  /// @param arg the serialized form of the metadata
  void onMetadata(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Called by the transport layer when data is availab;e
  /// @param status indication by transport layer of transfer status
  /// @param arg
  void onData(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Sets the new state of this exchange source using
  /// sequential consistency.
  /// @param newState the new state of the CudfExchangeSource.
  void setState(ReceiverState newState) {
    state_.store(newState, std::memory_order_seq_cst);
  }

  /// @brief Returns the state.
  ReceiverState getState() {
    return state_.load(std::memory_order_seq_cst);
  }

  /// @brief Sets the state to "desired" if and only if the current
  /// state is "expected".
  /// @param expected The expected state
  /// @param desired The desired state
  /// @return Returns true if state was changed, false otherwise.
  bool setStateIf(ReceiverState expected, ReceiverState desired);

  // The connection parameters
  const std::string host_;
  uint16_t port_;

  const PartitionKey partitionKey_;
  const uint32_t
      partitionKeyHash_; // A hash of above, used to create unique tags.

  std::atomic<ReceiverState> state_;

  uint32_t sequenceNumber_{0};

  // The shared queue of packed_columns that all CudfExchangeSources write to
  const std::shared_ptr<CudfExchangeQueue> queue_{nullptr};

  // the promise made to the upper layer initiating the communication
  VeloxPromise<ExchangeSource::Response> promise_{
      VeloxPromise<ExchangeSource::Response>::makeEmpty()};

  std::atomic<bool> requestIssuedBeforeHandshakeComplete_{false};
  std::atomic<bool> requestPending_{false};
  std::atomic<bool> closed_{false};
  bool atEnd_{false}; // set when "atEnd" is being received.

  // Some metrics/counters:
  CudfExchangeMetrics metrics_;

  // The outstanding request - there can only be one outstanding request
  // at any point in time. Used for handshake, metadata and data.
  // NOTE: The request owns/holds a reference to the upcall function
  // and must therefore exist until the upcall is done.
  std::shared_ptr<ucxx::Request> request_{nullptr};
};

} // namespace facebook::velox::cudf_exchange
