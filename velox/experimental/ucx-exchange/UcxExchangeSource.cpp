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

#include <thread>

#include <cudf/contiguous_split.hpp>
#include <folly/String.h>
#include <folly/Uri.h>
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"

using namespace facebook::velox::exec;
namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<UcxExchangeSource::ReceiverState, std::string_view>&
receiverStateNames() {
  static const folly::F14FastMap<
      UcxExchangeSource::ReceiverState,
      std::string_view>
      kNames = {
          {UcxExchangeSource::ReceiverState::Created, "Created"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeComplete,
           "WaitingForHandshakeComplete"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeResponse,
           "WaitingForHandshakeResponse"},
          {UcxExchangeSource::ReceiverState::ReadyToReceive, "ReadyToReceive"},
          {UcxExchangeSource::ReceiverState::WaitingForMetadata,
           "WaitingForMetadata"},
          {UcxExchangeSource::ReceiverState::WaitingForData, "WaitingForData"},
          {UcxExchangeSource::ReceiverState::WaitingForIntraNodeData,
           "WaitingForIntraNodeData"},
          {UcxExchangeSource::ReceiverState::Done, "Done"},
      };
  return kNames;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeSource,
    ReceiverState,
    receiverStateNames)

void UcxExchangeSource::setState(ReceiverState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private.
UcxExchangeSource::UcxExchangeSource(
    const std::shared_ptr<Communicator> communicator,
    std::string_view taskId,
    std::string_view host,
    uint16_t port,
    const PartitionKey& partitionKey,
    const std::shared_ptr<UcxExchangeQueue> queue)
    : CommElement(communicator),
      host_(host),
      port_(port),
      taskId_(taskId),
      partitionKey_(partitionKey),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queue_(std::move(queue)) {
  setState(ReceiverState::Created);
}

/*static*/
std::shared_ptr<UcxExchangeSource> UcxExchangeSource::create(
    std::string_view taskId,
    std::string_view url,
    const std::shared_ptr<UcxExchangeQueue>& queue) {
  folly::Uri uri(url);
  // Note that there is no distinct schema for the UCXX exchange.
  // The approach is to ignore the schema and not check for HTTP or HTTPS.
  // FIXME: Can't use the HTTP port as this conflicts with Prestissimo!
  // For the time being, there's an ugly hack that just increases the port by 3.
  const std::string host = uri.host();
  int port = uri.port() + 3;
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto key = extractTaskAndDestinationId(uri.path());
  auto source = std::shared_ptr<UcxExchangeSource>(
      new UcxExchangeSource(communicator, taskId, host, port, key, queue));
  // register the exchange source with the communicator. This makes sure that
  // "progress" is called.
  communicator->registerCommElement(source);
  VLOG(3) << source->toString()
          << " creating UcxExchangeSource for url: " << url;
  return source;
}

void UcxExchangeSource::process() {
  if (closed_) {
    // Driver thread called closed
    cleanUp();
    return;
  }

  switch (state_) {
    case ReceiverState::Created: {
      // Get the endpoint.
      HostPort hp{host_, port_};
      std::shared_ptr<UcxExchangeSource> selfPtr = getSelfPtr();
      auto epRef = communicator_->assocEndpointRef(selfPtr, hp);
      if (epRef) {
        setEndpoint(epRef);
        setStateIf(
            ReceiverState::Created, ReceiverState::WaitingForHandshakeComplete);
        sendHandshake();
      } else {
        // connection failed.
        VLOG(0) << toString() << " Failed to connect to " << host_ << ":"
                << std::to_string(port_);
        deliverEndMarker();
        setState(ReceiverState::Done);
      }
      communicator_->addToWorkQueue(getSelfPtr());
    } break;
    case ReceiverState::WaitingForHandshakeComplete:
      // Waiting for handshake send completion is handled by callback.
      break;
    case ReceiverState::WaitingForHandshakeResponse:
      // Waiting for HandshakeResponse is handled by callback.
      break;
    case ReceiverState::ReadyToReceive: {
      // Backpressure: don't post the next receive if the consumer queue is
      // overloaded. The source goes dormant (not in work queue) and will be
      // woken by UcxExchangeClient::next() calling resumeFromBackpressure()
      // when the queue drains below the low water mark.
      //
      // This creates natural backpressure: the server's tagSend for data
      // will block at rendezvous until we post a matching tagRecv. For
      // intra-node: the server's publish future won't resolve until we poll.
      int32_t queueSize = queue_->size();
      if (queueSize > kBackpressureHighWaterMark) {
        if (!backpressureActive_.exchange(true, std::memory_order_acq_rel)) {
          VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
                  << "] pausing, queueSize=" << queueSize
                  << " > highWater=" << kBackpressureHighWaterMark;
        }
        // Go dormant — do NOT re-enqueue into work queue.
        // UcxExchangeClient::next() will call resumeFromBackpressure().
        break;
      }

      if (isIntraNodeTransfer_) {
        // INTRA-NODE TRANSFER: Use registry instead of UCXX
        setStateIf(
            ReceiverState::ReadyToReceive,
            ReceiverState::WaitingForIntraNodeData);
        waitForIntraNodeData();
      } else {
        // REMOTE EXCHANGE: Use UCXX for metadata and data
        setStateIf(
            ReceiverState::ReadyToReceive, ReceiverState::WaitingForMetadata);
        getMetadata();
      }
    } break;
    case ReceiverState::WaitingForMetadata:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::WaitingForData:
      // Waiting for data is handled by an upcall from UCXX. Nothing to do.
      break;
    case ReceiverState::WaitingForIntraNodeData:
      // Poll for intra-node transfer data
      waitForIntraNodeData();
      break;
    case ReceiverState::Done:
      // We need to call clean-up in this thread to remove any state
      cleanUp();
      break;
  }
}

void UcxExchangeSource::cleanUp() {
  uint32_t value = static_cast<uint32_t>(getState());
  if (value != static_cast<uint32_t>(ReceiverState::Done)) {
    // Unexpected cleanup
    VLOG(3) << toString()
            << " In UcxExchangeSource::cleanUp state == " << value;
  }

  // Cancel any outstanding request. With weak_ptr callbacks, the callback
  // will safely no-op if we're destroyed before it completes.
  if (request_ && !request_->isCompleted()) {
    // The Task has failed and we may need to cancel outstanding requests
    request_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  if (communicator_) {
    if (request_) {
      communicator_->deferRequestCleanup(std::move(request_));
    }
    for (auto& req : completedRequests_) {
      communicator_->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  if (communicator_) {
    communicator_->unregister(getSelfPtr());
  }
}

void UcxExchangeSource::close() {
  // This is called by the driver thread so we need to be careful to
  // indicate to the process thread that we are closing and
  // let it do the actual cleaning up.

  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }

  VLOG(1) << toString() << " UcxExchangeSource::close called.";

  // Guarantee the end marker is delivered before transitioning to Done.
  deliverEndMarker();

  // Let the Communicator progress thread do the actual clean-up.
  setState(ReceiverState::Done);
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::resumeFromBackpressure() {
  bool expected = true;
  if (backpressureActive_.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel)) {
    VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
            << "] resumed by consumer, queueSize=" << queue_->size();
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

folly::F14FastMap<std::string, int64_t> UcxExchangeSource::stats() const {
  VELOX_UNREACHABLE();
}

folly::F14FastMap<std::string, RuntimeMetric> UcxExchangeSource::metrics()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> map;

  // these metrics will be aggregated over all exchange sources of the same
  // exchange client.
  map["ucxExchangeSource.numPackedColumns"] = metrics_.numPackedColumns_;
  map["ucxExchangeSource.totalBytes"] = metrics_.totalBytes_;
  map["ucxExchangeSource.rttPerRequest"] = metrics_.rttPerRequest_;
  return map;
}

// private methods ---
PartitionKey UcxExchangeSource::extractTaskAndDestinationId(
    std::string_view path) {
  // The URL path has the form: /v1/task/<taskId>/results/<destinationId>"
  std::vector<folly::StringPiece> components;
  folly::split('/', path, components, true);

  VELOX_CHECK_EQ(components[0], "v1");
  VELOX_CHECK_EQ(components[1], "task");
  VELOX_CHECK_EQ(components[3], "results");

  uint32_t destinationId;
  try {
    destinationId = static_cast<uint32_t>(std::stoul(components[4].str()));
  } catch (const std::exception& e) {
    VELOX_UNSUPPORTED("Illegal destination in task URL: {}", path);
  }

  return PartitionKey{components[2].str(), destinationId};
}

std::shared_ptr<UcxExchangeSource> UcxExchangeSource::getSelfPtr() {
  std::shared_ptr<UcxExchangeSource> ptr;
  try {
    ptr = shared_from_this();
  } catch (std::bad_weak_ptr& exp) {
    ptr = nullptr;
  }
  return ptr;
}

void UcxExchangeSource::enqueue(PackedTableWithStreamPtr data) {
  std::vector<velox::ContinuePromise> queuePromises;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    queue_->enqueueLocked(std::move(data), queuePromises);
  }
  // wake up consumers of the UcxExchangeQueue
  for (auto& promise : queuePromises) {
    promise.setValue();
  }
}

void UcxExchangeSource::deliverEndMarker() {
  if (!registered_.load(std::memory_order_acquire)) {
    // Never registered with queue -- don't deliver end marker to avoid
    // spurious numCompleted_ increments.
    return;
  }
  bool expected = false;
  if (!endMarkerDelivered_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    // Already delivered by another thread/path.
    return;
  }
  VLOG(3) << toString() << " delivering end-of-stream marker to queue";
  enqueue(nullptr);
}

void UcxExchangeSource::setEndpoint(std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

void UcxExchangeSource::sendHandshake() {
  std::shared_ptr<HandshakeMsg> handshakeReq = std::make_shared<HandshakeMsg>();
  handshakeReq->destination = partitionKey_.destination;
  // Use sizeof(...) - 1 and explicitly null-terminate to prevent buffer
  // overread if taskId is longer than the destination buffer.
  strncpy(
      handshakeReq->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(handshakeReq->taskId) - 1);
  handshakeReq->taskId[sizeof(handshakeReq->taskId) - 1] = '\0';
  handshakeReq->workerId = communicator_->getWorkerId();

  VLOG(3) << toString() << " Sending handshake with initial value: "
          << partitionKey_.toString() << " to server";

  // Create the handshake which will register client's existence with the server
  ucxx::AmReceiverCallbackInfo info(
      communicator_->kAmCallbackOwner, communicator_->kAmCallbackId);
  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  if (request_) {
    completedRequests_.push_back(std::move(request_));
  }
  // Pass handshakeReq as the callback arg to keep the send buffer alive until
  // the async amSend completes. UCXX stores it as shared_ptr<void> but the
  // type-erased deleter still calls ~HandshakeMsg correctly.
  request_ = endpointRef_->endpoint_->amSend(
      handshakeReq.get(),
      sizeof(*handshakeReq),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onHandshake(status, arg);
        }
      },
      handshakeReq);
}

void UcxExchangeSource::onHandshake(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  // arg holds the HandshakeMsg that was sent — it is unused here because this
  // is a send completion callback (the outgoing data has already been
  // transmitted). The parameter exists only because UCXX uses it as a lifetime
  // handle; letting it go out of scope releases the send buffer.

  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onHandshake called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeComplete) {
    VLOG(2) << toString() << " onHandshake called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send handshake to host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  } else {
    VLOG(3) << toString() << "+ onHandshake " << ucs_status_string(status);
    // Now wait for the HandshakeResponse from the server
    setStateIf(
        ReceiverState::WaitingForHandshakeComplete,
        ReceiverState::WaitingForHandshakeResponse);
    receiveHandshakeResponse();
  }
}

void UcxExchangeSource::getMetadata() {
  // Use kMaxMetaBufSize to support tables with many columns.
  // The sender allocates exact size needed; receiver pre-allocates max.
  auto metadataReq = std::make_shared<std::vector<uint8_t>>(kMaxMetaBufSize);
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber_);

  VLOG(3) << toString()
          << " waiting for metadata for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << metadataTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  if (request_) {
    completedRequests_.push_back(std::move(request_));
  }
  request_ = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      kMaxMetaBufSize,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onMetadata(status, arg);
        }
      },
      metadataReq);
}

void UcxExchangeSource::onMetadata(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onMetadata called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForMetadata) {
    VLOG(2) << toString() << " onMetadata called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onMetadata " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  } else {
    VELOX_CHECK_NOT_NULL(arg, "Didn't get metadata");

    // arg contains the actual serialized metadata, deserialize the metadata
    std::shared_ptr<std::vector<uint8_t>> metadataMsg =
        std::static_pointer_cast<std::vector<uint8_t>>(arg);

    auto ptr = std::make_shared<DataAndMetadata>();

    ptr->metadata =
        std::move(MetadataMsg::deserializeMetadataMsg(metadataMsg->data()));

    VLOG(3) << toString()
            << " Datasize bytes == " << ptr->metadata.dataSizeBytes;

    if (ptr->metadata.atEnd) {
      // It seems that all data has been transferred
      atEnd_ = true;
      // enqueue a nullpointer to mark the end for this source.
      VLOG(3) << "There is no more data to transfer for " << toString();
      deliverEndMarker();
      setStateIf(ReceiverState::WaitingForMetadata, ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      // jump out of this function.
      return;
    }

    // REMOTE EXCHANGE PATH: Allocate buffer and receive via UCXX
    // Get a stream from the global stream pool
    auto stream =
        facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
    // Store the stream in the DataAndMetadata struct so it can be used later
    // in onData() when creating the PackedTableWithStream.
    ptr->stream = stream;
    try {
      ptr->dataBuf = std::make_unique<rmm::device_buffer>(
          ptr->metadata.dataSizeBytes, stream);
    } catch (const rmm::bad_alloc& e) {
      VLOG(0) << toString() << " *** RMM  failed to allocate: " << e.what();
      queue_->setError("Failed to alloc GPU memory"); // Let the operator know
                                                      // via the queue
      deliverEndMarker();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    }

    // sync after allocating.
    stream.synchronize();

    VLOG(3) << toString() << " Allocated " << ptr->metadata.dataSizeBytes
            << " bytes of device memory";

    // Initiate the transfer of the actual data from GPU-2-GPU
    uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
    VLOG(3) << toString() << " waiting for data for chunk: " << sequenceNumber_
            << " using tag: " << std::hex << dataTag << std::dec;

    if (!setStateIf(
            ReceiverState::WaitingForMetadata, ReceiverState::WaitingForData)) {
      VLOG(1) << toString() << " onMetadata Invalid previous state ";
      return;
    }
    // Use weak_ptr to prevent use-after-free if close() is called during
    // callback
    std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
    if (request_) {
      completedRequests_.push_back(std::move(request_));
    }
    request_ = endpointRef_->endpoint_->tagRecv(
        ptr->dataBuf->data(),
        ptr->metadata.dataSizeBytes,
        ucxx::Tag{dataTag},
        ucxx::TagMaskFull,
        false,
        [weak](ucs_status_t status, std::shared_ptr<void> arg) {
          if (auto self = weak.lock()) {
            self->onData(status, arg);
          }
        },
        ptr // DataAndMetadata
    );
  }
}

void UcxExchangeSource::onData(ucs_status_t status, std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onData called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForData) {
    VLOG(2) << toString() << " onData called in state " << toName(getState())
            << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onData " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive data from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << toString() << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << " got chunk: " << sequenceNumber_;

    this->sequenceNumber_++;

    std::shared_ptr<DataAndMetadata> ptr =
        std::static_pointer_cast<DataAndMetadata>(arg);

    metrics_.numPackedColumns_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    // Create packed_columns from the received metadata and data buffer
    cudf::packed_columns packedCols(
        std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));

    // Unpack to get the table_view and create a packed_table
    cudf::table_view tableView = cudf::unpack(packedCols);
    auto packedTable = std::make_unique<cudf::packed_table>(
        cudf::packed_table{tableView, std::move(packedCols)});

    // Bundle the packed_table with the stream that was used for allocation
    auto data = std::make_unique<PackedTableWithStream>(
        std::move(packedTable), ptr->stream);

    enqueue(std::move(data));
    setStateIf(ReceiverState::WaitingForData, ReceiverState::ReadyToReceive);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::receiveHandshakeResponse() {
  auto responseBuffer = std::make_shared<HandshakeResponse>();
  uint64_t responseTag = getHandshakeResponseTag(partitionKeyHash_);

  VLOG(3) << toString()
          << " waiting for HandshakeResponse with tag: " << std::hex
          << responseTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  if (request_) {
    completedRequests_.push_back(std::move(request_));
  }
  request_ = endpointRef_->endpoint_->tagRecv(
      responseBuffer.get(),
      sizeof(*responseBuffer),
      ucxx::Tag{responseTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onHandshakeResponse(status, arg);
        }
      },
      responseBuffer);
}

void UcxExchangeSource::onHandshakeResponse(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " onHandshakeResponse called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeResponse) {
    VLOG(2) << toString() << " onHandshakeResponse called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive HandshakeResponse from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  std::shared_ptr<HandshakeResponse> response =
      std::static_pointer_cast<HandshakeResponse>(arg);

  isIntraNodeTransfer_ = response->isIntraNodeTransfer;

  VLOG(3) << toString() << " + onHandshakeResponse isIntraNodeTransfer="
          << isIntraNodeTransfer_;

  setStateIf(
      ReceiverState::WaitingForHandshakeResponse,
      ReceiverState::ReadyToReceive);
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::waitForIntraNodeData() {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " waitForIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  IntraNodeTransferKey key{
      partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};

  auto result = IntraNodeTransferRegistry::getInstance()->poll(key);

  if (!result.has_value()) {
    // Data not ready yet, re-queue to try again
    ++intraNodePollCount_;
    if (intraNodePollCount_ % 100 == 0) {
      VLOG(2) << "[INTRA] [ExSrc " << toString() << " seq=" << sequenceNumber_
              << "] still polling for data, polls=" << intraNodePollCount_;
    }
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  intraNodePollCount_ = 0;
  onIntraNodeData(std::move(result->data), result->atEnd);
}

void UcxExchangeSource::onIntraNodeData(
    std::shared_ptr<cudf::packed_columns> data,
    bool atEnd) {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  if (atEnd) {
    // End of stream
    atEnd_ = true;
    VLOG(3) << toString() << " Intra-node transfer: end of stream";
    deliverEndMarker();
    setState(ReceiverState::Done);

    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  if (!data) {
    // Error - should not happen if atEnd is false
    std::string errorMsg = fmt::format(
        "Intra-node transfer data is null for task {}, dest {}, seq {}",
        partitionKey_.taskId,
        partitionKey_.destination,
        sequenceNumber_);
    VLOG(0) << toString() << " " << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  VLOG(3) << toString()
          << " Intra-node transfer: received data for seq=" << sequenceNumber_
          << " size=" << data->gpu_data->size();

  metrics_.numPackedColumns_.addValue(1);
  metrics_.totalBytes_.addValue(data->gpu_data->size());

  // Convert packed_columns to PackedTableWithStream for the queue.
  // Create packed_columns from the shared data.
  cudf::packed_columns packedCols(
      std::move(data->metadata), std::move(data->gpu_data));

  // Unpack to get the table_view and create a packed_table
  cudf::table_view tableView = cudf::unpack(packedCols);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedCols)});

  // Get a stream from the pool so downstream cuDF operations on this data
  // run on a dedicated stream, not the default stream. The producer already
  // synchronized before enqueuing, so the GPU data is ready. This matches
  // the inter-node (UCX) receive path which also allocates a pool stream.
  auto stream =
      facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
  auto tableWithStream =
      std::make_unique<PackedTableWithStream>(std::move(packedTable), stream);

  enqueue(std::move(tableWithStream));

  this->sequenceNumber_++;
  setStateIf(
      ReceiverState::WaitingForIntraNodeData, ReceiverState::ReadyToReceive);
  communicator_->addToWorkQueue(getSelfPtr());
}

bool UcxExchangeSource::setStateIf(
    UcxExchangeSource::ReceiverState expected,
    UcxExchangeSource::ReceiverState desired) {
  ReceiverState exp = expected;
  // since spurious failures can happen even if state_ == expected, we need
  // to do this in a loop.
  while (!state_.compare_exchange_strong(
      exp, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    if (exp != expected) {
      // no spurious failure, state isn't what we've expected.
      return false;
    }
    // spurious failure.
    exp = expected; // reset for the next try
  }
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(expected) << " -> " << toName(desired);
  return true;
}

} // namespace facebook::velox::ucx_exchange
