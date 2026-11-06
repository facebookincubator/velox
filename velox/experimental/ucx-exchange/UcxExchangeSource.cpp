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

#include <cstring>
#include <thread>

#include <cudf/contiguous_split.hpp>
#include <folly/ScopeGuard.h>
#include <folly/String.h>
#include <folly/Uri.h>
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"

using namespace facebook::velox::exec;
namespace facebook::velox::ucx_exchange {

namespace {
struct GpuCallbackOnce {
  bool tryClaim() {
    bool expected = false;
    return claimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }

  std::atomic<bool> claimed{false};
};

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
          {UcxExchangeSource::ReceiverState::WaitingForReceiveCredit,
           "WaitingForReceiveCredit"},
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
  VLOG(3) << source->toString()
          << " creating UcxExchangeSource for url: " << url;
  return source;
}

void UcxExchangeSource::start() {
  auto self = getSelfPtr();
  VELOX_CHECK_NOT_NULL(self);

  std::lock_guard<std::mutex> lock(lifecycleMutex_);
  if (started_ || closed_.load(std::memory_order_acquire)) {
    return;
  }

  // Mark this before publication. If registerCommElement() partially succeeds
  // and then throws (for example while allocating its work-queue node), the
  // caller's close() will still schedule cleanup and balance queue completion.
  started_ = true;
  communicator_->registerCommElement(std::move(self));
}

void UcxExchangeSource::process() {
  try {
    processStateMachine();
  } catch (const std::exception& e) {
    failSource("state-machine processing", e.what());
  } catch (...) {
    failSource("state-machine processing", nullptr);
  }
}

void UcxExchangeSource::processStateMachine() {
  // A completion is one state-machine turn. In particular, a failed receive
  // allocation must not be retried (and decrease the adaptive window again)
  // by the switch below in the same call to process(). Callback paths enqueue
  // another turn whenever immediate progress is required.
  const bool drainedStateEvents = drainStateEvents();

  // Callback handoff failure can mark the source closed while leaving a
  // previously inserted event in the local queue. Always prioritize cleanup
  // after draining so an event cannot consume the only scheduled cleanup turn.
  if (closed_) {
    cleanUp();
    return;
  }

  if (drainedStateEvents) {
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
        if (setStateIf(
                ReceiverState::Created,
                ReceiverState::WaitingForHandshakeComplete)) {
          sendHandshake();
        }
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
      if (pauseForBackpressureIfNeeded()) {
        // Go dormant — do NOT re-enqueue into work queue.
        // UcxExchangeClient::next() will call resumeFromBackpressure().
        break;
      }

      if (isIntraNodeTransfer_) {
        // INTRA-NODE TRANSFER: Use registry instead of UCXX
        if (setStateIf(
                ReceiverState::ReadyToReceive,
                ReceiverState::WaitingForIntraNodeData)) {
          waitForIntraNodeData();
        }
      } else {
        // REMOTE EXCHANGE: Use UCXX for metadata and data
        if (setStateIf(
                ReceiverState::ReadyToReceive,
                ReceiverState::WaitingForMetadata)) {
          getMetadata();
        }
      }
    } break;
    case ReceiverState::WaitingForMetadata:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::WaitingForReceiveCredit:
      if (pendingDataReceive_) {
        startDataReceive(std::move(pendingDataReceive_));
      }
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
  {
    std::lock_guard<std::mutex> lock(lifecycleMutex_);
    if (cleanupComplete_) {
      return;
    }
  }

  uint32_t value = static_cast<uint32_t>(getState());
  if (value != static_cast<uint32_t>(ReceiverState::Done)) {
    // Unexpected cleanup
    VLOG(3) << toString()
            << " In UcxExchangeSource::cleanUp state == " << value;
  }

  // A source that stops before consuming its next intra-node page must
  // acknowledge that exact transfer. The per-key tombstone also covers a
  // producer publish racing with this cleanup.
  if (isIntraNodeTransfer_ && !atEnd_) {
    IntraNodeTransferRegistry::getInstance()->cancelTransfer(
        {partitionKey_.taskId, partitionKey_.destination, sequenceNumber_});
  }

  // Release pending and active reservations before request cancellation. A
  // later callback sees receiveBytesReserved=false and cannot double-release.
  releaseReceiveBytes(pendingDataReceive_);
  releaseReceiveBytes(activeDataReceive_);
  pendingDataReceive_.reset();
  activeDataReceive_.reset();

  // Copy requests into the Communicator's deferred list, then clear each
  // member only after insertion succeeds. deferRequestCleanup() takes its
  // argument by value, so moving the sole reference into a failed vector
  // insertion would otherwise destroy callback state while UCX can use it.
  if (communicator_) {
    // Sends cannot be cancelled with UCP's tag-receive cancellation API.
    auto deferRequest = [this](std::shared_ptr<ucxx::Request>& request) {
      if (!request) {
        return;
      }
      if (request->isCompleted()) {
        request.reset();
        return;
      }
      communicator_->deferRequestCleanup(request);
      request.reset();
    };
    deferRequest(handshakeRequest_);

    auto cancelReceive = [this](std::shared_ptr<ucxx::Request>& request) {
      if (!request) {
        return;
      }
      if (!request->isCompleted()) {
        request->cancel();
        communicator_->deferRequestCleanup(request);
      }
      request.reset();
    };
    cancelReceive(handshakeResponseRequest_);
    cancelReceive(metadataRequest_);
    cancelReceive(dataRequest_);
  }

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  if (communicator_) {
    communicator_->unregister(getSelfPtr());
  }

  std::lock_guard<std::mutex> lock(lifecycleMutex_);
  cleanupComplete_ = true;
}

void UcxExchangeSource::close() {
  // This is called by the driver thread so we need to be careful to
  // indicate to the process thread that we are closing and
  // let it do the actual cleaning up.

  bool firstClose = false;
  bool scheduleCleanup = false;
  {
    // Serialize close with initial publication. If close wins, start() observes
    // closed_ and never publishes the source; if start wins, close queues a
    // cleanup turn for the registered source.
    std::lock_guard<std::mutex> lock(lifecycleMutex_);
    bool expected = false;
    firstClose = closed_.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
    scheduleCleanup = started_ && !cleanupComplete_;
  }

  if (!firstClose && !scheduleCleanup) {
    return;
  }

  if (!firstClose) {
    // A previous callback-side attempt may have failed to allocate a work-queue
    // node. Repeated close() calls intentionally retry cleanup scheduling.
    try {
      communicator_->addToWorkQueue(getSelfPtr());
    } catch (...) {
      LOG(ERROR) << "Failed again to schedule UCX source cleanup";
    }
    return;
  }

  VLOG(1) << toString() << " UcxExchangeSource::close called.";
  setState(ReceiverState::Done);
  try {
    deliverEndMarker();
  } catch (...) {
    // Promise bookkeeping may allocate. Preserve a terminal signal even under
    // host allocation pressure rather than abandoning the source half-closed.
    try {
      queue_->setError("Failed to close UCX exchange source");
    } catch (...) {
      LOG(ERROR) << "Failed to close UCX exchange source and to record the "
                    "queue error";
    }
  }

  if (scheduleCleanup) {
    // Let the Communicator progress thread do the actual clean-up.
    try {
      communicator_->addToWorkQueue(getSelfPtr());
    } catch (...) {
      // Keep cleanupComplete_ false. A repeated close() (including client
      // teardown) will retry this allocation rather than returning solely
      // because closed_ is already set.
      LOG(ERROR) << "Failed to schedule UCX source cleanup";
    }
  }
}

void UcxExchangeSource::resumeFromBackpressure() noexcept {
  bool expected = true;
  if (backpressureActive_.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel)) {
    try {
      VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
              << "] resumed by consumer";
      auto self = getSelfPtr();
      if (!self) {
        backpressureActive_.store(true, std::memory_order_release);
        return;
      }
      communicator_->addToWorkQueue(std::move(self));
    } catch (...) {
      // A work-queue node can fail to allocate. Restore the dormant flag so a
      // later dequeue or receive-credit release retries the wake-up instead of
      // leaving this source asleep forever. If insertion succeeded and only
      // worker signaling failed, an extra later work item is harmless.
      backpressureActive_.store(true, std::memory_order_release);
      try {
        LOG(ERROR) << "Failed to resume UCX exchange source after "
                      "backpressure";
      } catch (...) {
      }
    }
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

void UcxExchangeSource::enqueue(
    PackedTableWithStreamPtr data,
    uint64_t reservedReceiveBytes,
    bool* receiveReservationActive) {
  std::vector<velox::ContinuePromise> queuePromises;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    queue_->enqueueLocked(
        std::move(data),
        queuePromises,
        reservedReceiveBytes,
        receiveReservationActive);
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

void UcxExchangeSource::enqueueSerializedCallback(
    CallbackPhase phase,
    ucs_status_t status,
    std::shared_ptr<void> arg) noexcept {
  try {
    auto self = getSelfPtr();
    if (!self) {
      return;
    }
    const bool enqueued = enqueueStateEvent(
        self,
        [raw = self.get(), phase, status, arg = std::move(arg)]() mutable {
          raw->runSerializedCallback(phase, status, std::move(arg));
        });
    if (!enqueued) {
      failCallbackDispatch();
    }
  } catch (...) {
    // Includes allocation of the type-erased state event itself. Never let a
    // C++ exception escape through UCXX's C callback boundary.
    failCallbackDispatch();
  }
}

void UcxExchangeSource::failCallbackDispatch() noexcept {
  closed_.store(true, std::memory_order_release);
  try {
    queue_->setError("Failed to dispatch UCX receive callback");
  } catch (...) {
    LOG(ERROR) << "Failed to dispatch UCX receive callback and to record the "
                  "queue error";
  }
  try {
    deliverEndMarker();
  } catch (...) {
    LOG(ERROR) << "Failed to deliver UCX end marker after callback handoff "
                  "failure";
  }
  state_.store(ReceiverState::Done, std::memory_order_release);
  try {
    communicator_->addToWorkQueue(getSelfPtr());
  } catch (...) {
    // The queue error above wakes consumers. There is no further safe action
    // at a C callback boundary if even scheduling cleanup cannot allocate.
    LOG(ERROR) << "Failed to schedule UCX source cleanup after callback "
                  "handoff failure";
  }
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
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  // Pass handshakeReq as the callback arg to keep the send buffer alive until
  // the async amSend completes. UCXX stores it as shared_ptr<void> but the
  // type-erased deleter still calls ~HandshakeMsg correctly.
  handshakeRequest_ =
      endpointRef_->endpoint_
          ->amSendBuilder(
              handshakeReq.get(), sizeof(*handshakeReq), UCS_MEMORY_TYPE_HOST)
          .receiverCallbackInfo(info)
          .callbackFunction(
              [weak, callbackOnce](
                  ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                if (!callbackOnce->tryClaim()) {
                  return;
                }
                if (auto self = weak.lock()) {
                  self->enqueueSerializedCallback(
                      CallbackPhase::Handshake, status, std::move(arg));
                }
              })
          .callbackData(handshakeReq)
          .build();
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
  // Ignore a duplicate or late completion after this phase advanced.
  if (getState() != ReceiverState::WaitingForHandshakeComplete) {
    VLOG(2) << toString() << " onHandshake called in state "
            << toName(getState()) << ", ignoring duplicate/late completion";
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
    if (setStateIf(
            ReceiverState::WaitingForHandshakeComplete,
            ReceiverState::WaitingForHandshakeResponse)) {
      receiveHandshakeResponse();
    }
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
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  retireRequest(metadataRequest_);
  metadataRequest_ =
      endpointRef_->endpoint_
          ->tagRecvBuilder(
              reinterpret_cast<void*>(metadataReq->data()),
              kMaxMetaBufSize,
              ucxx::Tag{metadataTag},
              ucxx::TagMaskFull)
          .callbackFunction(
              [weak, callbackOnce](
                  ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                if (!callbackOnce->tryClaim()) {
                  return;
                }
                if (auto self = weak.lock()) {
                  self->enqueueSerializedCallback(
                      CallbackPhase::Metadata, status, std::move(arg));
                }
              })
          .callbackData(metadataReq)
          .build();
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
  // Ignore a duplicate or late completion after this phase advanced.
  if (getState() != ReceiverState::WaitingForMetadata) {
    VLOG(2) << toString() << " onMetadata called in state "
            << toName(getState()) << ", ignoring duplicate/late completion";
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
    return;
  }

  VELOX_CHECK_NOT_NULL(arg, "Didn't get metadata");
  auto metadataMsg = std::static_pointer_cast<std::vector<uint8_t>>(arg);
  VELOX_CHECK_GE(metadataMsg->size(), kMetaHeaderSize);
  uint32_t serializedSize = 0;
  std::memcpy(
      &serializedSize,
      metadataMsg->data() + sizeof(kMagicNumber),
      sizeof(serializedSize));
  VELOX_CHECK_GE(serializedSize, kMetaHeaderSize);
  VELOX_CHECK_LE(
      serializedSize,
      metadataMsg->size(),
      "UCX metadata size exceeds the receive buffer");
  auto ptr = std::make_shared<DataAndMetadata>();
  ptr->metadata = std::move(
      MetadataMsg::deserializeMetadataMsg(metadataMsg->data(), serializedSize));
  // The callback has consumed the fixed-size receive buffer. Drop its storage
  // now; the phase request itself is retired before the next receive is posted.
  std::vector<uint8_t>().swap(*metadataMsg);

  VELOX_CHECK_GE(
      ptr->metadata.dataSizeBytes, 0, "UCX metadata data size is negative");
  VELOX_CHECK_GE(
      ptr->metadata.numRows, 0, "UCX metadata row count is negative");

  VLOG(3) << toString() << " Datasize bytes == " << ptr->metadata.dataSizeBytes;

  if (ptr->metadata.atEnd) {
    atEnd_ = true;
    VLOG(3) << "There is no more data to transfer for " << toString();
    deliverEndMarker();
    setStateIf(ReceiverState::WaitingForMetadata, ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  startDataReceive(std::move(ptr));
}

bool UcxExchangeSource::tryReserveReceiveBytes(
    const std::shared_ptr<DataAndMetadata>& ptr) {
  VELOX_CHECK_NOT_NULL(ptr);
  {
    std::lock_guard<std::mutex> lock(queue_->mutex());
    const auto dataSizeBytes =
        static_cast<uint64_t>(ptr->metadata.dataSizeBytes);
    if (queue_->tryReserveReceiveBytesLocked(dataSizeBytes)) {
      ptr->receiveBytesReserved = true;
      return true;
    }

    // Publish the dormant state while holding the same mutex consumers use to
    // decide whether a source needs waking. This closes the lost-resume race.
    pendingDataReceive_ = ptr;
    backpressureActive_.store(true, std::memory_order_release);
  }

  const auto state = getState();
  if (state == ReceiverState::WaitingForMetadata) {
    setStateIf(
        ReceiverState::WaitingForMetadata,
        ReceiverState::WaitingForReceiveCredit);
  } else {
    VELOX_CHECK(
        state == ReceiverState::WaitingForReceiveCredit,
        "Unexpected state {} while waiting for receive credit",
        toName(state));
  }
  return false;
}

void UcxExchangeSource::releaseReceiveBytes(
    const std::shared_ptr<DataAndMetadata>& ptr) {
  if (!ptr || !ptr->receiveBytesReserved) {
    return;
  }

  std::lock_guard<std::mutex> lock(queue_->mutex());
  queue_->releaseReceiveBytesLocked(
      static_cast<uint64_t>(ptr->metadata.dataSizeBytes));
  ptr->receiveBytesReserved = false;
}

void UcxExchangeSource::retireRequest(std::shared_ptr<ucxx::Request>& request) {
  if (!request) {
    return;
  }
  if (request->isCompleted()) {
    request.reset();
    return;
  }

  // Preserve our reference until insertion succeeds. UCXX documents that the
  // callback data must remain alive until callback completion; an incomplete
  // request therefore moves to communicator-owned deferred cleanup.
  communicator_->deferRequestCleanup(request);
  request.reset();
}

void UcxExchangeSource::startDataReceive(std::shared_ptr<DataAndMetadata> ptr) {
  VELOX_CHECK_NOT_NULL(ptr);
  if (closed_.load(std::memory_order_acquire)) {
    releaseReceiveBytes(ptr);
    deliverEndMarker();
    return;
  }

  const auto expectedState = getState();
  VELOX_CHECK(
      expectedState == ReceiverState::WaitingForMetadata ||
          expectedState == ReceiverState::WaitingForReceiveCredit,
      "Unexpected state {} before starting UCX data receive",
      toName(expectedState));

  if (!tryReserveReceiveBytes(ptr)) {
    return;
  }
  auto releaseReservationOnFailure = folly::makeGuard([&]() noexcept {
    try {
      releaseReceiveBytes(ptr);
    } catch (...) {
      // Preserve the original allocation/transport exception. A counter
      // invariant failure will also be reported by the terminal source path.
    }
  });

  auto stream =
      facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
  ptr->stream = stream;
  try {
    ptr->dataBuf = std::make_unique<rmm::device_buffer>(
        ptr->metadata.dataSizeBytes,
        stream,
        cudf::get_current_device_resource_ref());
  } catch (const rmm::bad_alloc& e) {
    VLOG(0) << toString()
            << " RMM failed to allocate receive buffer: " << e.what();

    bool retryLater = false;
    {
      std::lock_guard<std::mutex> lock(queue_->mutex());
      // Return the failed attempt's reservation and publish the dormant retry
      // state in one critical section. A consumer deciding whether to wake
      // sources under this mutex must observe either both changes or neither;
      // otherwise it can miss the only retry notification.
      if (ptr->receiveBytesReserved) {
        // The failed allocation did not free previously retained memory. Do
        // not perform additive window growth immediately before recording
        // multiplicative pressure.
        queue_->consumeReceiveReservationLocked(
            static_cast<uint64_t>(ptr->metadata.dataSizeBytes));
        ptr->receiveBytesReserved = false;
      }
      retryLater = queue_->recordReceiveAllocationPressureLocked(
          static_cast<uint64_t>(ptr->metadata.dataSizeBytes));
      if (retryLater) {
        pendingDataReceive_ = ptr;
        // Publish sleep under the queue mutex to pair with consumer wake-up
        // decisions and avoid losing the only retry notification.
        backpressureActive_.store(true, std::memory_order_release);
      }
    }
    if (retryLater) {
      if (expectedState == ReceiverState::WaitingForMetadata) {
        setStateIf(
            ReceiverState::WaitingForMetadata,
            ReceiverState::WaitingForReceiveCredit);
      }
      return;
    }

    queue_->setError("Failed to allocate GPU memory for UCX receive");
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  // UCX receives into this raw pointer on a UCX-owned CUDA stream. Complete
  // the async allocation before handing that pointer to UCX.
  try {
    stream.sync();
  } catch (const std::exception& e) {
    releaseReceiveBytes(ptr);
    queue_->setError(
        fmt::format(
            "Failed to prepare GPU UCX receive buffer for {}: {}",
            partitionKey_.toString(),
            e.what()));
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  VLOG(3) << toString() << " Allocated " << ptr->metadata.dataSizeBytes
          << " bytes of device memory";

  const uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
  VLOG(3) << toString() << " waiting for data for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << dataTag << std::dec;

  if (!setStateIf(expectedState, ReceiverState::WaitingForData)) {
    releaseReceiveBytes(ptr);
    VLOG(1) << toString() << " startDataReceive invalid previous state "
            << toName(getState());
    return;
  }

  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  retireRequest(dataRequest_);
  activeDataReceive_ = ptr;
  try {
    dataRequest_ =
        endpointRef_->endpoint_
            ->tagRecvBuilder(
                ptr->dataBuf->data(),
                ptr->metadata.dataSizeBytes,
                ucxx::Tag{dataTag},
                ucxx::TagMaskFull)
            .callbackFunction(
                [weak, callbackOnce](
                    ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                  if (!callbackOnce->tryClaim()) {
                    return;
                  }
                  if (auto self = weak.lock()) {
                    self->enqueueSerializedCallback(
                        CallbackPhase::Data, status, std::move(arg));
                  }
                })
            .callbackData(ptr)
            .build();
    releaseReservationOnFailure.dismiss();
  } catch (const std::exception& e) {
    releaseReceiveBytes(ptr);
    activeDataReceive_.reset();
    queue_->setError(
        fmt::format(
            "Failed to post GPU UCX data receive for {}: {}",
            partitionKey_.toString(),
            e.what()));
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  } catch (...) {
    releaseReceiveBytes(ptr);
    activeDataReceive_.reset();
    queue_->setError(
        fmt::format(
            "Failed to post GPU UCX data receive for {}",
            partitionKey_.toString()));
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }
}

void UcxExchangeSource::onData(ucs_status_t status, std::shared_ptr<void> arg) {
  auto ptr = std::static_pointer_cast<DataAndMetadata>(arg);
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onData called after close, ignoring";
    releaseReceiveBytes(ptr ? ptr : activeDataReceive_);
    deliverEndMarker();
    return;
  }
  // Ignore a duplicate or late completion after this phase advanced.
  if (getState() != ReceiverState::WaitingForData) {
    VLOG(2) << toString() << " onData called in state " << toName(getState())
            << ", ignoring duplicate/late completion";
    releaseReceiveBytes(ptr);
    return;
  }
  VLOG(3) << toString() << " + onData " << ucs_status_string(status);

  if (status != UCS_OK) {
    releaseReceiveBytes(ptr ? ptr : activeDataReceive_);
    if (!ptr || activeDataReceive_ == ptr) {
      activeDataReceive_.reset();
    }
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
    if (!ptr) {
      releaseReceiveBytes(activeDataReceive_);
      activeDataReceive_.reset();
      queue_->setError(
          fmt::format(
              "UCX data callback returned no receive buffer for {}",
              partitionKey_.toString()));
      deliverEndMarker();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    }
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << " got chunk: " << sequenceNumber_;

    this->sequenceNumber_++;

    const uint64_t reservedReceiveBytes = ptr->receiveBytesReserved
        ? static_cast<uint64_t>(ptr->metadata.dataSizeBytes)
        : 0;

    metrics_.numPackedColumns_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    PackedTableWithStreamPtr data;
    try {
      cudf::packed_columns packedCols(
          std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));
      cudf::table_view tableView = cudf::unpack(packedCols);
      auto packedTable = std::make_unique<cudf::packed_table>(
          cudf::packed_table{tableView, std::move(packedCols)});
      data = std::make_unique<PackedTableWithStream>(
          std::move(packedTable), ptr->stream, ptr->metadata.numRows);
    } catch (const std::exception& e) {
      releaseReceiveBytes(ptr);
      activeDataReceive_.reset();
      queue_->setError(
          fmt::format(
              "Failed to unpack GPU UCX receive for {}: {}",
              partitionKey_.toString(),
              e.what()));
      deliverEndMarker();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    } catch (...) {
      releaseReceiveBytes(ptr);
      activeDataReceive_.reset();
      queue_->setError(
          fmt::format(
              "Failed to unpack GPU UCX receive for {}",
              partitionKey_.toString()));
      deliverEndMarker();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    }

    enqueue(std::move(data), reservedReceiveBytes, &ptr->receiveBytesReserved);
    if (activeDataReceive_ == ptr) {
      activeDataReceive_.reset();
    }
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
  auto callbackOnce = std::make_shared<GpuCallbackOnce>();
  handshakeResponseRequest_ =
      endpointRef_->endpoint_
          ->tagRecvBuilder(
              responseBuffer.get(),
              sizeof(*responseBuffer),
              ucxx::Tag{responseTag},
              ucxx::TagMaskFull)
          .callbackFunction(
              [weak, callbackOnce](
                  ucs_status_t status, std::shared_ptr<void> arg) noexcept {
                if (!callbackOnce->tryClaim()) {
                  return;
                }
                if (auto self = weak.lock()) {
                  self->enqueueSerializedCallback(
                      CallbackPhase::HandshakeResponse, status, std::move(arg));
                }
              })
          .callbackData(responseBuffer)
          .build();
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
  // Ignore a duplicate or late completion after this phase advanced.
  if (getState() != ReceiverState::WaitingForHandshakeResponse) {
    VLOG(2) << toString() << " onHandshakeResponse called in state "
            << toName(getState()) << ", ignoring duplicate/late completion";
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
  VELOX_CHECK_NOT_NULL(
      response, "UCX handshake response callback returned no buffer");

  isIntraNodeTransfer_ = response->isIntraNodeTransfer;

  VLOG(3) << toString() << " + onHandshakeResponse isIntraNodeTransfer="
          << isIntraNodeTransfer_;

  setStateIf(
      ReceiverState::WaitingForHandshakeResponse,
      ReceiverState::ReadyToReceive);
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeSource::failSource(
    std::string_view operation,
    const char* detail) noexcept {
  try {
    releaseReceiveBytes(activeDataReceive_);
    activeDataReceive_.reset();
    releaseReceiveBytes(pendingDataReceive_);
    pendingDataReceive_.reset();
  } catch (...) {
    LOG(ERROR) << "Failed to release UCX receive reservations after "
               << operation << " failure";
  }

  try {
    const auto error = detail ? fmt::format(
                                    "UCX exchange source {} failed for {}: {}",
                                    operation,
                                    partitionKey_.toString(),
                                    detail)
                              : fmt::format(
                                    "UCX exchange source {} failed for {}",
                                    operation,
                                    partitionKey_.toString());
    LOG(ERROR) << error;
    queue_->setError(error);
  } catch (...) {
    try {
      queue_->setError("UCX exchange source failed");
    } catch (...) {
      LOG(ERROR) << "Failed to record UCX exchange source error";
    }
  }

  try {
    deliverEndMarker();
  } catch (...) {
    LOG(ERROR) << "Failed to deliver UCX end marker after " << operation
               << " failure";
  }
  state_.store(ReceiverState::Done, std::memory_order_release);
  try {
    communicator_->addToWorkQueue(getSelfPtr());
  } catch (...) {
    LOG(ERROR) << "Failed to schedule UCX source cleanup after " << operation
               << " failure";
  }
}

void UcxExchangeSource::runSerializedCallback(
    CallbackPhase phase,
    ucs_status_t status,
    std::shared_ptr<void> arg) noexcept {
  const char* phaseName = "unknown";
  try {
    switch (phase) {
      case CallbackPhase::Handshake:
        phaseName = "handshake";
        onHandshake(status, std::move(arg));
        return;
      case CallbackPhase::Metadata:
        phaseName = "metadata";
        onMetadata(status, std::move(arg));
        return;
      case CallbackPhase::Data:
        phaseName = "data";
        onData(status, std::move(arg));
        return;
      case CallbackPhase::HandshakeResponse:
        phaseName = "handshake response";
        onHandshakeResponse(status, std::move(arg));
        return;
    }
  } catch (const std::exception& e) {
    failSource(phaseName, e.what());
  } catch (...) {
    failSource(phaseName, nullptr);
  }
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
  // poll() acknowledges this sequence to the producer, which may immediately
  // publish the next one. Advance ownership before any unpacking or queue
  // allocation can throw so terminal cleanup cancels that next transfer rather
  // than an already-acknowledged key.
  const auto completedSequence = sequenceNumber_;
  if (!result->atEnd) {
    ++sequenceNumber_;
  }
  onIntraNodeData(
      std::move(result->data),
      result->numRows,
      result->atEnd,
      completedSequence);
}

void UcxExchangeSource::onIntraNodeData(
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    bool atEnd,
    uint32_t completedSequence) {
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
        completedSequence);
    VLOG(0) << toString() << " " << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  VLOG(3) << toString()
          << " Intra-node transfer: received data for seq=" << completedSequence
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
  auto tableWithStream = std::make_unique<PackedTableWithStream>(
      std::move(packedTable), stream, numRows);

  enqueue(std::move(tableWithStream));

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

bool UcxExchangeSource::pauseForBackpressureIfNeeded() {
  std::lock_guard<std::mutex> lock(queue_->mutex());
  const bool shouldPause = !queue_->receiveBytesBelowPrefetchLimitLocked() ||
      queue_->size() >= kBackpressureHighWaterMark;
  if (shouldPause) {
    // The consumer makes its resume decision under this same mutex. Publishing
    // the dormant state before unlocking prevents a dequeue from missing the
    // transition and leaving this source asleep indefinitely.
    backpressureActive_.store(true, std::memory_order_release);
  }
  return shouldPause;
}

} // namespace facebook::velox::ucx_exchange
