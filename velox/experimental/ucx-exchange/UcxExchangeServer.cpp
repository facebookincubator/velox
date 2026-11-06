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
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include <glog/logging.h>
#include <cuda/stream>
#include <limits>
#include <sstream>
#include "cuda_runtime.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<UcxExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxExchangeServer::ServerState::Created, "Created"},
              {UcxExchangeServer::ServerState::ReadyToTransfer,
               "ReadyToTransfer"},
              {UcxExchangeServer::ServerState::WaitingForDataFromQueue,
               "WaitingForDataFromQueue"},
              {UcxExchangeServer::ServerState::DataReady, "DataReady"},
              {UcxExchangeServer::ServerState::WaitingForSendComplete,
               "WaitingForSendComplete"},
              {UcxExchangeServer::ServerState::WaitingForIntraNodeRetrieve,
               "WaitingForIntraNodeRetrieve"},
              {UcxExchangeServer::ServerState::Done, "Done"},
          };
  return kNames;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeServer,
    ServerState,
    serverStateNames)

// Context wrappers for UCXX tagSend callbackData. These decouple the request
// lifetime from the buffer lifetime so payload memory is freed promptly after
// DMA completes even if cleanup must temporarily retain an in-flight request.
//
// The Request holds a shared_ptr to the context via callbackData. The
// context holds a shared_ptr to the actual buffer. When the send completion
// callback fires, it moves the buffer out of the context, releasing the GPU
// (or CPU) memory. The context remains alive as an empty shell for the
// lifetime of the Request, which is safe and costs negligible memory.
struct MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
  std::atomic<bool> callbackClaimed{false};

  bool tryClaimCallback() {
    bool expected = false;
    return callbackClaimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }
};

struct DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
  std::shared_ptr<UcxOutputQueue> outputQueue;
  int destination{0};
  int64_t bytes{0};
  std::atomic<bool> callbackClaimed{false};

  bool tryClaimCallback() {
    bool expected = false;
    return callbackClaimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }

  void releaseAccounting() {
    data.reset();
    outputQueue->releaseInFlightBytes(destination, bytes, 1);
  }
};

struct DequeuedDataContext {
  std::shared_ptr<UcxOutputQueue> outputQueue;
  std::shared_ptr<cudf::packed_columns> data;
  vector_size_t numRows{0};
  int destination{0};
  std::atomic<bool> callbackClaimed{false};

  bool tryClaimCallback() {
    bool expected = false;
    return callbackClaimed.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
  }

  void releaseAccounting() {
    if (!data) {
      return;
    }
    outputQueue->releaseInFlightBytes(
        destination, static_cast<int64_t>(data->gpu_data->size()), 1);
    data.reset();
  }
};

void UcxExchangeServer::setState(ServerState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private
UcxExchangeServer::UcxExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer)
    : CommElement(communicator, endpointRef),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      isIntraNodeTransfer_(isIntraNodeTransfer),
      queueMgr_(UcxOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);

  if (isIntraNodeTransfer_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Detected same-node source (intra-node transfer) for "
            << partitionKey_.toString();
  }
}

// static
std::shared_ptr<UcxExchangeServer> UcxExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer) {
  auto ptr = std::shared_ptr<UcxExchangeServer>(new UcxExchangeServer(
      communicator, endpointRef, key, isIntraNodeTransfer));
  return ptr;
}

void UcxExchangeServer::process() {
  std::lock_guard<std::recursive_mutex> processLock(processMutex_);
  processing_.store(true, std::memory_order_release);
  struct ProcessingGuard {
    std::atomic<bool>& processing;
    ~ProcessingGuard() {
      processing.store(false, std::memory_order_release);
    }
  } processingGuard{processing_};

  try {
    // Callback threads only enqueue state events. Apply those events under the
    // same exclusion as the state machine and close/error cleanup.
    try {
      drainStateEvents();
    } catch (const std::exception& error) {
      LOG(ERROR) << "@" << partitionKey_.taskId
                 << " failed to dispatch exchange-server callback: "
                 << error.what();
      setState(ServerState::Done);
    } catch (...) {
      LOG(ERROR) << "@" << partitionKey_.taskId
                 << " failed to dispatch exchange-server callback";
      setState(ServerState::Done);
    }

    // Check if close() was called - avoid processing if we're shutting down.
    if (closed_.load(std::memory_order_acquire)) {
      if (!cleanupComplete_) {
        close();
      }
      return;
    }
    switch (state_) {
      case ServerState::Created:
        setState(ServerState::ReadyToTransfer);
        communicator_->addToWorkQueue(getSelfPtr());
        break;
      case ServerState::ReadyToTransfer: {
        // Fetch the data from UcxQueueManager and store it in the dataPtr_;
        setState(ServerState::WaitingForDataFromQueue);
        // Register the callback with the destination queue to get data.
        // If the queue doesn't exist yet, getData will create an empty
        // queue and the callback will be triggered once the corresponding
        // source task has initialized the queue and added data to it.
        // Use weak_ptr to prevent use-after-free if close() is called during
        // callback
        std::weak_ptr<UcxExchangeServer> weakQueue = weak_from_this();
        const auto destination = partitionKey_.destination;
        auto callbackContext = std::make_shared<DequeuedDataContext>();
        auto outputQueue = queueMgr_->getDataWithQueue(
            partitionKey_.taskId,
            partitionKey_.destination,
            [weakQueue, destination, callbackContext](
                std::shared_ptr<UcxOutputQueue> stableQueue,
                std::shared_ptr<cudf::packed_columns> data,
                vector_size_t numRows,
                std::vector<int64_t> /*remainingBytes*/) {
              auto self = weakQueue.lock();
              if (!self) {
                if (data) {
                  stableQueue->releaseInFlightBytes(
                      destination,
                      static_cast<int64_t>(data->gpu_data->size()),
                      1);
                }
                return;
              }
              // Serialize the enqueue-or-release decision with close(). This
              // prevents close from draining and unregistering between a
              // callback's closed check and its state-event enqueue.
              std::lock_guard<std::recursive_mutex> processLock(
                  self->processMutex_);
              if (self->closed_.load(std::memory_order_acquire)) {
                if (data) {
                  stableQueue->releaseInFlightBytes(
                      destination,
                      static_cast<int64_t>(data->gpu_data->size()),
                      1);
                }
                return;
              }
              callbackContext->outputQueue = std::move(stableQueue);
              callbackContext->data = std::move(data);
              callbackContext->numRows = numRows;
              callbackContext->destination = destination;
              const bool dispatched = self->enqueueStateEvent(
                  self, [raw = self.get(), callbackContext]() {
                    try {
                      if (!callbackContext->tryClaimCallback()) {
                        return;
                      }
                      if (raw->closed_.load(std::memory_order_acquire)) {
                        callbackContext->releaseAccounting();
                        return;
                      }

                      std::lock_guard<std::recursive_mutex> lock(
                          raw->dataMutex_);
                      if (raw->dataPtr_) {
                        LOG(ERROR) << "@" << raw->partitionKey_.taskId
                                   << " received data while a prior payload is "
                                      "still pending";
                        callbackContext->releaseAccounting();
                        raw->setState(ServerState::Done);
                        return;
                      }
                      raw->outputQueue_ = callbackContext->outputQueue;
                      raw->dataPtr_ = std::move(callbackContext->data);
                      raw->dataNumRows_ = callbackContext->numRows;
                      VLOG(3) << "@" << raw->partitionKey_.taskId
                              << " Found data for client: "
                              << raw->partitionKey_.toString();
                      raw->setState(ServerState::DataReady);
                    } catch (...) {
                      // If ownership was not moved into dataPtr_, make one
                      // exact accounting attempt before terminating.
                      if (callbackContext->data) {
                        try {
                          callbackContext->releaseAccounting();
                        } catch (...) {
                        }
                      }
                      raw->handleCallbackDispatchFailure(
                          "output queue state event");
                    }
                  });
              if (!dispatched) {
                // deque::push_back has a strong exception guarantee. The claim
                // also protects against a work-queue failure after insertion:
                // a later dispatch observes that cleanup already won.
                if (callbackContext->tryClaimCallback()) {
                  try {
                    callbackContext->releaseAccounting();
                  } catch (...) {
                    // Preserve the callback boundary even if invariant cleanup
                    // itself detects corrupted accounting.
                  }
                }
                self->handleCallbackDispatchFailure("output queue");
              }
            });
        {
          std::lock_guard<std::recursive_mutex> lock(dataMutex_);
          outputQueue_ = std::move(outputQueue);
        }
        this->communicator_->addToWorkQueue(getSelfPtr());
      } break;
      case ServerState::WaitingForDataFromQueue:
        // Waiting for data is handled by an upcall from the data queue. Nothing
        // to do
        break;
      case ServerState::DataReady:
        sendData();
        break;
      case ServerState::WaitingForSendComplete:
        // Waiting for send complete is handled by an upcall from UCXX. Nothing
        // to do
        break;
      case ServerState::WaitingForIntraNodeRetrieve:
        // Intra-node transfer: check if the source has retrieved the data
        if (intraNodeRetrieveFuture_.valid()) {
          auto status =
              intraNodeRetrieveFuture_.wait_for(std::chrono::milliseconds(0));
          if (status == std::future_status::ready) {
            try {
              intraNodeRetrieveFuture_.get(); // Clear the future
            } catch (const IntraNodeTransferCancelled&) {
              VLOG(2) << "@" << partitionKey_.taskId
                      << " intra-node destination cancelled sequence "
                      << sequenceNumber_;
              releaseIntraNodeInFlightBytes();
              setState(ServerState::Done);
              communicator_->addToWorkQueue(getSelfPtr());
              break;
            } catch (const std::exception& error) {
              LOG(ERROR) << "@" << partitionKey_.taskId
                         << " intra-node retrieval failed: " << error.what();
              releaseIntraNodeInFlightBytes();
              setState(ServerState::Done);
              communicator_->addToWorkQueue(getSelfPtr());
              break;
            } catch (...) {
              LOG(ERROR) << "@" << partitionKey_.taskId
                         << " intra-node retrieval failed";
              releaseIntraNodeInFlightBytes();
              setState(ServerState::Done);
              communicator_->addToWorkQueue(getSelfPtr());
              break;
            }
            intraNodePollCount_ = 0;
            onIntraNodeRetrieveComplete();
          } else {
            // Not ready yet, re-queue to check later
            ++intraNodePollCount_;
            if (intraNodePollCount_ % 100 == 0) {
              VLOG(2) << "[INTRA] [ExSrv " << partitionKey_.toString()
                      << " seq=" << sequenceNumber_
                      << "] still waiting for source retrieval, polls="
                      << intraNodePollCount_;
            }
            communicator_->addToWorkQueue(getSelfPtr());
          }
        }
        break;
      case ServerState::Done:
        close();
        break;
    };
    // Inline callbacks can fail their work-queue handoff while sendData() is
    // still publishing the Request member. Finish that call first, then close
    // safely in this same communicator turn.
    if (getState() == ServerState::Done && !cleanupComplete_) {
      close();
    }
  } catch (const std::exception& error) {
    handleProcessFailure(error.what());
  } catch (...) {
    handleProcessFailure("unknown exchange-server exception");
  }
}

void UcxExchangeServer::close() noexcept {
  try {
    std::lock_guard<std::recursive_mutex> processLock(processMutex_);
    if (cleanupComplete_) {
      return;
    }
    closed_.store(true, std::memory_order_release);

    auto logCleanupFailure = [&](const char* operation) noexcept {
      try {
        LOG(ERROR) << "@" << partitionKey_.taskId
                   << " exchange-server cleanup failed during " << operation;
      } catch (...) {
      }
    };
    auto scheduleRetry = [&]() noexcept {
      try {
        communicator_->addToWorkQueue(getSelfPtr());
      } catch (...) {
        // The communicator still owns this registered server. A later shutdown
        // or endpoint drain can retry close even if host OOM prevents requeue.
      }
    };

    // A queue callback can have dequeued a page immediately before close won
    // processMutex_. Dispatch it in the closed state so it releases accounting
    // rather than stranding the page in the event queue.
    try {
      drainStateEvents();
    } catch (...) {
      logCleanupFailure("callback drain");
    }
    VLOG(3) << "@" << partitionKey_.taskId
            << " Close UcxExchangeServer to remote "
            << partitionKey_.toString();

    // The registry owns the published GPU page between publish() and poll().
    // Closing either side must remove that entry before producer accounting is
    // released, otherwise the page and retrieval future can live forever while
    // the flow-control counters incorrectly report the bytes as free.
    if (isIntraNodeTransfer_ && intraNodeRetrieveFuture_.valid()) {
      try {
        IntraNodeTransferRegistry::getInstance()->cancelTransfer(
            {partitionKey_.taskId, partitionKey_.destination, sequenceNumber_});
        try {
          intraNodeRetrieveFuture_.get();
        } catch (const IntraNodeTransferCancelled&) {
          // This close initiated the cancellation. The exceptional future is
          // only a control signal for an otherwise-running producer.
        }
      } catch (...) {
        logCleanupFailure("intra-node transfer cancellation");
        scheduleRetry();
        return;
      }
    }

    // Tag sends cannot be safely cancelled after UCX has committed them.
    // Pass a copy to deferred cleanup and clear our reference only after the
    // vector insertion succeeds. This makes host-allocation failure retryable
    // without exposing the Request callback to use-after-free.
    bool requestsDeferred = true;
    auto deferRequest = [&](std::shared_ptr<ucxx::Request>& request) noexcept {
      if (!request) {
        return;
      }
      try {
        if (request->isCompleted()) {
          request.reset();
          return;
        }
        communicator_->deferRequestCleanup(request);
        request.reset();
      } catch (...) {
        requestsDeferred = false;
        logCleanupFailure("request deferral");
      }
    };
    if (communicator_) {
      deferRequest(metaRequest_);
      deferRequest(dataRequest_);
    }
    if (!requestsDeferred) {
      scheduleRetry();
      return;
    }

    // A dequeued page is accounted until it is either claimed by a transport
    // callback or explicitly abandoned here. Both helpers are exact-once.
    try {
      releasePendingData();
    } catch (...) {
      logCleanupFailure("pending-data accounting");
    }
    try {
      releaseIntraNodeInFlightBytes();
    } catch (...) {
      logCleanupFailure("intra-node accounting");
    }
    if (outputQueue_ && !outputResultsDeleted_) {
      try {
        // Clear an installed getData callback and discard queued output on
        // every error/cancellation close, not only the ordinary end marker.
        outputQueue_->deleteResults(partitionKey_.destination);
        outputResultsDeleted_ = true;
      } catch (...) {
        logCleanupFailure("output deletion");
        scheduleRetry();
        return;
      }
    }

    if (endpointRef_) {
      try {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      } catch (...) {
        logCleanupFailure("endpoint unregister");
        scheduleRetry();
        return;
      }
    }
    try {
      communicator_->unregister(getSelfPtr());
    } catch (...) {
      logCleanupFailure("communicator unregister");
      scheduleRetry();
      return;
    }
    cleanupComplete_ = true;
  } catch (...) {
    // close() is reachable from endpoint and UCXX callback paths. Never let an
    // exchange-specific cleanup failure unwind through those boundaries.
    try {
      communicator_->addToWorkQueue(getSelfPtr());
    } catch (...) {
      // The registered communicator/endpoint references still retain this
      // server for a later shutdown retry.
    }
  }
}

std::string UcxExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<UcxExchangeServer> UcxExchangeServer::getSelfPtr() {
  return shared_from_this();
}

void UcxExchangeServer::sendData() {
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);

  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_
          << "] sendData hasData=" << (dataPtr_ != nullptr)
          << (dataPtr_ && dataPtr_->gpu_data
                  ? " size=" + std::to_string(dataPtr_->gpu_data->size())
                  : "");

  if (isIntraNodeTransfer_) {
    // INTRA-NODE TRANSFER PATH: Use registry for all communication, no UCXX
    // needed
    sendStart_ = std::chrono::high_resolution_clock::now();

    if (dataPtr_) {
      bytes_ = dataPtr_->gpu_data->size();
      VELOX_CHECK_LE(
          bytes_,
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
          "UCX payload size exceeds signed accounting range");
      VELOX_CHECK_NOT_NULL(
          outputQueue_, "Dequeued payload has no stable output queue");

      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing data for sequence "
              << sequenceNumber_ << " of size " << bytes_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      // dataPtr_ is already a shared_ptr, pass directly to share ownership.
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key, dataPtr_, dataNumRows_, /*atEnd=*/false);
      bool expected = false;
      VELOX_CHECK(
          intraNodeBytesInFlight_.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel),
          "Previous intra-node payload is still in flight");
      dataPtr_.reset();
      dataNumRows_ = 0;
      intraNodeAtEndPublished_ = false;

      // Transition to WaitingForIntraNodeRetrieve state
      setState(ServerState::WaitingForIntraNodeRetrieve);
      communicator_->addToWorkQueue(getSelfPtr());
    } else {
      // Data pointer is null, so no more data will be coming.
      // Publish atEnd marker to registry
      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing atEnd for sequence "
              << sequenceNumber_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key, nullptr, /*numRows=*/0, /*atEnd=*/true);
      intraNodeAtEndPublished_ = true;

      if (outputQueue_) {
        outputQueue_->deleteResults(partitionKey_.destination);
      } else {
        queueMgr_->deleteResults(
            partitionKey_.taskId, partitionKey_.destination);
      }
      outputResultsDeleted_ = true;

      // Wait for source to acknowledge atEnd before finishing
      setState(ServerState::WaitingForIntraNodeRetrieve);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  } else {
    // REMOTE EXCHANGE PATH: Use UCXX for metadata and data transfer
    std::shared_ptr<MetadataMsg> metadataMsg = std::make_shared<MetadataMsg>();

    if (dataPtr_) {
      // Copy metadata (not move) because in broadcast mode, the same
      // packed_columns may be shared across multiple destination queues.
      // Metadata is small (CPU-side), so copying is negligible.
      metadataMsg->cudfMetadata =
          std::make_unique<std::vector<uint8_t>>(*dataPtr_->metadata);
      metadataMsg->dataSizeBytes = dataPtr_->gpu_data->size();
      metadataMsg->numRows = dataNumRows_;
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = false;
    } else {
      VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
              << partitionKey_.toString();
      metadataMsg->cudfMetadata = nullptr;
      metadataMsg->dataSizeBytes = 0;
      metadataMsg->numRows = 0;
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = true;
    }

    auto [serializedMetadata, serMetaSize] = metadataMsg->serialize();

    // send metadata.
    uint64_t metadataTag =
        getMetadataTag(this->partitionKeyHash_, this->sequenceNumber_);
    // Use weak_ptr to prevent use-after-free if close() is called during
    // callback
    std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
    retireRequest(metaRequest_);

    // Wrap serialized metadata so the completion can release its payload
    // independently of the request shell.
    auto metaCtx = std::make_shared<MetaSendContext>();
    metaCtx->metadata = serializedMetadata;

    try {
      metaRequest_ =
          endpointRef_->endpoint_
              ->tagSendBuilder(
                  metaCtx->metadata.get(), serMetaSize, ucxx::Tag{metadataTag})
              .callbackFunction([metadataTag, weakMeta](
                                    ucs_status_t status,
                                    std::shared_ptr<void> arg) noexcept {
                auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
                if (!ctx->tryClaimCallback()) {
                  return;
                }
                // Release the payload once; a duplicate callback must not
                // advance the server state or release ownership twice.
                ctx->metadata.reset();
                if (auto self = weakMeta.lock();
                    self && !self->closed_.load(std::memory_order_acquire)) {
                  if (!self->enqueueStateEvent(
                          self, [raw = self.get(), status, metadataTag]() {
                            try {
                              raw->metadataSendComplete(status, metadataTag);
                            } catch (...) {
                              raw->handleCallbackDispatchFailure(
                                  "metadata state event");
                            }
                          })) {
                    self->handleCallbackDispatchFailure("metadata send");
                  }
                }
              })
              .callbackData(metaCtx)
              .build();
    } catch (const std::exception& error) {
      LOG(ERROR) << "@" << partitionKey_.taskId
                 << " failed to submit metadata send: " << error.what();
      releasePendingData();
      setState(ServerState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    } catch (...) {
      LOG(ERROR) << "@" << partitionKey_.taskId
                 << " failed to submit metadata send";
      releasePendingData();
      setState(ServerState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    }

    // An inline callback that could not enqueue its state event uses Done as
    // the fail-safe. Do not overwrite it by committing a data send.
    if (getState() == ServerState::Done) {
      releasePendingData();
      return;
    }

    // send the data chunk (if any)
    if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ = dataPtr_->gpu_data->size();
      VELOX_CHECK_LE(
          bytes_,
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
          "UCX payload size exceeds signed accounting range");
      VELOX_CHECK_NOT_NULL(
          outputQueue_, "Dequeued payload has no stable output queue");

      VLOG(3) << "@" << partitionKey_.taskId
              << " Sending rmm::buffer: " << std::hex
              << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task "
              << partitionKey_.toString() << ":" << this->sequenceNumber_
              << std::dec << " of size " << bytes_;

      setState(ServerState::WaitingForSendComplete);
      uint64_t dataTag =
          getDataTag(this->partitionKeyHash_, this->sequenceNumber_);
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();
      retireRequest(dataRequest_);

      // Wrap the GPU data buffer so completion can release both payload and
      // accounting independently of the request shell.
      auto dataCtx = std::make_shared<DataSendContext>();
      dataCtx->data = dataPtr_;
      dataCtx->outputQueue = outputQueue_;
      dataCtx->destination = partitionKey_.destination;
      dataCtx->bytes = static_cast<int64_t>(bytes_);

      try {
        dataRequest_ =
            endpointRef_->endpoint_
                ->tagSendBuilder(
                    dataCtx->data->gpu_data->data(),
                    dataCtx->data->gpu_data->size(),
                    ucxx::Tag{dataTag})
                .callbackFunction([weakData](
                                      ucs_status_t status,
                                      std::shared_ptr<void> arg) noexcept {
                  auto ctx = std::static_pointer_cast<DataSendContext>(arg);
                  if (!ctx->tryClaimCallback()) {
                    return;
                  }
                  // Zero-byte, zero-column-schema pages still release one
                  // packed column accounting unit.
                  auto self = weakData.lock();
                  try {
                    ctx->releaseAccounting();
                  } catch (...) {
                    if (self) {
                      self->handleCallbackDispatchFailure(
                          "data send accounting");
                    }
                    return;
                  }
                  if (self && !self->closed_.load(std::memory_order_acquire)) {
                    if (!self->enqueueStateEvent(
                            self, [raw = self.get(), status]() {
                              try {
                                raw->sendComplete(status);
                              } catch (...) {
                                raw->handleCallbackDispatchFailure(
                                    "data state event");
                              }
                            })) {
                      self->handleCallbackDispatchFailure("data send");
                    }
                  }
                })
                .callbackData(dataCtx)
                .build();
      } catch (const std::exception& error) {
        if (dataCtx->tryClaimCallback()) {
          try {
            dataCtx->releaseAccounting();
          } catch (...) {
            LOG(ERROR) << "@" << partitionKey_.taskId
                       << " failed to release rejected data send accounting";
          }
        }
        dataPtr_.reset();
        dataNumRows_ = 0;
        LOG(ERROR) << "@" << partitionKey_.taskId
                   << " failed to submit data send: " << error.what();
        setState(ServerState::Done);
        communicator_->addToWorkQueue(getSelfPtr());
        return;
      } catch (...) {
        if (dataCtx->tryClaimCallback()) {
          try {
            dataCtx->releaseAccounting();
          } catch (...) {
            LOG(ERROR) << "@" << partitionKey_.taskId
                       << " failed to release rejected data send accounting";
          }
        }
        dataPtr_.reset();
        dataNumRows_ = 0;
        LOG(ERROR) << "@" << partitionKey_.taskId
                   << " failed to submit data send";
        setState(ServerState::Done);
        communicator_->addToWorkQueue(getSelfPtr());
        return;
      }
      // UCXX's callback context now owns the payload until DMA completion.
      dataPtr_.reset();
      dataNumRows_ = 0;
    } else {
      // Data pointer is null, so no more data will be coming.
      VLOG(3) << "@" << partitionKey_.taskId
              << " Finished transferring partition for task "
              << partitionKey_.toString();
      if (outputQueue_) {
        outputQueue_->deleteResults(partitionKey_.destination);
      } else {
        queueMgr_->deleteResults(
            partitionKey_.taskId, partitionKey_.destination);
      }
      outputResultsDeleted_ = true;
      setState(ServerState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  }
}

void UcxExchangeServer::handleCallbackDispatchFailure(
    const char* callback) noexcept {
  try {
    LOG(ERROR) << "@" << partitionKey_.taskId << " could not dispatch "
               << callback << " callback to the server state machine";
  } catch (...) {
  }
  try {
    setState(ServerState::Done);
  } catch (...) {
    state_.store(ServerState::Done, std::memory_order_release);
  }
  try {
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  } catch (...) {
  }

  // An inline completion can run before tagSend() publishes its Request into
  // the member field. The active process turn observes Done and closes after
  // tagSend returns. An asynchronous completion has no such unpublished
  // request and can make one best-effort, noexcept cleanup attempt here.
  if (!processing_.load(std::memory_order_acquire) &&
      !closed_.load(std::memory_order_acquire)) {
    close();
  }
}

void UcxExchangeServer::handleProcessFailure(const char* error) noexcept {
  try {
    LOG(ERROR) << "@" << partitionKey_.taskId
               << " exchange-server state machine failed: " << error;
  } catch (...) {
  }

  try {
    setState(ServerState::Done);
  } catch (...) {
    state_.store(ServerState::Done, std::memory_order_release);
  }

  try {
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  } catch (...) {
    // If even scheduling cleanup fails (for example under host OOM), perform
    // one best-effort close now. closed_ makes this terminal rather than a
    // repeatedly requeued failure.
  }
  try {
    close();
  } catch (...) {
  }
}

void UcxExchangeServer::metadataSendComplete(
    ucs_status_t status,
    uint64_t metadataTag) {
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " metadata send completion dispatched after close, ignoring";
    return;
  }
  if (status == UCS_OK) {
    VLOG(3) << "@" << partitionKey_.taskId << " metadata successfully sent to "
            << partitionKey_.toString() << " with tag: " << std::hex
            << metadataTag;
    return;
  }

  VLOG(0) << "@" << partitionKey_.taskId << " Error in sendData, send metadata "
          << ucs_status_string(status)
          << " failed for task: " << partitionKey_.toString();
  setState(ServerState::Done);
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::sendComplete(ucs_status_t status) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " sendComplete called after close, ignoring";
    return;
  }
  if (getState() == ServerState::Done) {
    return;
  }
  if (status == UCS_OK) {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = micros > 0 ? bytes_ / micros : 0;

    VLOG(3) << "@" << partitionKey_.taskId << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
            << " MByte/s";

    this->sequenceNumber_++;
    VLOG(3) << "@" << partitionKey_.taskId
            << " Send complete; advancing to next sequence.";
    setState(ServerState::ReadyToTransfer);
  } else {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Error in sendComplete, send complete "
            << ucs_status_string(status);
    setState(ServerState::Done);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::onIntraNodeRetrieveComplete() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " onIntraNodeRetrieveComplete called after close, ignoring";
    return;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - sendStart_;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

  VLOG(3)
      << "@" << partitionKey_.taskId << " Intra-node transfer duration: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms ";
  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer throughput: " << throughput << " MByte/s";

  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer complete for sequence " << sequenceNumber_;

  releaseIntraNodeInFlightBytes();

  if (intraNodeAtEndPublished_) {
    // This was the final atEnd marker, we're done
    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: atEnd acknowledged, finishing";
    setState(ServerState::Done);
  } else {
    // More data may be coming, continue transfer loop
    this->sequenceNumber_++;
    setState(ServerState::ReadyToTransfer);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::releaseIntraNodeInFlightBytes() {
  if (!intraNodeBytesInFlight_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  VELOX_CHECK_LE(
      bytes_, static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  std::shared_ptr<UcxOutputQueue> outputQueue;
  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    outputQueue = outputQueue_;
  }
  VELOX_CHECK_NOT_NULL(
      outputQueue, "Intra-node payload has no stable output queue");
  outputQueue->releaseInFlightBytes(
      partitionKey_.destination, static_cast<int64_t>(bytes_), 1);
}

void UcxExchangeServer::releasePendingData() {
  std::shared_ptr<UcxOutputQueue> outputQueue;
  int64_t bytes = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    if (!dataPtr_) {
      return;
    }
    VELOX_CHECK_LE(
        dataPtr_->gpu_data->size(),
        static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
    bytes = static_cast<int64_t>(dataPtr_->gpu_data->size());
    outputQueue = outputQueue_;
    dataPtr_.reset();
    dataNumRows_ = 0;
  }
  VELOX_CHECK_NOT_NULL(
      outputQueue, "Dequeued payload has no stable output queue");
  outputQueue->releaseInFlightBytes(partitionKey_.destination, bytes, 1);
}

void UcxExchangeServer::retireRequest(std::shared_ptr<ucxx::Request>& request) {
  if (!request) {
    return;
  }
  if (request->isCompleted()) {
    request.reset();
    return;
  }

  // Preserve our reference until insertion succeeds. Incomplete UCXX callback
  // state remains communicator-owned until isCompleted() becomes true.
  communicator_->deferRequestCleanup(request);
  request.reset();
}

} // namespace facebook::velox::ucx_exchange
