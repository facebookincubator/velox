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
#include "cuda_runtime.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxChannelState.h"
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

// Context wrappers for UCXX tagSend callbackData. These decouple the
// ucxx::Request lifetime (which must survive for UCP wireup replay) from
// the buffer lifetime (which should be freed promptly after DMA completes).
//
// The Request holds a shared_ptr to the context via callbackData. The
// context holds a shared_ptr to the actual buffer. When the send completion
// callback fires, it moves the buffer out of the context, releasing the GPU
// (or CPU) memory, and records the send's status for process() to act on.
// The context remains alive as an empty shell for the lifetime of the
// Request, which is safe and costs negligible memory.
struct UcxExchangeServer::MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
  // UCS_INPROGRESS until the send completes.
  std::atomic<ucs_status_t> status{UCS_INPROGRESS};
};

struct UcxExchangeServer::DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
  // UCS_INPROGRESS until the send completes.
  std::atomic<ucs_status_t> status{UCS_INPROGRESS};
};

namespace {
// Records the status a send completed with. Keeps the first one, since UCP can
// replay the completion of a request. Runs in a UCX callback.
void recordSendStatus(std::atomic<ucs_status_t>& slot, ucs_status_t status) {
  auto expected = UCS_INPROGRESS;
  slot.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
}

bool hasFailed(ucs_status_t status) {
  return status != UCS_OK && status != UCS_INPROGRESS;
}
} // namespace

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
  std::weak_ptr<UcxExchangeServer> weakServer = ptr;
  // Runs on any thread, possibly before bindReader() returns, so it only marks
  // the server and never touches 'entry_'. process() acts on the mark.
  ptr->entry_ =
      ptr->queueMgr_->bindReader(key.taskId, key.destination, [weakServer]() {
        if (auto self = weakServer.lock()) {
          self->released_.store(true, std::memory_order_release);
          self->communicator_->addToWorkQueue(self);
        }
      });
  return ptr;
}

void UcxExchangeServer::process() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  if (released_.load(std::memory_order_acquire)) {
    // The destination's consumer is gone, or its output was deleted.
    setState(ServerState::Done);
    close();
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
      // Register the callback with the destination queue to get data. If the
      // task's entry is still Pending, the callback is triggered once the
      // source task has initialized it and added data to it.
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      std::weak_ptr<UcxExchangeServer> weakQueue = weak_from_this();
      entry_->getData(partitionKey_.destination, [weakQueue](ReaderStep step) {
        auto self = weakQueue.lock();
        if (!self) {
          return; // Object was destroyed, safe to ignore
        }
        // Check if close() was called - avoid processing if we're shutting
        // down
        if (self->closed_.load(std::memory_order_acquire)) {
          VLOG(3) << "@" << self->partitionKey_.taskId
                  << " getData callback called after close, ignoring";
          return;
        }
        // This upcall may be called from another thread than the
        // communicator thread. It is called
        // when data on the queue becomes available.
        VLOG(3) << "@" << self->partitionKey_.taskId
                << " Found data for client: " << self->partitionKey_.toString();
        std::lock_guard<std::mutex> lock(self->dataMutex_);
        VELOX_CHECK_NULL(self->dataPtr_, "Data pointer exists: Illegal state!");
        switch (step.kind) {
          case ReaderStep::Kind::kChunk:
            self->dataPtr_ = std::move(step.data);
            self->dataNumRows_ = step.numRows;
            self->setState(ServerState::DataReady);
            break;
          case ReaderStep::Kind::kEnd:
            // A null dataPtr_ is the end of the channel.
            self->dataNumRows_ = 0;
            self->endOutcome_ = std::move(step.outcome);
            self->setState(ServerState::DataReady);
            break;
          case ReaderStep::Kind::kStop:
            self->released_.store(true, std::memory_order_release);
            break;
        }
        self->communicator_->addToWorkQueue(self);
      });
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
      // Each send's completion records its status and requeues the server.
      onSendsCompleted();
      break;
    case ServerState::WaitingForIntraNodeRetrieve:
      // Intra-node transfer: check if the source has retrieved the data
      if (intraNodeRetrieveFuture_.valid()) {
        auto status =
            intraNodeRetrieveFuture_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
          intraNodeRetrieveFuture_.get(); // Clear the future
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
}

void UcxExchangeServer::onEndpointClosed(std::string_view reason) {
  VLOG(1) << toString() << " " << reason;
  entry_->consumerGone(partitionKey_.destination);
  close();
}

void UcxExchangeServer::close() {
  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }
  VLOG(3) << "@" << partitionKey_.taskId
          << " Close UcxExchangeServer to remote " << partitionKey_.toString();

  // Cancel any outstanding requests. With weak_ptr callbacks, the callbacks
  // will safely no-op if we're destroyed before they complete.
  if (metaRequest_ && !metaRequest_->isCompleted()) {
    metaRequest_->cancel();
  }
  if (dataRequest_ && !dataRequest_->isCompleted()) {
    dataRequest_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  if (communicator_) {
    if (metaRequest_) {
      communicator_->deferRequestCleanup(std::move(metaRequest_));
    }
    if (dataRequest_) {
      communicator_->deferRequestCleanup(std::move(dataRequest_));
    }
    for (auto& req : completedRequests_) {
      communicator_->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }
  metaContext_.reset();
  dataContext_.reset();

  queueMgr_->releaseReader(entry_, partitionKey_.destination);

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  communicator_->unregister(getSelfPtr());
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
  // A send may complete inside the call that posts it, on this thread with
  // this lock held. Its completion therefore only records its status.
  std::unique_lock<std::mutex> lock(dataMutex_);

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

      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing data for sequence "
              << sequenceNumber_ << " of size " << bytes_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      // dataPtr_ is already a shared_ptr, pass directly to share ownership.
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key, dataPtr_, dataNumRows_, /*atEnd=*/false);
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

      // Wait for source to acknowledge atEnd before finishing
      setState(ServerState::WaitingForIntraNodeRetrieve);
      // deleteResults() may finish the producer task, which must not happen
      // under 'dataMutex_'.
      lock.unlock();
      entry_->deleteResults(partitionKey_.destination);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  } else {
    // REMOTE EXCHANGE PATH: Use UCXX for metadata and data transfer
    std::pair<std::shared_ptr<uint8_t>, size_t> serialized;

    if (dataPtr_) {
      MetadataMsg metadataMsg;
      // Copy metadata (not move) because in broadcast mode, the same
      // packed_columns may be shared across multiple destination queues.
      // Metadata is small (CPU-side), so copying is negligible.
      metadataMsg.cudfMetadata =
          std::make_unique<std::vector<uint8_t>>(*dataPtr_->metadata);
      metadataMsg.dataSizeBytes = dataPtr_->gpu_data->size();
      metadataMsg.numRows = dataNumRows_;
      metadataMsg.remainingBytes = {};
      serialized = metadataMsg.serialize();
    } else {
      VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
              << partitionKey_.toString();
      serialized = serializeEndMessage(endOutcome_);
    }

    auto [serializedMetadata, serMetaSize] = std::move(serialized);

    // Set before posting, since the send may complete inside the post.
    setState(ServerState::WaitingForSendComplete);

    // send metadata.
    uint64_t metadataTag =
        getMetadataTag(this->partitionKeyHash_, this->sequenceNumber_);
    // Use weak_ptr to prevent use-after-free if close() is called during
    // callback
    std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
    if (metaRequest_) {
      completedRequests_.push_back(std::move(metaRequest_));
    }

    // Wrap the serialized metadata in a context so the callback can release
    // it after the send completes, while the Request (and context shell)
    // stays alive for UCP wireup replay.
    auto metaCtx = std::make_shared<MetaSendContext>();
    metaCtx->metadata = serializedMetadata;
    metaContext_ = metaCtx;
    dataContext_ = nullptr;

    metaRequest_ =
        endpointRef_->endpoint_
            ->tagSendBuilder(
                metaCtx->metadata.get(), serMetaSize, ucxx::Tag{metadataTag})
            .callbackFunction(
                [tid = partitionKey_.toString(), metadataTag, weakMeta](
                    ucs_status_t status, std::shared_ptr<void> arg) {
                  // Release the metadata buffer from the context. The context
                  // shell stays alive with the Request; only the payload is
                  // freed.
                  auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
                  auto metaHolder =
                      std::move(ctx->metadata); // release CPU buffer
                  recordSendStatus(ctx->status, status);
                  VLOG(3) << "Metadata send to " << tid << " with tag "
                          << std::hex << metadataTag << std::dec
                          << " completed: " << ucs_status_string(status);

                  auto self = weakMeta.lock();
                  // process() acts on the status, outside this callback.
                  if (self && !self->closed_.load(std::memory_order_acquire)) {
                    self->communicator_->addToWorkQueue(self);
                  }
                })
            .callbackData(metaCtx)
            .build();

    // send the data chunk (if any)
    if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ = dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Sending rmm::buffer: " << std::hex
              << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task "
              << partitionKey_.toString() << ":" << this->sequenceNumber_
              << std::dec << " of size " << bytes_;

      uint64_t dataTag =
          getDataTag(this->partitionKeyHash_, this->sequenceNumber_);
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();
      if (dataRequest_) {
        completedRequests_.push_back(std::move(dataRequest_));
      }

      // Wrap the GPU data buffer in a context so the callback can release
      // it after the DMA completes, while the Request (and context shell)
      // stays alive for UCP wireup replay.
      auto dataCtx = std::make_shared<DataSendContext>();
      dataCtx->data = dataPtr_;
      dataContext_ = dataCtx;

      dataRequest_ =
          endpointRef_->endpoint_
              ->tagSendBuilder(
                  dataCtx->data->gpu_data->data(),
                  dataCtx->data->gpu_data->size(),
                  ucxx::Tag{dataTag})
              .callbackFunction(
                  [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
                    // Release the GPU data buffer from the context. The DMA has
                    // completed by the time this callback fires, so the buffer
                    // is safe to free. The context shell stays alive with the
                    // Request.
                    auto ctx = std::static_pointer_cast<DataSendContext>(arg);
                    auto dataHolder = std::move(ctx->data);
                    recordSendStatus(ctx->status, status);

                    auto self = weakData.lock();
                    // process() acts on the status, outside this callback.
                    if (self &&
                        !self->closed_.load(std::memory_order_acquire)) {
                      self->communicator_->addToWorkQueue(self);
                    }
                  })
              .callbackData(dataCtx)
              .build();
    }
  }
}

void UcxExchangeServer::onSendsCompleted() {
  VELOX_CHECK_NOT_NULL(metaContext_, "No send to wait for");
  const auto metaStatus = metaContext_->status.load(std::memory_order_acquire);
  const auto dataStatus = dataContext_ != nullptr
      ? dataContext_->status.load(std::memory_order_acquire)
      : UCS_OK;
  if (hasFailed(metaStatus)) {
    onSendFailed(metaStatus);
    return;
  }
  if (hasFailed(dataStatus)) {
    onSendFailed(dataStatus);
    return;
  }
  if (metaStatus == UCS_INPROGRESS || dataStatus == UCS_INPROGRESS) {
    // The completion of the send still pending requeues the server.
    return;
  }

  if (dataContext_ == nullptr) {
    // The consumer received the end message.
    VLOG(3) << "@" << partitionKey_.taskId
            << " Finished transferring partition for task "
            << partitionKey_.toString();
    metaContext_.reset();
    entry_->deleteResults(partitionKey_.destination);
    setState(ServerState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
    return;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - sendStart_;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

  VLOG(3)
      << "@" << partitionKey_.taskId << " duration: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms ";
  VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
          << " MByte/s";

  this->sequenceNumber_++;
  metaContext_.reset();
  dataContext_.reset();
  {
    std::lock_guard<std::mutex> lock(dataMutex_);
    dataPtr_.reset(); // release memory.
    dataNumRows_ = 0;
  }
  setState(ServerState::ReadyToTransfer);
  communicator_->addToWorkQueue(getSelfPtr());
}

void UcxExchangeServer::onSendFailed(ucs_status_t status) {
  const bool endpointAlive = endpointRef_ != nullptr &&
      endpointRef_->endpoint_ != nullptr && endpointRef_->endpoint_->isAlive();
  const auto cause = classifySendFailure(
      closed_.load(std::memory_order_acquire) ||
          released_.load(std::memory_order_acquire),
      endpointAlive);
  VLOG(1) << toString() << " send failed: " << ucs_status_string(status)
          << ", endpoint alive: " << endpointAlive;
  switch (cause) {
    case SendFailureCause::kNone:
      break;
    case SendFailureCause::kConsumerGone:
      entry_->consumerGone(partitionKey_.destination);
      break;
    case SendFailureCause::kSendFailed:
      if (auto error = entry_->sendFailed(
              partitionKey_.destination, ucs_status_string(status))) {
        LOG(ERROR) << *error;
        if (auto task = entry_->task()) {
          task->setError(*error);
        }
      }
      break;
  }
  setState(ServerState::Done);
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

} // namespace facebook::velox::ucx_exchange
