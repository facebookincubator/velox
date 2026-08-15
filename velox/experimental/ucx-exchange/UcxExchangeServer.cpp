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
#include <rmm/cuda_stream_view.hpp>
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

// Context wrappers for UCXX tagSend callbackData. These decouple the
// ucxx::Request lifetime (which must survive for UCP wireup replay) from
// the buffer lifetime (which should be freed promptly after DMA completes).
//
// The Request holds a shared_ptr to the context via callbackData. The
// context holds a shared_ptr to the actual buffer. When the send completion
// callback fires, it moves the buffer out of the context, releasing the GPU
// (or CPU) memory. The context remains alive as an empty shell for the
// lifetime of the Request, which is safe and costs negligible memory.
struct MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
};

struct DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
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
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
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
      queueMgr_->getData(
          partitionKey_.taskId,
          partitionKey_.destination,
          [weakQueue](
              std::shared_ptr<cudf::packed_columns> data,
              std::vector<int64_t> remainingBytes) {
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
                    << " Found data for client: "
                    << self->partitionKey_.toString();
            std::lock_guard<std::recursive_mutex> lock(self->dataMutex_);
            VELOX_CHECK_NULL(
                self->dataPtr_, "Data pointer exists: Illegal state!");
            self->dataPtr_ = std::move(data);
            self->setState(ServerState::DataReady);
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
      // Waiting for send complete is handled by an upcall from UCXX. Nothing to
      // do
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
      if (endpointRef_) {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      }
      break;
  };
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

      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing data for sequence "
              << sequenceNumber_ << " of size " << bytes_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      // dataPtr_ is already a shared_ptr, pass directly to share ownership.
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key, dataPtr_, /*atEnd=*/false);
      dataPtr_.reset();
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
              key, nullptr, /*atEnd=*/true);
      intraNodeAtEndPublished_ = true;

      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);

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
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = false;
    } else {
      VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
              << partitionKey_.toString();
      metadataMsg->cudfMetadata = nullptr;
      metadataMsg->dataSizeBytes = 0;
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
    if (metaRequest_) {
      completedRequests_.push_back(std::move(metaRequest_));
    }

    // Wrap the serialized metadata in a context so the callback can release
    // it after the send completes, while the Request (and context shell)
    // stays alive for UCP wireup replay.
    auto metaCtx = std::make_shared<MetaSendContext>();
    metaCtx->metadata = serializedMetadata;

    metaRequest_ = endpointRef_->endpoint_->tagSend(
        metaCtx->metadata.get(),
        serMetaSize,
        ucxx::Tag{metadataTag},
        false,
        [tid = partitionKey_.toString(), metadataTag, weakMeta](
            ucs_status_t status, std::shared_ptr<void> arg) {
          // Release the metadata buffer from the context. The context
          // shell stays alive with the Request; only the payload is freed.
          auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
          auto metaHolder = std::move(ctx->metadata); // release CPU buffer

          auto self = weakMeta.lock();
          if (!self) {
            return; // Object was destroyed, safe to ignore
          }
          // Check if close() was called
          if (self->closed_.load(std::memory_order_acquire)) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata send callback called after close, ignoring";
            return;
          }
          if (status == UCS_OK) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata successfully sent to " << tid
                    << " with tag: " << std::hex << metadataTag;
          } else {
            VLOG(0) << "@" << self->partitionKey_.taskId
                    << " Error in sendData, send metadata "
                    << ucs_status_string(status) << " failed for task: " << tid;
            self->setState(ServerState::Done);
            self->communicator_->addToWorkQueue(self);
          }
        },
        metaCtx);

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

      setState(ServerState::WaitingForSendComplete);
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

      dataRequest_ = endpointRef_->endpoint_->tagSend(
          dataCtx->data->gpu_data->data(),
          dataCtx->data->gpu_data->size(),
          ucxx::Tag{dataTag},
          false,
          [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
            // Release the GPU data buffer from the context. The DMA has
            // completed by the time this callback fires, so the buffer is
            // safe to free. The context shell stays alive with the Request.
            auto ctx = std::static_pointer_cast<DataSendContext>(arg);
            auto dataHolder = std::move(ctx->data);

            if (auto self = weakData.lock()) {
              self->sendComplete(status, arg);
            }
            // dataHolder is destroyed here, releasing the GPU buffer if
            // sendComplete() already reset the server's dataPtr_.
          },
          dataCtx);
    } else {
      // Data pointer is null, so no more data will be coming.
      VLOG(3) << "@" << partitionKey_.taskId
              << " Finished transferring partition for task "
              << partitionKey_.toString();
      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
      setState(ServerState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  }
}

void UcxExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " sendComplete called after close, ignoring";
    return;
  }
  if (status == UCS_OK) {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    VELOX_CHECK_NOT_NULL(dataPtr_, "dataPtr_ is null");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = bytes_ / micros;

    VLOG(3) << "@" << partitionKey_.taskId << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
            << " MByte/s";

    this->sequenceNumber_++;
    dataPtr_.reset(); // release memory.
    VLOG(3) << "@" << partitionKey_.taskId
            << " Releasing dataPtr_ in sendComplete.";
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
