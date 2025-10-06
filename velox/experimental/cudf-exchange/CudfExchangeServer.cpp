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
#include "velox/experimental/cudf-exchange/CudfExchangeServer.h"
#include <glog/logging.h>
#include "cuda_runtime.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/CudfExchangeProtocol.h"

namespace facebook::velox::cudf_exchange {

// This constructor is private
CudfExchangeServer::CudfExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key)
    : CommElement(communicator, endpointRef),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queueMgr_(CudfOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);
}

// static
std::shared_ptr<CudfExchangeServer> CudfExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key) {
  auto ptr = std::shared_ptr<CudfExchangeServer>(
      new CudfExchangeServer(communicator, endpointRef, key));
  return ptr;
}

void CudfExchangeServer::process() {
  switch (state_) {
    case ServerState::Created:
      setState(ServerState::ReadyToTransfer);
      communicator_->addToWorkQueue(getSelfPtr());
      break;
    case ServerState::ReadyToTransfer:
      // Fetch the data from CudfQueueManager and store it in the dataPtr_;
      setState(ServerState::WaitingForDataFromQueue);
      // Register the callback with the destination queue to get data.
      // If the queue doesn't exist yet, getData will create an empty
      // queue and the callback will be triggered once the corresponding
      // source task has initialized the queue and added data to it.
      queueMgr_->getData(
          partitionKey_.taskId,
          partitionKey_.destination,
          [this](
              std::unique_ptr<cudf::packed_columns> data,
              std::vector<int64_t> remainingBytes) {
            // This upcall may be called from another thread than the
            // communicator thread. It is called
            // when data on the queue becomes available.
            VLOG(3) << "Found data for client: "
                    << this->partitionKey_.toString();
            std::lock_guard<std::recursive_mutex> lock(this->dataMutex_);
            VELOX_CHECK(
                dataPtr_ == nullptr, "Data pointer exists: Illegal state!");
            VLOG(3)
                << "Assigning dataPtr_ in getData callback. dataptr_ != nullptr: "
                << (data != nullptr);
            this->dataPtr_ = std::move(data);
            this->setState(ServerState::DataReady);
            this->communicator_->addToWorkQueue(getSelfPtr());
          });
      this->communicator_->addToWorkQueue(getSelfPtr());
      break;
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
    case ServerState::Done:
      // unregister.
      communicator_->unregister(getSelfPtr());
      close();
      break;
  };
}

void CudfExchangeServer::close() {
  VLOG(3) << "Close CudfExchangeServer to remote " << partitionKey_.toString()
          << ".";
  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
  }
  communicator_->unregister(getSelfPtr());
}

std::string CudfExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<CudfExchangeServer> CudfExchangeServer::getSelfPtr() {
  return shared_from_this();
}

void CudfExchangeServer::sendData() {
  // Create the MetaDataRecord.
  std::shared_ptr<MetadataMsg> metadataMsg = std::make_shared<MetadataMsg>();

  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    if (dataPtr_) {
      metadataMsg->cudfMetadata = std::move(dataPtr_->metadata);
      metadataMsg->dataSizeBytes = dataPtr_->gpu_data->size();
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = false;
    } else {
      VLOG(3) << "Final exchange for " << partitionKey_.toString();
      metadataMsg->cudfMetadata = nullptr;
      metadataMsg->dataSizeBytes = 0;
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = true;
    }
  }

  auto [serializedMetadata, serMetaSize] = metadataMsg->serialize();

  // send metadata, no callback needed.
  uint64_t metadataTag =
      getMetadataTag(this->partitionKeyHash_, this->sequenceNumber_);
  metaRequest_ = endpointRef_->endpoint_->tagSend(
      serializedMetadata.get(),
      serMetaSize,
      ucxx::Tag{metadataTag},
      false,
      [tid = partitionKey_.toString(), metadataTag](
          ucs_status_t status, std::shared_ptr<void> arg) {
        VLOG(3) << "metadata successfully sent to " << tid
                << " with tag: " << std::hex << metadataTag;
      },
      serializedMetadata);

  metaRequest_->checkError();
  auto s = metaRequest_->getStatus();
  if (s != UCS_INPROGRESS && s != UCS_OK) {
    VLOG(0) << "Error in sendData, send metadata " << ucs_status_string(s)
            << " failed for task: " << partitionKey_.toString();
    setState(ServerState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  }

  // send the data chunk (if any)
  {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ = dataPtr_->gpu_data->size();
      VLOG(3) << "Sending rmm::buffer: " << std::hex << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task "
              << partitionKey_.toString() << ":" << this->sequenceNumber_
              << std::dec << " of size " << bytes_;

      setState(ServerState::WaitingForSendComplete);
      uint64_t dataTag =
          getDataTag(this->partitionKeyHash_, this->sequenceNumber_);
      dataRequest_ = endpointRef_->endpoint_->tagSend(
          dataPtr_->gpu_data->data(),
          dataPtr_->gpu_data->size(),
          ucxx::Tag{dataTag},
          false,
          std::bind(
              &CudfExchangeServer::sendComplete,
              this,
              std::placeholders::_1,
              std::placeholders::_2));

      dataRequest_->checkError();
      s = dataRequest_->getStatus();
      if (s != UCS_INPROGRESS && s != UCS_OK) {
        VLOG(0) << "Error in sendData, send rmm::buffer "
                << ucs_status_string(s)
                << " failed for task: " << partitionKey_.toString();
        setState(ServerState::Done);
        communicator_->addToWorkQueue(getSelfPtr());
      }
    } else {
      // Data pointer is null, so no more data will be coming.
      VLOG(3) << "Finished transferring partition for task "
              << partitionKey_.toString();
      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
      setState(ServerState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  }
}

void CudfExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (status == UCS_OK) {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    VELOX_CHECK(dataPtr_ != nullptr, "dataPtr_ is null");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = bytes_ / micros;

    VLOG(3) << "duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "throughput: " << throughput << " MByte/s";

    this->sequenceNumber_++;
    dataPtr_.reset(); // release memory.
    VLOG(3) << "Releasing dataPtr_ in sendComplete.";
    setState(ServerState::ReadyToTransfer);
  } else {
    VLOG(3) << "Error in sendComplete, send complete "
            << ucs_status_string(status);
    setState(ServerState::Done);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

} // namespace facebook::velox::cudf_exchange
