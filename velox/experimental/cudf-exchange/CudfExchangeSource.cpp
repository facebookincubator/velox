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
#include "velox/experimental/cudf-exchange/CudfExchangeSource.h"
#include "velox/experimental/cudf/exec/Utilities.h"

using namespace facebook::velox::exec;
namespace facebook::velox::cudf_exchange {

// This constructor is private.
CudfExchangeSource::CudfExchangeSource(
    const std::shared_ptr<Communicator> communicator,
    const std::string& host,
    uint16_t port,
    const PartitionKey& partitionKey,
    const std::shared_ptr<CudfExchangeQueue> queue)
    : CommElement(communicator),
      host_(host),
      port_(port),
      partitionKey_(partitionKey),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queue_(std::move(queue)) {
  setState(ReceiverState::Created);
}

/*static*/
std::shared_ptr<CudfExchangeSource> CudfExchangeSource::create(
    const std::string& url,
    const std::shared_ptr<CudfExchangeQueue>& queue) {
  folly::Uri uri(url);
  // Note that there is no distinct schema for the UCXX exchange.
  // The approach is to ignore the schema and not check for HTTP or HTTPS.
  // FIXME: Can't use the HTTP port as this conflicts with Prestissimo!
  // For the time being, there's an ugly hack that just increases the port by 3.
  VLOG(3) << " Creating CudfExchangeSource " << url;
  const std::string host = uri.host();
  int port = uri.port() + 3;
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto key = extractTaskAndDestinationId(uri.path());
  auto source = std::shared_ptr<CudfExchangeSource>(
      new CudfExchangeSource(communicator, host, port, key, queue));
  // register the exchange source with the communicator. This makes sure that
  // "progress" is called.
  communicator->registerCommElement(source);
  return source;
}

bool CudfExchangeSource::shouldRequestLocked() {
  VLOG(3) << "*** should request locked called atEnd_ :" << atEnd_;
  if (atEnd_) {
    return false;
  }

  if (!requestPending_) {
    VELOX_CHECK(!promise_.valid() || promise_.isFulfilled());
    VLOG(3) << "requestPending_ set to true since promise valid: "
            << promise_.valid()
            << " and promise fulfilled: " << promise_.isFulfilled();
    requestPending_ = true;
    return true;
  }

  // We are still processing previous request.
  return false;
}

folly::SemiFuture<ExchangeSource::Response> CudfExchangeSource::request(
    std::chrono::microseconds maxWait) {
  // Highlayer requested data transfer
  VLOG(3) << "+ recv request for data";

  // Before calling 'request', the caller should have called
  // 'shouldRequestLocked' and received 'true' response. Hence, we expect
  // requestPending_ == true, atEnd_ == false.
  VELOX_CHECK(requestPending_);

  // if the receiver is not yet in state HandshakeComplete, we need to remember
  // this.
  requestIssuedBeforeHandshakeComplete_ = !setStateIf(
      ReceiverState::HandshakeComplete, ReceiverState::ReadyToReceive);
  communicator_->addToWorkQueue(getSelfPtr());

  promise_ =
      VeloxPromise<ExchangeSource::Response>("CudfExchangeSource::request");
  VLOG(3) << "Created new promise_";
  auto future = promise_.getSemiFuture();

  VLOG(3) << "- recv request for data";
  return future;
}

void CudfExchangeSource::process() {
  switch (state_) {
    case ReceiverState::Created: {
      // Get the endpoint.
      HostPort hp{host_, port_};
      std::shared_ptr<CudfExchangeSource> selfPtr = getSelfPtr();
      auto epRef = communicator_->assocEndpointRef(selfPtr, hp);
      if (epRef) {
        setEndpoint(epRef);
        setState(ReceiverState::WaitingForHandshakeComplete);
        sendHandshake();
      } else {
        // connection failed.
        VLOG(0) << "Failed to connect to " << host_ << ":"
                << std::to_string(port_);
        setState(ReceiverState::Done);
      }
      communicator_->addToWorkQueue(getSelfPtr());
    } break;
    case ReceiverState::WaitingForHandshakeComplete:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::HandshakeComplete:
      if (requestIssuedBeforeHandshakeComplete_) {
        // Need to transition to ReadyToReceive since request has been called.
        setState(ReceiverState::ReadyToReceive);
        requestIssuedBeforeHandshakeComplete_ = false;
      }
      break;
    case ReceiverState::ReadyToReceive:
      // change state before calling getMetadata since immediate upcalls in
      // getMetadata will also change state.
      setState(ReceiverState::WaitingForMetadata);
      getMetadata();
      break;
    case ReceiverState::WaitingForMetadata:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
      break;
    case ReceiverState::WaitingForData:
      // Waiting for data is handled by an upcall from UCXX. Nothing to do.
      break;
    case ReceiverState::Done:
      close();
      // unregister.
      communicator_->unregister(getSelfPtr());
      break;
  }
}

void CudfExchangeSource::close() {
  VLOG(1) << "CudfExchangeSource::close called.";
  VLOG(1) << fmt::format("closing task: {}", partitionKey_.toString());
  closed_.store(true);
  checkSetRequestPromise(); // fulfill any outstanding open request promise.
  VLOG(3) << "Close receiver to remote " << partitionKey_.toString() << ".";
  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
  }
  communicator_->unregister(getSelfPtr());
}

folly::F14FastMap<std::string, int64_t> CudfExchangeSource::stats() const {
  VELOX_UNREACHABLE();
}

folly::F14FastMap<std::string, RuntimeMetric> CudfExchangeSource::metrics()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> map;

  // these metrics will be aggregated over all exchange sources of the same
  // exchange client.
  map["cudfExchangeSource.numPackedColumns"] = metrics_.numPackedColumns_;
  map["cudfExchangeSource.totalBytes"] = metrics_.totalBytes_;
  map["cudfExchangeSource.rttPerRequest"] = metrics_.rttPerRequest_;
  return map;
}

// private methods ---
PartitionKey CudfExchangeSource::extractTaskAndDestinationId(
    const std::string& path) {
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
    std::string msg = "Illegal destination in task URL: " + path;
    VELOX_UNSUPPORTED(msg);
  }

  return PartitionKey{components[2].str(), destinationId};
}

std::shared_ptr<CudfExchangeSource> CudfExchangeSource::getSelfPtr() {
  return shared_from_this();
}

void CudfExchangeSource::enqueueAndFulfillPromise(
    std::unique_ptr<cudf::packed_columns> columns,
    MetadataMsg& metadata) {
  std::vector<velox::ContinuePromise> queuePromises;
  velox::VeloxPromise<ExchangeSource::Response> requestPromise;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    queue_->enqueueLocked(std::move(columns), queuePromises);
    // enqueueLocked returns one or more promises of waiting CudfExchanges that
    // can get fulfilled by the reading from the queue.
    requestPromise =
        std::move(promise_); // move promise_ while holding the lock.
    requestPending_ = false;
  }
  // wake up consumers of the CudfExchangeQueue
  for (auto& promise : queuePromises) {
    promise.setValue();
  }

  // Tell the caller of request that the request has been fulfilled.
  if (requestPromise.valid() && !requestPromise.isFulfilled()) {
    VLOG(3) << "Fulfilling promise_";
    requestPromise.setValue(ExchangeSource::Response{
        metadata.dataSizeBytes, // 0
        metadata.atEnd, // True
        metadata.remainingBytes // Empty vector
    });
  } else {
    // The source must have been closed.
    VELOX_CHECK(closed_.load());
  }
}

bool CudfExchangeSource::checkSetRequestPromise() {
  velox::VeloxPromise<ExchangeSource::Response> promise;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    VLOG(3) << "On close: finalizing promise_";
    promise = std::move(promise_);
  }
  if (promise.valid() && !promise.isFulfilled()) {
    promise.setValue(ExchangeSource::Response{0, false, {}});
    return true;
  }

  return false;
}

void CudfExchangeSource::setEndpoint(std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

void CudfExchangeSource::sendHandshake() {
  std::shared_ptr<HandshakeMsg> handshakeReq = std::make_shared<HandshakeMsg>();
  handshakeReq->destination = partitionKey_.destination;
  strncpy(
      handshakeReq->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(handshakeReq->taskId));

  VLOG(3) << toString() << " Sending handshake with initial value: "
          << partitionKey_.toString() << " to server";

  // Create the handshake which will register client's existence with the server
  ucxx::AmReceiverCallbackInfo info(
      communicator_->kAmCallbackOwner, communicator_->kAmCallbackId);
  request_ = endpointRef_->endpoint_->amSend(
      handshakeReq.get(),
      sizeof(HandshakeMsg),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      std::bind(
          &CudfExchangeSource::onHandshake,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
      handshakeReq);

  request_->checkError();
  auto s = request_->getStatus();
  if (s != UCS_INPROGRESS && s != UCS_OK) {
    VLOG(0) << "Error in sendHandshake " << ucs_status_string(s)
            << " failed for task: " << partitionKey_.toString();
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  }
}

void CudfExchangeSource::onHandshake(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send handshake to host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << "+ onHandshake " << ucs_status_string(status);
    setState(ReceiverState::ReadyToReceive);
  }
  // more work to do
  communicator_->addToWorkQueue(getSelfPtr());
}

void CudfExchangeSource::getMetadata() {
  uint32_t sizeMetadata = 4096; // shouldn't be a fixed size.
  auto metadataReq = std::make_shared<std::vector<uint8_t>>(sizeMetadata);
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber_);

  VLOG(3) << toString()
          << " waiting for metadata for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << metadataTag << std::dec;

  request_ = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      sizeMetadata,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      std::bind(
          &CudfExchangeSource::onMetadata,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
      metadataReq);

  request_->checkError();
  auto s = request_->getStatus();
  if (s != UCS_INPROGRESS && s != UCS_OK) {
    VLOG(3) << "Error in getMetadata, receive metadata " << ucs_status_string(s)
            << " failed for task: " << partitionKey_.toString();
  }
}

void CudfExchangeSource::onMetadata(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  VLOG(3) << "+ onMetadata " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    setState(ReceiverState::Done);
    communicator_->addToWorkQueue(getSelfPtr());
  } else {
    VELOX_CHECK(arg != nullptr, "Didn't get metadata");

    // arg contains the actual serialized metadata, deserialize the metadata
    std::shared_ptr<std::vector<uint8_t>> metadataMsg =
        std::static_pointer_cast<std::vector<uint8_t>>(arg);

    auto ptr = std::make_shared<DataAndMetadata>();

    ptr->metadata =
        std::move(MetadataMsg::deserializeMetadataMsg(metadataMsg->data()));
    // auto m = std::unique_ptr<MetadataMsg>(new MetadataMsg(msg));

    VLOG(3) << "Datasize bytes == " << ptr->metadata.dataSizeBytes;

    if (ptr->metadata.atEnd) {
      // It seems that all data has been transferred
      atEnd_ = true;
      // enqueue a nullpointer to mark the end for this source.
      VLOG(3) << "There is no more data to transfer";
      enqueueAndFulfillPromise(nullptr, ptr->metadata);
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      // jump out of this function.
      return;
    }

    // Now allocate memory for the CudaVector
    // Get a stream from the global stream pool
    auto stream =
        facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
    try {
      ptr->dataBuf = std::make_unique<rmm::device_buffer>(
          ptr->metadata.dataSizeBytes, stream);
    } catch (const rmm::bad_alloc& e) {
      VLOG(0) << "!!! RMM bad_alloc: " << e.what();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    } catch (const rmm::cuda_error& e) {
      VLOG(0) << "!!! General allocation exception: " << e.what();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
      return;
    }

    // sync after allocating.
    stream.synchronize();

    VLOG(3) << "Allocated " << ptr->metadata.dataSizeBytes
            << " bytes of device memory";

    // Initiate the transfer of the actual data from GPU-2-GPU
    uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
    VLOG(3) << toString() << " waiting for data for chunk: " << sequenceNumber_
            << " using tag: " << std::hex << dataTag << std::dec;

    setState(ReceiverState::WaitingForData);
    request_ = endpointRef_->endpoint_->tagRecv(
        ptr->dataBuf->data(),
        ptr->metadata.dataSizeBytes,
        ucxx::Tag{dataTag},
        ucxx::TagMaskFull,
        false,
        std::bind(
            &CudfExchangeSource::onData,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        ptr // DataAndMetadata
    );

    request_->checkError();
    auto s = request_->getStatus();
    if (s != UCS_INPROGRESS && s != UCS_OK) {
      VLOG(0) << "Error in onMetadata, receive data " << ucs_status_string(s)
              << " failed for task: " << partitionKey_.toString();
      setState(ReceiverState::Done);
      communicator_->addToWorkQueue(getSelfPtr());
    }
  }
}

void CudfExchangeSource::onData(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  VLOG(3) << "+ onDdata " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive data from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << toString() << errorMsg;
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << "got chunk: " << sequenceNumber_;

    this->sequenceNumber_++;

    std::shared_ptr<DataAndMetadata> ptr =
        std::static_pointer_cast<DataAndMetadata>(arg);

    metrics_.numPackedColumns_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    std::unique_ptr<cudf::packed_columns> columns =
        std::make_unique<cudf::packed_columns>(
            std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));
    enqueueAndFulfillPromise(std::move(columns), ptr->metadata);
    setState(ReceiverState::HandshakeComplete);
  }
  communicator_->addToWorkQueue(getSelfPtr());
}

bool CudfExchangeSource::setStateIf(
    CudfExchangeSource::ReceiverState expected,
    CudfExchangeSource::ReceiverState desired) {
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
  return true;
}

} // namespace facebook::velox::cudf_exchange
