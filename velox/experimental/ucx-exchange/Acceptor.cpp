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
#include "velox/experimental/ucx-exchange/Acceptor.h"

#include <fmt/format.h>
#include <algorithm>
#include <cstring>

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {

namespace {

// Sends 'response' to the source of 'key' over 'epRef'. Does not wait for the
// send to complete.
void sendHandshakeResponse(
    const std::shared_ptr<EndpointRef>& epRef,
    const PartitionKey& key,
    std::shared_ptr<HandshakeResponse> response) {
  const uint64_t responseTag =
      getHandshakeResponseTag(fnv1a_32(key.toString()));
  VLOG(3) << "Sending HandshakeResponse to " << key.toString()
          << " status=" << static_cast<int>(response->status)
          << " isIntraNodeTransfer=" << response->isIntraNodeTransfer
          << " tag=" << std::hex << responseTag;

  [[maybe_unused]] auto sendRequest =
      epRef->endpoint_
          ->tagSendBuilder(
              response.get(), sizeof(*response), ucxx::Tag{responseTag})
          .callbackFunction([keyStr = key.toString()](
                                ucs_status_t status,
                                std::shared_ptr<void> /*arg*/) {
            if (status == UCS_OK) {
              VLOG(3) << "HandshakeResponse sent successfully to " << keyStr;
            } else {
              VLOG(0) << "Failed to send HandshakeResponse to " << keyStr
                      << ": " << ucs_status_string(status);
            }
          })
          .callbackData(response)
          .build();
}

} // namespace

/*static*/
void Acceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  try {
    handleHandshake(request, ep);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Cannot handle a UCX exchange handshake: " << e.what();
  }
}

/*static*/
void Acceptor::handleHandshake(
    const std::shared_ptr<ucxx::Request>& request,
    ucp_ep_h ep) {
  if (request == nullptr || !request->isCompleted()) {
    LOG(ERROR) << "Dropping a UCX exchange handshake without a completed "
                  "request";
    return;
  }
  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  if (buffer == nullptr) {
    LOG(ERROR) << "Dropping a UCX exchange handshake without a receive buffer";
    return;
  }
  const size_t size = buffer->getSize();
  // A peer of an older protocol version sends kLegacyHandshakeSize bytes, which
  // hold every field up to workerId. A shorter handshake cannot name its
  // channel, so there is no one to answer.
  if (size < kLegacyHandshakeSize) {
    LOG(ERROR) << "Dropping a UCX exchange handshake shorter than "
               << kLegacyHandshakeSize << " bytes: " << size;
    return;
  }

  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto epRef = communicator->findEndpointRefByHandle(ep);
  if (epRef == nullptr) {
    LOG(ERROR) << "Dropping a UCX exchange handshake from an unknown endpoint";
    return;
  }

  // Copy no more than was received. The copy is aligned, unlike the buffer.
  HandshakeMsg handshake{};
  std::memcpy(&handshake, buffer->data(), std::min(size, sizeof(handshake)));
  // A handshake before version 1 ends before its version field.
  const uint32_t peerVersion =
      size < sizeof(HandshakeMsg) ? 0 : handshake.protocolVersion;
  // The task ID need not end in a NUL.
  const PartitionKey key{
      std::string(
          handshake.taskId,
          strnlen(handshake.taskId, sizeof(handshake.taskId))),
      handshake.destination};

  if (peerVersion != kProtocolVersion) {
    auto response = std::make_shared<HandshakeResponse>();
    response->status = HandshakeStatus::kVersionMismatch;
    sendHandshakeResponse(epRef, key, std::move(response));
    std::string error = fmt::format(
        "UCX exchange peers must run the same protocol version: this worker "
        "runs {}, consumer worker {} runs {}",
        kProtocolVersion,
        epRef->getPeerIp(),
        peerVersion);
    LOG(ERROR) << error << ". Task: " << key.toString();
    // Task::setError() terminates the task synchronously, which must not
    // happen inside UCX progress.
    communicator->postToLoop([taskId = key.taskId, error = std::move(error)] {
      UcxOutputQueueManager::getInstanceRef()->failTask(taskId, error);
    });
    return;
  }

  // Determine if this is an intra-process transfer by comparing the source's
  // workerId with our Communicator's workerId. A match means both source and
  // server are in the same Communicator singleton (same process), so
  // IntraNodeTransferRegistry (in-process std::promise/future) can be used.
  //
  // Previous approach used IP comparison (getLocalIpAddresses), which fails
  // when multiple Docker containers share the same host IP address.
  bool isIntraNodeTransfer =
      cudf_velox::CudfConfig::getInstance().intraNodeExchange &&
      (handshake.workerId == communicator->getWorkerId());

  // Disable intra-node when the task is not yet initialized (a Pending
  // output queue from sinks connecting before initializeTask) or when the task
  // uses broadcast mode (all destination servers share the same
  // packed_columns — the intra-node source's destructive move would
  // corrupt it for other servers).
  if (isIntraNodeTransfer) {
    if (!UcxOutputQueueManager::getInstanceRef()->canUseIntraNode(key.taskId)) {
      VLOG(2) << "[ACCEPTOR] Disabling intra-node for task " << key.taskId
              << " (not initialized or broadcast)";
      isIntraNodeTransfer = false;
    }
  }

  auto exchangeServer =
      UcxExchangeServer::create(communicator, epRef, key, isIntraNodeTransfer);

  // Add this exchangeServer to the endpoint reference.
  epRef->addCommElem(exchangeServer);

  // Register exchangeServer with communicator.
  communicator->registerCommElement(exchangeServer);
  VLOG(2) << "[ACCEPTOR] new server: " << exchangeServer->toString()
          << " peerIp=" << epRef->getPeerIp()
          << " isIntraNodeTransfer=" << isIntraNodeTransfer;

  // Send HandshakeResponse back to the source to inform about intra-node
  // transfer. This allows the source to bypass UCXX for all subsequent data
  // transfers.
  auto response = std::make_shared<HandshakeResponse>();
  response->status = HandshakeStatus::kAccepted;
  response->isIntraNodeTransfer = exchangeServer->isIntraNodeTransfer();
  sendHandshakeResponse(epRef, key, std::move(response));
}

void Acceptor::registerEndpointRef(
    ucp_ep_h handle,
    std::shared_ptr<EndpointRef> endpointRef) {
  auto [it, inserted] = handleToEndpointRef_.try_emplace(handle, endpointRef);
  if (!inserted) {
    LOG(WARNING) << "Replacing the UCX endpoint registered for a reused "
                    "endpoint handle";
    it->second = std::move(endpointRef);
  }
}

void Acceptor::unregisterEndpointRef(
    const std::shared_ptr<EndpointRef>& endpointRef) {
  // By value, not by handle: UCXX no longer reports the handle once the
  // endpoint is closed.
  std::erase_if(handleToEndpointRef_, [&](const auto& entry) {
    return entry.second == endpointRef;
  });
}

std::shared_ptr<EndpointRef> Acceptor::findEndpointRef(ucp_ep_h handle) const {
  auto it = handleToEndpointRef_.find(handle);
  return it != handleToEndpointRef_.end() ? it->second : nullptr;
}
} // namespace facebook::velox::ucx_exchange
