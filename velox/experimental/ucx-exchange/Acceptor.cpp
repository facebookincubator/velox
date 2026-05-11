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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {

/*static*/
void Acceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  VELOX_CHECK_NOT_NULL(request, "AMCallback called with nullptr request!");
  VELOX_CHECK(
      request->isCompleted(), "AMCallback called with incomplete request!");
  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  VELOX_CHECK_NOT_NULL(buffer, "AMCallback: failed to get receive buffer.");
  // Validate buffer size BEFORE casting to prevent reading past buffer bounds.
  VELOX_CHECK_GE(
      buffer->getSize(),
      sizeof(HandshakeMsg),
      "AMCallback: received buffer size ({}) is smaller than HandshakeMsg ({}). "
      "Possible protocol mismatch or truncated message.",
      buffer->getSize(),
      sizeof(HandshakeMsg));
  HandshakeMsg* handshakePtr = reinterpret_cast<HandshakeMsg*>(buffer->data());

  // Create a exchangeServer based on the information received in the initial
  // handshake.
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();

  auto epRef = communicator->findEndpointRefByHandle(ep);
  VELOX_CHECK_NOT_NULL(epRef, "Could not find endpoint reference");

  const PartitionKey key = {handshakePtr->taskId, handshakePtr->destination};

  // Determine if this is an intra-process transfer by comparing the source's
  // workerId with our Communicator's workerId. A match means both source and
  // server are in the same Communicator singleton (same process), so
  // IntraNodeTransferRegistry (in-process std::promise/future) can be used.
  //
  // Previous approach used IP comparison (getLocalIpAddresses), which fails
  // when multiple Docker containers share the same host IP address.
  bool isIntraNodeTransfer =
      cudf_velox::CudfConfig::getInstance().intraNodeExchange &&
      (handshakePtr->workerId == communicator->getWorkerId());

  // Disable intra-node when the task is not yet initialized (placeholder
  // queue from sinks connecting before initializeTask) or when the task
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

  std::string peerIp = epRef->getPeerIp();

  auto exchangeServer =
      UcxExchangeServer::create(communicator, epRef, key, isIntraNodeTransfer);

  // Add this exchangeServer to the endpoint reference.
  epRef->addCommElem(exchangeServer);

  // Register exchangeServer with communicator.
  communicator->registerCommElement(exchangeServer);
  VLOG(2) << "[ACCEPTOR] new server: " << exchangeServer->toString()
          << " peerIp=" << peerIp
          << " isIntraNodeTransfer=" << isIntraNodeTransfer;

  // Send HandshakeResponse back to the source to inform about intra-node
  // transfer. This allows the source to bypass UCXX for all subsequent data
  // transfers.
  auto response = std::make_shared<HandshakeResponse>();
  response->isIntraNodeTransfer = exchangeServer->isIntraNodeTransfer();

  uint32_t keyHash = fnv1a_32(key.toString());
  uint64_t responseTag = getHandshakeResponseTag(keyHash);

  VLOG(3) << "Sending HandshakeResponse to " << key.toString()
          << " isIntraNodeTransfer=" << response->isIntraNodeTransfer
          << " tag=" << std::hex << responseTag;

  // Fire-and-forget: we don't need to track this request completion
  epRef->endpoint_->tagSend(
      response.get(),
      sizeof(*response),
      ucxx::Tag{responseTag},
      false,
      [response, keyStr = key.toString()](
          ucs_status_t status, std::shared_ptr<void> arg) {
        if (status == UCS_OK) {
          VLOG(3) << "HandshakeResponse sent successfully to " << keyStr;
        } else {
          VLOG(0) << "Failed to send HandshakeResponse to " << keyStr << ": "
                  << ucs_status_string(status);
        }
      },
      response);
}

// Add endpoint reference to ucp_cp -> epRef map.
void Acceptor::registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef) {
  auto epHandle = endpointRef->endpoint_->getHandle();
  auto res = handleToEndpointRef_.insert(std::pair{epHandle, endpointRef});
  VELOX_CHECK(res.second, "Endpoint handle already exists!");
}
} // namespace facebook::velox::ucx_exchange
