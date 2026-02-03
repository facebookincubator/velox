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
#include "velox/experimental/cudf-exchange/Acceptor.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/CudfExchangeProtocol.h"
#include "velox/experimental/cudf-exchange/CudfExchangeServer.h"
#include "velox/experimental/cudf-exchange/EndpointRef.h"

namespace facebook::velox::cudf_exchange {

/*static*/
void Acceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  VELOX_CHECK(request, "AMCallback called with nullptr request!");
  VELOX_CHECK(
      request->isCompleted(), "AMCallback called with incomplete request!");
  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  HandshakeMsg* handshakePtr = reinterpret_cast<HandshakeMsg*>(buffer->data());
  VELOX_CHECK(
      handshakePtr != nullptr, "AMCallback: could not cast to HandhsakeMsg.");

  // Create a exchangeServer based on the information received in the initial
  // handshake.
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();

  auto it = communicator->acceptor_.handleToEndpointRef_.find(ep);
  VELOX_CHECK(
      it != communicator->acceptor_.handleToEndpointRef_.end(),
      "Could not find endpoint reference");
  std::shared_ptr<EndpointRef> epRef = it->second;

  const PartitionKey key = {handshakePtr->taskId, handshakePtr->destination};

  // Extract source's listener address for same-node detection.
  std::string sourceListenerIp(handshakePtr->sourceListenerIp);
  uint16_t sourceListenerPort = handshakePtr->sourceListenerPort;

  auto exchangeServer = CudfExchangeServer::create(
      communicator, epRef, key, sourceListenerIp, sourceListenerPort);

  // Add this exchangeServer to the endpoint reference.
  epRef->addCommElem(exchangeServer);

  // Register exchangeServer with communicator.
  communicator->registerCommElement(exchangeServer);
  VLOG(3) << "Registered new exchange server task: "
          << exchangeServer->toString()
          << " (sourceListener: " << sourceListenerIp << ":"
          << sourceListenerPort << ")";

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
      sizeof(HandshakeResponse),
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
} // namespace facebook::velox::cudf_exchange
