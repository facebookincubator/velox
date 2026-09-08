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

#include <atomic>

namespace facebook::velox::ucx_exchange {

namespace {

std::atomic<uint64_t> gNumIntraNodeHandshakes{0};
std::atomic<uint64_t> gNumRemoteHandshakes{0};

void completeHandshake(
    const std::shared_ptr<Communicator>& communicator,
    const std::shared_ptr<EndpointRef>& epRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer) {
  // Eligibility may be resolved after the active-message callback returns.
  // Do not issue UCXX operations if the endpoint closed in the meantime.
  if (!epRef->endpoint_ || !epRef->endpoint_->isAlive()) {
    VLOG(2) << "[ACCEPTOR] Dropping resolved handshake for " << key.toString()
            << " because its endpoint is closed";
    return;
  }

  const std::string peerIp = epRef->getPeerIp();
  auto exchangeServer =
      UcxExchangeServer::create(communicator, epRef, key, isIntraNodeTransfer);

  if (isIntraNodeTransfer) {
    gNumIntraNodeHandshakes.fetch_add(1, std::memory_order_relaxed);
  } else {
    gNumRemoteHandshakes.fetch_add(1, std::memory_order_relaxed);
  }

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

  const uint32_t keyHash = fnv1a_32(key.toString());
  const uint64_t responseTag = getHandshakeResponseTag(keyHash);

  VLOG(3) << "Sending HandshakeResponse to " << key.toString()
          << " isIntraNodeTransfer=" << response->isIntraNodeTransfer
          << " tag=" << std::hex << responseTag;

  // Fire-and-forget: we don't need to track this request completion.
  epRef->endpoint_->tagSend(
      response.get(),
      sizeof(*response),
      ucxx::Tag{responseTag},
      false,
      [response, keyStr = key.toString()](
          ucs_status_t status, std::shared_ptr<void> /*arg*/) {
        if (status == UCS_OK) {
          VLOG(3) << "HandshakeResponse sent successfully to " << keyStr;
        } else {
          VLOG(0) << "Failed to send HandshakeResponse to " << keyStr << ": "
                  << ucs_status_string(status);
        }
      },
      response);
}

} // namespace

uint64_t Acceptor::numIntraNodeHandshakes() {
  return gNumIntraNodeHandshakes.load(std::memory_order_relaxed);
}

uint64_t Acceptor::numRemoteHandshakes() {
  return gNumRemoteHandshakes.load(std::memory_order_relaxed);
}

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

  if (!isIntraNodeTransfer) {
    completeHandshake(communicator, epRef, key, false);
    return;
  }

  // The source is in this process, but its producer task may not have reached
  // initializeTask() yet. Do not permanently route it through remote UCX based
  // on a placeholder queue. Wait until the real output kind is known, then
  // finish the UCXX handshake on the Communicator thread. Broadcast remains on
  // UCX because its packed_columns object is shared by multiple destinations.
  std::weak_ptr<Communicator> weakCommunicator = communicator;
  std::weak_ptr<EndpointRef> weakEndpoint = epRef;
  UcxOutputQueueManager::getInstanceRef()->notifyOnIntraNodeEligibility(
      key.taskId, [weakCommunicator, weakEndpoint, key](bool canUseIntraNode) {
        auto communicator = weakCommunicator.lock();
        if (!communicator) {
          return;
        }
        VLOG(2) << "[ACCEPTOR] Resolved deferred handshake for "
                << key.toString() << " isIntraNodeTransfer=" << canUseIntraNode;
        communicator->deferAction(
            [weakCommunicator, weakEndpoint, key, canUseIntraNode]() {
              auto communicator = weakCommunicator.lock();
              auto epRef = weakEndpoint.lock();
              if (!communicator || !epRef) {
                return;
              }
              completeHandshake(communicator, epRef, key, canUseIntraNode);
            });
      });
}

// Add endpoint reference to ucp_cp -> epRef map.
void Acceptor::registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef) {
  auto epHandle = endpointRef->endpoint_->getHandle();
  auto res = handleToEndpointRef_.insert(std::pair{epHandle, endpointRef});
  VELOX_CHECK(res.second, "Endpoint handle already exists!");
}
} // namespace facebook::velox::ucx_exchange
