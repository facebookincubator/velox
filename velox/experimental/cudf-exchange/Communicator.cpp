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
#include "velox/experimental/cudf-exchange/Communicator.h"
#include <cuda_runtime.h>
#include <ucxx/api.h>
#include <ucxx/utils/ucx.h>
#include <iostream>
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/cudf-exchange/CommElement.h"
#include "velox/experimental/cudf-exchange/EndpointRef.h"

namespace facebook::velox::cudf_exchange {

// static
std::once_flag Communicator::onceFlag;
std::shared_ptr<Communicator> Communicator::instancePtr_ = nullptr;

/* static */
std::shared_ptr<Communicator> Communicator::initAndGet(uint16_t port) {
  std::call_once(onceFlag, [&] {
    instancePtr_ = std::shared_ptr<Communicator>(new Communicator());
    instancePtr_->port_ = port;
  });
  VELOX_CHECK(
      instancePtr_->port_ == port,
      "Cannot initialize communicator again with different port");
  return instancePtr_;
}

/* static */
std::shared_ptr<Communicator> Communicator::getInstance() {
  if (!instancePtr_) {
    throw std::logic_error(
        "Communicator not initialized. Call init(port) first.");
  }
  return instancePtr_;
}

/* static */ void Communicator::cStyleListenerCallback(
    ucp_conn_request_h conn_request,
    void* arg) {
  // cast the argument back to our instance variable:
  Communicator* instance = static_cast<Communicator*>(arg);
  instance->listenerCallback(conn_request);
}

/// @brief Run doesn't return until stop() is called.
/// All operations of the communicator will be carried out in the thread
/// that calls run.
void Communicator::run() {
  // Force CUDA context creation
  cudaFree(0);

  // create the UCXX context, worker, listener-context etc.
  context_ = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  worker_ = context_->createWorker();

  // Communicator is using blocking progress mode.
  worker_->initBlockingProgressMode();

  listener_ = worker_->createListener(
      port_, Communicator::cStyleListenerCallback, this);

  // Setup the active message callback that handles the
  // initial handshake and creates the senders.
  ucxx::AmReceiverCallbackInfo info(kAmCallbackOwner, kAmCallbackId);
  worker_->registerAmReceiverCallback(info, &Acceptor::cStyleAMCallback);

  VLOG(3) << "Communicator running.";

  running_.store(true);
  while (running_) {
    // wait for progress.
    worker_->progressWorkerEvent(0);

    // process the work queue. Make sure that communication is progressed
    // after each call to a comms element, otherwise we will deadlock.
    while (!workQueue_.empty()) {
      auto comms = workQueue_.pop();
      comms->process();
      worker_->progressWorkerEvent(0);
    }
  }
}

/// @brief Stops the communicator, called from an outside thread.
void Communicator::stop() {
  VLOG(3) << "In Communicator::stop "
          << " elements_.size(): " << elements_.size()
          << " endpoints_.size(): " << endpoints_.size()
          << " workQueue_._size(): " << workQueue_.size();
  worker_->stopProgressThread();
  running_.store(false);
}

void Communicator::registerCommElement(std::shared_ptr<CommElement> comms) {
  std::lock_guard<std::mutex> lock(elemMutex_);
  auto ret = elements_.insert(comms);
  VELOX_CHECK(ret.second, "CommElement already registered!");
  // Also put the comms element into the work queue.
  workQueue_.push(comms);
}

void Communicator::addToWorkQueue(std::shared_ptr<CommElement> comms) {
  workQueue_.push(comms);
}

void Communicator::unregister(std::shared_ptr<CommElement> comms) {
  std::lock_guard<std::mutex> lock(elemMutex_);
  workQueue_.erase(comms);
  elements_.erase(comms);
}

std::shared_ptr<EndpointRef> Communicator::assocEndpointRef(
    std::shared_ptr<CommElement> comms,
    HostPort hostPort) {
  auto it = endpoints_.find(hostPort);
  if (it != endpoints_.end()) {
    std::shared_ptr<EndpointRef> ep = it->second;
    ep->addCommElem(comms);
    return ep;
  }
  // endpoint doesn't exist. Need to connect. Enable error handling.
  auto ep = worker_->createEndpointFromHostname(
      hostPort.hostname, hostPort.port, true);
  std::shared_ptr<EndpointRef> epRef = nullptr;
  if (ep != nullptr) {
    epRef = std::make_shared<EndpointRef>(ep);
    epRef->addCommElem(comms);
    // register on close callback.
    ep->setCloseCallback(EndpointRef::onClose, epRef);
    endpoints_.insert(std::pair{hostPort, epRef});
  }
  return epRef;
}

void Communicator::removeEndpointRef(std::shared_ptr<EndpointRef> ep) {
  VLOG(3) << "+ Communicator::removeEndpointRef";

  if (ep->endpoint_ && ep->endpoint_->isAlive()) {
    VLOG(3) << "In Communicator::removeEndpointRef call closeBlocking";
    ep->endpoint_->closeBlocking();
  }
  for (auto it = endpoints_.begin(); it != endpoints_.end();) {
    if (it->second == ep) {
      it = endpoints_.erase(it); // erase returns the next iterator
    } else {
      ++it;
    }
  }
  VLOG(3) << "- Communicator::removeEndpointRef";
}

/// @brief The callback method that is invoked when a client connects.
void Communicator::listenerCallback(ucp_conn_request_h conn_request) {
  char ip_str[INET6_ADDRSTRLEN];
  char port_str[INET6_ADDRSTRLEN];
  ucp_conn_request_attr_t attr{};

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucxx::utils::ucsErrorThrow(ucp_conn_request_query(conn_request, &attr));
  ucxx::utils::sockaddr_get_ip_port_str(
      &attr.client_address, ip_str, port_str, INET6_ADDRSTRLEN);
  VLOG(3)
      << "Communicator received a connection request from client at address "
      << ip_str << ":" << port_str;

  // incoming endpoints are not shared. Outgoing endpoints to the same node are
  // shared. This guarantees that between any two nodes, there will be at most 2
  // endpoints, one per direction. For compatibility reasons, both incoming and
  // outgoing endpoints are represented using the EndpointRef.
  auto endpoint = listener_->createEndpointFromConnRequest(conn_request, true);
  auto epRef = std::make_shared<EndpointRef>(endpoint);
  endpoint->setCloseCallback(EndpointRef::onClose, epRef);
  // Add this endpoint reference to the list of endpoints.
  unsigned long val = std::strtoul(port_str, nullptr, 10);
  VELOX_CHECK(
      val <= static_cast<unsigned long>(std::numeric_limits<uint16_t>::max()),
      "Port out of range for uint16_t!");

  uint16_t port = static_cast<uint16_t>(val);
  HostPort hp(ip_str, port);
  auto res = endpoints_.insert(std::pair{hp, epRef});
  VELOX_CHECK(res.second, "Endpoint already exists!");
  acceptor_.registerEndpointRef(epRef);
}
} // namespace facebook::velox::cudf_exchange
