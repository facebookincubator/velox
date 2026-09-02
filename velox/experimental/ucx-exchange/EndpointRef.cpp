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
#include "velox/experimental/ucx-exchange/EndpointRef.h"

#include <array>

#include <glog/logging.h>

#include "velox/experimental/ucx-exchange/Communicator.h"

namespace facebook::velox::ucx_exchange {

/* static */
void EndpointRef::onClose(ucs_status_t status, std::shared_ptr<void> arg) {
  // NOTE: This callback is called from within the UCX progress thread.
  // We must NOT call any blocking operations, progress functions, or
  // iterate communicators_ here. All work is deferred to the main
  // Communicator loop via deferEndpointCleanup().

  std::shared_ptr<EndpointRef> ep = std::static_pointer_cast<EndpointRef>(arg);

  // Defer ALL cleanup to the main progress loop.
  // The main loop will:
  //   1. Close all communicators registered with this endpoint
  //   2. Clean up the endpoint itself (closeBlocking, etc.)
  auto c = Communicator::getInstance();
  c->deferEndpointCleanup(ep);
}

bool EndpointRef::addCommElem(std::shared_ptr<CommElement> commElem) {
  if (!commElem) {
    return false; // nothing to do, no commElem.
  }
  std::lock_guard<std::mutex> lock(commMutex_);
  cleanup();
  auto ret = communicators_.insert(commElem);
  return ret.second;
}

void EndpointRef::removeCommElem(std::shared_ptr<CommElement> commElem) {
  if (!commElem) {
    return;
  }
  std::lock_guard<std::mutex> lock(commMutex_);
  communicators_.erase(commElem);
}

void EndpointRef::closeAndDrainCommunicators() {
  // Swap communicators_ to a local copy under the lock, then iterate
  // the local copy without holding the lock. This prevents:
  // - Data races with concurrent addCommElem/removeCommElem
  // - Re-entrancy: close() may eventually trigger removeCommElem on
  //   this same EndpointRef, but communicators_ is already empty
  //   so removeCommElem will be a no-op.
  std::set<
      std::weak_ptr<CommElement>,
      std::owner_less<std::weak_ptr<CommElement>>>
      localCopy;
  {
    std::lock_guard<std::mutex> lock(commMutex_);
    localCopy.swap(communicators_);
  }

  // Now iterate the local copy -- no lock held, no contention.
  for (auto& weakElem : localCopy) {
    if (std::shared_ptr<CommElement> spt = weakElem.lock()) {
      spt->close();
    }
  }
  // localCopy is destroyed here, releasing all weak_ptrs.
}

std::optional<bool> EndpointRef::usesTransport(
    std::string_view transportName) const {
  const bool cacheCudaIpc = transportName == "cuda_ipc";
  std::unique_lock<std::mutex> cacheLock(transportMutex_, std::defer_lock);
  if (cacheCudaIpc) {
    cacheLock.lock();
    if (cudaIpcTransport_) {
      return cudaIpcTransport_;
    }
  }

  constexpr std::size_t kMaxTransports = 32;
  std::array<ucp_transport_entry_t, kMaxTransports> transports{};
  ucp_ep_attr_t attributes{};
  attributes.field_mask = UCP_EP_ATTR_FIELD_TRANSPORTS;
  attributes.transports.entries = transports.data();
  attributes.transports.num_entries = transports.size();
  attributes.transports.entry_size = sizeof(ucp_transport_entry_t);

  const auto status = ucp_ep_query(endpoint_->getHandle(), &attributes);
  if (status != UCS_OK) {
    VLOG(1) << "Unable to query UCP endpoint transports: "
            << ucs_status_string(status);
    return std::nullopt;
  }

  bool found = false;
  for (unsigned i = 0; i < attributes.transports.num_entries; ++i) {
    const auto* name = attributes.transports.entries[i].transport_name;
    if (name != nullptr && transportName == name) {
      found = true;
      break;
    }
  }
  if (cacheCudaIpc) {
    cudaIpcTransport_ = found;
  }
  VLOG(1) << "[UCX-ENDPOINT-TRANSPORT] transport=" << transportName
          << " present=" << found;
  return found;
}

bool EndpointRef::operator<(EndpointRef const& other) {
  if (endpoint_ == other.endpoint_) {
    return false; // covers the case where both are nullptr
  }
  if (endpoint_ == nullptr) {
    return true; // nullptr comes before anything else.
  }
  if (other.endpoint_ == nullptr) {
    return false;
  }
  return endpoint_->getHandle() < other.endpoint_->getHandle();
}

void EndpointRef::cleanup() {
  for (auto it = communicators_.begin(); it != communicators_.end();) {
    if (it->expired()) {
      it = communicators_.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace facebook::velox::ucx_exchange
