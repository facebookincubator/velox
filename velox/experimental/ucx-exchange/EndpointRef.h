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
#pragma once

#include <ucxx/api.h>
#include <memory>
#include <mutex>
#include <set>

#include "velox/experimental/ucx-exchange/CommElement.h"

namespace facebook::velox::ucx_exchange {

/// @brief The endpoint reference keeps track of the communication elements that
/// use a given UCXX endpoint. When this endpoint is closed, the elements
/// (UcxExchangeSources and UcxExchangeServers) are notified. When an element
/// is done, it notifies the endpoint. Access to the communicators_ set is
/// protected by commMutex_. The onClose callback defers all work to the
/// Communicator main loop thread via deferEndpointCleanup().
class EndpointRef : public std::enable_shared_from_this<EndpointRef> {
 public:
  EndpointRef(
      const std::shared_ptr<ucxx::Endpoint> endpoint,
      std::string peerIp = "")
      : endpoint_{endpoint}, peerIp_{std::move(peerIp)}, communicators_{} {}

  /// @brief Static method that is called when the underlying UCXX system closes
  /// the endpoint. In this case, all communication elements are informed that
  /// the endpoint has been closed.
  /// @param status The status (reason) why the endpoint has been closed.
  /// @param arg A reference to the EndpointRef (since this is a static method)
  static void onClose(ucs_status_t status, std::shared_ptr<void> arg);

  /// @brief Adds a new CommElement that is using this endpoint.
  /// @param commElem A shared pointer to the UcxExchangeSource or
  /// UcxExchangeServer.
  /// @return True, if commElem could be added.
  bool addCommElem(std::shared_ptr<CommElement> commElem);

  /// @brief Removes a CommElement from this endpoint again.
  /// @param commElem A shared pointer to the UcxExchangeSource or
  /// UcxExchangeServer.
  void removeCommElem(std::shared_ptr<CommElement> commElem);

  /// @brief Closes all registered communicators and drains the set.
  /// Uses swap-and-drain pattern: swaps communicators_ into a local copy
  /// under the lock, then iterates the copy without the lock.
  /// Must be called from the Communicator main loop thread ONLY.
  void closeAndDrainCommunicators();

  /// implement < operator such that this endpoint can be used in a
  /// std::map
  bool operator<(EndpointRef const& other);

  /// @brief Get the peer's IP address as seen from the connection.
  /// Used for reliable intra-node detection.
  const std::string& getPeerIp() const {
    return peerIp_;
  }

  const std::shared_ptr<ucxx::Endpoint> endpoint_;

 private:
  /// The peer's actual IP address extracted from the connection request.
  /// Used for server-side intra-node detection instead of client-reported IPs.
  std::string peerIp_;
  void cleanup(); // cleans up expired communication elements.

  std::set<
      std::weak_ptr<CommElement>,
      std::owner_less<std::weak_ptr<CommElement>>>
      communicators_;
  std::mutex commMutex_; // Protects communicators_
};
} // namespace facebook::velox::ucx_exchange
