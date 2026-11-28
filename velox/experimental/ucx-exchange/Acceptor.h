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
#include <map>
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"

namespace facebook::velox::ucx_exchange {

/// @brief The acceptor creates a new UcxExchangeServer each time a handshake
/// message is received. Handshakes are sent as active messages, using a
/// worker-wide handler. The acceptor is a passive component that is used by the
/// Communicator.
class Acceptor {
 public:
  /// Handles an incoming handshake. Runs inside UCX progress, so it throws
  /// nothing: a handshake it cannot handle is logged and dropped.
  static void cStyleAMCallback(
      std::shared_ptr<ucxx::Request> request,
      ucp_ep_h ep);

  /// Maps 'handle', the UCP handle of an incoming endpoint, to 'endpointRef',
  /// so that handshakes on the endpoint resolve to it. Replaces an earlier
  /// mapping of the handle: UCP can reuse the handle of an endpoint that closed
  /// for a new connection. Runs inside the UCX listener callback, so it throws
  /// nothing.
  void registerEndpointRef(
      ucp_ep_h handle,
      std::shared_ptr<EndpointRef> endpointRef);

  /// Erases every mapping to 'endpointRef', which is being removed. A mapping
  /// of its handle that a newer endpoint replaced stays.
  void unregisterEndpointRef(const std::shared_ptr<EndpointRef>& endpointRef);

  /// Returns the endpoint reference mapped to 'handle', or nullptr.
  std::shared_ptr<EndpointRef> findEndpointRef(ucp_ep_h handle) const;

  /// Returns true if the send of a HandshakeResponse to the source of 'key',
  /// which completed with 'status', may not have reached the source. That
  /// source then neither reads from its server nor sends the close message,
  /// so the Acceptor reports the destination's consumer gone. A cancelled send
  /// is not lost: UCX cancels sends when the endpoint closes, which tells the
  /// endpoint's servers, or when the worker shuts down. Logs the outcome.
  static bool isResponseLost(ucs_status_t status, const PartitionKey& key);

 private:
  // Creates a server for a handshake of this protocol version, or refuses a
  // handshake of another version. May throw; cStyleAMCallback() catches.
  static void handleHandshake(
      const std::shared_ptr<ucxx::Request>& request,
      ucp_ep_h ep);

  // Maps the UCP handle of each incoming endpoint to its reference. Used only
  // on the Communicator thread: registerEndpointRef() runs in the listener
  // callback and lookups in the active message callback, both inside the
  // worker's progress, and unregisterEndpointRef() in the Communicator's
  // endpoint removal. No mutex needed.
  std::map<ucp_ep_h, std::shared_ptr<EndpointRef>> handleToEndpointRef_;
};

} // namespace facebook::velox::ucx_exchange
