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

namespace facebook::velox::ucx_exchange {

/// @brief The acceptor creates a new UcxExchangeServer each time a handshake
/// message is received. Handshakes are sent as active messages, using a
/// worker-wide handler. The acceptor is a passive component that is used by the
/// Communicator.
struct Acceptor {
  // The static callback function for incoming handshake requests.
  static void cStyleAMCallback(
      std::shared_ptr<ucxx::Request> request,
      ucp_ep_h ep);

  /// @brief Adds the endpoint reference to the handleToEndpointRef_ map such
  /// that endpoint handles can be resolved
  void registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef);

  // Maps the lower-layer UCP endpoint handle to an endpoint reference.
  // Accessed only from the Communicator main loop thread: writes happen in
  // registerEndpointRef() (called from listenerCallback during
  // worker_->progress()), and reads happen in cStyleAMCallback (also fired
  // during worker_->progress()). No mutex needed.
  std::map<ucp_ep_h, std::shared_ptr<EndpointRef>> handleToEndpointRef_;
};

} // namespace facebook::velox::ucx_exchange
