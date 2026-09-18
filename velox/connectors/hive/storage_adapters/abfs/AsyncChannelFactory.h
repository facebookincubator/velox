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

#include <folly/SocketAddress.h>
#include <folly/io/async/AsyncTransport.h>

#include <chrono>
#include <memory>
#include <string>

namespace facebook::velox::filesystems {

/// Selects the transport security used for an endpoint connection.
enum class AsyncChannelSecurity { kPlaintext, kTls };

/// Carries resolved connection details without performing DNS resolution.
struct AsyncChannelEndpoint {
  /// Holds the pre-resolved numeric address used for the socket connection.
  folly::SocketAddress connectAddress;
  /// Holds the HTTP host and TLS server name.
  std::string serverName;
  /// Selects plaintext or TLS transport creation.
  AsyncChannelSecurity security{AsyncChannelSecurity::kPlaintext};
  /// Bounds the connection attempt.
  std::chrono::milliseconds connectTimeout{std::chrono::seconds(30)};
  /// Bounds the TLS handshake independently from the TCP connection.
  std::chrono::milliseconds tlsHandshakeTimeout{std::chrono::seconds(30)};
  /// Adds a trusted PEM CA without replacing system trust roots.
  std::string additionalTrustedCaPath;
};

/// Creates asynchronous transport channels for an already resolved endpoint.
class AsyncChannelFactory {
 public:
  /// Destroys the channel factory interface.
  virtual ~AsyncChannelFactory() = default;

  /// Creates a transport without performing name resolution on the event base.
  virtual folly::AsyncTransportWrapper::UniquePtr connect(
      const AsyncChannelEndpoint& endpoint) = 0;
};

/// Shares an asynchronous channel factory with an Azure transport.
using AsyncChannelFactoryPtr = std::shared_ptr<AsyncChannelFactory>;

} // namespace facebook::velox::filesystems
