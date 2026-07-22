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

#include "velox/connectors/hive/storage_adapters/abfs/AsyncChannelFactory.h"
#include "velox/connectors/hive/storage_adapters/abfs/HttpConnection.h"

#include <azure/core/http/http.hpp>
#include <azure/core/http/transport.hpp>

#include <cstddef>
#include <memory>

namespace facebook::velox::filesystems {

/// Bridges one EventBase-affine ABFS connection to Azure Core HTTP.
class FollyHttpTransport final : public Azure::Core::Http::HttpTransport {
 public:
  /// Reports the EventBase-affine Stage 1 connection-pool counters.
  struct PoolMetrics {
    /// Holds the configured physical connection limit.
    size_t maxConnections{0};
    /// Counts open physical connections, including leased connections.
    size_t totalConnections{0};
    /// Counts connections currently owned by requests or response bodies.
    size_t leasedConnections{0};
    /// Counts fully released connections available for reuse.
    size_t idleConnections{0};
    /// Counts fibers currently suspended waiting for a connection.
    size_t waitingFibers{0};
    /// Records the largest simultaneous leased-connection count.
    size_t peakLeasedConnections{0};
    /// Counts idle connections closed after reaching their deadline.
    size_t idleConnectionEvictions{0};
  };

  /// Configures one pre-resolved endpoint and the HTTP resource limits.
  FollyHttpTransport(
      AsyncChannelFactoryPtr factory,
      AsyncChannelEndpoint endpoint,
      HttpLimits limits,
      HttpTimeouts timeouts,
      size_t maxConnectionsPerEndpoint);

  /// Sends an Azure request after validating its endpoint and method.
  std::unique_ptr<Azure::Core::Http::RawResponse> Send(
      Azure::Core::Http::Request& request,
      const Azure::Core::Context& context) override;

  /// Returns an immutable snapshot of this transport's private pool state.
  PoolMetrics poolMetrics() const;

 private:
  AsyncChannelFactoryPtr factory_;
  AsyncChannelEndpoint endpoint_;
  HttpLimits limits_;
  HttpTimeouts timeouts_;
  size_t maxConnectionsPerEndpoint_{0};
  std::shared_ptr<void> pool_;
};

} // namespace facebook::velox::filesystems
