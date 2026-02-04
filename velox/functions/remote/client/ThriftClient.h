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

#include <folly/io/async/EventBase.h>
#include "velox/functions/remote/if/gen-cpp2/RemoteFunctionServiceAsyncClient.h"

namespace facebook::velox::functions {

using RemoteFunctionClient =
    apache::thrift::Client<remote::RemoteFunctionService>;

/// Abstract interface for the remote function client, enabling dependency
/// injection and mocking in tests.
class IRemoteFunctionClient {
 public:
  virtual ~IRemoteFunctionClient() = default;

  /// Invokes the remote function synchronously.
  virtual void invokeFunction(
      remote::RemoteFunctionResponse& response,
      const remote::RemoteFunctionRequest& request) = 0;
};

/// Default implementation that wraps the actual thrift client.
class ThriftRemoteFunctionClient : public IRemoteFunctionClient {
 public:
  explicit ThriftRemoteFunctionClient(
      std::unique_ptr<RemoteFunctionClient> client)
      : client_(std::move(client)) {}

  void invokeFunction(
      remote::RemoteFunctionResponse& response,
      const remote::RemoteFunctionRequest& request) override {
    client_->sync_invokeFunction(response, request);
  }

 private:
  std::unique_ptr<RemoteFunctionClient> client_;
};

/// Factory function type for creating remote function clients.
/// Parameters: location (socket address), eventBase (for async operations)
/// Returns: A unique_ptr to an IRemoteFunctionClient implementation.
using RemoteFunctionClientFactory = std::function<std::unique_ptr<
    IRemoteFunctionClient>(folly::SocketAddress, folly::EventBase*)>;

std::unique_ptr<RemoteFunctionClient> getThriftClient(
    folly::SocketAddress location,
    folly::EventBase* eventBase);

/// Default factory that creates ThriftRemoteFunctionClient instances.
inline std::unique_ptr<IRemoteFunctionClient> getDefaultRemoteFunctionClient(
    folly::SocketAddress location,
    folly::EventBase* eventBase) {
  return std::make_unique<ThriftRemoteFunctionClient>(
      getThriftClient(location, eventBase));
}

} // namespace facebook::velox::functions
