/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/functions/remote/client/RemoteVectorFunction.h"
#include "velox/functions/remote/client/ThriftClient.h"

namespace facebook::velox::functions {

struct RemoteThriftVectorFunctionMetadata
    : public RemoteVectorFunctionMetadata {
  /// Network address of the server to communicate with using a thrift client.
  /// Note that this can hold a network location (ip/port pair) or a unix domain
  /// socket path (see SocketAddress::makeFromPath()).
  folly::SocketAddress location;

  /// Optional factory for creating remote function clients. If not set, the
  /// default thrift client factory is used. This enables dependency injection
  /// for testing with mock clients.
  RemoteFunctionClientFactory clientFactory;
};

/// Registers a new remote function. It will use the meatadata defined in
/// `RemoteThriftVectorFunctionMetadata` to control the serialization format,
/// remote server address, and communicate with it using a thrift client.
//
/// Remote functions are registered as regular statufull functions (using the
/// same internal catalog), and hence conflict if there already exists a
/// (non-remote) function registered with the same name. The `overwrite` flag
/// controls whether to overwrite in these cases.
void registerRemoteFunction(
    const std::string& name,
    std::vector<exec::FunctionSignaturePtr> signatures,
    const RemoteThriftVectorFunctionMetadata& metadata = {},
    bool overwrite = true);

} // namespace facebook::velox::functions
