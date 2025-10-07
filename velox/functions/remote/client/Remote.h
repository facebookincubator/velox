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

#include "velox/functions/remote/client/RemoteVectorFunction.h"

namespace facebook::velox::functions {

struct RemoteThriftVectorFunctionMetadata
    : public RemoteVectorFunctionMetadata {
  // TODO: Move `folly::SocketAddress location` and other thrift options here
  // once call sites are updated.
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

// TODO: Remove once call sites are updated.
void registerRemoteFunction(
    const std::string& name,
    std::vector<exec::FunctionSignaturePtr> signatures,
    const RemoteVectorFunctionMetadata& metadata = {},
    bool overwrite = true);

} // namespace facebook::velox::functions
