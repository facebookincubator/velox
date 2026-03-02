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

#include "velox/functions/remote/client/Remote.h"

#include <folly/io/async/EventBase.h>
#include "velox/functions/remote/client/RemoteVectorFunction.h"
#include "velox/functions/remote/client/ThriftClient.h"
#include "velox/functions/remote/if/GetSerde.h"

namespace facebook::velox::functions {
namespace {

class RemoteThriftFunction : public RemoteVectorFunction {
 public:
  RemoteThriftFunction(
      const std::string& functionName,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const RemoteThriftVectorFunctionMetadata& metadata)
      : RemoteVectorFunction(functionName, inputArgs, metadata),
        location_(metadata.location),
        client_(createClient(metadata)) {}

  std::unique_ptr<remote::RemoteFunctionResponse> invokeRemoteFunction(
      const remote::RemoteFunctionRequest& request) const override {
    auto remoteResponse = std::make_unique<remote::RemoteFunctionResponse>();
    client_->invokeFunction(*remoteResponse, request);
    return remoteResponse;
  }

  std::string remoteLocationToString() const override {
    return location_.describe();
  }

 private:
  std::unique_ptr<IRemoteFunctionClient> createClient(
      const RemoteThriftVectorFunctionMetadata& metadata) {
    if (metadata.clientFactory) {
      return metadata.clientFactory(metadata.location, &eventBase_);
    }
    return getDefaultRemoteFunctionClient(metadata.location, &eventBase_);
  }

  folly::SocketAddress location_;
  folly::EventBase eventBase_;

  std::unique_ptr<IRemoteFunctionClient> client_;
};

std::shared_ptr<exec::VectorFunction> createRemoteFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/,
    const RemoteThriftVectorFunctionMetadata& metadata) {
  return std::make_unique<RemoteThriftFunction>(name, inputArgs, metadata);
}

} // namespace

void registerRemoteFunction(
    const std::string& name,
    std::vector<exec::FunctionSignaturePtr> signatures,
    const RemoteThriftVectorFunctionMetadata& metadata,
    bool overwrite) {
  exec::registerStatefulVectorFunction(
      name,
      signatures,
      std::bind(
          createRemoteFunction,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3,
          metadata),
      metadata,
      overwrite);
}

} // namespace facebook::velox::functions
