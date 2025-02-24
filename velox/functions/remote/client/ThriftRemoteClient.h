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

#include <folly/SocketAddress.h>
#include <folly/io/async/EventBase.h>

#include "velox/functions/remote/client/RemoteClient.h"
#include "velox/functions/remote/client/ThriftClient.h"

namespace facebook::velox::functions {

class ThriftRemoteClient : public RemoteClient {
 public:
  ThriftRemoteClient(
      const folly::SocketAddress& address,
      const std::string& functionName,
      RowTypePtr remoteInputType,
      std::vector<std::string> serializedInputTypes,
      const RemoteVectorFunctionMetadata& metadata);

  void applyRemote(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override;

 private:
  folly::EventBase eventBase_;
  std::unique_ptr<RemoteFunctionClient> thriftClient_;
};

} // namespace facebook::velox::functions
