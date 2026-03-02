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

#include "velox/expression/VectorFunction.h"
#include "velox/functions/remote/if/gen-cpp2/RemoteFunction_types.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {

struct RemoteVectorFunctionMetadata : public exec::VectorFunctionMetadata {
  /// The serialization format to be used to send batches of data to the remote
  /// process.
  remote::PageFormat serdeFormat{remote::PageFormat::PRESTO_PAGE};

  /// Whether to preserve the input vector encoding in the request sent to
  /// remote service.
  bool preserveEncoding{false};
};

/// Main vector function logic. Needs to be extended with the transport-specific
/// logic.
class RemoteVectorFunction : public exec::VectorFunction {
 public:
  RemoteVectorFunction(
      const std::string& functionName,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const RemoteVectorFunctionMetadata& metadata);

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override;

 protected:
  // The actual function to communicates with the remote host.
  virtual std::unique_ptr<remote::RemoteFunctionResponse> invokeRemoteFunction(
      const remote::RemoteFunctionRequest& request) const = 0;

  // A string representation of the remote host being connected to. Useful for
  // exception messages.
  virtual std::string remoteLocationToString() const = 0;

 private:
  void applyRemote(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const;

  const std::string functionName_;

  remote::PageFormat serdeFormat_;
  std::unique_ptr<VectorSerde> serde_;
  std::unique_ptr<VectorSerde::Options> serdeOptions_;
  bool preserveEncoding_;

  // Structures we construct once to cache:
  RowTypePtr remoteInputType_;
  std::vector<std::string> serializedInputTypes_;
};

} // namespace facebook::velox::functions
