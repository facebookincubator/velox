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

#include "functions/remote/if/GetSerde.h"
#include "vector/VectorStream.h"
#include "velox/expression/EvalCtx.h"
#include "velox/functions/remote/client/Remote.h"
#include "velox/functions/remote/if/gen-cpp2/RemoteFunctionServiceAsyncClient.h"

namespace facebook::velox::functions {
class RemoteClient {
 public:
  RemoteClient(
      const std::string& functionName,
      RowTypePtr remoteInputType,
      std::vector<std::string> serializedInputTypes,
      const RemoteVectorFunctionMetadata& metadata)
      : functionName_(functionName),
        remoteInputType_(std::move(remoteInputType)),
        serializedInputTypes_(std::move(serializedInputTypes)),
        serdeFormat_(metadata.serdeFormat),
        metadata_(metadata),
        serde_(getSerde(serdeFormat_)) {}

  virtual ~RemoteClient() = default;

  virtual void applyRemote(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const = 0;

 protected:
  std::string functionName_;
  RowTypePtr remoteInputType_;
  std::vector<std::string> serializedInputTypes_;
  remote::PageFormat serdeFormat_;
  RemoteVectorFunctionMetadata metadata_;
  std::unique_ptr<VectorSerde> serde_;
};
} // namespace facebook::velox::functions
