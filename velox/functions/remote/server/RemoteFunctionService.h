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

#include <thrift/lib/cpp2/server/ThriftServer.h>
#include "velox/common/memory/Memory.h"
#include "velox/functions/remote/if/gen-cpp2/RemoteFunctionService.h"

namespace facebook::velox::functions {

// Simple implementation of the thrift server handler.
class RemoteFunctionServiceHandler
    : virtual public apache::thrift::ServiceHandler<
          remote::RemoteFunctionService> {
 public:
  RemoteFunctionServiceHandler(const std::string& functionPrefix = "")
      : functionPrefix_(functionPrefix) {}

  void invokeFunction(
      remote::RemoteFunctionResponse& response,
      std::unique_ptr<remote::RemoteFunctionRequest> request) override;

 private:
  std::shared_ptr<memory::MemoryPool> pool_{memory::addDefaultLeafMemoryPool()};
  const std::string functionPrefix_;
};

} // namespace facebook::velox::functions
