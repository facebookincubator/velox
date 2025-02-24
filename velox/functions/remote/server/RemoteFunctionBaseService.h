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

#include <memory>
#include "velox/common/memory/Memory.h"
#include "velox/expression/Expr.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {

class RemoteFunctionBaseService {
 public:
  virtual ~RemoteFunctionBaseService() = default;

 protected:
  RemoteFunctionBaseService(
      const std::string& functionPrefix,
      std::shared_ptr<memory::MemoryPool> pool)
      : functionPrefix_(functionPrefix), pool_(std::move(pool)) {
    if (!pool_) {
      pool_ = memory::memoryManager()->addLeafPool();
    }
  }

  RowVectorPtr invokeFunctionInternal(
      const folly::IOBuf& payload,
      const std::vector<std::string>& argTypeNames,
      const std::string& returnTypeName,
      const std::string& functionName,
      bool throwOnError,
      VectorSerde* serde);

  exec::EvalErrors* getEvalErrors_() {
    return evalCtx_ ? evalCtx_->errors() : nullptr;
  }

  std::string functionPrefix_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  std::unique_ptr<exec::EvalCtx> evalCtx_;
};

} // namespace facebook::velox::functions
