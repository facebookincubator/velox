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

#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {
class QueryMetadataWriter {
 public:
  explicit QueryMetadataWriter(std::string traceOutputDir);

  void write(
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      const core::PlanNodePtr& planNode) const;

 private:
  memory::MemoryPool* const pool_{
      memory::MemoryManager::getInstance()->tracePool()};
  const std::string traceOutputDir_;
  std::shared_ptr<filesystems::FileSystem> fileSystem_;
  std::string configFilePath_;
};
} // namespace facebook::velox::exec
