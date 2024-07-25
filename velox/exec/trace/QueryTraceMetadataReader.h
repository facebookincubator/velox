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

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {
class QueryTraceMetadataReader {
 public:
  explicit QueryTraceMetadataReader(std::string traceOutputDir);

  void read(
      std::unordered_map<std::string, std::string>& queryConfigs,
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>& connectorProperties,
      core::PlanNodePtr& queryPlan) const;

 private:
  memory::MemoryPool* const pool_;
  const std::string traceOutputDir_;
  std::shared_ptr<filesystems::FileSystem> fileSystem_;
  std::string configFilePath_;
};
} // namespace facebook::velox::exec
