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

#include "velox/exec/trace/QueryTraceMetadataWriter.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
namespace {
const std::string kQueryConfigKey = "queryConfig";
const std::string kConnectorPropertiesKey = "connectorProperties";
const std::string kPlanNodeKey = "planNode";
const std::string kQueryConfigFileName = "query_config.json";
} // namespace

QueryMetadataWriter::QueryMetadataWriter(std::string traceOutputDir)
    : traceOutputDir_(std::move(traceOutputDir)) {
  fileSystem_ = filesystems::getFileSystem(traceOutputDir_, nullptr);
  VELOX_CHECK_NOT_NULL(fileSystem_);
  if (!fileSystem_->exists(traceOutputDir_)) {
    fileSystem_->mkdir(traceOutputDir_);
  }
  configFilePath_ = fmt::format("{}/{}", traceOutputDir_, kQueryConfigFileName);
}

void QueryMetadataWriter::write(
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    const core::PlanNodePtr& planNode) const {
  folly::dynamic queryConfigObj = folly::dynamic::object;
  const auto configValues = queryCtx->queryConfig().rawConfigsCopy();
  for (const auto& [key, value] : configValues) {
    queryConfigObj[key] = value;
  }

  folly::dynamic connectorPropertiesObj = folly::dynamic::object;
  for (const auto& [connectorId, configs] :
       queryCtx->connectorSessionProperties()) {
    folly::dynamic obj = folly::dynamic::object;
    for (const auto& [key, value] : configs->rawConfigsCopy()) {
      obj[key] = value;
    }
    connectorPropertiesObj[connectorId] = obj;
  }

  folly::dynamic configObj = folly::dynamic::object;
  configObj[kQueryConfigKey] = queryConfigObj;
  configObj[kConnectorPropertiesKey] = connectorPropertiesObj;
  configObj[kPlanNodeKey] = planNode->serialize();

  const auto configStr = folly::toJson(configObj);
  const auto file = fileSystem_->openFileForWrite(configFilePath_);
  file->append(configStr);
  file->close();
}

} // namespace facebook::velox::exec
