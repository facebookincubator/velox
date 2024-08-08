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

#include "velox/exec/trace/QueryTraceMetadataReader.h"

#include <utility>
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
namespace {
const std::string kQueryConfigKey = "queryConfig";
const std::string kConnectorPropertiesKey = "connectorProperties";
const std::string kPlanNodeKey = "planNode";
} // namespace

QueryTraceMetadataReader::QueryTraceMetadataReader(std::string traceOutputDir)
    : pool_(memory::MemoryManager::getInstance()->tracePool()),
      traceOutputDir_(std::move(traceOutputDir)) {
  fileSystem_ = filesystems::getFileSystem(traceOutputDir_, nullptr);
  VELOX_CHECK_NOT_NULL(fileSystem_);
  if (!fileSystem_->exists(traceOutputDir_)) {
    fileSystem_->mkdir(traceOutputDir_);
  }
  configFilePath_ = fmt::format("{}/query_config.json", traceOutputDir_);

  Type::registerSerDe();
  common::Filter::registerSerDe();
  connector::hive::HiveTableHandle::registerSerDe();
  connector::hive::LocationHandle::registerSerDe();
  connector::hive::HiveColumnHandle::registerSerDe();
  connector::hive::HiveInsertTableHandle::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  registerPartitionFunctionSerDe();
}

void QueryTraceMetadataReader::read(
    std::unordered_map<std::string, std::string>& queryConfigs,
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>& connectorProperties,
    core::PlanNodePtr& queryPlan) const {
  const auto file = fileSystem_->openFileForRead(configFilePath_);
  const auto metadata = file->pread(0, file->size());
  VELOX_USER_CHECK(!metadata.empty());
  folly::dynamic obj = folly::parseJson(metadata);

  const auto& querConfigObj = obj[kQueryConfigKey];
  for (const auto& [key, value] : querConfigObj.items()) {
    queryConfigs[key.asString()] = value.asString();
  }

  const auto& connectorPropertiesObj = obj[kConnectorPropertiesKey];
  for (const auto& [connectorId, configs] : connectorPropertiesObj.items()) {
    connectorProperties[connectorId.asString()] = {};
    for (const auto& [key, value] : configs.items()) {
      connectorProperties[connectorId.asString()][key.asString()] =
          value.asString();
    }
  }

  queryPlan =
      ISerializable::deserialize<core::PlanNode>(obj[kPlanNodeKey], pool_);
}
} // namespace facebook::velox::exec
