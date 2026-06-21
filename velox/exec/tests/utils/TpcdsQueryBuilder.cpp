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
#include "velox/exec/tests/utils/TpcdsQueryBuilder.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <filesystem>
namespace fs = std::filesystem;

namespace {

/// Try to find data files for a table name. First tries exact match,
/// then strips any schema prefix (e.g. "tpcds.store_sales" ->
/// "store_sales").
const std::vector<std::string>* findDataFiles(
    const std::unordered_map<std::string, std::vector<std::string>>&
        tableDataFiles,
    const std::string& tableName) {
  auto it = tableDataFiles.find(tableName);
  if (it != tableDataFiles.end() && !it->second.empty()) {
    return &it->second;
  }

  auto dotPos = tableName.rfind('.');
  if (dotPos != std::string::npos) {
    it = tableDataFiles.find(tableName.substr(dotPos + 1));
    if (it != tableDataFiles.end() && !it->second.empty()) {
      return &it->second;
    }
  }

  return nullptr;
}

} // namespace

namespace facebook::velox::exec::test {

TpcdsQueryBuilder::TpcdsQueryBuilder(dwio::common::FileFormat format)
    : format_(format) {}

void TpcdsQueryBuilder::initialize(const std::string& dataPath) {
  tableDataFiles_.clear();

  std::error_code ec;
  for (auto const& tableEntry :
       fs::directory_iterator{dataPath, fs::directory_options(), ec}) {
    if (!tableEntry.is_directory()) {
      continue;
    }
    const auto tableName = tableEntry.path().filename().string();

    // Skip hidden directories.
    if (tableName.empty() || tableName[0] == '.') {
      continue;
    }

    auto& files = tableDataFiles_[tableName];

    std::error_code fileEc;
    for (auto const& fileEntry : fs::directory_iterator{
             tableEntry.path(), fs::directory_options(), fileEc}) {
      if (!fileEntry.is_regular_file()) {
        continue;
      }
      // Skip hidden files.
      if (fileEntry.path().filename().c_str()[0] == '.') {
        continue;
      }
      files.push_back(fileEntry.path().string());
    }

    // Sort for deterministic ordering.
    std::sort(files.begin(), files.end());
  }

  VELOX_USER_CHECK(
      !tableDataFiles_.empty(),
      "No table subdirectories found in data path: {}",
      dataPath);
}

void TpcdsQueryBuilder::registerHiveConnector(
    const std::string& connectorId,
    folly::Executor* /*ioExecutor*/) {
  auto hiveConnector = connector::hive::HiveConnectorFactory().newConnector(
      connectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  connector::ConnectorRegistry::global().insert(
      hiveConnector->connectorId(), hiveConnector);
}

VeloxPlan TpcdsQueryBuilder::getQueryPlan(
    int queryId,
    const std::string& planDir,
    memory::MemoryPool* pool) {
  VeloxPlanLoader loader(planDir, pool);
  auto veloxPlan = loader.loadPlanByQueryId(queryId);

  // Collect all TableScan nodes from the plan tree.
  auto scanNodes = VeloxPlanLoader::collectTableScanNodes(veloxPlan.plan);

  // Detect connector ID from the plan's TableScan nodes.
  // Presto plans typically use "hive" while the test fixture registers
  // "test-hive". We need to register a connector under the plan's ID.
  if (!scanNodes.empty() && connectorId_.empty()) {
    connectorId_ = scanNodes[0]->tableHandle()->connectorId();
    LOG(INFO) << "TpcdsQueryBuilder: detected connector ID '" << connectorId_
              << "' from plan";

    if (connector::ConnectorRegistry::tryGet(connectorId_) == nullptr) {
      registerHiveConnector(connectorId_);
      ownedConnector_ = true;
      LOG(INFO) << "TpcdsQueryBuilder: registered connector under ID '"
                << connectorId_ << "'";
    }
  }

  // For each TableScan, look up the table name and populate data files.
  for (const auto& scanNode : scanNodes) {
    const auto& tableName = scanNode->tableHandle()->name();

    const auto* files = findDataFiles(tableDataFiles_, tableName);
    if (files) {
      veloxPlan.dataFiles[scanNode->id()] = *files;
    } else {
      LOG(WARNING) << "TpcdsQueryBuilder: no data files found for table '"
                   << tableName << "' (scan node " << scanNode->id() << ")";
    }
  }

  veloxPlan.dataFileFormat = format_;
  return veloxPlan;
}

std::shared_ptr<connector::ConnectorSplit> TpcdsQueryBuilder::makeSplit(
    const std::string& filePath) const {
  const auto& id = connectorId_.empty() ? kHiveConnectorId : connectorId_;
  return connector::hive::HiveConnectorSplitBuilder(filePath)
      .connectorId(id)
      .fileFormat(format_)
      .build();
}

void TpcdsQueryBuilder::shutdown() {
  if (ownedConnector_ && !connectorId_.empty()) {
    connector::ConnectorRegistry::global().erase(connectorId_);
    ownedConnector_ = false;
  }
  connectorId_.clear();
}

} // namespace facebook::velox::exec::test
