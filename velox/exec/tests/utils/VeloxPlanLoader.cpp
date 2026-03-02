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
#include "velox/exec/tests/utils/VeloxPlanLoader.h"
#include <folly/json.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "velox/common/base/Exceptions.h"
#include "velox/common/serialization/Serializable.h"
#include "velox/core/PlanNode.h"

#if __has_include("filesystem")
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace facebook::velox::exec::test {

namespace {

void collectTableScanNodesRecursive(
    const core::PlanNodePtr& node,
    std::vector<core::TableScanNodePtr>& out) {
  if (auto scan = std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
    out.push_back(scan);
  }
  for (const auto& source : node->sources()) {
    collectTableScanNodesRecursive(source, out);
  }
}

std::string readFileToString(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    VELOX_FAIL("Failed to open plan file: {}", path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

} // namespace

// static
std::string VeloxPlanLoader::resolvePlanDirectory(
    const std::string& defaultDir,
    const std::string& envVarName) {
  if (!envVarName.empty()) {
    const char* env = std::getenv(envVarName.c_str());
    if (env && *env != '\0') {
      return std::string(env);
    }
  }
  return defaultDir;
}

VeloxPlanLoader::VeloxPlanLoader(
    const std::string& planDirectory,
    memory::MemoryPool* pool,
    bool stripPartitionedOutput)
    : planDir_(planDirectory),
      pool_(pool),
      stripPartitionedOutput_(stripPartitionedOutput) {
  if (pool_ == nullptr) {
    ownedPool_ = memory::memoryManager()->addLeafPool("VeloxPlanLoader");
    pool_ = ownedPool_.get();
  }
}

std::string VeloxPlanLoader::pathForQuery(int queryId) const {
  const std::string path = planDir_ + "/Q" + std::to_string(queryId) + ".json";
  if (fs::exists(path)) {
    return path;
  }
  VELOX_FAIL("Plan file does not exist: {}", path);
}

void VeloxPlanLoader::maybeStripPartitionedOutput(
    core::PlanNodePtr& plan) const {
  if (!stripPartitionedOutput_ || !plan) {
    return;
  }
  if (auto partitionedOutput =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(plan)) {
    VELOX_CHECK_EQ(
        partitionedOutput->sources().size(),
        1,
        "PartitionedOutput should have exactly one source");
    plan = partitionedOutput->sources()[0];
    LOG(INFO) << "VeloxPlanLoader: stripped PartitionedOutput root from plan";
  }
}

VeloxPlan VeloxPlanLoader::loadPlan(const std::string& path) const {
  const std::string contents = readFileToString(path);
  folly::dynamic planJson;
  try {
    planJson = folly::parseJson(contents);
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to parse plan JSON from {}: {}", path, e.what());
  }
  core::PlanNodePtr plan;
  try {
    plan = velox::ISerializable::deserialize<core::PlanNode>(planJson, pool_);
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to deserialize plan from {}: {}", path, e.what());
  }
  maybeStripPartitionedOutput(plan);
  VeloxPlan result;
  result.plan = std::move(plan);
  result.dataFiles = {};
  result.dataFileFormat = dwio::common::FileFormat::PARQUET;
  return result;
}

VeloxPlan VeloxPlanLoader::loadPlanByQueryId(int queryId) const {
  VELOX_USER_CHECK(
      queryId >= 1 && queryId <= 99, "queryId must be 1..99, got {}", queryId);
  std::string path = pathForQuery(queryId);
  return loadPlan(path);
}

// static
std::vector<core::TableScanNodePtr> VeloxPlanLoader::collectTableScanNodes(
    const core::PlanNodePtr& plan) {
  std::vector<core::TableScanNodePtr> out;
  if (plan) {
    collectTableScanNodesRecursive(plan, out);
  }
  return out;
}

} // namespace facebook::velox::exec::test
