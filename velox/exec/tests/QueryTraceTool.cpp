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

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include "velox/common/memory/Memory.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/trace/QueryTraceDataReader.h"
#include "velox/exec/trace/QueryTraceMetadataReader.h"
#include "velox/exec/trace/QueryTraceRestore.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {
void restoreAndRun(
    const std::string& metadataDir,
    const std::string& traceDataDir,
    const std::string& planNodeId) {
  const auto metadataReader = QueryTraceMetadataReader(metadataDir);
  std::unordered_map<std::string, std::string> actualQueryConfigs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      actualConnectorConfigs;
  core::PlanNodePtr queryPlan = nullptr;
  metadataReader.read(actualQueryConfigs, actualConnectorConfigs, queryPlan);
  const auto targetPlanNode = findPlanNodeById(queryPlan, planNodeId);

  std::shared_ptr<exec::Task> task;
  const auto restoredPlanNode =
      PlanBuilder()
          .traceScan(traceDataDir)
          .addNode(addTableWriter(
              std::dynamic_pointer_cast<const core::TableWriteNode>(
                  targetPlanNode)))
          .planNode();
  AssertQueryBuilder(restoredPlanNode)
      .maxDrivers(1)
      .configs(actualQueryConfigs)
      .connectorSessionProperties(actualConnectorConfigs)
      .config(core::QueryConfig::kQueryTraceEnabled, false)
      .copyResults(memory::MemoryManager::getInstance()->tracePool(), task);
}

} // namespace

DEFINE_string(metadata_dir, "", "");

DEFINE_string(trace_data_dir, "", "");

DEFINE_string(plan_node_id, "", "Target PlanNodeId");

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv, false};

  // Initializes the process-wide memory-manager with the default options.
  memory::initializeMemoryManager({});

  // TODO: Make it configuerable or linkable.
  filesystems::registerLocalFileSystem();

  const auto ioExecutor = std::make_unique<folly::IOThreadPoolExecutor>(3);
  // TODO: make it configurable or linkable.
  const auto hiveConnector =
      connector::getConnectorFactory("hive")->newConnector(
          "test-hive",
          std::make_shared<config::ConfigBase>(
              std::unordered_map<std::string, std::string>()),
          ioExecutor.get());
  connector::registerConnector(hiveConnector);

  restoreAndRun(FLAGS_metadata_dir, FLAGS_trace_data_dir, FLAGS_plan_node_id);
  return 0;
}
