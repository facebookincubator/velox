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

#include "substrait.h" // @manual
#include <google/protobuf/util/json_util.h>
#include <velox/common/base/Exceptions.h>
#include <velox/exec/tests/utils/AssertQueryBuilder.h>
#include <velox/exec/tests/utils/HiveConnectorTestBase.h>
#include <velox/exec/tests/utils/TempDirectoryPath.h>
#include <velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h>
#include <velox/functions/prestosql/registration/RegistrationFunctions.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "context.h"

namespace facebook::velox::py {

namespace py = pybind11;

static void readFromFile(
    const std::string& msgPath,
    google::protobuf::Message& msg) {
  // Read json file and resume the Substrait plan.
  std::ifstream msgJson(msgPath);
  VELOX_CHECK(
      !msgJson.fail(), "Failed to open file: {}. {}", msgPath, strerror(errno));
  std::stringstream buffer;
  buffer << msgJson.rdbuf();
  std::string msgData = buffer.str();
  auto status = google::protobuf::util::JsonStringToMessage(msgData, &msg);
  VELOX_CHECK(
      status.ok(),
      "Failed to parse Substrait JSON: {} {}",
      status.code(),
      status.message());
}

static inline void initializeSubstrait() {
  PySubstraitContext::getInstance().initialize();
}

static inline void finalizeSubstrait() {
  PySubstraitContext::getInstance().finalize();
}

std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
makeSplits(
    const facebook::velox::substrait::SubstraitVeloxPlanConverter& converter,
    std::shared_ptr<const core::PlanNode> planNode,
    const std::string& dirPath) {
  const auto& splitInfos = converter.splitInfos();
  auto leafPlanNodeIds = planNode->leafPlanNodeIds();
  // Only one leaf node is expected here.
  EXPECT_EQ(1, leafPlanNodeIds.size());
  const auto& splitInfo = splitInfos.at(*leafPlanNodeIds.begin());

  const auto& paths = splitInfo->paths;
  const auto& starts = splitInfo->starts;
  const auto& lengths = splitInfo->lengths;
  const auto fileFormat = splitInfo->format;

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      splits;
  splits.reserve(paths.size());

  for (int i = 0; i < paths.size(); i++) {
    auto path = fmt::format("{}{}", dirPath, paths[i]);
    auto start = starts[i];
    auto length = lengths[i];
    auto split = facebook::velox::exec::test::HiveConnectorSplitBuilder(path)
                     .fileFormat(fileFormat)
                     .start(start)
                     .length(length)
                     .build();
    splits.emplace_back(split);
  }
  return splits;
}

static inline RowVectorPtr runSubstraitQuery(
    const std::string& planPath,
    bool enableSplits,
    const std::string& dirPath) {
  memory::MemoryPool* pool = PyVeloxContext::getSingletonInstance().pool();

  facebook::velox::substrait::SubstraitVeloxPlanConverter planConverter(pool);

  ::substrait::Plan substraitPlan;
  readFromFile(planPath, substraitPlan);

  auto planNode = planConverter.toVeloxPlan(substraitPlan);

  if (enableSplits) {
    return facebook::velox::exec::test::AssertQueryBuilder(planNode)
        .splits(makeSplits(planConverter, planNode, dirPath))
        .copyResults(pool);
  } else {
    return facebook::velox::exec::test::AssertQueryBuilder(planNode)
        .copyResults(pool);
  }
}

void addSubstraitBindings(py::module& m, bool asModuleLocalDefinitions) {
  using namespace facebook::velox;
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  m.def(
      "initialize_substrait",
      &initializeSubstrait,
      "Initializes the modules required for Substrait.");
  m.def(
      "run_substrait_query",
      &runSubstraitQuery,
      "Runs a Substrait query and return output.",
      py::arg("plan_path"),
      py::arg("enable_splits") = false,
      py::arg("dir_path") = "");
  m.def(
      "finalize_substrait",
      &finalizeSubstrait,
      "Finalizes the modules required for Substrait.");
}
} // namespace facebook::velox::py
