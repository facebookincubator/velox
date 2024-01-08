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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "context.h"

namespace facebook::velox::py {

namespace py = pybind11;

namespace {

bool isFilePath(const std::string& path) {
  return std::filesystem::exists(path);
}

static void readFromJSON(
    const std::string& msg,
    google::protobuf::Message& protoMsg) {
  auto status = google::protobuf::util::JsonStringToMessage(msg, &protoMsg);
  VELOX_CHECK(
      status.ok(),
      "Failed to parse Substrait JSON: {} {}",
      status.code(),
      status.message());
}

bool isJSON(std::string content) {
  // Remove leading whitespaces
  auto start =
      std::find_if_not(content.begin(), content.end(), [](unsigned char c) {
        return std::isspace(c);
      });
  content.erase(content.begin(), start);

  // Remove trailing whitespaces
  auto end =
      std::find_if_not(content.rbegin(), content.rend(), [](unsigned char c) {
        return std::isspace(c);
      });
  content.erase(end.base(), content.end());

  if (content.empty()) {
    return false; // If the file is empty, it's not a JSON file
  }

  // Check if the first character is '{' or '[' and the last character is '}' or
  // ']'
  return (
      (content.front() == '{' && content.back() == '}') ||
      (content.front() == '[' && content.back() == ']'));
}

bool isJsonFile(const std::string& filePath) {
  std::ifstream file(filePath);
  std::string content(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return isJSON(content);
}

} // namespace

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
    const std::shared_ptr<const core::PlanNode> planNode,
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
    const std::string& plan,
    bool enableSplits,
    const std::string& dirPath) {
  memory::MemoryPool* pool = PyVeloxContext::getSingletonInstance().pool();

  facebook::velox::substrait::SubstraitVeloxPlanConverter planConverter(pool);

  ::substrait::Plan substraitPlan;
  // convert plan to protobuf
  // read from file
  if (isJSON(plan)) {
    readFromJSON(plan, substraitPlan);
  } else if (isFilePath(plan)) {
    // check whether input file is in JSON format
    if (isJsonFile(plan)) {
      readFromFile(plan, substraitPlan);
    } else {
      /// other formats are not yet supported.
      /// NOTE: To support Protobuf format, we should generate Substrait proto
      /// in Python format too. Then we should be able to add proto format too.
      throw py::value_error("plan should be path to a plan in JSON format.");
    }
  } else {
    throw py::value_error("Invalid Substrait plan.");
  }

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
      R"delimiter(
        Runs a Substrait query and return output.

        Parameters
        ----------
        plan : str
          The path of Substrait plan or Substrait plan JSON.
        enable_splits: bool
        	Flag to enable splits.
        file_path: str
        	The path to which the vector will be saved. Must specify if `enable_splits`
        	is set to True.

        Returns
        -------
        RowVector

        Examples
        --------

        >>> import pyvelox.pyvelox as pv
        >>> pv.initialize_substrait()
        >>> res = pv.run_substrait_query('path_to_substrait_plan')
        >>> pv.finalize_substrait()
        >>> type(res)
        <class 'pyvelox.pyvelox.RowVector'>
      )delimiter",
      py::arg("plan"),
      py::arg("enable_splits") = false,
      py::arg("dir_path") = "");
  m.def(
      "finalize_substrait",
      &finalizeSubstrait,
      "Finalizes the modules required for Substrait.");
}
} // namespace facebook::velox::py
