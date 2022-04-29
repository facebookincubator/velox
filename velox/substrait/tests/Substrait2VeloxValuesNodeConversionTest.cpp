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

#include <google/protobuf/util/json_util.h>
#include <fstream>
#include <sstream>

#include "velox/common/base/tests/Fs.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "velox/substrait/SubstraitToVeloxPlan.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::substrait;

class VeloxConverter : public OperatorTestBase {
 public:
  void assertPlanConversion(
      const std::shared_ptr<const core::PlanNode>& plan,
      const std::string& duckDbSql) {
    if (auto vValuesNode =
            std::dynamic_pointer_cast<const facebook::velox::core::ValuesNode>(
                plan)) {
      vectors = vValuesNode->values();
    }
    createDuckDbTable(vectors);
    assertQuery(plan, duckDbSql);
  }

  std::shared_ptr<facebook::velox::substrait::SubstraitVeloxPlanConverter>
      planConverter = std::make_shared<
          facebook::velox::substrait::SubstraitVeloxPlanConverter>();
  std::vector<RowVectorPtr> vectors;

  std::unique_ptr<facebook::velox::memory::ScopedMemoryPool> pool_{
      memory::getDefaultScopedMemoryPool()};
};

// SELECT * FROM tmp
TEST_F(VeloxConverter, valuesNode) {
  // Find the Velox path according current path.
  std::string veloxPath;
  std::string currentPath = fs::current_path().c_str();
  size_t pos = 0;

  if ((pos = currentPath.find("project")) != std::string::npos) {
    // In Github test, the Velox home is /root/project.
    veloxPath = currentPath.substr(0, pos) + "project";
  } else if ((pos = currentPath.find("velox")) != std::string::npos) {
    veloxPath = currentPath.substr(0, pos) + "velox";
  } else if ((pos = currentPath.find("fbcode")) != std::string::npos) {
    veloxPath = currentPath;
  } else {
    throw std::runtime_error("Current path is not a valid Velox path.");
  }

  // Find and deserialize Substrait plan json file.
  std::string subPlanPath =
      veloxPath + "/velox/substrait/tests/substrait_virtualTable.json";

  // Read json and resume the Substrait plan.
  std::ifstream subJson(subPlanPath);
  std::stringstream buffer;
  buffer << subJson.rdbuf();
  std::string subData = buffer.str();
  ::substrait::Plan subPlan;
  google::protobuf::util::JsonStringToMessage(subData, &subPlan);

  auto veloxPlan = planConverter->toVeloxPlan(subPlan, pool_.get());

  assertPlanConversion(veloxPlan, "SELECT * FROM tmp");
}