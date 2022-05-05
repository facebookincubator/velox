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
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "velox/substrait/SubstraitToVeloxPlan.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::substrait;

class Substrait2VeloxValuesNodeConversionTest : public OperatorTestBase {
 public:
  void assertPlanConversion(
      const PlanNodePtr& plan,
      const std::string& duckDbSql) {
    const auto& valuesNode =
        std::dynamic_pointer_cast<const core::ValuesNode>(plan);
    ASSERT_TRUE(valuesNode != nullptr);
    const auto& vectors = valuesNode->values();

    createDuckDbTable(vectors);
    assertQuery(plan, duckDbSql);
  }

  void parseJson(const std::string& filePath, ::substrait::Plan* subPlan) {
    // Read json and resume the Substrait plan.
    std::ifstream subJson(filePath);
    std::stringstream buffer;
    buffer << subJson.rdbuf();
    std::string subData = buffer.str();

    google::protobuf::util::JsonStringToMessage(subData, subPlan);
  };

  std::shared_ptr<SubstraitVeloxPlanConverter> planConverter_ =
      std::make_shared<SubstraitVeloxPlanConverter>();

  std::unique_ptr<memory::ScopedMemoryPool> pool_{
      memory::getDefaultScopedMemoryPool()};
};

// SELECT * FROM tmp
TEST_F(Substrait2VeloxValuesNodeConversionTest, valuesNode) {
  auto subPlanPath = getDataFilePath(
      "velox/substrait/tests", "data/substrait_virtualTable.json");

  std::shared_ptr<::substrait::Plan> subPlan =
      std::make_shared<::substrait::Plan>();

  parseJson(subPlanPath, subPlan.get());

  auto veloxPlan = planConverter_->toVeloxPlan(*subPlan, pool_.get());

  assertPlanConversion(veloxPlan, "SELECT * FROM tmp");
}