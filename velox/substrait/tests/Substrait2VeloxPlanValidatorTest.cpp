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

#include "velox/substrait/tests/JsonToProtoConverter.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/substrait/SubstraitToVeloxPlanValidator.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::exec;
namespace vestrait = facebook::velox::substrait;

class Substrait2VeloxPlanConversionTest
    : public exec::test::HiveConnectorTestBase {
 protected:

  std::shared_ptr<vestrait::SubstraitVeloxPlanConverter> planConverter_ =
      std::make_shared<vestrait::SubstraitVeloxPlanConverter>(
          memoryPool_.get());

  bool validatePlan(std::string file) {
    std::string subPlanPath = getDataFilePath("velox/substrait/tests", file);

    ::substrait::Plan substraitPlan;
    JsonToProtoConverter::readFromFile(subPlanPath, substraitPlan);
    return validatePlan(substraitPlan);
  }

  bool validatePlan(::substrait::Plan& plan) {
    std::shared_ptr<core::QueryCtx> queryCtx =
        std::make_shared<core::QueryCtx>();

    // An execution context used for function validation.
    std::unique_ptr<core::ExecCtx> execCtx =
        std::make_unique<core::ExecCtx>(pool_.get(), queryCtx.get());

    auto planValidator = std::make_shared<
        facebook::velox::substrait::SubstraitToVeloxPlanValidator>(
        pool_.get(), execCtx.get());
    return planValidator->validate(plan);
  }

 private:
  std::shared_ptr<memory::MemoryPool> memoryPool_{
      memory::getDefaultMemoryPool()};
};

TEST_F(Substrait2VeloxPlanConversionTest, group) {
  std::string subPlanPath =
      getDataFilePath("velox/substrait/tests", "group.json");

  ::substrait::Plan substraitPlan;
  JsonToProtoConverter::readFromFile(subPlanPath, substraitPlan);

  ASSERT_FALSE(validatePlan(substraitPlan));
  // Convert to Velox PlanNode.
  facebook::velox::substrait::SubstraitVeloxPlanConverter planConverter(
      pool_.get());
  EXPECT_ANY_THROW(planConverter.toVeloxPlan(substraitPlan));
}
