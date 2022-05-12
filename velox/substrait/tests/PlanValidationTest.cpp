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

#include "TestUtils.h"

#include "velox/common/base/tests/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/substrait/SubstraitToVeloxPlanValidator.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
namespace vestrait = facebook::velox::substrait;

class PlanValidationTest : public ::testing::Test {
 protected:
  std::shared_ptr<core::QueryCtx> queryCtx_ = core::QueryCtx::createForTest();

  std::unique_ptr<memory::ScopedMemoryPool> pool_ =
      memory::getDefaultScopedMemoryPool();

  std::shared_ptr<TestUtils> testUtils_ = std::make_shared<TestUtils>();
};

// This test will validate the Substrait plan of Read.
TEST_F(PlanValidationTest, read) {
  core::ExecCtx execCtx{pool_.get(), queryCtx_.get()};
  auto planValidator =
      std::make_shared<vestrait::SubstraitToVeloxPlanValidator>(&execCtx);
  std::string subPlanPath =
      getDataFilePath("velox/substrait/tests", "data/read_validation.json");
  ::substrait::Plan subPlan;
  testUtils_->getMsg(subPlanPath, subPlan);
  bool validated = planValidator->validate(subPlan);
  ASSERT_EQ(validated, true);
}

// This test will validate the Substrait plan of Project.
TEST_F(PlanValidationTest, project) {
  core::ExecCtx execCtx{pool_.get(), queryCtx_.get()};
  auto planValidator =
      std::make_shared<vestrait::SubstraitToVeloxPlanValidator>(&execCtx);
  std::string subPlanPath =
      getDataFilePath("velox/substrait/tests", "data/project_validation.json");
  ::substrait::Plan subPlan;
  testUtils_->getMsg(subPlanPath, subPlan);
  bool validated = planValidator->validate(subPlan);
  ASSERT_EQ(validated, true);
}

// This test will validate the Substrait plan of Aggregate.
TEST_F(PlanValidationTest, aggregate) {
  core::ExecCtx execCtx{pool_.get(), queryCtx_.get()};
  auto planValidator =
      std::make_shared<vestrait::SubstraitToVeloxPlanValidator>(&execCtx);
  std::string subPlanPath =
      getDataFilePath("velox/substrait/tests", "data/agg_validation.json");
  ::substrait::Plan subPlan;
  testUtils_->getMsg(subPlanPath, subPlan);
  bool validated = planValidator->validate(subPlan);
  ASSERT_EQ(validated, true);
}
