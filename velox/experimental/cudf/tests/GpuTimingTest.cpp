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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class GpuTimingTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::CudfConfig::getInstance().gpuTimingEnabled = true;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::CudfConfig::getInstance().gpuTimingEnabled = false;
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }
};

TEST_F(GpuTimingTest, projectReportsGpuWallTime) {
  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<int32_t>(10000, [](auto row) { return row; })});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 * c0 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& opStats = stats.at(projNodeId);

  ASSERT_TRUE(opStats.customStats.count("gpuWallTimeNanos"));
  auto& gpuStat = opStats.customStats.at("gpuWallTimeNanos");
  EXPECT_GT(gpuStat.count, 0);
  EXPECT_GT(gpuStat.sum, 1000L);
  EXPECT_LT(gpuStat.sum, 10'000'000'000L);
}

TEST_F(GpuTimingTest, disabledNoGpuWallTime) {
  cudf_velox::CudfConfig::getInstance().gpuTimingEnabled = false;

  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<int32_t>(100, [](auto row) { return row; })});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 + 1 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& opStats = stats.at(projNodeId);

  EXPECT_EQ(opStats.customStats.count("gpuWallTimeNanos"), 0);
}

TEST_F(GpuTimingTest, multipleBatchesAccumulate) {
  std::vector<RowVectorPtr> batches;
  for (int i = 0; i < 10; i++) {
    batches.push_back(makeRowVector(
        {"c0"}, {makeFlatVector<int32_t>(100, [](auto row) { return row; })}));
  }

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values(batches)
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& opStats = stats.at(projNodeId);

  ASSERT_TRUE(opStats.customStats.count("gpuWallTimeNanos"));
  auto& gpuStat = opStats.customStats.at("gpuWallTimeNanos");
  EXPECT_GT(gpuStat.count, 1);
  EXPECT_GT(gpuStat.sum, 0);
}
