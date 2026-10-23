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
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

namespace facebook::velox::cudf_velox::test {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

class CudfPlanNodeTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    CudfConfig::getInstance().allowCpuFallback = true;
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    OperatorTestBase::TearDown();
  }
};

TEST_F(CudfPlanNodeTest, executesExplicitOrderByPlan) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({3, 1, 2}),
       makeFlatVector<std::string>({"c", "a", "b"})});
  auto values = PlanBuilder().values({input}).planNode();
  auto upload = std::make_shared<CudfFromVeloxNode>("upload", values);
  auto orderBy = std::make_shared<CudfOrderByNode>(
      "order-by",
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      std::vector<core::SortOrder>{core::kAscNullsLast},
      false,
      upload);
  auto plan = std::make_shared<CudfToVeloxNode>("download", orderBy);

  std::shared_ptr<exec::Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(CudfConfig::kCudfEnabled, true)
                    .config(CudfConfig::kCudfAllowCpuFallback, false)
                    .maxDrivers(1)
                    .copyResults(pool(), task);
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<std::string>({"a", "b", "c"})});
  facebook::velox::test::assertEqualVectors(expected, result);

  const auto stats = exec::test::toOperatorStats(task->taskStats());
  EXPECT_EQ(stats.count("CudfFromVelox"), 1);
  EXPECT_EQ(stats.count("CudfOrderBy"), 1);
  EXPECT_EQ(stats.count("CudfToVelox"), 1);
  EXPECT_EQ(stats.count("OrderBy"), 0);
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
