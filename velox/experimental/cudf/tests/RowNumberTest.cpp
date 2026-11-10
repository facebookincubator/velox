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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <limits>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

class CudfRowNumberTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Runs 'plan' on the GPU with one driver and one GPU batch per input batch,
  // and checks that it ran as CudfRowNumber and returned exactly the CPU
  // result, in the same row order.
  std::shared_ptr<Task> assertGpuMatchesCpu(const core::PlanNodePtr& plan) {
    auto expected = AssertQueryBuilder(plan)
                        .config(cudf_velox::CudfConfig::kCudfEnabled, false)
                        .copyResults(pool());
    std::shared_ptr<Task> task;
    auto actual = AssertQueryBuilder(plan)
                      .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1")
                      .maxDrivers(1)
                      .copyResults(pool(), task);
    facebook::velox::test::assertEqualVectors(expected, actual);
    const auto stats = toOperatorStats(task->taskStats());
    EXPECT_EQ(stats.count("CudfRowNumber"), 1);
    EXPECT_EQ(stats.count("RowNumber"), 0);
    return task;
  }
};

} // namespace

TEST_F(CudfRowNumberTest, partitionedWithNullKeys) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 1}),
       makeFlatVector<int64_t>({0, 1, 2, 3})});
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({std::nullopt, 2, 3, 1}),
       makeFlatVector<int64_t>({4, 5, 6, 7})});
  assertGpuMatchesCpu(
      PlanBuilder().values({batch1, batch2}).rowNumber({"p"}).planNode());
}

TEST_F(CudfRowNumberTest, mixedGroupSizesAcrossBatches) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>(
           {30, 10, 30, std::nullopt, 10, 30, std::nullopt}),
       makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6})});
  // New groups sort between old groups. Unequal sizes and prior counts must
  // align with group starts, including the null-key group's COUNT_ALL.
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>(
           {20, 30, std::nullopt, 20, 10, 20, 40, 30, 20, std::nullopt}),
       makeFlatVector<int64_t>({7, 8, 9, 10, 11, 12, 13, 14, 15, 16})});
  // Only seen groups: update counts without rebuilding the state keys/join.
  auto batch3 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({40, 10, 20, 40, 30, std::nullopt}),
       makeFlatVector<int64_t>({17, 18, 19, 20, 21, 22})});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch1, batch2, batch3})
          .rowNumber({"p"})
          .planNode());
}

TEST_F(CudfRowNumberTest, singlePartitionAcrossBatches) {
  auto singleton = makeRowVector(
      {"p", "v"}, {makeFlatVector<int32_t>({7}), makeFlatVector<int64_t>({0})});
  auto batch = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({7, 7, 7}), makeFlatVector<int64_t>({1, 2, 3})});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({singleton, batch, singleton})
          .rowNumber({"p"})
          .planNode());
}

TEST_F(CudfRowNumberTest, multipleKeysWithVarchar) {
  auto batch1 = makeRowVector(
      {"s", "k", "v"},
      {makeNullableFlatVector<std::string>({"a", "b", std::nullopt, "a"}),
       makeFlatVector<int64_t>({1, 1, 2, 1}),
       makeFlatVector<int64_t>({0, 1, 2, 3})});
  auto batch2 = makeRowVector(
      {"s", "k", "v"},
      {makeNullableFlatVector<std::string>({"a", std::nullopt, "b", "a"}),
       makeFlatVector<int64_t>({2, 2, 1, 1}),
       makeFlatVector<int64_t>({4, 5, 6, 7})});
  assertGpuMatchesCpu(
      PlanBuilder().values({batch1, batch2}).rowNumber({"s", "k"}).planNode());
}

TEST_F(CudfRowNumberTest, noKeys) {
  auto batch = makeRowVector({"v"}, {makeFlatVector<int64_t>({0, 1, 2})});
  assertGpuMatchesCpu(
      PlanBuilder().values({batch, batch, batch}).rowNumber({}).planNode());
}

TEST_F(CudfRowNumberTest, noKeysZeroColumnInput) {
  // A zero-column cuDF table holds no row count, so the operator must use the
  // vector size.
  auto batch = makeRowVector(ROW({}), 3);
  for (const auto& limit :
       std::vector<std::optional<int32_t>>{std::nullopt, 4}) {
    SCOPED_TRACE(limit.has_value() ? "limit 4" : "no limit");
    assertGpuMatchesCpu(
        PlanBuilder()
            .values({batch, batch, batch})
            .rowNumber({}, limit)
            .planNode());
    assertGpuMatchesCpu(
        PlanBuilder()
            .values({batch, batch, batch})
            .rowNumber({}, limit, false)
            .planNode());
  }
}

TEST_F(CudfRowNumberTest, limitWithKeys) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({1, 2, 1, 1}),
       makeFlatVector<int64_t>({0, 1, 2, 3})});
  // Partition 1 is already at the limit, so this batch produces no output.
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({1, 1}), makeFlatVector<int64_t>({4, 5})});
  auto batch3 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({2, 3, 2, 3, 3}),
       makeFlatVector<int64_t>({6, 7, 8, 9, 10})});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch1, batch2, batch3})
          .rowNumber({"p"}, 2)
          .planNode());
}

TEST_F(CudfRowNumberTest, limitNoKeysFinishesEarly) {
  auto batch = makeRowVector({"v"}, {makeFlatVector<int64_t>({0, 1, 2})});
  auto task = assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch, batch, batch, batch})
          .rowNumber({}, 4)
          .planNode());
  // The limit is reached in the second batch, so the operator stops pulling
  // input before the last two batches.
  const auto stats = toOperatorStats(task->taskStats()).at("CudfRowNumber");
  EXPECT_EQ(stats.outputPositions, 4);
  EXPECT_EQ(stats.inputPositions, 6);
}

TEST_F(CudfRowNumberTest, partialModeWithLimit) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({1, 2, 1, 1}),
       makeFlatVector<int64_t>({0, 1, 2, 3})});
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({2, 2, 1}), makeFlatVector<int64_t>({4, 5, 6})});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch1, batch2})
          .rowNumber({"p"}, 2, false)
          .planNode());
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch1, batch2})
          .rowNumber({}, 5, false)
          .planNode());
}

TEST_F(CudfRowNumberTest, duplicateKeys) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({1, 2, 1}), makeFlatVector<int64_t>({0, 1, 2})});
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({2, 1, 3}), makeFlatVector<int64_t>({3, 4, 5})});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch1, batch2})
          .rowNumber({"p", "v", "p"}, 1)
          .planNode());
  assertGpuMatchesCpu(
      PlanBuilder().values({batch1, batch2}).rowNumber({"p", "p"}).planNode());
}

TEST_F(CudfRowNumberTest, emptyInputAndEmptyBatches) {
  auto empty = makeRowVector(
      {"p", "v"}, {makeFlatVector<int32_t>({}), makeFlatVector<int64_t>({})});
  auto batch = makeRowVector(
      {"p", "v"},
      {makeFlatVector<int32_t>({1, 1, 2}), makeFlatVector<int64_t>({0, 1, 2})});
  assertGpuMatchesCpu(
      PlanBuilder().values({empty}).rowNumber({"p"}).planNode());
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({empty, batch, empty, batch})
          .rowNumber({"p"})
          .planNode());
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({empty, batch, empty, batch})
          .rowNumber({}, 4)
          .planNode());
}

TEST_F(CudfRowNumberTest, manyPartitionsAcrossBatches) {
  // About one new partition per row in every batch, with repeats across
  // batches.
  std::vector<RowVectorPtr> batches;
  for (int32_t batch = 0; batch < 4; ++batch) {
    batches.push_back(makeRowVector(
        {"p", "v"},
        {makeFlatVector<int64_t>(
             1'000, [&](auto row) { return (batch * 1'000 + row) / 3; }),
         makeFlatVector<int64_t>(
             1'000, [&](auto row) { return batch * 1'000 + row; })}));
  }
  assertGpuMatchesCpu(
      PlanBuilder().values(batches).rowNumber({"p"}, 2).planNode());
  assertGpuMatchesCpu(
      PlanBuilder().values(batches).rowNumber({"p"}).planNode());
}

TEST_F(CudfRowNumberTest, floatKeysNanAndSignedZero) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto otherNan = -std::numeric_limits<double>::quiet_NaN();
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<double>({nan, 0.0, -0.0, 1.0}),
       makeFlatVector<int64_t>({0, 1, 2, 3})});
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeFlatVector<double>({-0.0, otherNan, 0.0, nan}),
       makeFlatVector<int64_t>({4, 5, 6, 7})});
  assertGpuMatchesCpu(
      PlanBuilder().values({batch1, batch2}).rowNumber({"p"}).planNode());
}
