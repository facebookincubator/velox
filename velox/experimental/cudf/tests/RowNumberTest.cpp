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
#include "velox/experimental/cudf/exec/CudfRowNumber.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Driver.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/BackpressureTestNode.h"
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
    std::shared_ptr<Task> expectedTask;
    auto expected =
        AssertQueryBuilder(plan)
            .config(cudf_velox::CudfConfig::kCudfEnabled, false)
            .config(cudf_velox::CudfConfig::kCudfAllowCpuFallback, true)
            .maxDrivers(1)
            .copyResults(pool(), expectedTask);
    const auto stats = toOperatorStats(expectedTask->taskStats());
    EXPECT_EQ(stats.count("RowNumber"), 1);
    EXPECT_EQ(stats.count("CudfRowNumber"), 0);
    return assertGpuResults(plan, expected);
  }

  std::shared_ptr<Task> assertGpuResults(
      const core::PlanNodePtr& plan,
      const RowVectorPtr& expected) {
    std::shared_ptr<Task> task;
    auto actual =
        AssertQueryBuilder(plan)
            .config(cudf_velox::CudfConfig::kCudfEnabled, true)
            .config(cudf_velox::CudfConfig::kCudfAllowCpuFallback, false)
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

TEST_F(CudfRowNumberTest, newPartitionsAcrossBatches) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({2, std::nullopt, 2}),
       makeFlatVector<int64_t>({0, 1, 2})});
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({3, 1, 3}),
       makeFlatVector<int64_t>({3, 4, 5})});
  auto batch3 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({1, 2, 3, std::nullopt}),
       makeFlatVector<int64_t>({6, 7, 8, 9})});
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
      {makeFlatVector<int32_t>(10'000, [](auto /*row*/) { return 7; }),
       makeFlatVector<int64_t>(10'000, [](auto row) { return row + 1; })});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({singleton, batch, batch, singleton})
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
  for (bool generateRowNumber : {true, false}) {
    SCOPED_TRACE(generateRowNumber);
    assertGpuMatchesCpu(
        PlanBuilder()
            .values({batch, batch, batch})
            .rowNumber({}, std::nullopt, generateRowNumber)
            .planNode());
  }
}

TEST_F(CudfRowNumberTest, noKeysZeroColumnInput) {
  // A zero-column cuDF table holds no row count, so the operator must use the
  // vector size.
  auto batch = makeRowVector(ROW({}), 3);
  for (const auto& limit :
       std::vector<std::optional<int32_t>>{std::nullopt, 4}) {
    SCOPED_TRACE(limit.has_value() ? "limit 4" : "no limit");
    for (bool generateRowNumber : {true, false}) {
      SCOPED_TRACE(generateRowNumber);
      auto plan = PlanBuilder()
                      .values({batch, batch, batch})
                      .rowNumber({}, limit, generateRowNumber)
                      .planNode();
      // CPU RowNumber does not advance its no-key counter without generating
      // numbers. Use the intended cardinality for a limit spanning batches.
      auto task = !generateRowNumber && limit
          ? assertGpuResults(plan, makeRowVector(ROW({}), 4))
          : assertGpuMatchesCpu(plan);
      EXPECT_EQ(
          toOperatorStats(task->taskStats())
              .at("CudfRowNumber")
              .outputPositions,
          limit ? 4 : 9);
    }
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
  for (const auto limit : {1, 2}) {
    SCOPED_TRACE(limit);
    for (bool generateRowNumber : {true, false}) {
      SCOPED_TRACE(generateRowNumber);
      assertGpuMatchesCpu(
          PlanBuilder()
              .values({batch1, batch2, batch3})
              .rowNumber({"p"}, limit, generateRowNumber)
              .planNode());
    }
  }
}

TEST_F(CudfRowNumberTest, limitNoKeysFinishesEarly) {
  auto batch = makeRowVector({"v"}, {makeFlatVector<int64_t>({0, 1, 2})});
  for (const auto limit : {4, 6}) {
    SCOPED_TRACE(limit);
    for (bool generateRowNumber : {true, false}) {
      SCOPED_TRACE(generateRowNumber);
      auto plan = PlanBuilder()
                      .values({batch, batch, batch, batch})
                      .rowNumber({}, limit, generateRowNumber)
                      .planNode();
      // CPU RowNumber's no-key counter advances only when generating numbers.
      // Independently check the correct partial-mode limit across batches.
      auto task = generateRowNumber
          ? assertGpuMatchesCpu(plan)
          : assertGpuResults(
                plan,
                makeRowVector(
                    {"v"}, {makeFlatVector<int64_t>(limit, [](auto row) {
                      return row % 3;
                    })}));
      // Stop in or at the end of the second batch, leaving two unread batches.
      const auto stats = toOperatorStats(task->taskStats()).at("CudfRowNumber");
      EXPECT_EQ(stats.outputPositions, limit);
      EXPECT_EQ(stats.inputPositions, 6);
    }
  }
}

TEST_F(CudfRowNumberTest, partialModeWithoutLimit) {
  auto batch1 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({2, std::nullopt, 1, 2}),
       makeFlatVector<int64_t>({0, 1, 2, 3})});
  auto batch2 = makeRowVector(
      {"p", "v"},
      {makeNullableFlatVector<int32_t>({1, 3, std::nullopt, 2}),
       makeFlatVector<int64_t>({4, 5, 6, 7})});
  assertGpuMatchesCpu(
      PlanBuilder()
          .values({batch1, batch2})
          .rowNumber({"p"}, std::nullopt, false)
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

TEST_F(CudfRowNumberTest, emptyInput) {
  auto empty = makeRowVector(
      {"p", "v"}, {makeFlatVector<int32_t>({}), makeFlatVector<int64_t>({})});
  assertGpuMatchesCpu(
      PlanBuilder().values({empty}).rowNumber({"p"}).planNode());
}

TEST_F(CudfRowNumberTest, noRowLossWithBackpressuringDownstream) {
  using namespace facebook::velox::cudf_velox;
  constexpr int32_t kNumBatches = 5;
  constexpr int32_t kRowsPerBatch = 100;
  std::vector<RowVectorPtr> batches;
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    batches.push_back(makeRowVector(
        {"p", "v"},
        {makeFlatVector<int64_t>(
             kRowsPerBatch,
             [&](auto row) { return (batch * kRowsPerBatch + row) % 7; }),
         makeFlatVector<int64_t>(kRowsPerBatch, [&](auto row) {
           return batch * kRowsPerBatch + row;
         })}));
  }
  auto plan = PlanBuilder().values(batches).rowNumber({"p"}).planNode();
  auto downstream = std::make_shared<BackpressureNode>("2", 3, plan);
  core::PlanFragment fragment;
  fragment.planNode = downstream;
  auto task = Task::create(
      "row-number-backpressure",
      std::move(fragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfRowNumber rowNumber(
      0,
      &driverCtx,
      std::dynamic_pointer_cast<const core::RowNumberNode>(plan));
  BackpressureOperator consumer(1, &driverCtx, downstream);
  EXPECT_EQ(rowNumber.operatorType(), "CudfRowNumber");

  int32_t numOutputBatches = 0;
  auto drainConsumer = [&]() {
    auto output = std::dynamic_pointer_cast<CudfVector>(consumer.getOutput());
    ASSERT_NE(output, nullptr);
    auto actual = with_arrow::toVeloxColumn(
        output->getTableView(),
        pool(),
        plan->outputType(),
        output->stream(),
        get_output_mr());
    auto expected = makeRowVector(
        {"p", "v", "row_number"},
        {batches[numOutputBatches]->childAt(0),
         batches[numOutputBatches]->childAt(1),
         makeFlatVector<int64_t>(kRowsPerBatch, [&](auto row) {
           return (numOutputBatches * kRowsPerBatch + row) / 7 + 1;
         })});
    facebook::velox::test::assertEqualVectors(expected, actual);
    ++numOutputBatches;
  };
  for (const auto& batch : batches) {
    ASSERT_TRUE(rowNumber.needsInput());
    auto stream = cudfGlobalStreamPool().get_stream();
    rowNumber.addInput(
        std::make_shared<CudfVector>(
            pool(),
            batch->type(),
            batch->size(),
            with_arrow::toCudfTable(batch, pool(), stream, get_output_mr()),
            stream));
    if (!consumer.needsInput()) {
      // Direct GPU-vector handoff: no always-accepting CudfToVelox queue can
      // mask refusal. Keep RowNumber's next batch pending for all three cycles.
      for (int32_t cycle = 0; cycle < 3; ++cycle) {
        EXPECT_FALSE(consumer.needsInput());
        EXPECT_FALSE(rowNumber.needsInput());
        EXPECT_EQ(consumer.getOutput(), nullptr);
      }
      drainConsumer();
    }
    ASSERT_TRUE(consumer.needsInput());
    consumer.addInput(rowNumber.getOutput());
  }
  rowNumber.noMoreInput();
  for (int32_t cycle = 0; cycle < 3; ++cycle) {
    EXPECT_EQ(consumer.getOutput(), nullptr);
  }
  drainConsumer();
  consumer.noMoreInput();
  EXPECT_TRUE(rowNumber.isFinished());
  EXPECT_TRUE(consumer.isFinished());
  EXPECT_EQ(numOutputBatches, kNumBatches);
  consumer.close();
  rowNumber.close();
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
