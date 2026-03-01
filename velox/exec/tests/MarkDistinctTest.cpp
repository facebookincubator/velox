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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;
using facebook::velox::exec::Operator;
using facebook::velox::exec::TestScopedSpillInjection;
using facebook::velox::memory::testingRunArbitration;

class MarkDistinctTest : public OperatorTestBase {
 public:
  void runBasicTest(const VectorPtr& base) {
    const vector_size_t size = base->size() * 2;
    auto indices = makeIndices(size, [](auto row) { return row / 2; });
    auto data = wrapInDictionary(indices, size, base);

    auto isDistinct =
        makeFlatVector<bool>(size, [&](auto row) { return row % 2 == 0; });

    auto expectedResults = makeRowVector({data, isDistinct});

    auto plan = PlanBuilder()
                    .values({makeRowVector({data})})
                    .markDistinct("c0_distinct", {"c0"})
                    .planNode();

    auto results = AssertQueryBuilder(plan).copyResults(pool());
    assertEqualVectors(expectedResults, results);
  }

 protected:
  MarkDistinctTest() {
    filesystems::registerLocalFileSystem();
  }

  RowTypePtr rowType_{ROW({"c0", "c1"}, {BIGINT(), BIGINT()})};

  VectorFuzzer::Options fuzzerOpts_{
      .vectorSize = 1024,
      .nullRatio = 0,
      .allowLazyVector = false};
};

template <typename T>
class MarkDistinctPODTest : public MarkDistinctTest {};

using MyTypes = ::testing::Types<int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(MarkDistinctPODTest, MyTypes);

TYPED_TEST(MarkDistinctPODTest, basic) {
  auto data = VectorTestBase::makeFlatVector<TypeParam>(
      1'000, [](auto row) { return row; }, [](auto row) { return row == 7; });

  MarkDistinctTest::runBasicTest(data);
}

TEST_F(MarkDistinctTest, tinyint) {
  auto data = makeFlatVector<int8_t>(
      256, [](auto row) { return row; }, [](auto row) { return row == 7; });

  runBasicTest(data);
}

TEST_F(MarkDistinctTest, boolean) {
  auto data = makeNullableFlatVector<bool>({true, false, std::nullopt});

  runBasicTest(data);
}

TEST_F(MarkDistinctTest, varchar) {
  auto base = makeFlatVector<StringView>({
      "{1, 2, 3, 4, 5}",
      "{1, 2, 3}",
  });
  runBasicTest(base);
}

TEST_F(MarkDistinctTest, array) {
  auto base = makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5},
      {1, 2, 3},
  });
  runBasicTest(base);
}

TEST_F(MarkDistinctTest, map) {
  auto base = makeMapVector<int8_t, int32_t>(
      {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {{1, 1}, {1, 1}, {1, 1}}});
  runBasicTest(base);
}

TEST_F(MarkDistinctTest, row) {
  auto base = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3, 4, 5},
          {1, 2, 3},
      }),
      makeMapVector<int8_t, int32_t>({
          {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
          {{1, 1}, {1, 1}, {1, 1}},
      }),
  });
  runBasicTest(base);
}

TEST_F(MarkDistinctTest, aggregation) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatVector<int32_t>({1, 1}),
          makeFlatVector<int32_t>({1, 1}),
          makeFlatVector<int32_t>({1, 2}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({1, 2}),
          makeFlatVector<int32_t>({1, 1}),
          makeFlatVector<int32_t>({1, 2}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({2, 2}),
          makeFlatVector<int32_t>({2, 3}),
          makeFlatVector<int32_t>({1, 2}),
      }),
  };

  createDuckDbTable(vectors);

  auto plan =
      PlanBuilder()
          .values(vectors)
          .markDistinct("c1_distinct", {"c0", "c1"})
          .markDistinct("c2_distinct", {"c0", "c2"})
          .singleAggregation(
              {"c0"}, {"sum(c1)", "sum(c2)"}, {"c1_distinct", "c2_distinct"})
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT c0, sum(distinct c1), sum(distinct c2) FROM tmp GROUP BY 1");
}

TEST_F(MarkDistinctTest, spill) {
  auto vectors = createVectors(8, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    uint32_t spillPartitionBits;
    uint32_t numSpills;
    uint32_t cpuTimeSliceLimitMs;

    std::string debugString() const {
      return fmt::format(
          "spillPartitionBits {}, numSpills {}, cpuTimeSliceLimitMs {}",
          spillPartitionBits,
          numSpills,
          cpuTimeSliceLimitMs);
    }
  } testSettings[] = {{2, 1, 0}, {3, 1, 0}, {2, 2, 0}, {2, 2, 10}, {2, 3, 10}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    TestScopedSpillInjection scopedSpillInjection(
        100, ".*", testData.numSpills);

    core::PlanNodeId markDistinctId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
            .config(
                core::QueryConfig::kSpillNumPartitionBits,
                testData.spillPartitionBits)
            .config(
                core::QueryConfig::kDriverCpuTimeSliceLimitMs,
                testData.cpuTimeSliceLimitMs)
            .config(core::QueryConfig::kAggregationSpillEnabled, false)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .markDistinct("c1_distinct", {"c0", "c1"})
                    .capturePlanNodeId(markDistinctId)
                    .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                    .planNode())
            .assertResults("SELECT c0, count(distinct c1) FROM tmp GROUP BY 1");

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(markDistinctId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);
    ASSERT_GT(planStats.spilledPartitions, 0);

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(MarkDistinctTest, reclaimDuringInputOrOutput) {
  auto vectors = createVectors(8, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    std::string spillInjectionPoint;
    uint32_t spillPartitionBits;

    std::string debugString() const {
      return fmt::format(
          "spillInjectionPoint {}, spillPartitionBits {}",
          spillInjectionPoint,
          spillPartitionBits);
    }
  } testSettings[] = {
      {"facebook::velox::exec::Driver::runInternal::addInput", 2},
      {"facebook::velox::exec::Driver::runInternal::getOutput", 2},
      {"facebook::velox::exec::Driver::runInternal::addInput", 3},
      {"facebook::velox::exec::Driver::runInternal::getOutput", 3}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());

    std::atomic_int numRound{0};
    SCOPED_TESTVALUE_SET(
        testData.spillInjectionPoint,
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "MarkDistinct") {
            return;
          }
          if (++numRound != 5) {
            return;
          }
          testingRunArbitration(op->pool(), 0);
        })));

    core::PlanNodeId markDistinctId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
            .config(
                core::QueryConfig::kSpillNumPartitionBits,
                testData.spillPartitionBits)
            .config(core::QueryConfig::kAggregationSpillEnabled, false)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .markDistinct("c1_distinct", {"c0", "c1"})
                    .capturePlanNodeId(markDistinctId)
                    .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                    .planNode())
            .assertResults("SELECT c0, count(distinct c1) FROM tmp GROUP BY 1");

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(markDistinctId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);
    ASSERT_GT(planStats.spilledPartitions, 0);

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(MarkDistinctTest, recursiveSpill) {
  auto vectors = createVectors(32, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    int32_t numSpills;
    int32_t maxSpillLevel;

    std::string debugString() const {
      return fmt::format(
          "numSpills {}, maxSpillLevel {}", numSpills, maxSpillLevel);
    }
  } testSettings[] = {{2, 3}, {8, 4}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());

    std::atomic_int numSpills{0};
    std::atomic_int numInputs{0};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "MarkDistinct") {
            return;
          }
          if (++numInputs != 5) {
            return;
          }
          ++numSpills;
          testingRunArbitration(op->pool(), 0);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::getOutput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "MarkDistinct") {
            return;
          }
          if (!op->testingNoMoreInput()) {
            return;
          }
          if (numSpills++ >= testData.numSpills) {
            return;
          }
          testingRunArbitration(op->pool(), 0);
        })));

    core::PlanNodeId markDistinctId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
            .config(
                core::QueryConfig::kMaxSpillLevel, testData.maxSpillLevel - 1)
            .config(core::QueryConfig::kAggregationSpillEnabled, false)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .markDistinct("c1_distinct", {"c0", "c1"})
                    .capturePlanNodeId(markDistinctId)
                    .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                    .planNode())
            .assertResults("SELECT c0, count(distinct c1) FROM tmp GROUP BY 1");

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(markDistinctId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);

    auto runTimeStats =
        task->taskStats().pipelineStats.back().operatorStats.at(1).runtimeStats;
    if (testData.numSpills > testData.maxSpillLevel) {
      ASSERT_GT(runTimeStats["exceededMaxSpillLevel"].sum, 0);
    } else {
      ASSERT_EQ(runTimeStats.count("exceededMaxSpillLevel"), 0);
    }

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

TEST_F(MarkDistinctTest, memoryUsage) {
  const auto rowType =
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), VARCHAR()});
  const auto vectors = createVectors(rowType, 1024, 100 << 20);

  core::PlanNodeId markDistinctId;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .markDistinct("c1_distinct", {"c0", "c1"})
                  .capturePlanNodeId(markDistinctId)
                  .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                  .planNode();

  struct {
    uint8_t numSpills;

    std::string debugString() const {
      return fmt::format("numSpills {}", numSpills);
    }
  } testSettings[] = {{1}, {3}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    int64_t peakBytesWithSpilling = 0;
    int64_t peakBytesWithOutSpilling = 0;

    for (const auto& spillEnable : {false, true}) {
      auto queryCtx = core::QueryCtx::create(executor_.get());
      auto spillDirectory = exec::test::TempDirectoryPath::create();
      const std::string spillEnableConfig = std::to_string(spillEnable);

      std::shared_ptr<exec::Task> task;
      TestScopedSpillInjection scopedSpillInjection(
          100, ".*", testData.numSpills);
      AssertQueryBuilder(plan)
          .spillDirectory(spillDirectory->getPath())
          .queryCtx(queryCtx)
          .config(core::QueryConfig::kSpillEnabled, spillEnableConfig)
          .config(
              core::QueryConfig::kMarkDistinctSpillEnabled, spillEnableConfig)
          .config(core::QueryConfig::kAggregationSpillEnabled, "false")
          .copyResults(pool_.get(), task);

      if (spillEnable) {
        peakBytesWithSpilling = queryCtx->pool()->peakBytes();
        auto taskStats = exec::toPlanStats(task->taskStats());
        const auto& stats = taskStats.at(markDistinctId);

        ASSERT_GT(stats.spilledBytes, 0);
        ASSERT_GT(stats.spilledRows, 0);
        ASSERT_GT(stats.spilledFiles, 0);
        ASSERT_GT(stats.spilledPartitions, 0);
      } else {
        peakBytesWithOutSpilling = queryCtx->pool()->peakBytes();
      }
    }

    ASSERT_GT(peakBytesWithOutSpilling, peakBytesWithSpilling);
  }
}

TEST_F(MarkDistinctTest, maxSpillBytes) {
  auto rowType = ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), VARCHAR()});
  auto vectors = createVectors(rowType, 1024, 15 << 20);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .markDistinct("c1_distinct", {"c0", "c1"})
                  .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                  .planNode();

  struct {
    int32_t maxSpilledBytes;
    bool expectedExceedLimit;

    std::string debugString() const {
      return fmt::format("maxSpilledBytes {}", maxSpilledBytes);
    }
  } testSettings[] = {{1 << 30, false}, {1 << 20, true}, {0, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    try {
      TestScopedSpillInjection scopedSpillInjection(100, ".*", 1);
      AssertQueryBuilder(plan)
          .spillDirectory(spillDirectory->getPath())
          .queryCtx(queryCtx)
          .config(core::QueryConfig::kSpillEnabled, true)
          .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
          .config(core::QueryConfig::kAggregationSpillEnabled, false)
          .config(core::QueryConfig::kMaxSpillBytes, testData.maxSpilledBytes)
          .copyResults(pool_.get());
      ASSERT_FALSE(testData.expectedExceedLimit);
    } catch (const VeloxRuntimeError& e) {
      ASSERT_TRUE(testData.expectedExceedLimit);
      ASSERT_NE(
          e.message().find(
              "Query exceeded per-query local spill limit of 1.00MB"),
          std::string::npos);
      ASSERT_EQ(
          e.errorCode(), facebook::velox::error_code::kSpillLimitExceeded);
    }
  }
}
