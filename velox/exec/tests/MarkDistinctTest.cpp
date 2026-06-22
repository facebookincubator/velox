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

#include <gmock/gmock.h>

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

namespace {
// Returns the id of the first MarkDistinctNode in 'plan', or std::nullopt if
// none exists in the subtree.
std::optional<core::PlanNodeId> findMarkDistinctId(
    const core::PlanNodePtr& plan) {
  if (auto* node = core::PlanNode::findFirstNode(plan.get(), [](const auto* n) {
        return dynamic_cast<const core::MarkDistinctNode*>(n) != nullptr;
      })) {
    return node->id();
  }
  return std::nullopt;
}
} // namespace

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

  // Expects a 3-column input (distinct key + 2 boolean masks) and a 3-column
  // 'expectedMarkers' (no-mask marker + 2 per-mask markers). Runs MarkDistinct
  // and asserts the output matches input columns followed by the expected
  // markers.
  void assertMarkers(
      const RowVectorPtr& input,
      const std::vector<VectorPtr>& expectedMarkers) {
    VELOX_CHECK_EQ(expectedMarkers.size(), 3);
    auto expectedColumns = input->children();
    expectedColumns.insert(
        expectedColumns.end(), expectedMarkers.begin(), expectedMarkers.end());
    auto expected = makeRowVector(expectedColumns);

    auto plan = PlanBuilder()
                    .values({input})
                    .markDistinct({"nomask", "m0", "m1"}, {"c0"}, {"c1", "c2"})
                    .planNode();

    AssertQueryBuilder(plan).assertResults(expected);
  }

  // End-to-end multi-mask query shape for aggregation and spill tests:
  // 2 keys + 2 boolean masks → 3 markers (nomask + 2 per-mask). Returns {plan,
  // duckDbSql}.
  std::pair<core::PlanNodePtr, std::string> makeMultiMaskCase(
      const std::vector<RowVectorPtr>& vectors) {
    auto plan =
        PlanBuilder()
            .values(vectors)
            .markDistinct({"nomask", "m0", "m1"}, {"c0", "c1"}, {"c2", "c3"})
            .singleAggregation(
                {"c0"},
                {"count(c1)", "count(c1)", "count(c1)"},
                {"nomask", "m0", "m1"})
            .planNode();
    return {
        std::move(plan),
        "SELECT c0, "
        "count(DISTINCT c1), "
        "count(DISTINCT c1) FILTER (WHERE c2), "
        "count(DISTINCT c1) FILTER (WHERE c3) "
        "FROM tmp GROUP BY 1"};
  }

  // Runs 'plan' that contains a MarkDistinct node with spilling enabled
  // and validates the result against DuckDB. Asserts that the MarkDistinct
  // node actually spilled. 'configure' lets callers add per-test config
  // overrides on the AssertQueryBuilder.
  void runSpillTest(
      const core::PlanNodePtr& plan,
      const std::string& duckDbSql,
      uint32_t numSpills = 1,
      std::function<void(AssertQueryBuilder&)> configure = nullptr) {
    const auto markDistinctId = findMarkDistinctId(plan);
    VELOX_CHECK(markDistinctId.has_value(), "No MarkDistinct node in plan");

    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    TestScopedSpillInjection scopedSpillInjection(100, ".*", numSpills);

    AssertQueryBuilder builder(duckDbQueryRunner_);
    builder.spillDirectory(spillDirectory->getPath())
        .config(core::QueryConfig::kSpillEnabled, true)
        .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
        .config(core::QueryConfig::kAggregationSpillEnabled, false)
        .queryCtx(queryCtx)
        .plan(plan);
    if (configure) {
      configure(builder);
    }
    auto task = builder.assertResults(duckDbSql);

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(*markDistinctId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledRows, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledPartitions, 0);

    task.reset();
    waitForAllTasksToBeDeleted();
  }

 protected:
  MarkDistinctTest() {
    filesystems::registerLocalFileSystem();
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3"},
          {BIGINT(), BIGINT(), BOOLEAN(), BOOLEAN()})};

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
  // Simulate the input over 3 splits.
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
    auto configure = [&](AssertQueryBuilder& builder) {
      builder
          .config(
              core::QueryConfig::kSpillNumPartitionBits,
              testData.spillPartitionBits)
          .config(
              core::QueryConfig::kDriverCpuTimeSliceLimitMs,
              testData.cpuTimeSliceLimitMs);
    };

    {
      SCOPED_TRACE("single-marker");
      auto vectors = createVectors(8, rowType_, fuzzerOpts_);
      createDuckDbTable(vectors);
      auto plan = PlanBuilder()
                      .values(vectors)
                      .markDistinct("c1_distinct", {"c0", "c1"})
                      .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                      .planNode();
      runSpillTest(
          plan,
          "SELECT c0, count(distinct c1) FROM tmp GROUP BY 1",
          testData.numSpills,
          configure);
    }
    {
      SCOPED_TRACE("multi-mask");
      auto vectors = createVectors(8, rowType_, fuzzerOpts_);
      createDuckDbTable(vectors);
      auto [plan, sql] = makeMultiMaskCase(vectors);
      runSpillTest(plan, sql, testData.numSpills, configure);
    }
  }
}

DEBUG_ONLY_TEST_F(MarkDistinctTest, reclaimDuringInputOrOutput) {
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

  auto runShape = [&](const std::vector<RowVectorPtr>& vectors,
                      const core::PlanNodePtr& plan,
                      const std::string& duckDbSql,
                      const std::string& spillInjectionPoint,
                      uint32_t spillPartitionBits) {
    createDuckDbTable(vectors);
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());

    std::atomic_int numRound{0};
    SCOPED_TESTVALUE_SET(
        spillInjectionPoint, std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "MarkDistinct") {
            return;
          }
          if (++numRound != 5) {
            return;
          }
          testingRunArbitration(op->pool(), 0);
        })));

    const auto markDistinctId = findMarkDistinctId(plan);
    VELOX_CHECK(markDistinctId.has_value(), "No MarkDistinct node in plan");
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
            .config(
                core::QueryConfig::kSpillNumPartitionBits, spillPartitionBits)
            .config(core::QueryConfig::kAggregationSpillEnabled, false)
            .queryCtx(queryCtx)
            .plan(plan)
            .assertResults(duckDbSql);

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(*markDistinctId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);
    ASSERT_GT(planStats.spilledPartitions, 0);

    task.reset();
    waitForAllTasksToBeDeleted();
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    {
      SCOPED_TRACE("single-marker");
      auto vectors = createVectors(8, rowType_, fuzzerOpts_);
      auto plan = PlanBuilder()
                      .values(vectors)
                      .markDistinct("c1_distinct", {"c0", "c1"})
                      .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                      .planNode();
      runShape(
          vectors,
          plan,
          "SELECT c0, count(distinct c1) FROM tmp GROUP BY 1",
          testData.spillInjectionPoint,
          testData.spillPartitionBits);
    }
    {
      SCOPED_TRACE("multi-mask");
      auto vectors = createVectors(8, rowType_, fuzzerOpts_);
      auto [plan, sql] = makeMultiMaskCase(vectors);
      runShape(
          vectors,
          plan,
          sql,
          testData.spillInjectionPoint,
          testData.spillPartitionBits);
    }
  }
}

DEBUG_ONLY_TEST_F(MarkDistinctTest, recursiveSpill) {
  struct {
    int32_t numSpills;
    int32_t maxSpillLevel;

    std::string debugString() const {
      return fmt::format(
          "numSpills {}, maxSpillLevel {}", numSpills, maxSpillLevel);
    }
  } testSettings[] = {{2, 3}, {8, 4}};

  auto runShape = [&](const std::vector<RowVectorPtr>& vectors,
                      const core::PlanNodePtr& plan,
                      const std::string& duckDbSql,
                      int32_t targetNumSpills,
                      int32_t maxSpillLevel) {
    createDuckDbTable(vectors);
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
          if (numSpills++ >= targetNumSpills) {
            return;
          }
          testingRunArbitration(op->pool(), 0);
        })));

    const auto markDistinctId = findMarkDistinctId(plan);
    VELOX_CHECK(markDistinctId.has_value(), "No MarkDistinct node in plan");
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kMarkDistinctSpillEnabled, true)
            .config(core::QueryConfig::kMaxSpillLevel, maxSpillLevel - 1)
            .config(core::QueryConfig::kAggregationSpillEnabled, false)
            .queryCtx(queryCtx)
            .plan(plan)
            .assertResults(duckDbSql);

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(*markDistinctId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);
    ASSERT_GT(planStats.spilledPartitions, 0);

    auto runTimeStats =
        task->taskStats().pipelineStats.back().operatorStats.at(1).runtimeStats;
    if (targetNumSpills > maxSpillLevel) {
      ASSERT_GT(runTimeStats["exceededMaxSpillLevel"].sum, 0);
    } else {
      ASSERT_EQ(runTimeStats.count("exceededMaxSpillLevel"), 0);
    }

    task.reset();
    waitForAllTasksToBeDeleted();
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    {
      SCOPED_TRACE("single-marker");
      auto vectors = createVectors(32, rowType_, fuzzerOpts_);
      auto plan = PlanBuilder()
                      .values(vectors)
                      .markDistinct("c1_distinct", {"c0", "c1"})
                      .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                      .planNode();
      runShape(
          vectors,
          plan,
          "SELECT c0, count(distinct c1) FROM tmp GROUP BY 1",
          testData.numSpills,
          testData.maxSpillLevel);
    }
    {
      SCOPED_TRACE("multi-mask");
      auto vectors = createVectors(32, rowType_, fuzzerOpts_);
      auto [plan, sql] = makeMultiMaskCase(vectors);
      runShape(vectors, plan, sql, testData.numSpills, testData.maxSpillLevel);
    }
  }
}

TEST_F(MarkDistinctTest, spillWithDuplicateKeys) {
  // Verifies correctness when the same key appears in both pre-spill and
  // post-spill input batches. The hash table state must be preserved through
  // spill/restore so that keys already marked as distinct before spill are not
  // re-marked during restore.
  auto vectors = {
      makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      }),
      makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      }),
      makeRowVector({
          makeFlatVector<int64_t>({1, 2, 6, 7, 8}),
          makeFlatVector<int64_t>({10, 20, 60, 70, 80}),
      }),
      makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3, 9, 10}),
          makeFlatVector<int64_t>({10, 20, 30, 90, 100}),
      }),
  };

  createDuckDbTable(vectors);
  auto plan = PlanBuilder()
                  .values(vectors)
                  .markDistinct("c1_distinct", {"c0", "c1"})
                  .singleAggregation({"c0"}, {"count(c1)"}, {"c1_distinct"})
                  .planNode();
  runSpillTest(plan, "SELECT c0, count(distinct c1) FROM tmp GROUP BY 1");
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

TEST_F(MarkDistinctTest, multiMaskWithNulls) {
  // Null keys are treated as valid distinct values (ignoreNullKeys = false).
  // Mask values are chosen so each per-mask marker diverges from the no-mask
  // marker and from the other per-mask marker.
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {1, std::nullopt, 1, std::nullopt, std::nullopt}),
      makeFlatVector<bool>({true, false, true, true, false}),
      makeFlatVector<bool>({false, true, true, false, true}),
  });

  assertMarkers(
      data,
      {
          makeFlatVector<bool>({true, true, false, false, false}),
          makeFlatVector<bool>({true, false, false, true, false}),
          makeFlatVector<bool>({false, true, true, false, false}),
      });
}

TEST_F(MarkDistinctTest, multiMaskNullMasks) {
  // Null mask values are treated as false. Specifically test the case where
  // the first row of a key has a null mask: the per-mask marker must NOT
  // fire on that row, even though no other row has set the bit yet.
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 1, 1}),
      makeNullableFlatVector<bool>({std::nullopt, true, false, true}),
      makeNullableFlatVector<bool>({true, std::nullopt, std::nullopt, false}),
  });

  assertMarkers(
      data,
      {
          makeFlatVector<bool>({true, false, false, false}),
          makeFlatVector<bool>({false, true, false, false}),
          makeFlatVector<bool>({true, false, false, false}),
      });
}

TEST_F(MarkDistinctTest, multiMaskAllMasksFalse) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<bool>({false, false, false}),
      makeFlatVector<bool>({false, false, false}),
  });

  assertMarkers(
      data,
      {
          makeFlatVector<bool>({true, true, true}),
          makeFlatVector<bool>({false, false, false}),
          makeFlatVector<bool>({false, false, false}),
      });
}

TEST_F(MarkDistinctTest, multiMaskEncodedMasks) {
  // Dictionary-encoded mask0 with non-trivial indices (reversed) and
  // constant-encoded mask1 exercise the DecodedVector decode path.
  // Base: [F, T, T, F, T]. Reversed indices [4,3,2,1,0] → logical [T,F,T,T,F].
  auto baseMask = makeFlatVector<bool>({false, true, true, false, true});
  auto indices = makeIndices(5, [](auto row) { return 4 - row; });
  auto dictMask = wrapInDictionary(indices, 5, baseMask);
  auto constMask =
      BaseVector::wrapInConstant(5, 0, makeFlatVector<bool>({true}));

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
      dictMask,
      constMask,
  });

  // Per-row expected markers:
  //   nomask:  key=1 first at row 0, key=2 first at row 3 → [T,F,F,T,F].
  //   result0: dict mask logical [T,F,T,T,F]; key=1 mask=T at row 0 fires,
  //            row 2 mask=T but key already seen → F; key=2 mask=T at row 3
  //            fires; row 4 mask=F → F.
  //   result1: const mask always true; mirrors no-mask marker.
  assertMarkers(
      data,
      {
          makeFlatVector<bool>({true, false, false, true, false}),
          makeFlatVector<bool>({true, false, false, true, false}),
          makeFlatVector<bool>({true, false, false, true, false}),
      });
}

TEST_F(MarkDistinctTest, multiMaskAggregation) {
  // End-to-end with downstream aggregation, validated against DuckDB.
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatVector<int32_t>({1, 1, 2, 2}),
          makeFlatVector<int32_t>({10, 20, 30, 40}),
          makeFlatVector<bool>({true, true, true, false}),
          makeFlatVector<bool>({true, false, true, true}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({1, 2, 1}),
          makeFlatVector<int32_t>({10, 30, 30}),
          makeFlatVector<bool>({true, true, true}),
          makeFlatVector<bool>({true, true, false}),
      }),
  };

  createDuckDbTable(vectors);
  auto [plan, sql] = makeMultiMaskCase(vectors);
  AssertQueryBuilder(plan, duckDbQueryRunner_).assertResults(sql);
}

TEST_F(MarkDistinctTest, chainedMarkDistinctSharedMask) {
  // Two chained MarkDistinct nodes that both reference the same mask column.
  // Verifies the mask column flows through the first MarkDistinct unchanged
  // and remains addressable by the second.
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<int32_t>({10, 20, 10}),
          makeFlatVector<int32_t>({100, 100, 200}),
          makeFlatVector<bool>({true, false, true}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({2, 1}),
          makeFlatVector<int32_t>({20, 10}),
          makeFlatVector<int32_t>({200, 100}),
          makeFlatVector<bool>({true, false}),
      }),
  };
  createDuckDbTable(vectors);

  auto plan =
      PlanBuilder()
          .values(vectors)
          .markDistinct({"c1_nomask", "c1_filtered"}, {"c0", "c1"}, {"c3"})
          .markDistinct({"c2_nomask", "c2_filtered"}, {"c0", "c2"}, {"c3"})
          .singleAggregation(
              {"c0"}, {"sum(c1)", "sum(c2)"}, {"c1_filtered", "c2_filtered"})
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT c0, "
          "sum(DISTINCT c1) FILTER (WHERE c3), "
          "sum(DISTINCT c2) FILTER (WHERE c3) "
          "FROM tmp GROUP BY 1");
}

TEST_F(MarkDistinctTest, multiMaskNonContiguousMaskChannels) {
  // Mask channels at non-contiguous positions in the input.
  // Input schema: (g:int, mask0:bool, x:int, mask1:bool).
  // Use distinct x values so the two masks produce different results —
  // a swap of mask channel indices would be detected.
  auto data = makeRowVector(
      {"g", "mask0", "x", "mask1"},
      {
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<bool>({true, true, true}),
          makeFlatVector<int32_t>({10, 20, 30}),
          makeFlatVector<bool>({true, false, true}),
      });

  auto expected = makeRowVector(
      {"g", "mask0", "x", "mask1", "nomask", "m0", "m1"},
      {
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<bool>({true, true, true}),
          makeFlatVector<int32_t>({10, 20, 30}),
          makeFlatVector<bool>({true, false, true}),
          // Markers below.
          makeFlatVector<bool>({true, true, true}),
          makeFlatVector<bool>({true, true, true}),
          makeFlatVector<bool>({true, false, true}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .markDistinct({"nomask", "m0", "m1"}, {"g", "x"}, {"mask0", "mask1"})
          .planNode();

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(MarkDistinctTest, multiMaskSpillWithOverlappingKeys) {
  // Same keys appear before and after spill with different masks active.
  // Pre-spill: mask0=T, mask1=F. Post-spill: mask0=F, mask1=T.
  // After restore, bitmask must preserve pre-spill state.
  const int32_t batchSize = 100;
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 4; ++batch) {
    vectors.push_back(makeRowVector({
        makeFlatVector<int64_t>(batchSize, [](auto row) { return row + 1; }),
        makeFlatVector<int64_t>(batchSize, [](auto row) { return row + 1; }),
        makeFlatVector<bool>(batchSize, [](auto /*row*/) { return true; }),
        makeFlatVector<bool>(batchSize, [](auto /*row*/) { return false; }),
    }));
  }
  for (int32_t batch = 0; batch < 4; ++batch) {
    vectors.push_back(makeRowVector({
        makeFlatVector<int64_t>(batchSize, [](auto row) { return row + 1; }),
        makeFlatVector<int64_t>(batchSize, [](auto row) { return row + 1; }),
        makeFlatVector<bool>(batchSize, [](auto /*row*/) { return false; }),
        makeFlatVector<bool>(batchSize, [](auto /*row*/) { return true; }),
    }));
  }

  createDuckDbTable(vectors);
  auto [plan, sql] = makeMultiMaskCase(vectors);
  runSpillTest(plan, sql);
}

TEST_F(MarkDistinctTest, multiMaskLargeN) {
  // Exercise multi-byte bitmask handling. N=9 crosses the first byte boundary
  // (bits::nbytes(9) == 2). N=64 is exactly 8 bytes. N=100 spans 13 bytes.
  // Mask values vary per channel so a bit-position bug produces wrong values
  // for at least one mask.
  // mask_i[row 0] = bit 0 of i, mask_i[row 1] = bit 1 of i, row 2 = true.
  // This gives each mask a unique per-mask result and ensures every bit
  // position in the per-group bitmask is set by at least one mask.
  auto maskBits = [](int32_t i) {
    return std::pair<bool, bool>{(i & 1) != 0, (i & 2) != 0};
  };

  for (int32_t numMasks : {9, 64, 100}) {
    SCOPED_TRACE(fmt::format("numMasks: {}", numMasks));
    std::vector<VectorPtr> inputColumns;
    inputColumns.push_back(makeFlatVector<int32_t>({1, 1, 2}));

    std::vector<std::string> markerNames;
    markerNames.push_back("nomask");
    std::vector<std::string> maskNames;
    for (int32_t i = 0; i < numMasks; ++i) {
      auto [firstRow, secondRow] = maskBits(i);
      inputColumns.push_back(makeFlatVector<bool>({firstRow, secondRow, true}));
      markerNames.push_back(fmt::format("m{}", i));
      maskNames.push_back(fmt::format("c{}", i + 1));
    }
    auto data = makeRowVector(inputColumns);

    // Expected: nomask = [T, F, T] (first occurrence of each key).
    // For mask_i: result[0] = mask_i[0]; result[1] fires only if mask_i[0]
    // was false and mask_i[1] is true; result[2] = mask_i[2] = true.
    std::vector<VectorPtr> expectedColumns = inputColumns;
    expectedColumns.push_back(makeFlatVector<bool>({true, false, true}));
    for (int32_t i = 0; i < numMasks; ++i) {
      auto [firstRow, secondRow] = maskBits(i);
      expectedColumns.push_back(
          makeFlatVector<bool>({firstRow, !firstRow && secondRow, true}));
    }
    auto expected = makeRowVector(expectedColumns);

    auto plan = PlanBuilder()
                    .values({data})
                    .markDistinct(markerNames, {"c0"}, maskNames)
                    .planNode();

    AssertQueryBuilder(plan).assertResults(expected);
  }
}
