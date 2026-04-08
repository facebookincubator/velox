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

#include <re2/re2.h>

#include <fmt/format.h>
#include "folly/synchronization/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashJoinBridge.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/ArbitratorTestUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HashJoinTestBase.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/VectorTestUtil.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

using facebook::velox::test::BatchMaker;

namespace facebook::velox::exec {
namespace {

class HashJoinTest : public HashJoinTestBase,
                     public testing::WithParamInterface<TestParam> {
 public:
  HashJoinTest() : HashJoinTestBase(GetParam()) {}

  explicit HashJoinTest(const TestParam& param) : HashJoinTestBase(param) {}

  static std::vector<TestParam> getTestParams() {
    return std::vector<TestParam>({TestParam{1, false}, TestParam{1, true}});
  }
};

class MultiThreadedHashJoinTest : public HashJoinTest {
 public:
  MultiThreadedHashJoinTest() : HashJoinTest(GetParam()) {}

  static std::vector<TestParam> getTestParams() {
    return std::vector<TestParam>(
        {TestParam{1, false},
         TestParam{1, true},
         TestParam{3, false},
         TestParam{3, true}});
  }
};

TEST_P(MultiThreadedHashJoinTest, bigintArray) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t.t_k0 = u.u_k0")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, outOfJoinKeyColumnOrder) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeType(probeType_)
      .probeKeys({"t_k2"})
      .probeVectors(5, 10)
      .buildType(buildType_)
      .buildKeys({"u_k2"})
      .buildVectors(64, 15)
      .joinOutputLayout({"t_k1", "t_k2", "u_k1", "u_k2", "u_v1"})
      .referenceQuery(
          "SELECT t_k1, t_k2, u_k1, u_k2, u_v1 FROM t, u WHERE t_k2 = u_k2")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, joinWithCancellation) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .injectTaskCancellation(true)
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t.t_k0 = u.u_k0")
      .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
        auto stats = task->taskStats();
        EXPECT_GT(stats.terminationTimeMs, 0);
      })
      .run();
}

TEST_P(MultiThreadedHashJoinTest, testJoinWithSpillenabledCancellation) {
  auto spillDirectory = TempDirectoryPath::create();
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .injectTaskCancellation(true)
      .injectSpill(false)
      // Need spill directory so that canSpill() is true for HashProbe
      .spillDirectory(spillDirectory->getPath())
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t.t_k0 = u.u_k0")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, emptyBuild) {
  const std::vector<bool> finishOnEmptys = {false, true};
  for (const auto finishOnEmpty : finishOnEmptys) {
    SCOPED_TRACE(fmt::format("finishOnEmpty: {}", finishOnEmpty));

    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .hashProbeFinishEarlyOnEmptyBuild(finishOnEmpty)
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .keyTypes({BIGINT()})
        .probeVectors(1600, 5)
        .buildVectors(0, 5)
        .referenceQuery(
            "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t_k0 = u_k0")
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          // Check the hash probe has processed probe input rows.
          if (finishOnEmpty) {
            ASSERT_EQ(getInputPositions(task, 1), 0);
          } else {
            ASSERT_GT(getInputPositions(task, 1), 0);
          }
        })
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, emptyProbe) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT()})
      .probeVectors(0, 5)
      .buildVectors(1500, 5)
      .checkSpillStats(false)
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t_k0 = u_k0")
      .verifier([&](const std::shared_ptr<Task>& task, bool hasSpill) {
        const auto statsPair = taskSpilledStats(*task);
        if (hasSpill) {
          ASSERT_GT(statsPair.first.spilledRows, 0);
          ASSERT_GT(statsPair.first.spilledBytes, 0);
          ASSERT_GT(statsPair.first.spilledPartitions, 0);
          ASSERT_GT(statsPair.first.spilledFiles, 0);
          // There is no spilling at empty probe side.
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_GT(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
        } else {
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
        }
      })
      .run();
}

TEST_P(MultiThreadedHashJoinTest, normalizedKey) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT(), VARCHAR()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .referenceQuery(
          "SELECT t_k0, t_k1, t_data, u_k0, u_k1, u_data FROM t, u WHERE t_k0 = u_k0 AND t_k1 = u_k1")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, normalizedKeyOverflow) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .keyTypes({BIGINT(), VARCHAR(), BIGINT(), BIGINT(), BIGINT(), BIGINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .referenceQuery(
          "SELECT t_k0, t_k1, t_k2, t_k3, t_k4, t_k5, t_data, u_k0, u_k1, u_k2, u_k3, u_k4, u_k5, u_data FROM t, u WHERE t_k0 = u_k0 AND t_k1 = u_k1 AND t_k2 = u_k2 AND t_k3 = u_k3 AND t_k4 = u_k4 AND t_k5 = u_k5")
      .run();
}

DEBUG_ONLY_TEST_P(MultiThreadedHashJoinTest, parallelJoinBuildCheck) {
  std::atomic<bool> isParallelBuild{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashTable::parallelJoinBuild",
      std::function<void(void*)>([&](void*) { isParallelBuild = true; }));
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT(), VARCHAR()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .referenceQuery(
          "SELECT t_k0, t_k1, t_data, u_k0, u_k1, u_data FROM t, u WHERE t_k0 = u_k0 AND t_k1 = u_k1")
      .injectSpill(false)
      .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
        auto joinStats = task->taskStats()
                             .pipelineStats.back()
                             .operatorStats.back()
                             .runtimeStats;
        ASSERT_GT(joinStats["hashtable.buildWallNanos"].sum, 0);
        ASSERT_GE(joinStats["hashtable.buildWallNanos"].count, 1);
      })
      .run();
  ASSERT_EQ(numDrivers_ == 1, !isParallelBuild);
}

DEBUG_ONLY_TEST_P(
    MultiThreadedHashJoinTest,
    raceBetweenTaskTerminateAndTableBuild) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::finishHashBuild",
      std::function<void(Operator*)>([&](Operator* op) {
        auto task = op->operatorCtx()->task();
        task->requestAbort();
      }));
  VELOX_ASSERT_THROW(
      HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
          .numDrivers(numDrivers_)
          .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
          .keyTypes({BIGINT(), VARCHAR()})
          .probeVectors(1600, 5)
          .buildVectors(1500, 5)
          .referenceQuery(
              "SELECT t_k0, t_k1, t_data, u_k0, u_k1, u_data FROM t, u WHERE t_k0 = u_k0 AND t_k1 = u_k1")
          .injectSpill(false)
          .run(),
      "Aborted for external error");
}

TEST_P(MultiThreadedHashJoinTest, allTypes) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .keyTypes(
          {BIGINT(),
           VARCHAR(),
           REAL(),
           DOUBLE(),
           INTEGER(),
           SMALLINT(),
           TINYINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .referenceQuery(
          "SELECT t_k0, t_k1, t_k2, t_k3, t_k4, t_k5, t_k6, t_data, u_k0, u_k1, u_k2, u_k3, u_k4, u_k5, u_k6, u_data FROM t, u WHERE t_k0 = u_k0 AND t_k1 = u_k1 AND t_k2 = u_k2 AND t_k3 = u_k3 AND t_k4 = u_k4 AND t_k5 = u_k5 AND t_k6 = u_k6")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, filter) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .joinFilter("((t_k0 % 100) + (u_k0 % 100)) % 40 < 20")
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t_k0 = u_k0 AND ((t_k0 % 100) + (u_k0 % 100)) % 40 < 20")
      .run();
}

// Regression test for a JoinFuzzer-found bug where HashProbe::evalFilter
// produces a DictionaryVector<bool> with indices pointing past the base
// vector's size. The issue involves the filter "t_N = true" on a
// dictionary-encoded probe boolean column combined with expression
// memoization across multiple output batches from the same probe input.
// In debug builds, this triggers a validation failure in
// DictionaryVector::validate().
//
// This test exercises the same code path: hash join with boolean filter on
// a probe column that is also a join key, using dictionary-encoded probe
// input (matching the fuzzer's ENCODED input type) and small output batch
// sizes to force multiple output batches from the same probe input.
TEST_P(HashJoinTest, booleanJoinFilterDictionaryValidation) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 0.1;

  for (int seed = 0; seed < 20; ++seed) {
    SCOPED_TRACE(fmt::format("seed: {}", seed));
    opts.vectorSize = 10 + (seed % 20);
    VectorFuzzer fuzzer(opts, pool_.get(), seed);

    auto probeType = ROW({"t0", "t1"}, {INTEGER(), BOOLEAN()});
    auto buildType = ROW({"u0", "u1"}, {INTEGER(), BOOLEAN()});

    // Use fuzzRow which wraps columns in dictionary/constant encoding,
    // matching the JoinFuzzer's ENCODED input type.
    std::vector<RowVectorPtr> probeVectors = {fuzzer.fuzzRow(probeType)};
    std::vector<RowVectorPtr> buildVectors = {fuzzer.fuzzRow(buildType)};

    for (int batchSize : {3, 5}) {
      HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
          .numDrivers(1)
          .probeKeys({"t0", "t1"})
          .probeVectors(std::vector<RowVectorPtr>(probeVectors))
          .buildKeys({"u0", "u1"})
          .buildVectors(std::vector<RowVectorPtr>(buildVectors))
          .joinFilter("t1 = true")
          .joinOutputLayout({"t0", "t1", "u0", "u1"})
          .config(
              core::QueryConfig::kPreferredOutputBatchRows,
              std::to_string(batchSize))
          .config(
              core::QueryConfig::kMaxOutputBatchRows, std::to_string(batchSize))
          .injectSpill(false)
          .referenceQuery(
              "SELECT t.t0, t.t1, u.u0, u.u1 FROM t, u "
              "WHERE t.t0 = u.u0 AND t.t1 = u.u1 AND t.t1 = true")
          .run();
    }
  }
}

// Same as above but with the boolean filter column as a non-key payload
// column, exercising a slightly different code path where only integer
// keys are used for join matching.
TEST_P(HashJoinTest, booleanPayloadFilterDictionaryValidation) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 0.1;

  for (int seed = 0; seed < 20; ++seed) {
    SCOPED_TRACE(fmt::format("seed: {}", seed));
    opts.vectorSize = 10 + (seed % 20);
    VectorFuzzer fuzzer(opts, pool_.get(), seed);

    auto probeType = ROW({"t0", "t1"}, {INTEGER(), BOOLEAN()});
    auto buildType = ROW({"u0", "u1"}, {INTEGER(), BOOLEAN()});

    std::vector<RowVectorPtr> probeVectors = {fuzzer.fuzzRow(probeType)};
    std::vector<RowVectorPtr> buildVectors = {fuzzer.fuzzRow(buildType)};

    for (int batchSize : {3, 5}) {
      HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
          .numDrivers(1)
          .probeKeys({"t0"})
          .probeVectors(std::vector<RowVectorPtr>(probeVectors))
          .buildKeys({"u0"})
          .buildVectors(std::vector<RowVectorPtr>(buildVectors))
          .joinFilter("t1 = true")
          .joinOutputLayout({"t0", "t1", "u0", "u1"})
          .config(
              core::QueryConfig::kPreferredOutputBatchRows,
              std::to_string(batchSize))
          .config(
              core::QueryConfig::kMaxOutputBatchRows, std::to_string(batchSize))
          .injectSpill(false)
          .referenceQuery(
              "SELECT t.t0, t.t1, u.u0, u.u1 FROM t, u "
              "WHERE t.t0 = u.u0 AND t.t1 = true")
          .run();
    }
  }
}

DEBUG_ONLY_TEST_P(MultiThreadedHashJoinTest, filterSpillOnFirstProbeInput) {
  auto spillDirectory = TempDirectoryPath::create();
  std::atomic_bool injectProbeSpillOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::getOutput",
      std::function<void(Operator*)>([&](Operator* op) {
        if (!isHashProbeMemoryPool(*op->pool())) {
          return;
        }
        HashProbe* probeOp = static_cast<HashProbe*>(op);
        if (!probeOp->testingHasPendingInput()) {
          return;
        }
        if (!injectProbeSpillOnce.exchange(false)) {
          return;
        }
        testingRunArbitration(op->pool());
        ASSERT_EQ(op->pool()->usedBytes(), 40960);
        ASSERT_EQ(op->pool()->reservedBytes(), 1048576);
      }));

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .keyTypes({BIGINT()})
      .numDrivers(1)
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .injectSpill(false)
      .spillDirectory(spillDirectory->getPath())
      .joinFilter("((t_k0 % 100) + (u_k0 % 100)) % 40 < 20")
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t_k0 = u_k0 AND ((t_k0 % 100) + (u_k0 % 100)) % 40 < 20")
      .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
        const auto statsPair = taskSpilledStats(*task);
        ASSERT_EQ(statsPair.first.spilledRows, 0);
        ASSERT_EQ(statsPair.first.spilledBytes, 0);
        ASSERT_EQ(statsPair.first.spilledPartitions, 0);
        ASSERT_EQ(statsPair.first.spilledFiles, 0);
        ASSERT_GT(statsPair.second.spilledRows, 0);
        ASSERT_GT(statsPair.second.spilledBytes, 0);
        ASSERT_GT(statsPair.second.spilledPartitions, 0);
        ASSERT_GT(statsPair.second.spilledFiles, 0);
      })
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithNull) {
  struct {
    double probeNullRatio;
    double buildNullRatio;

    std::string debugString() const {
      return fmt::format(
          "probeNullRatio: {}, buildNullRatio: {}",
          probeNullRatio,
          buildNullRatio);
    }
  } testSettings[] = {
      {0.0, 1.0}, {0.0, 0.1}, {0.1, 1.0}, {0.1, 0.1}, {1.0, 1.0}, {1.0, 0.1}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::vector<RowVectorPtr> probeVectors =
        makeBatches(5, 3, probeType_, pool_.get(), testData.probeNullRatio);

    // The first half number of build batches having no nulls to trigger it
    // later during the processing.
    std::vector<RowVectorPtr> buildVectors = mergeBatches(
        makeBatches(5, 6, buildType_, pool_.get(), 0.0),
        makeBatches(5, 6, buildType_, pool_.get(), testData.buildNullRatio));

    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeType(probeType_)
        .probeKeys({"t_k2"})
        .probeVectors(std::move(probeVectors))
        .buildType(buildType_)
        .buildKeys({"u_k2"})
        .buildVectors(std::move(buildVectors))
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinOutputLayout({"t_k1", "t_k2"})
        .referenceQuery(
            "SELECT t_k1, t_k2 FROM t WHERE t.t_k2 NOT IN (SELECT u_k2 FROM u)")
        // NOTE: we might not trigger spilling at build side if we detect the
        // null join key in the build rows early.
        .checkSpillStats(false)
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, rightSemiJoinFilterWithLargeOutput) {
  // Build the identical left and right vectors to generate large join
  // outputs.
  std::vector<RowVectorPtr> probeVectors =
      makeBatches(4, [&](uint32_t /*unused*/) {
        return makeRowVector(
            {"t0", "t1"},
            {makeFlatVector<int32_t>(2048, [](auto row) { return row; }),
             makeFlatVector<int32_t>(2048, [](auto row) { return row; })});
      });

  std::vector<RowVectorPtr> buildVectors =
      makeBatches(4, [&](uint32_t /*unused*/) {
        return makeRowVector(
            {"u0", "u1"},
            {makeFlatVector<int32_t>(2048, [](auto row) { return row; }),
             makeFlatVector<int32_t>(2048, [](auto row) { return row; })});
      });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kRightSemiFilter)
      .joinOutputLayout({"u1"})
      .referenceQuery("SELECT u.u1 FROM u WHERE u.u0 IN (SELECT t0 FROM t)")
      .run();
}

/// Test hash join where build-side keys come from a small range and allow for
/// array-based lookup instead of a hash table.
TEST_P(MultiThreadedHashJoinTest, arrayBasedLookup) {
  auto oddIndices = makeIndices(500, [](auto i) { return 2 * i + 1; });

  std::vector<RowVectorPtr> probeVectors = {
      // Join key vector is flat.
      makeRowVector({
          makeFlatVector<int32_t>(1'000, [](auto row) { return row; }),
          makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
      }),
      // Join key vector is constant. There is a match in the build side.
      makeRowVector({
          makeConstant(4, 2'000),
          makeFlatVector<int64_t>(2'000, [](auto row) { return row; }),
      }),
      // Join key vector is constant. There is no match.
      makeRowVector({
          makeConstant(5, 2'000),
          makeFlatVector<int64_t>(2'000, [](auto row) { return row; }),
      }),
      // Join key vector is a dictionary.
      makeRowVector({
          wrapInDictionary(
              oddIndices,
              500,
              makeFlatVector<int32_t>(1'000, [](auto row) { return row * 4; })),
          makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
      })};

  // 100 key values in [0, 198] range.
  std::vector<RowVectorPtr> buildVectors = {
      makeRowVector(
          {makeFlatVector<int32_t>(100, [](auto row) { return row / 2; })}),
      makeRowVector(
          {makeFlatVector<int32_t>(100, [](auto row) { return row * 2; })}),
      makeRowVector(
          {makeFlatVector<int32_t>(100, [](auto row) { return row; })})};

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeKeys({"c0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"c0"})
      .buildVectors(std::move(buildVectors))
      .joinOutputLayout({"c1"})
      .outputProjections({"c1 + 1"})
      .referenceQuery("SELECT t.c1 + 1 FROM t, u WHERE t.c0 = u.c0")
      .verifier([&](const std::shared_ptr<Task>& task, bool hasSpill) {
        if (hasSpill) {
          return;
        }
        auto joinStats = task->taskStats()
                             .pipelineStats.back()
                             .operatorStats.back()
                             .runtimeStats;
        ASSERT_EQ(151, joinStats["distinctKey0"].sum);
        ASSERT_EQ(200, joinStats["rangeKey0"].sum);
      })
      .run();
}

TEST_P(MultiThreadedHashJoinTest, joinSidesDifferentSchema) {
  // In this join, the tables have different schema. LHS table t has schema
  // {INTEGER, VARCHAR, INTEGER}. RHS table u has schema {INTEGER, REAL,
  // INTEGER}. The filter predicate uses
  // a column from the right table  before the left and the corresponding
  // columns at the same channel number(1) have different types. This has been
  // a source of crashes in the join logic.
  size_t batchSize = 100;

  std::vector<std::string> stringVector = {"aaa", "bbb", "ccc", "ddd", "eee"};
  std::vector<RowVectorPtr> probeVectors =
      makeBatches(5, [&](int32_t /*unused*/) {
        return makeRowVector({
            makeFlatVector<int32_t>(batchSize, [](auto row) { return row; }),
            makeFlatVector<StringView>(
                batchSize,
                [&](auto row) {
                  return StringView(stringVector[row % stringVector.size()]);
                }),
            makeFlatVector<int32_t>(batchSize, [](auto row) { return row; }),
        });
      });
  std::vector<RowVectorPtr> buildVectors =
      makeBatches(5, [&](int32_t /*unused*/) {
        return makeRowVector({
            makeFlatVector<int32_t>(batchSize, [](auto row) { return row; }),
            makeFlatVector<double>(
                batchSize, [](auto row) { return row * 5.0; }),
            makeFlatVector<int32_t>(batchSize, [](auto row) { return row; }),
        });
      });

  // In this hash join the 2 tables have a common key which is the
  // first channel in both tables.
  const std::string referenceQuery =
      "SELECT t.c0 * t.c2/2 FROM "
      "  t, u "
      "  WHERE t.c0 = u.c0 AND "
      // TODO: enable ltrim test after the race condition in expression
      // execution gets fixed.
      //"  u.c2 > 10 AND ltrim(t.c1) = 'aaa'";
      "  u.c2 > 10";

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeKeys({"t_c0"})
      .probeVectors(std::move(probeVectors))
      .probeProjections({"c0 AS t_c0", "c1 AS t_c1", "c2 AS t_c2"})
      .buildKeys({"u_c0"})
      .buildVectors(std::move(buildVectors))
      .buildProjections({"c0 AS u_c0", "c1 AS u_c1", "c2 AS u_c2"})
      //.joinFilter("u_c2 > 10 AND ltrim(t_c1) == 'aaa'")
      .joinFilter("u_c2 > 10")
      .joinOutputLayout({"t_c0", "t_c2"})
      .outputProjections({"t_c0 * t_c2/2"})
      .referenceQuery(referenceQuery)
      .run();
}

TEST_P(MultiThreadedHashJoinTest, innerJoinWithEmptyBuild) {
  const std::vector<bool> finishOnEmptys = {false, true};
  for (auto finishOnEmpty : finishOnEmptys) {
    SCOPED_TRACE(fmt::format("finishOnEmpty: {}", finishOnEmpty));

    std::vector<RowVectorPtr> probeVectors = makeBatches(5, [&](int32_t batch) {
      return makeRowVector({
          makeFlatVector<int32_t>(
              123,
              [batch](auto row) { return row * 11 / std::max(batch, 1); },
              nullEvery(13)),
          makeFlatVector<int32_t>(1'234, [](auto row) { return row; }),
      });
    });
    std::vector<RowVectorPtr> buildVectors =
        makeBatches(10, [&](int32_t batch) {
          return makeRowVector({makeFlatVector<int32_t>(
              123,
              [batch](auto row) { return row % std::max(batch, 1); },
              nullEvery(7))});
        });

    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .hashProbeFinishEarlyOnEmptyBuild(finishOnEmpty)
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"c0"})
        .probeVectors(std::move(probeVectors))
        .buildKeys({"c0"})
        .buildVectors(std::move(buildVectors))
        .buildFilter("c0 < 0")
        .joinOutputLayout({"c1"})
        .referenceQuery("SELECT null LIMIT 0")
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
          // Check the hash probe has processed probe input rows.
          if (finishOnEmpty) {
            ASSERT_EQ(getInputPositions(task, 1), 0);
          } else {
            ASSERT_GT(getInputPositions(task, 1), 0);
          }
        })
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, leftSemiJoinFilter) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeType(probeType_)
      .probeVectors(174, 5)
      .probeKeys({"t_k1"})
      .buildType(buildType_)
      .buildVectors(133, 4)
      .buildKeys({"u_k1"})
      .joinType(core::JoinType::kLeftSemiFilter)
      .joinOutputLayout({"t_k2"})
      .referenceQuery("SELECT t_k2 FROM t WHERE t_k1 IN (SELECT u_k1 FROM u)")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, leftSemiJoinFilterWithEmptyBuild) {
  const std::vector<bool> finishOnEmptys = {false, true};
  for (const auto finishOnEmpty : finishOnEmptys) {
    SCOPED_TRACE(fmt::format("finishOnEmpty: {}", finishOnEmpty));

    std::vector<RowVectorPtr> probeVectors =
        makeBatches(10, [&](int32_t /*unused*/) {
          return makeRowVector({
              makeFlatVector<int32_t>(
                  1'234, [](auto row) { return row % 11; }, nullEvery(13)),
              makeFlatVector<int32_t>(1'234, [](auto row) { return row; }),
          });
        });
    std::vector<RowVectorPtr> buildVectors =
        makeBatches(10, [&](int32_t /*unused*/) {
          return makeRowVector({
              makeFlatVector<int32_t>(
                  123, [](auto row) { return row % 5; }, nullEvery(7)),
          });
        });

    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .hashProbeFinishEarlyOnEmptyBuild(finishOnEmpty)
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"c0"})
        .probeVectors(std::move(probeVectors))
        .buildKeys({"c0"})
        .buildVectors(std::move(buildVectors))
        .joinType(core::JoinType::kLeftSemiFilter)
        .joinFilter("c0 < 0")
        .joinOutputLayout({"c1"})
        .referenceQuery(
            "SELECT t.c1 FROM t WHERE t.c0 IN (SELECT c0 FROM u WHERE c0 < 0)")
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, leftSemiJoinFilterWithExtraFilter) {
  std::vector<RowVectorPtr> probeVectors = makeBatches(5, [&](int32_t batch) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeFlatVector<int32_t>(
                250, [batch](auto row) { return row % (11 + batch); }),
            makeFlatVector<int32_t>(
                250, [batch](auto row) { return row * batch; }),
        });
  });

  std::vector<RowVectorPtr> buildVectors = makeBatches(5, [&](int32_t batch) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(
                123, [batch](auto row) { return row % (5 + batch); }),
            makeFlatVector<int32_t>(
                123, [batch](auto row) { return row * batch; }),
        });
  });

  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kLeftSemiFilter)
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(
            "SELECT t.* FROM t WHERE EXISTS (SELECT u0 FROM u WHERE t0 = u0)")
        .run();
  }

  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kLeftSemiFilter)
        .joinFilter("t1 != u1")
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(
            "SELECT t.* FROM t WHERE EXISTS (SELECT u0, u1 FROM u WHERE t0 = u0 AND t1 <> u1)")
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, rightSemiJoinFilter) {
  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeType(probeType_)
      .probeVectors(133, 3)
      .probeKeys({"t_k1"})
      .buildType(buildType_)
      .buildVectors(174, 4)
      .buildKeys({"u_k1"})
      .joinType(core::JoinType::kRightSemiFilter)
      .joinOutputLayout({"u_k2"})
      .referenceQuery("SELECT u_k2 FROM u WHERE u_k1 IN (SELECT t_k1 FROM t)")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, rightSemiJoinFilterWithEmptyBuild) {
  const std::vector<bool> finishOnEmptys = {false, true};
  for (const auto finishOnEmpty : finishOnEmptys) {
    SCOPED_TRACE(fmt::format("finishOnEmpty: {}", finishOnEmpty));

    // probeVectors size is greater than buildVector size.
    std::vector<RowVectorPtr> probeVectors =
        makeBatches(5, [&](uint32_t /*unused*/) {
          return makeRowVector(
              {"t0", "t1"},
              {makeFlatVector<int32_t>(
                   431, [](auto row) { return row % 11; }, nullEvery(13)),
               makeFlatVector<int32_t>(431, [](auto row) { return row; })});
        });

    std::vector<RowVectorPtr> buildVectors =
        makeBatches(5, [&](uint32_t /*unused*/) {
          return makeRowVector(
              {"u0", "u1"},
              {
                  makeFlatVector<int32_t>(
                      434, [](auto row) { return row % 5; }, nullEvery(7)),
                  makeFlatVector<int32_t>(434, [](auto row) { return row; }),
              });
        });

    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .hashProbeFinishEarlyOnEmptyBuild(finishOnEmpty)
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(probeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(buildVectors))
        .buildFilter("u0 < 0")
        .joinType(core::JoinType::kRightSemiFilter)
        .joinOutputLayout({"u1"})
        .referenceQuery(
            "SELECT u.u1 FROM u WHERE u.u0 IN (SELECT t0 FROM t) AND u.u0 < 0")
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
          // Check the hash probe has processed probe input rows.
          if (finishOnEmpty) {
            ASSERT_EQ(getInputPositions(task, 1), 0);
          } else {
            ASSERT_GT(getInputPositions(task, 1), 0);
          }
        })
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, rightSemiJoinFilterWithAllMatches) {
  // Make build side larger to test all rows are returned.
  std::vector<RowVectorPtr> probeVectors =
      makeBatches(3, [&](uint32_t /*unused*/) {
        return makeRowVector(
            {"t0", "t1"},
            {
                makeFlatVector<int32_t>(
                    123, [](auto row) { return row % 5; }, nullEvery(7)),
                makeFlatVector<int32_t>(123, [](auto row) { return row; }),
            });
      });

  std::vector<RowVectorPtr> buildVectors =
      makeBatches(5, [&](uint32_t /*unused*/) {
        return makeRowVector(
            {"u0", "u1"},
            {makeFlatVector<int32_t>(
                 314, [](auto row) { return row % 11; }, nullEvery(13)),
             makeFlatVector<int32_t>(314, [](auto row) { return row; })});
      });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kRightSemiFilter)
      .joinOutputLayout({"u1"})
      .referenceQuery("SELECT u.u1 FROM u WHERE u.u0 IN (SELECT t0 FROM t)")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, rightSemiJoinFilterWithExtraFilter) {
  auto probeVectors = makeBatches(4, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeFlatVector<int32_t>(345, [](auto row) { return row; }),
            makeFlatVector<int32_t>(345, [](auto row) { return row; }),
        });
  });

  auto buildVectors = makeBatches(4, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(250, [](auto row) { return row; }),
            makeFlatVector<int32_t>(250, [](auto row) { return row; }),
        });
  });

  // Always true filter.
  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kRightSemiFilter)
        .joinFilter("t1 > -1")
        .joinOutputLayout({"u0", "u1"})
        .referenceQuery(
            "SELECT u.* FROM u WHERE EXISTS (SELECT t0 FROM t WHERE u0 = t0 AND t1 > -1)")
        .verifier([&](const std::shared_ptr<Task>& task, bool hasSpill) {
          ASSERT_EQ(
              getOutputPositions(task, "HashProbe"), 200 * 5 * numDrivers_);
        })
        .run();
  }

  // Always false filter.
  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kRightSemiFilter)
        .joinFilter("t1 > 100000")
        .joinOutputLayout({"u0", "u1"})
        .referenceQuery(
            "SELECT u.* FROM u WHERE EXISTS (SELECT t0 FROM t WHERE u0 = t0 AND t1 > 100000)")
        .verifier([&](const std::shared_ptr<Task>& task, bool hasSpill) {
          ASSERT_EQ(getOutputPositions(task, "HashProbe"), 0);
        })
        .run();
  }

  // Selective filter.
  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kRightSemiFilter)
        .joinFilter("t1 % 5 = 0")
        .joinOutputLayout({"u0", "u1"})
        .referenceQuery(
            "SELECT u.* FROM u WHERE EXISTS (SELECT t0 FROM t WHERE u0 = t0 AND t1 % 5 = 0)")
        .verifier([&](const std::shared_ptr<Task>& task, bool hasSpill) {
          ASSERT_EQ(
              getOutputPositions(task, "HashProbe"), 200 / 5 * 5 * numDrivers_);
        })
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, semiFilterOverLazyVectors) {
  auto probeVectors = makeBatches(1, [&](auto /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeFlatVector<int32_t>(1'000, [](auto row) { return row; }),
            makeFlatVector<int64_t>(1'000, [](auto row) { return row * 10; }),
        });
  });

  auto buildVectors = makeBatches(3, [&](auto /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(
                1'000, [](auto row) { return -100 + (row / 5); }),
            makeFlatVector<int64_t>(
                1'000, [](auto row) { return -1000 + (row / 5) * 10; }),
        });
  });

  std::shared_ptr<TempFilePath> probeFile = TempFilePath::create();
  writeToFile(probeFile->getPath(), probeVectors);

  std::shared_ptr<TempFilePath> buildFile = TempFilePath::create();
  writeToFile(buildFile->getPath(), buildVectors);

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  core::PlanNodeId probeScanId;
  core::PlanNodeId buildScanId;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .tableScan(asRowType(probeVectors[0]->type()))
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .tableScan(asRowType(buildVectors[0]->type()))
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "",
                      {"t0", "t1"},
                      core::JoinType::kLeftSemiFilter)
                  .planNode();

  SplitPath splitPaths = {
      {probeScanId, {probeFile->getPath()}},
      {buildScanId, {buildFile->getPath()}},
  };

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .planNode(plan)
      .inputSplits(splitPaths)
      .checkSpillStats(false)
      .referenceQuery("SELECT t0, t1 FROM t WHERE t0 IN (SELECT u0 FROM u)")
      .run();

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .planNode(flipJoinSides(plan))
      .inputSplits(splitPaths)
      .checkSpillStats(false)
      .referenceQuery("SELECT t0, t1 FROM t WHERE t0 IN (SELECT u0 FROM u)")
      .run();

  // With extra filter.
  planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  plan = PlanBuilder(planNodeIdGenerator)
             .tableScan(asRowType(probeVectors[0]->type()))
             .capturePlanNodeId(probeScanId)
             .hashJoin(
                 {"t0"},
                 {"u0"},
                 PlanBuilder(planNodeIdGenerator)
                     .tableScan(asRowType(buildVectors[0]->type()))
                     .capturePlanNodeId(buildScanId)
                     .planNode(),
                 "(t1 + u1) % 3 = 0",
                 {"t0", "t1"},
                 core::JoinType::kLeftSemiFilter)
             .planNode();

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .planNode(plan)
      .inputSplits(splitPaths)
      .checkSpillStats(false)
      .referenceQuery(
          "SELECT t0, t1 FROM t WHERE t0 IN (SELECT u0 FROM u WHERE (t1 + u1) % 3 = 0)")
      .run();

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .planNode(flipJoinSides(plan))
      .inputSplits(splitPaths)
      .checkSpillStats(false)
      .referenceQuery(
          "SELECT t0, t1 FROM t WHERE t0 IN (SELECT u0 FROM u WHERE (t1 + u1) % 3 = 0)")
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoin) {
  std::vector<RowVectorPtr> probeVectors =
      makeBatches(5, [&](uint32_t /*unused*/) {
        return makeRowVector({
            makeFlatVector<int32_t>(
                1'000, [](auto row) { return row % 11; }, nullEvery(13)),
            makeFlatVector<int32_t>(1'000, [](auto row) { return row; }),
        });
      });

  std::vector<RowVectorPtr> buildVectors =
      makeBatches(5, [&](uint32_t /*unused*/) {
        return makeRowVector({
            makeFlatVector<int32_t>(
                1'234, [](auto row) { return row % 5; }, nullEvery(7)),
        });
      });

  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"c0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"c0"})
        .buildVectors(std::move(testBuildVectors))
        .buildFilter("c0 IS NOT NULL")
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinOutputLayout({"c1"})
        .referenceQuery(
            "SELECT t.c1 FROM t WHERE t.c0 NOT IN (SELECT c0 FROM u WHERE c0 IS NOT NULL)")
        .checkSpillStats(false)
        .run();
  }

  // Empty build side.
  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"c0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"c0"})
        .buildVectors(std::move(testBuildVectors))
        .buildFilter("c0 < 0")
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinOutputLayout({"c1"})
        .referenceQuery(
            "SELECT t.c1 FROM t WHERE t.c0 NOT IN (SELECT c0 FROM u WHERE c0 < 0)")
        .checkSpillStats(false)
        .run();
  }

  // Build side with nulls. Null-aware Anti join always returns nothing.
  {
    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"c0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"c0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinOutputLayout({"c1"})
        .referenceQuery(
            "SELECT t.c1 FROM t WHERE t.c0 NOT IN (SELECT c0 FROM u)")
        .checkSpillStats(false)
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilter) {
  std::vector<RowVectorPtr> probeVectors =
      makeBatches(5, [&](int32_t /*unused*/) {
        return makeRowVector(
            {"t0", "t1"},
            {
                makeFlatVector<int32_t>(128, [](auto row) { return row % 11; }),
                makeFlatVector<int32_t>(128, [](auto row) { return row; }),
            });
      });

  std::vector<RowVectorPtr> buildVectors =
      makeBatches(5, [&](int32_t /*unused*/) {
        return makeRowVector(
            {"u0", "u1"},
            {
                makeFlatVector<int32_t>(123, [](auto row) { return row % 5; }),
                makeFlatVector<int32_t>(123, [](auto row) { return row; }),
            });
      });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kAnti)
      .nullAware(true)
      .joinFilter("t1 != u1")
      .joinOutputLayout({"t0", "t1"})
      .referenceQuery(
          "SELECT t.* FROM t WHERE NOT EXISTS (SELECT * FROM u WHERE t0 = u0 AND t1 <> u1)")
      .checkSpillStats(false)
      .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
        // Verify spilling is not triggered in case of null-aware anti-join
        // with filter.
        const auto statsPair = taskSpilledStats(*task);
        ASSERT_EQ(statsPair.first.spilledRows, 0);
        ASSERT_EQ(statsPair.first.spilledBytes, 0);
        ASSERT_EQ(statsPair.first.spilledPartitions, 0);
        ASSERT_EQ(statsPair.first.spilledFiles, 0);
        ASSERT_EQ(statsPair.second.spilledRows, 0);
        ASSERT_EQ(statsPair.second.spilledBytes, 0);
        ASSERT_EQ(statsPair.second.spilledPartitions, 0);
        ASSERT_EQ(statsPair.second.spilledFiles, 0);
        verifyTaskSpilledRuntimeStats(*task, false);
        ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
      })
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterAndEmptyBuild) {
  const std::vector<bool> finishOnEmptys = {false, true};
  for (const auto finishOnEmpty : finishOnEmptys) {
    SCOPED_TRACE(fmt::format("finishOnEmpty: {}", finishOnEmpty));

    auto probeVectors = makeBatches(4, [&](int32_t /*unused*/) {
      return makeRowVector(
          {"t0", "t1"},
          {
              makeNullableFlatVector<int32_t>({std::nullopt, 1, 2}),
              makeFlatVector<int32_t>({0, 1, 2}),
          });
    });
    auto buildVectors = makeBatches(4, [&](int32_t /*unused*/) {
      return makeRowVector(
          {"u0", "u1"},
          {
              makeNullableFlatVector<int32_t>({3, 2, 3}),
              makeFlatVector<int32_t>({0, 2, 3}),
          });
    });

    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .hashProbeFinishEarlyOnEmptyBuild(finishOnEmpty)
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::vector<RowVectorPtr>(probeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::vector<RowVectorPtr>(buildVectors))
        .buildFilter("u0 < 0")
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinFilter("u1 > t1")
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(
            "SELECT t.* FROM t WHERE NOT EXISTS (SELECT * FROM u WHERE u0 < 0 AND u.u0 = t.t0)")
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          // Verify spilling is not triggered in case of null-aware anti-join
          // with filter.
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
        })
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterAndNullKey) {
  auto probeVectors = makeBatches(4, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeNullableFlatVector<int32_t>({std::nullopt, 1, 2}),
            makeFlatVector<int32_t>({0, 1, 2}),
        });
  });
  auto buildVectors = makeBatches(4, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeNullableFlatVector<int32_t>({std::nullopt, 2, 3}),
            makeFlatVector<int32_t>({0, 2, 3}),
        });
  });

  std::vector<std::string> filters({"u1 > t1", "u1 * t1 > 0"});
  for (const std::string& filter : filters) {
    const auto referenceSql = fmt::format(
        "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE {})",
        filter);

    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinFilter(filter)
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(referenceSql)
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          // Verify spilling is not triggered in case of null-aware anti-join
          // with filter.
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
        })
        .run();
  }
}

TEST_P(
    MultiThreadedHashJoinTest,
    hashModeNullAwareAntiJoinWithFilterAndNullKey) {
  // Use float type keys to trigger hash mode table.
  auto probeVectors = makeBatches(50, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeNullableFlatVector<float>({std::nullopt, 1, 2}),
            makeFlatVector<int32_t>({1, 1, 2}),
        });
  });
  auto buildVectors = makeBatches(5, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeNullableFlatVector<float>({std::nullopt, 2, 3}),
            makeFlatVector<int32_t>({0, 2, 3}),
        });
  });

  std::vector<std::string> filters({"u1 < t1", "u1 + t1 = 0"});
  for (const std::string& filter : filters) {
    const auto referenceSql = fmt::format(
        "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE {})",
        filter);

    auto testProbeVectors = probeVectors;
    auto testBuildVectors = buildVectors;
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(testProbeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(testBuildVectors))
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinFilter(filter)
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(referenceSql)
        .checkSpillStats(false)
        .run();
  }
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterOnNullableColumn) {
  const std::string referenceSql =
      "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE t1 <> u1)";
  const std::string joinFilter = "t1 <> u1";
  {
    SCOPED_TRACE("null filter column");
    auto probeVectors = makeBatches(3, [&](int32_t /*unused*/) {
      return makeRowVector(
          {"t0", "t1"},
          {
              makeFlatVector<int32_t>(200, [](auto row) { return row % 11; }),
              makeFlatVector<int32_t>(200, folly::identity, nullEvery(97)),
          });
    });
    auto buildVectors = makeBatches(3, [&](int32_t /*unused*/) {
      return makeRowVector(
          {"u0", "u1"},
          {
              makeFlatVector<int32_t>(234, [](auto row) { return row % 5; }),
              makeFlatVector<int32_t>(234, folly::identity, nullEvery(91)),
          });
    });
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(probeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(buildVectors))
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinFilter(joinFilter)
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(referenceSql)
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          // Verify spilling is not triggered in case of null-aware anti-join
          // with filter.
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
        })
        .run();
  }

  {
    SCOPED_TRACE("null filter and key column");
    auto probeVectors = makeBatches(3, [&](int32_t /*unused*/) {
      return makeRowVector(
          {"t0", "t1"},
          {
              makeFlatVector<int32_t>(
                  200, [](auto row) { return row % 11; }, nullEvery(23)),
              makeFlatVector<int32_t>(200, folly::identity, nullEvery(29)),
          });
    });
    auto buildVectors = makeBatches(3, [&](int32_t /*unused*/) {
      return makeRowVector(
          {"u0", "u1"},
          {
              makeFlatVector<int32_t>(
                  234, [](auto row) { return row % 5; }, nullEvery(31)),
              makeFlatVector<int32_t>(234, folly::identity, nullEvery(37)),
          });
    });
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .numDrivers(numDrivers_)
        .parallelizeJoinBuildRows(parallelBuildSideRowsEnabled_)
        .probeKeys({"t0"})
        .probeVectors(std::move(probeVectors))
        .buildKeys({"u0"})
        .buildVectors(std::move(buildVectors))
        .joinType(core::JoinType::kAnti)
        .nullAware(true)
        .joinFilter(joinFilter)
        .joinOutputLayout({"t0", "t1"})
        .referenceQuery(referenceSql)
        .checkSpillStats(false)
        .verifier([&](const std::shared_ptr<Task>& task, bool /*unused*/) {
          // Verify spilling is not triggered in case of null-aware anti-join
          // with filter.
          const auto statsPair = taskSpilledStats(*task);
          ASSERT_EQ(statsPair.first.spilledRows, 0);
          ASSERT_EQ(statsPair.first.spilledBytes, 0);
          ASSERT_EQ(statsPair.first.spilledPartitions, 0);
          ASSERT_EQ(statsPair.first.spilledFiles, 0);
          ASSERT_EQ(statsPair.second.spilledRows, 0);
          ASSERT_EQ(statsPair.second.spilledBytes, 0);
          ASSERT_EQ(statsPair.second.spilledPartitions, 0);
          ASSERT_EQ(statsPair.second.spilledFiles, 0);
          verifyTaskSpilledRuntimeStats(*task, false);
          ASSERT_EQ(maxHashBuildSpillLevel(*task), -1);
        })
        .run();
  }
}

TEST_P(
    MultiThreadedHashJoinTest,
    nullAwareAntiJoinWithFilterBatchedEvaluation) {
  // Use >1024 build rows to trigger multiple batches in
  // applyFilterOnTableRowsForNullAwareJoin (kBatchSize is 1024), exercising the
  // per-batch deselect of filterPassedRows from rows. Include null probe keys
  // so that crossJoinProbeRows is non-empty and the cross-join path iterates
  // all 2048 build rows across 2 batches.
  auto probeVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeFlatVector<int32_t>(
                256,
                [](auto row) { return row % 50; },
                [](auto row) { return row < 4; }),
            makeFlatVector<int32_t>(256, [](auto row) { return row; }),
        });
  });
  auto buildVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(2048, [](auto row) { return row % 25; }),
            makeFlatVector<int32_t>(2048, [](auto row) { return row * 2; }),
        });
  });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kAnti)
      .nullAware(true)
      .joinFilter("t1 <> u1")
      .joinOutputLayout({"t0", "t1"})
      .referenceQuery(
          "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE t1 <> u1)")
      .checkSpillStats(false)
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterEarlyTermination) {
  auto probeVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeFlatVector<int32_t>(100, [](auto row) { return row % 10; }),
            makeFlatVector<int32_t>(100, [](auto row) { return row; }),
        });
  });
  auto buildVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(500, [](auto row) { return row % 10; }),
            makeFlatVector<int32_t>(500, [](auto row) { return row; }),
        });
  });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kAnti)
      .nullAware(true)
      .joinFilter("t1 < u1")
      .joinOutputLayout({"t0", "t1"})
      .referenceQuery(
          "SELECT t.* FROM t WHERE NOT EXISTS (SELECT * FROM u WHERE t0 = u0 AND t1 < u1)")
      .checkSpillStats(false)
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterMixedNulls) {
  auto probeVectors = makeBatches(2, [&](int32_t batch) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeNullableFlatVector<int32_t>(
                {std::nullopt,
                 1,
                 2,
                 std::nullopt,
                 4,
                 5,
                 6,
                 std::nullopt,
                 8,
                 9}),
            makeFlatVector<int32_t>(
                10, [batch](auto row) { return batch * 10 + row; }),
        });
  });
  auto buildVectors = makeBatches(2, [&](int32_t batch) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeNullableFlatVector<int32_t>(
                {1,
                 std::nullopt,
                 3,
                 4,
                 std::nullopt,
                 6,
                 7,
                 8,
                 std::nullopt,
                 10}),
            makeFlatVector<int32_t>(
                10, [batch](auto row) { return batch * 5 + row; }),
        });
  });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kAnti)
      .nullAware(true)
      .joinFilter("t1 + u1 < 50")
      .joinOutputLayout({"t0", "t1"})
      .referenceQuery(
          "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE t1 + u1 < 50)")
      .checkSpillStats(false)
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterAllNullProbeKeys) {
  auto probeVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeNullConstant(TypeKind::INTEGER, 64),
            makeFlatVector<int32_t>(64, [](auto row) { return row; }),
        });
  });
  auto buildVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(32, [](auto row) { return row; }),
            makeFlatVector<int32_t>(32, [](auto row) { return row * 3; }),
        });
  });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kAnti)
      .nullAware(true)
      .joinFilter("t1 <> u1")
      .joinOutputLayout({"t0", "t1"})
      .referenceQuery(
          "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE t1 <> u1)")
      .checkSpillStats(false)
      .run();
}

TEST_P(MultiThreadedHashJoinTest, nullAwareAntiJoinWithFilterEmptyBatch) {
  auto probeVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"t0", "t1"},
        {
            makeFlatVector<int32_t>(32, [](auto row) { return row; }),
            makeFlatVector<int32_t>(32, [](auto row) { return row; }),
        });
  });
  auto buildVectors = makeBatches(1, [&](int32_t /*unused*/) {
    return makeRowVector(
        {"u0", "u1"},
        {
            makeFlatVector<int32_t>(32, [](auto row) { return row; }),
            makeFlatVector<int32_t>(32, [](auto row) { return 1000 + row; }),
        });
  });

  HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
      .numDrivers(numDrivers_)
      .probeKeys({"t0"})
      .probeVectors(std::move(probeVectors))
      .buildKeys({"u0"})
      .buildVectors(std::move(buildVectors))
      .joinType(core::JoinType::kAnti)
      .nullAware(true)
      .joinFilter("t1 > u1")
      .joinOutputLayout({"t0", "t1"})
      .referenceQuery(
          "SELECT t.* FROM t WHERE t0 NOT IN (SELECT u0 FROM u WHERE t1 > u1)")
      .checkSpillStats(false)
      .run();
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    MultiThreadedHashJoinTest,
    MultiThreadedHashJoinTest,
    testing::ValuesIn(MultiThreadedHashJoinTest::getTestParams()),
    [](const testing::TestParamInfo<TestParam>& info) {
      return TestParamToName(info.param);
    });

VELOX_INSTANTIATE_TEST_SUITE_P(
    HashJoinTest,
    HashJoinTest,
    testing::ValuesIn(HashJoinTest::getTestParams()),
    [](const testing::TestParamInfo<TestParam>& info) {
      return TestParamToName(info.param);
    });

} // namespace
} // namespace facebook::velox::exec
