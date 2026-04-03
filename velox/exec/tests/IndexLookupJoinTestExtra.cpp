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

#include "fmt/format.h"
#include "folly/synchronization/EventCount.h"
#include "gmock/gmock.h"
#include "gtest/gtest-matchers.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/Connector.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/IndexLookupJoin.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/IndexLookupJoinTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TestIndexStorageConnector.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace facebook::velox::exec::test {
namespace {
struct TestParam {
  bool asyncLookup;
  int32_t numPrefetches;
  bool serialExecution;
  bool hasNullKeys;
  bool needsIndexSplit;

  TestParam(
      bool _asyncLookup,
      int32_t _numPrefetches,
      bool _serialExecution,
      bool _hasNullKeys,
      bool _needsIndexSplit = false)
      : asyncLookup(_asyncLookup),
        numPrefetches(_numPrefetches),
        serialExecution(_serialExecution),
        hasNullKeys(_hasNullKeys),
        needsIndexSplit(_needsIndexSplit) {}

  std::string toString() const {
    return fmt::format(
        "asyncLookup={}, numPrefetches={}, serialExecution={}, hasNullKeys={}, needsIndexSplit={}",
        asyncLookup,
        numPrefetches,
        serialExecution,
        hasNullKeys,
        needsIndexSplit);
  }
};

class IndexLookupJoinTest : public IndexLookupJoinTestBase,
                            public testing::WithParamInterface<TestParam> {
 public:
  static std::vector<TestParam> getTestParams() {
    std::vector<TestParam> testParams;
    for (bool asyncLookup : {false, true}) {
      for (int numPrefetches : {0, 3}) {
        for (bool serialExecution : {false, true}) {
          for (bool hasNullKeys : {false, true}) {
            for (bool needsIndexSplit : {false, true}) {
              // Serial execution doesn't support index split as it requires
              // single-threaded execution which is incompatible with the
              // split-based parallelism used by index lookup join.
              if (serialExecution && needsIndexSplit) {
                continue;
              }
              testParams.emplace_back(
                  asyncLookup,
                  numPrefetches,
                  serialExecution,
                  hasNullKeys,
                  needsIndexSplit);
            }
          }
        }
      }
    }
    return testParams;
  }

 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    core::PlanNode::registerSerDe();
    connector::hive::HiveColumnHandle::registerSerDe();
    Type::registerSerDe();
    core::ITypedExpr::registerSerDe();
    TestIndexConnectorFactory::registerConnector(connectorCpuExecutor_.get());

    keyType_ = ROW({"u0", "u1", "u2"}, {BIGINT(), BIGINT(), BIGINT()});
    valueType_ = ROW({"u3", "u4", "u5"}, {BIGINT(), BIGINT(), VARCHAR()});
    tableType_ = concat(keyType_, valueType_);
    probeType_ = ROW(
        {"t0", "t1", "t2", "t3", "t4", "t5"},
        {BIGINT(), BIGINT(), BIGINT(), BIGINT(), ARRAY(BIGINT()), VARCHAR()});
  }

  void TearDown() override {
    connector::unregisterConnector(kTestIndexConnectorName);
    HiveConnectorTestBase::TearDown();
  }

  void testSerde(const core::PlanNodePtr& plan) {
    auto serialized = plan->serialize();
    auto copy = ISerializable::deserialize<core::PlanNode>(serialized, pool());
    ASSERT_EQ(plan->toString(true, true), copy->toString(true, true));
  }

  // Makes index table handle with the specified index table and async lookup
  // flag.
  static std::shared_ptr<TestIndexTableHandle> makeIndexTableHandle(
      const std::shared_ptr<TestIndexTable>& indexTable,
      bool asyncLookup,
      bool needsIndexSplit = false) {
    return std::make_shared<TestIndexTableHandle>(
        kTestIndexConnectorName, indexTable, asyncLookup, needsIndexSplit);
  }

  static connector::ColumnHandleMap makeIndexColumnHandles(
      const std::vector<std::string>& names) {
    connector::ColumnHandleMap handles;
    for (const auto& name : names) {
      handles.emplace(name, std::make_shared<TestIndexColumnHandle>(name));
    }

    return handles;
  }

  const std::unique_ptr<folly::CPUThreadPoolExecutor> connectorCpuExecutor_{
      std::make_unique<folly::CPUThreadPoolExecutor>(128)};
};

// Verifies that when splitOutput_ is false, trailing input rows that have no
// lookup matches are included in the current output batch rather than being
// emitted in a separate batch via produceRemainingOutputForLeftJoin.
TEST_P(IndexLookupJoinTest, leftJoinTrailingMissesWithNoSplitOutput) {
  IndexTableData tableData;
  generateIndexTableData({500, 1, 1}, tableData, pool_);

  struct {
    int numProbeBatches;
    int numRowsPerProbeBatch;
    int maxBatchRows;
    int equalMatchPct;
    bool splitOutput;

    std::string debugString() const {
      return fmt::format(
          "numProbeBatches: {}, numRowsPerProbeBatch: {}, maxBatchRows: {}, equalMatchPct: {}, splitOutput: {}",
          numProbeBatches,
          numRowsPerProbeBatch,
          maxBatchRows,
          equalMatchPct,
          splitOutput);
    }
  } testSettings[] = {
      // With splitOutput=false, trailing misses should be folded into the
      // current batch. With splitOutput=true, they are emitted separately.
      {10, 100, 200, 10, false},
      {10, 100, 200, 50, false},
      {10, 100, 200, 2, false},
      {1, 500, 1000, 10, false},
      {1, 500, 1000, 50, false},
      {10, 50, 200, 10, false},
      // With no matches, all rows are misses.
      {10, 100, 200, 0, false},
      // 100% matches - no trailing misses exist.
      {10, 100, 200, 100, false},
      // splitOutput=true as comparison.
      {10, 100, 200, 10, true},
      {10, 100, 200, 50, true},
      {10, 100, 200, 0, true},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    const auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numRowsPerProbeBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1", "t2"},
        GetParam().hasNullKeys,
        {},
        {},
        testData.equalMatchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/3,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType({"u0", "u1", "u2", "u5"}),
        makeIndexColumnHandles({"u0", "u1", "u2", "u5"}));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        /*filter=*/"",
        /*hasMarker=*/false,
        core::JoinType::kLeft,
        {"t4", "u5"});
    AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
    queryBuilder.plan(plan)
        .config(
            core::QueryConfig::kIndexLookupJoinMaxPrefetchBatches,
            std::to_string(GetParam().numPrefetches))
        .config(
            core::QueryConfig::kPreferredOutputBatchRows,
            std::to_string(testData.maxBatchRows))
        .config(
            core::QueryConfig::kPreferredOutputBatchBytes,
            std::to_string(1ULL << 30))
        .config(
            core::QueryConfig::kIndexLookupJoinSplitOutput,
            testData.splitOutput ? "true" : "false")
        .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
        .serialExecution(GetParam().serialExecution)
        .barrierExecution(GetParam().serialExecution);
    if (GetParam().needsIndexSplit) {
      queryBuilder.split(
          indexScanNodeId_,
          Split(
              std::make_shared<TestIndexConnectorSplit>(
                  kTestIndexConnectorName)));
    }
    const auto task = queryBuilder.assertResults(
        "SELECT t.c4, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

    // Verify match column correctness for all cases.
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        /*filter=*/"",
        /*hasMarker=*/true,
        core::JoinType::kLeft,
        {"t4", "u5"});
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

DEBUG_ONLY_TEST_P(IndexLookupJoinTest, runtimeStats) {
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{2};
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      100,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::asyncLookup",
      std::function<void(void*)>([&](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // NOLINT
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  auto task = runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      100,
      0,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

  auto taskStats = toPlanStats(task->taskStats());

  // Check IndexSource stats - lookup timing should be here, not on join stats.
  auto& indexSourceStats = taskStats.at(indexScanNodeId_);
  ASSERT_EQ(indexSourceStats.addInputTiming.count, numProbeBatches);
  ASSERT_GT(indexSourceStats.addInputTiming.cpuNanos, 0);
  ASSERT_GT(indexSourceStats.addInputTiming.wallNanos, 0);

  // Verify that backgroundTiming was cleared from join stats (moved to
  // IndexSource to avoid double counting).
  auto& operatorStats = taskStats.at(joinNodeId_);
  ASSERT_EQ(operatorStats.backgroundTiming.count, 0);
  ASSERT_EQ(operatorStats.backgroundTiming.cpuNanos, 0);
  ASSERT_EQ(operatorStats.backgroundTiming.wallNanos, 0);

  // Check runtime stats are present on IndexSource.
  auto runtimeStats = indexSourceStats.customStats;
  ASSERT_EQ(
      runtimeStats.at(std::string(IndexLookupJoin::kConnectorLookupWallTime))
          .count,
      numProbeBatches);
  ASSERT_GT(
      runtimeStats.at(std::string(IndexLookupJoin::kConnectorLookupWallTime))
          .sum,
      0);
  ASSERT_EQ(
      runtimeStats.at(std::string(IndexLookupJoin::kClientLookupWaitWallTime))
          .count,
      numProbeBatches);
  ASSERT_GT(
      runtimeStats.at(std::string(IndexLookupJoin::kClientLookupWaitWallTime))
          .sum,
      0);
  ASSERT_EQ(
      runtimeStats.at(std::string(IndexLookupJoin::kConnectorResultPrepareTime))
          .count,
      numProbeBatches);
  ASSERT_GT(
      runtimeStats.at(std::string(IndexLookupJoin::kConnectorResultPrepareTime))
          .sum,
      0);
  ASSERT_EQ(
      runtimeStats.count(
          std::string(IndexLookupJoin::kClientRequestProcessTime)),
      0);
  ASSERT_EQ(
      runtimeStats.count(
          std::string(IndexLookupJoin::kClientResultProcessTime)),
      0);
  ASSERT_EQ(
      runtimeStats.count(std::string(IndexLookupJoin::kClientLookupResultSize)),
      0);
  ASSERT_EQ(
      runtimeStats.count(
          std::string(IndexLookupJoin::kClientLookupResultRawSize)),
      0);
  ASSERT_THAT(
      indexSourceStats.toString(true, true),
      testing::MatchesRegex(".*Runtime stats.*connectorLookupWallNanos:.*"));
  ASSERT_THAT(
      indexSourceStats.toString(true, true),
      testing::MatchesRegex(".*Runtime stats.*clientlookupWaitWallNanos.*"));
  ASSERT_THAT(
      indexSourceStats.toString(true, true),
      testing::MatchesRegex(
          ".*Runtime stats.*connectorResultPrepareCpuNanos.*"));
}

/// Verifies that IndexLookupJoin's StatsSplitter correctly reports separate
/// operator stats for both the IndexLookupJoin node and the IndexSource node.
/// This ensures IndexSource appears with its own CPU/Scheduled/Output stats
/// in the query plan visualization.
DEBUG_ONLY_TEST_P(IndexLookupJoinTest, statsSplitter) {
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{2};
  const int batchSize{100};
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      batchSize,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  // Add a small delay in async lookup to ensure timing stats are captured.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::asyncLookup",
      std::function<void(void*)>([&](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  auto task = runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      100,
      0,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

  auto taskStats = toPlanStats(task->taskStats());

  // Verify that both the IndexLookupJoin node and IndexSource node have stats.
  ASSERT_TRUE(taskStats.count(joinNodeId_) > 0)
      << "IndexLookupJoin node stats missing";
  ASSERT_TRUE(taskStats.count(indexScanNodeId_) > 0)
      << "IndexSource node stats missing";

  const auto& joinStats = taskStats.at(joinNodeId_);
  const auto& indexSourceStats = taskStats.at(indexScanNodeId_);

  // Verify join stats have input from probe side.
  EXPECT_GT(joinStats.inputRows, 0);
  EXPECT_GT(joinStats.outputRows, 0);

  // Verify IndexSource stats have output positions (the lookup results).
  EXPECT_GT(indexSourceStats.outputRows, 0);
  EXPECT_EQ(indexSourceStats.outputRows, joinStats.outputRows);

  // Verify IndexSource stats have input positions (lookup keys sent to
  // connector).
  EXPECT_GT(indexSourceStats.inputRows, 0);
  EXPECT_GT(indexSourceStats.inputBytes, 0);
  // For inner join without filter, input rows should match join input rows
  // (all probe rows are sent as lookup keys).
  EXPECT_EQ(indexSourceStats.inputRows, joinStats.inputRows);

  // Verify IndexSource stats have timing from backgroundTiming (lookup time).
  // The addInputTiming should contain the lookup wall/cpu time.
  EXPECT_GT(indexSourceStats.addInputTiming.count, 0);

  // Verify runtime stats are present on IndexSource (connector metrics).
  // These include connector lookup wall time, etc.
  EXPECT_TRUE(
      indexSourceStats.customStats.count(
          std::string(IndexLookupJoin::kConnectorLookupWallTime)) > 0)
      << "IndexSource should have connector lookup wall time";
  EXPECT_GT(
      indexSourceStats.customStats
          .at(std::string(IndexLookupJoin::kConnectorLookupWallTime))
          .sum,
      0);

  // Verify that backgroundTiming was cleared from join stats (moved to
  // IndexSource to avoid double counting).
  EXPECT_EQ(joinStats.backgroundTiming.count, 0);
  EXPECT_EQ(joinStats.backgroundTiming.cpuNanos, 0);
  EXPECT_EQ(joinStats.backgroundTiming.wallNanos, 0);
}

/// Verifies that IndexSource stats report rows BEFORE the join filter is
/// applied, while IndexLookupJoin stats report rows AFTER the filter.
DEBUG_ONLY_TEST_P(IndexLookupJoinTest, statsSplitterWithFilter) {
  // Skip serial execution tests for simplicity - the stats behavior is the
  // same.
  if (GetParam().serialExecution || GetParam().hasNullKeys) {
    return;
  }

  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{2};
  const int batchSize{100};
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      batchSize,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      /*hasNullKeys=*/false,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  // Add a small delay in async lookup to ensure timing stats are captured.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::asyncLookup",
      std::function<void(void*)>([&](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  // Add a filter that should filter out approximately half the rows.
  // The filter "u3 % 2 = 0" will keep only even values of u3.
  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"u3 % 2 = 0",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  auto task = runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      100,
      0,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2 AND u.c3 % 2 = 0");

  auto taskStats = toPlanStats(task->taskStats());

  // Verify that both the IndexLookupJoin node and IndexSource node have stats.
  ASSERT_TRUE(taskStats.count(joinNodeId_) > 0)
      << "IndexLookupJoin node stats missing";
  ASSERT_TRUE(taskStats.count(indexScanNodeId_) > 0)
      << "IndexSource node stats missing";

  const auto& joinStats = taskStats.at(joinNodeId_);
  const auto& indexSourceStats = taskStats.at(indexScanNodeId_);

  // Verify join stats have input from probe side.
  EXPECT_GT(joinStats.inputRows, 0);
  EXPECT_GT(joinStats.outputRows, 0);

  // Verify IndexSource stats have output positions (rows before filter).
  EXPECT_GT(indexSourceStats.outputRows, 0);

  // KEY ASSERTION: IndexSource should have MORE rows than IndexLookupJoin
  // because IndexSource reports rows BEFORE the filter ("u3 % 2 = 0") is
  // applied.
  EXPECT_GT(indexSourceStats.outputRows, joinStats.outputRows)
      << "IndexSource should report more rows (before filter) than "
      << "IndexLookupJoin (after filter)";
}

TEST_P(IndexLookupJoinTest, DISABLED_barrier) {
  if (GetParam().needsIndexSplit) {
    return;
  }
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeSplits{5};
  const auto probeVectors = generateProbeInput(
      numProbeSplits,
      256,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});

  struct {
    int numPrefetches;
    bool barrierExecution;
    bool serialExecution;

    std::string debugString() const {
      return fmt::format(
          "numPrefetches {}, barrierExecution {}, serialExecution {}",
          numPrefetches,
          barrierExecution,
          serialExecution);
    }
  } testSettings[] = {
      {0, false, false},
      {0, false, true},
      {1, true, true},
      {1, false, true},
      {4, true, true},
      {4, false, true},
      {256, true, true},
      {256, false, true},
      {0, true, false},
      {0, false, false},
      {1, true, false},
      {1, false, false},
      {4, true, false},
      {4, false, false},
      {256, true, false},
      {256, false, false},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto task = runLookupQuery(
        plan,
        probeFiles,
        testData.serialExecution,
        testData.barrierExecution,
        32,
        testData.numPrefetches,
        GetParam().needsIndexSplit,
        "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

    const auto taskStats = task->taskStats();
    ASSERT_EQ(
        taskStats.numBarriers, testData.barrierExecution ? numProbeSplits : 0);
    ASSERT_EQ(taskStats.numFinishedSplits, numProbeSplits);
  }
}

TEST_P(IndexLookupJoinTest, nullKeys) {
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeSplits{5};
  const int probeBatchSize{256};
  const auto probeVectors = generateProbeInput(
      numProbeSplits,
      probeBatchSize,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      /*hasNullKeys=*/true,
      {},
      {},
      /*equalMatchPct=*/100);
  // Set some probe key vector to all nulls to trigger the case that entire
  // probe input is skipped.
  for (int i = 0; i < numProbeSplits; i += 2) {
    for (int row = 0; row < probeBatchSize; ++row) {
      probeVectors[i]->childAt(i % keyType_->size())->setNull(row, true);
    }
  }
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      columnHandles;
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  const auto innerPlan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});

  runLookupQuery(
      innerPlan,
      probeFiles,
      /*serialExecution=*/GetParam().serialExecution,
      /*barrierExecution=*/GetParam().serialExecution,
      32,
      GetParam().numPrefetches,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

  const auto leftPlan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kLeft,
      {"u3", "t5"});

  runLookupQuery(
      leftPlan,
      probeFiles,
      /*serialExecution=*/GetParam().serialExecution,
      /*barrierExecution=*/GetParam().serialExecution,
      32,
      GetParam().numPrefetches,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

  const auto probeScanId = probeScanNodeId_;
  auto planWithMatchColumn = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/true,
      core::JoinType::kLeft,
      {"u3", "t5"});
  verifyResultWithMatchColumn(
      leftPlan,
      probeScanId,
      planWithMatchColumn,
      probeScanNodeId_,
      probeFiles,
      GetParam().needsIndexSplit);
}

TEST_P(IndexLookupJoinTest, joinFuzzer) {
  IndexTableData tableData;
  generateIndexTableData({1024, 1, 1}, tableData, pool_);
  const auto probeVectors = generateProbeInput(
      50, 256, 1, tableData, pool_, {"t0", "t1", "t2"}, GetParam().hasNullKeys);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto scanOutput = tableType_->names();
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(scanOutput.begin(), scanOutput.end(), g);
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType(scanOutput),
      makeIndexColumnHandles(scanOutput));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0"},
      {"u0"},
      {"contains(t4, u1)", "u2 between t1 and t2"},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u0", "u4", "t0", "t1", "t4"});
  runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      32,
      GetParam().numPrefetches,
      GetParam().needsIndexSplit,
      "SELECT u.c0, u.c1, u.c2, u.c3, u.c4, u.c5, t.c0, t.c1, t.c2, t.c3, t.c4, t.c5 FROM t, u WHERE t.c0 = u.c0 AND array_contains(t.c4, u.c1) AND u.c2 BETWEEN t.c1 AND t.c2");
}

TEST_P(IndexLookupJoinTest, tableRowsWithDuplicateKeys) {
  IndexTableData tableData;
  generateIndexTableData({10, 1, 1}, tableData, pool_);
  for (int i = 0; i < keyType_->size(); ++i) {
    tableData.keyVectors->childAt(i) = makeFlatVector<int64_t>(
        tableData.keyVectors->childAt(i)->size(),
        [](auto /*unused*/) { return 1; });
    tableData.tableVectors->childAt(i) = makeFlatVector<int64_t>(
        tableData.keyVectors->childAt(i)->size(),
        [](auto /*unused*/) { return 1; });
  }

  auto probeVectors = generateProbeInput(
      4, 32, 1, tableData, pool_, {"t0", "t1", "t2"}, false, {}, {}, 100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto scanOutput = tableType_->names();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType(scanOutput),
      makeIndexColumnHandles(scanOutput));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      scanOutput);
  runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      32,
      GetParam().numPrefetches,
      GetParam().needsIndexSplit,
      "SELECT u.c0, u.c1, u.c2, u.c3, u.c4, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 = t.c2");
}

TEST_P(IndexLookupJoinTest, withFilter) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numRowsPerProbeBatch;
    int matchPct;
    std::vector<std::string> scanOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string filter;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}, numProbeBatches: {}, numRowsPerProbeBatch: {}, matchPct: {}, "
          "scanOutputColumns: {}, outputColumns: {}, joinType: {}, filter: {}, "
          "duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numRowsPerProbeBatch,
          matchPct,
          folly::join(",", scanOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          filter,
          duckDbVerifySql);
    }
  } testSettings[] = {
      // Inner join with filter on probe side
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "t3 % 2 = 0",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c3 % 2 = 0"},
      // Inner join with filter always be true.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "t3 = t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c3 = t.c3"},
      // Inner join with filter always be false.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "t3 != t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c3 != t.c3"},

      // Inner join with filter on lookup side
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "u3 % 2 = 0",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND u.c3 % 2 = 0"},
      // Inner join with filter always be true.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "u3 = u3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND u.c3 = u.c3"},
      // Inner join with filter always be false.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "u3 != u3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND u.c3 != u.c3"},

      // Inner join with filter on both side
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "u3 % 2 = 0 AND t3 % 2 = 0",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND u.c3 % 2 = 0 AND t.c3 % 2 = 0"},
      // Inner join with filter always be true.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "u3 = u3 AND t3 = t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND u.c3 = u.c3 AND t.c3 = t.c3"},
      // Inner join with filter always be false.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "u3 != u3 AND t3 != t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND u.c3 != u.c3 AND t.c3 != t.c3"},

      // Left join with filter on probe side
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "t3 % 2 = 0",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c3 % 2 = 0"},
      // Left join with filter always be true.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "t3 = t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c3 = t.c3"},
      // Inner join with filter always be false.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "t3 != t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c3 != t.c3"},

      // Left join with filter on lookup side
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "u3 % 2 = 0",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c3 % 2 = 0"},
      // Inner join with filter always be true.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "u3 = u3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c3 = u.c3"},
      // Left join with filter always be false.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "u3 != u3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c3 != u.c3"},

      // Left join with filter on both side
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "u3 % 2 = 0 AND t3 % 2 = 0",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c3 % 2 = 0 AND t.c3 % 2 = 0"},
      // Left join with filter always be true.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "u3 = u3 AND t3 = t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c3 = u.c3 AND t.c3 = t.c3"},
      // Left join with filter always be false.
      {{100, 1, 1},
       5,
       100,
       80,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "u3 != u3 AND t3 != t3",
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c3 != u.c3 AND t.c3 != t.c3"}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);
    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numRowsPerProbeBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1", "t2"},
        GetParam().hasNullKeys,
        {},
        {},
        testData.matchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/3,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.scanOutputColumns),
        makeIndexColumnHandles(testData.scanOutputColumns));

    // Create a plan with filter
    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        testData.filter,
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);

    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);

    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        testData.filter,
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, mixedFilterBatches) {
  // Create IndexTableData using VectorTestBase utilities
  IndexTableData tableData;

  const std::string dummyString("test");
  StringView dummyStringView(dummyString);
  // Create table key data (u0, u1, u2) using makeFlatVector
  auto u0 = makeFlatVector<int64_t>(64, [&](auto row) { return row % 8; });
  auto u1 = makeFlatVector<int64_t>(64, [&](auto row) { return row % 8; });
  auto u2 = makeFlatVector<int64_t>(64, [&](auto row) { return row % 8; });
  tableData.keyVectors = makeRowVector({"u0", "u1", "u2"}, {u0, u1, u2});

  // Create table value data (u3, u4, u5) using makeFlatVector
  auto u3 = makeFlatVector<int64_t>(64, [&](auto row) { return row; });
  auto u4 = makeFlatVector<int64_t>(64, [&](auto row) { return row; });
  auto u5 = makeFlatVector<StringView>(
      64, [&](auto /*unused*/) { return dummyStringView; });
  tableData.valueVectors = makeRowVector({"u3", "u4", "u5"}, {u3, u4, u5});

  // Create complete table data by combining key and value data
  tableData.tableVectors = makeRowVector(
      {"u0", "u1", "u2", "u3", "u4", "u5"}, {u0, u1, u2, u3, u4, u5});

  // Create probe vectors using makeArrayVectorFromJson in a loop
  std::vector<RowVectorPtr> probeVectors;
  probeVectors.reserve(5);
  for (int i = 0; i < 5; ++i) {
    probeVectors.push_back(makeRowVector(
        {"t0", "t1", "t2", "t3", "t4", "t5"},
        {makeFlatVector<int64_t>(128, [&](auto row) { return row; }),
         makeFlatVector<int64_t>(128, [&](auto row) { return row; }),
         makeFlatVector<int64_t>(128, [&](auto row) { return row; }),
         makeFlatVector<int64_t>(128, [&](auto row) { return row; }),
         makeArrayVector<int64_t>(
             128,
             [](vector_size_t /*unused*/) { return 1; },
             [](vector_size_t, vector_size_t) { return 1; }),
         makeFlatVector<StringView>(
             128, [&](auto /*unused*/) { return dummyStringView; })}));
  }

  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      "t3 > 4",
      /*hasMarker=*/false,
      core::JoinType::kLeft,
      {"t1", "u1", "u2", "u3", "u5"});

  AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
  queryBuilder.plan(plan)
      .config(
          core::QueryConfig::kIndexLookupJoinMaxPrefetchBatches,
          std::to_string(GetParam().numPrefetches))
      .config(core::QueryConfig::kPreferredOutputBatchRows, "4")
      .config(core::QueryConfig::kIndexLookupJoinSplitOutput, "true")
      .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
      .serialExecution(GetParam().serialExecution)
      .barrierExecution(GetParam().serialExecution);
  if (GetParam().needsIndexSplit) {
    queryBuilder.split(
        indexScanNodeId_,
        Split(
            std::make_shared<TestIndexConnectorSplit>(
                kTestIndexConnectorName)));
  }
  queryBuilder.assertResults(
      "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2 AND t.c3 > 4");
}

// Tests the index split handling behavior of the IndexLookupJoin operator.
// When needsIndexSplit is true:
// - The operator blocks waiting for splits until it receives them
// - Works correctly with various split counts (1, 2, 3 splits)
// - Fails when no splits are provided before the no-more-splits signal
// This test only runs when GetParam().needsIndexSplit is true.
DEBUG_ONLY_TEST_P(IndexLookupJoinTest, needsIndexSplit) {
  if (!GetParam().needsIndexSplit) {
    return;
  }
  keyType_ = ROW({"u0"}, {BIGINT()});
  valueType_ = ROW({"u1", "u2"}, {BIGINT(), VARCHAR()});
  tableType_ = concat(keyType_, valueType_);
  probeType_ = ROW({"t0", "t1", "t2"}, {BIGINT(), BIGINT(), VARCHAR()});

  IndexTableData tableData;
  generateIndexTableData({100}, tableData, pool_);
  const auto probeVectors =
      generateProbeInput(3, 100, 1, tableData, pool_, {"t0"});
  const auto probeFiles = createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());

  const auto duckDbVerifySql =
      "SELECT t.c0, t.c1, u.c1, u.c2 FROM t LEFT JOIN u ON t.c0 = u.c0";

  struct {
    int numIndexSplits;
    bool expectFailure;

    std::string debugString() const {
      return fmt::format(
          "numIndexSplits: {}, expectFailure: {}",
          numIndexSplits,
          expectFailure);
    }
  } testSettings[] = {
      {1, false}, // One split - should succeed
      {2, false}, // Two splits - should succeed
      {3, false}, // Three splits - should succeed
      {0, true}, // No splits - should fail
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    // Create index table handle that requires splits.
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, /*needsIndexSplit=*/true);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType({"u0", "u1", "u2"}),
        makeIndexColumnHandles({"u0", "u1", "u2"}));
    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0"},
        {"u0"},
        {},
        "",
        /*hasMarker=*/false,
        core::JoinType::kLeft,
        {"t0", "t1", "u1", "u2"});

    // Track the number of times collectIndexSplits is called.
    std::atomic_int collectSplitCallCount{0};
    folly::EventCount waitCollectSplit;
    std::atomic_bool waitCollectSplitFlag{true};

    std::mutex mutex;
    std::shared_ptr<Task> task;
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::IndexLookupJoin::collectIndexSplits",
        std::function<void(const IndexLookupJoin*)>(
            [&](const IndexLookupJoin* op) {
              {
                std::lock_guard<std::mutex> lock(mutex);
                if (task == nullptr) {
                  task = op->operatorCtx()->task();
                  // Signal that we've entered collectIndexSplits for the first
                  // time.
                  waitCollectSplitFlag = false;
                  waitCollectSplit.notifyAll();
                }
              }
              ++collectSplitCallCount;
            }));

    // Run the query in a separate thread without providing index splits
    // upfront. The main thread will provide splits after the task starts.
    std::thread queryThread([&] {
      AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
      queryBuilder.plan(plan).splits(
          probeScanNodeId_, makeHiveConnectorSplits(probeFiles));
      // Do NOT provide index splits here - they will be provided by the
      // main thread after the task starts.

      if (testData.expectFailure) {
        VELOX_ASSERT_THROW(queryBuilder.copyResults(pool()), "");
      } else {
        queryBuilder.assertResults(duckDbVerifySql);
      }
    });

    // Wait for collectIndexSplits to be called.
    waitCollectSplit.await([&] { return !waitCollectSplitFlag.load(); });
    // Wait for 1 second and expect the task to NOT finish since it's
    // waiting for splits.
    std::this_thread::sleep_for(std::chrono::seconds(1)); // NOLINT
    {
      std::lock_guard<std::mutex> lock(mutex);
      ASSERT_NE(task, nullptr);
      ASSERT_EQ(task->state(), TaskState::kRunning)
          << "Task should still be running while waiting for splits";

      if (testData.expectFailure) {
        // Signal no more splits immediately to trigger the failure.
        task->noMoreSplits(indexScanNodeId_);
      } else {
        // Add the specified number of index splits.
        for (int i = 0; i < testData.numIndexSplits; ++i) {
          task->addSplit(
              indexScanNodeId_,
              Split(
                  std::make_shared<TestIndexConnectorSplit>(
                      kTestIndexConnectorName)));
        }
        // Signal no more splits to allow the task to finish.
        task->noMoreSplits(indexScanNodeId_);
      }
    }

    queryThread.join();

    if (!testData.expectFailure) {
      // Verify collectIndexSplits was called the expected number of times:
      // once initially when blocked, plus once after splits are available.
      ASSERT_EQ(collectSplitCallCount.load(), 2)
          << "collectIndexSplits should be called once initially (blocked), "
             "then once when splits are available";
    }
  }
}

// Tests that when needsIndexSplit is false, the operator does NOT call
// collectIndexSplits. The query should complete successfully without waiting
// for any index splits.
// This test only runs when GetParam().needsIndexSplit is false.
DEBUG_ONLY_TEST_P(IndexLookupJoinTest, noNeedsIndexSplitNoCollect) {
  if (GetParam().needsIndexSplit) {
    return;
  }

  keyType_ = ROW({"u0"}, {BIGINT()});
  valueType_ = ROW({"u1", "u2"}, {BIGINT(), VARCHAR()});
  tableType_ = concat(keyType_, valueType_);
  probeType_ = ROW({"t0", "t1", "t2"}, {BIGINT(), BIGINT(), VARCHAR()});

  IndexTableData tableData;
  generateIndexTableData({100}, tableData, pool_);
  const auto probeVectors =
      generateProbeInput(3, 100, 1, tableData, pool_, {"t0"});
  const auto probeFiles = createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());

  const auto duckDbVerifySql =
      "SELECT t.c0, t.c1, u.c1, u.c2 FROM t LEFT JOIN u ON t.c0 = u.c0";

  // Create index table handle with needsIndexSplit=false.
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, /*needsIndexSplit=*/false);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2"}),
      makeIndexColumnHandles({"u0", "u1", "u2"}));
  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0"},
      {"u0"},
      {},
      "",
      /*hasMarker=*/false,
      core::JoinType::kLeft,
      {"t0", "t1", "u1", "u2"});

  // Track if collectIndexSplits is ever called (it should NOT be).
  std::atomic_bool collectSplitsCalled{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::IndexLookupJoin::collectIndexSplits",
      std::function<void(const IndexLookupJoin*)>(
          [&](const IndexLookupJoin* /*op*/) { collectSplitsCalled = true; }));

  // Run the query - it should complete without waiting for splits.
  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
      .assertResults(duckDbVerifySql);

  // Verify collectIndexSplits was never called since needsIndexSplit=false.
  ASSERT_FALSE(collectSplitsCalled.load())
      << "collectIndexSplits should not be called when needsIndexSplit is "
         "false";
}

// Tests that when needsIndexSplit is false, adding splits or signaling
// no-more-splits should fail because the index scan node is not registered
// for split collection.
// This test only runs when GetParam().needsIndexSplit is false.
DEBUG_ONLY_TEST_P(IndexLookupJoinTest, noNeedsIndexSplitSplitOperationFails) {
  if (GetParam().needsIndexSplit) {
    return;
  }

  keyType_ = ROW({"u0"}, {BIGINT()});
  valueType_ = ROW({"u1", "u2"}, {BIGINT(), VARCHAR()});
  tableType_ = concat(keyType_, valueType_);
  probeType_ = ROW({"t0", "t1", "t2"}, {BIGINT(), BIGINT(), VARCHAR()});

  IndexTableData tableData;
  generateIndexTableData({100}, tableData, pool_);
  const auto probeVectors =
      generateProbeInput(3, 100, 1, tableData, pool_, {"t0"});
  const auto probeFiles = createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());

  const auto duckDbVerifySql =
      "SELECT t.c0, t.c1, u.c1, u.c2 FROM t LEFT JOIN u ON t.c0 = u.c0";

  // Create index table handle with needsIndexSplit=false.
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, /*needsIndexSplit=*/false);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2"}),
      makeIndexColumnHandles({"u0", "u1", "u2"}));
  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0"},
      {"u0"},
      {},
      "",
      /*hasMarker=*/false,
      core::JoinType::kLeft,
      {"t0", "t1", "u1", "u2"});

  // Test both addSplit and noMoreSplits operations.
  for (bool testAddSplit : {true, false}) {
    SCOPED_TRACE(fmt::format("testAddSplit: {}", testAddSplit));

    // Use TestValue to block the Task from starting to allow us to verify
    // that split operations fail.
    folly::EventCount taskEnterWait;
    std::atomic_bool taskEnterWaitFlag{true};
    std::shared_ptr<Task> task;
    std::mutex mutex;

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::enter",
        std::function<void(Task*)>([&](Task* taskPtr) {
          {
            std::lock_guard<std::mutex> lock(mutex);
            if (task == nullptr) {
              task = taskPtr->shared_from_this();
            }
          }
          // Block until the test thread signals to continue.
          taskEnterWait.await([&] { return !taskEnterWaitFlag.load(); });
        }));

    // Run the query in a separate thread.
    std::thread queryThread([&] {
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
          .assertResults(duckDbVerifySql);
    });

    // Wait a bit for the task to start and hit the TestValue.
    std::this_thread::sleep_for(std::chrono::seconds(1)); // NOLINT

    {
      std::lock_guard<std::mutex> lock(mutex);
      if (task != nullptr) {
        if (testAddSplit) {
          // Try to add a split - this should fail because the index scan node
          // is not registered for split collection.
          VELOX_ASSERT_THROW(
              task->addSplit(
                  indexScanNodeId_,
                  Split(
                      std::make_shared<TestIndexConnectorSplit>(
                          kTestIndexConnectorName))),
              "Splits can be associated only with leaf plan nodes which require splits. Plan node ID 0 doesn't refer to such plan node");
        } else {
          // Try to signal no more splits - this should fail because the index
          // scan node is not registered for split collection.
          VELOX_ASSERT_THROW(
              task->noMoreSplits(indexScanNodeId_),
              "Splits can be associated only with leaf plan nodes which require splits. Plan node ID 0 doesn't refer to such plan node.");
        }
      }
    }

    // Allow the query to complete.
    taskEnterWaitFlag = false;
    taskEnterWait.notifyAll();

    queryThread.join();
  }
}
} // namespace

VELOX_INSTANTIATE_TEST_SUITE_P(
    IndexLookupJoinTest,
    IndexLookupJoinTest,
    testing::ValuesIn(IndexLookupJoinTest::getTestParams()),
    [](const testing::TestParamInfo<TestParam>& info) {
      return fmt::format(
          "{}_{}prefetches_{}_{}_{}",
          info.param.asyncLookup ? "async" : "sync",
          info.param.numPrefetches,
          info.param.serialExecution ? "serial" : "parallel",
          info.param.hasNullKeys ? "nullKeys" : "noNullKeys",
          info.param.needsIndexSplit ? "needsIndexSplit" : "noSplit");
    });
} // namespace facebook::velox::exec::test
