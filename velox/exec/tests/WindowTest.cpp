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
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/RowsStreamingWindowBuild.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::exec {

namespace {

class WindowTest : public OperatorTestBase {
 public:
  void SetUp() override {
    OperatorTestBase::SetUp();
    window::prestosql::registerAllWindowFunctions();
    filesystems::registerLocalFileSystem();
  }
};

TEST_F(WindowTest, spill) {
  const vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"d", "p", "s"},
      {
          // Payload.
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          // Partition key.
          makeFlatVector<int16_t>(size, [](auto row) { return row % 11; }),
          // Sorting key.
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      });

  createDuckDbTable({data});

  core::PlanNodeId windowId;
  auto plan = PlanBuilder()
                  .values(split(data, 10))
                  .window({"row_number() over (partition by p order by s)"})
                  .capturePlanNodeId(windowId)
                  .planNode();

  auto spillDirectory = TempDirectoryPath::create();
  TestScopedSpillInjection scopedSpillInjection(100);
  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kWindowSpillEnabled, "true")
          .spillDirectory(spillDirectory->getPath())
          .assertResults(
              "SELECT *, row_number() over (partition by p order by s) FROM tmp");

  auto taskStats = exec::toPlanStats(task->taskStats());
  const auto& stats = taskStats.at(windowId);

  ASSERT_GT(stats.spilledBytes, 0);
  ASSERT_GT(stats.spilledRows, 0);
  ASSERT_GT(stats.spilledFiles, 0);
  ASSERT_GT(stats.spilledPartitions, 0);
}

TEST_F(WindowTest, rowBasedStreamingWindowOOM) {
  const vector_size_t size = 1'000'000;
  auto data = makeRowVector(
      {"d", "p", "s"},
      {
          // Payload.
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          // Partition key.
          makeFlatVector<int16_t>(size, [](auto row) { return row; }),
          // Sorting key.
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      });

  createDuckDbTable({data});

  // Abstract the common values vector split.
  auto valuesSplit = split(data, 10);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  auto queryCtx = core::QueryCtx::create(executor_.get());
  queryCtx->testingOverrideMemoryPool(memory::memoryManager()->addRootPool(
      queryCtx->queryId(),
      8'388'608 /* 8MB */,
      exec::MemoryReclaimer::create()));

  params.queryCtx = queryCtx;

  auto testWindowBuild = [&](bool useStreamingWindow) {
    if (useStreamingWindow) {
      params.planNode =
          PlanBuilder(planNodeIdGenerator)
              .values(valuesSplit)
              .streamingWindow(
                  {"row_number() over (partition by p order by s)"})
              .project({"d"})
              .singleAggregation({}, {"sum(d)"})
              .planNode();

      readCursor(params, [](Task*) {});
    } else {
      params.planNode =
          PlanBuilder(planNodeIdGenerator)
              .values(valuesSplit)
              .window({"row_number() over (partition by p order by s)"})
              .project({"d"})
              .singleAggregation({}, {"sum(d)"})
              .planNode();

      VELOX_ASSERT_THROW(
          readCursor(params, [](Task*) {}),
          "Exceeded memory pool capacity after attempt to grow capacity through arbitration.");
    }
  };
  // RowStreamingWindow will not OOM.
  testWindowBuild(true);
  // SortBasedWindow will OOM.
  testWindowBuild(false);
}

TEST_F(WindowTest, rowBasedStreamingWindowMemoryUsage) {
  auto memoryUsage = [&](bool useStreamingWindow, vector_size_t size) {
    auto data = makeRowVector(
        {"d", "p", "s"},
        {
            // Payload.
            makeFlatVector<int64_t>(size, [](auto row) { return row; }),
            // Partition key.
            makeFlatVector<int16_t>(size, [](auto row) { return row % 11; }),
            // Sorting key.
            makeFlatVector<int32_t>(size, [](auto row) { return row; }),
        });

    createDuckDbTable({data});

    // Abstract the common values vector split.
    auto valuesSplit = split(data, 10);
    core::PlanNodeId windowId;
    auto builder = PlanBuilder().values(valuesSplit);
    if (useStreamingWindow) {
      builder.orderBy({"p", "s"}, false)
          .streamingWindow({"row_number() over (partition by p order by s)"});
    } else {
      builder.window({"row_number() over (partition by p order by s)"});
    }
    auto plan = builder.capturePlanNodeId(windowId).planNode();
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
            .assertResults(
                "SELECT *, row_number() over (partition by p order by s) FROM tmp");

    return exec::toPlanStats(task->taskStats()).at(windowId).peakMemoryBytes;
  };

  const vector_size_t smallSize = 100'000;
  const vector_size_t largeSize = 1'000'000;
  // As the volume of data increases, the peak memory usage of the sort-based
  // window will increase (2418624 vs 17098688). Since the peak memory usage of
  // the RowBased Window represents the one batch data in a single partition,
  // the peak memory usage will not increase as the volume of data grows.
  auto sortWindowSmallUsage = memoryUsage(false, smallSize);
  auto sortWindowLargeUsage = memoryUsage(false, largeSize);
  ASSERT_GT(sortWindowLargeUsage, sortWindowSmallUsage);

  auto rowWindowSmallUsage = memoryUsage(true, smallSize);
  auto rowWindowLargeUsage = memoryUsage(true, largeSize);
  ASSERT_EQ(rowWindowSmallUsage, rowWindowLargeUsage);
}

DEBUG_ONLY_TEST_F(WindowTest, rankRowStreamingWindowBuild) {
  auto data = makeRowVector(
      {"c1"},
      {makeFlatVector<int64_t>(std::vector<int64_t>{1, 1, 1, 1, 1, 2, 2})});

  createDuckDbTable({data});

  const std::vector<std::string> kClauses = {
      "rank() over (order by c1 rows unbounded preceding)"};

  auto plan = PlanBuilder()
                  .values({data})
                  .orderBy({"c1"}, false)
                  .streamingWindow(kClauses)
                  .planNode();

  std::atomic_bool isStreamCreated{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::RowsStreamingWindowBuild::RowsStreamingWindowBuild",
      std::function<void(RowsStreamingWindowBuild*)>(
          [&](RowsStreamingWindowBuild* windowBuild) {
            isStreamCreated.store(true);
          }));

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
      .config(core::QueryConfig::kPreferredOutputBatchRows, "2")
      .config(core::QueryConfig::kMaxOutputBatchRows, "2")
      .assertResults(
          "SELECT *, rank() over (order by c1 rows unbounded preceding) FROM tmp");

  ASSERT_TRUE(isStreamCreated.load());
}

DEBUG_ONLY_TEST_F(WindowTest, valuesRowsStreamingWindowBuild) {
  const vector_size_t size = 1'00;

  auto data = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
       makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
       makeFlatVector<int64_t>(
           size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
       makeFlatVector<int32_t>(size, [](auto row) { return row % 40; }),
       makeFlatVector<int32_t>(size, [](auto row) { return row; })});

  createDuckDbTable({data});

  const std::vector<std::string> kClauses = {
      "rank() over (partition by c0, c2 order by c1, c3)",
      "dense_rank() over (partition by c0, c2 order by c1, c3)",
      "row_number() over (partition by c0, c2 order by c1, c3)",
      "sum(c4) over (partition by c0, c2 order by c1, c3)"};

  auto plan = PlanBuilder()
                  .values({split(data, 10)})
                  .orderBy({"c0", "c2", "c1", "c3"}, false)
                  .streamingWindow(kClauses)
                  .planNode();

  std::atomic_bool isStreamCreated{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::RowsStreamingWindowBuild::RowsStreamingWindowBuild",
      std::function<void(RowsStreamingWindowBuild*)>(
          [&](RowsStreamingWindowBuild* windowBuild) {
            isStreamCreated.store(true);
          }));

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
      .assertResults(
          "SELECT *, rank() over (partition by c0, c2 order by c1, c3), dense_rank() over (partition by c0, c2 order by c1, c3), row_number() over (partition by c0, c2 order by c1, c3), sum(c4) over (partition by c0, c2 order by c1, c3) FROM tmp");
  ASSERT_TRUE(isStreamCreated.load());
}

DEBUG_ONLY_TEST_F(WindowTest, aggregationWithNonDefaultFrame) {
  const vector_size_t size = 1'00;

  auto data = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
       makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
       makeFlatVector<int64_t>(
           size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
       makeFlatVector<int32_t>(size, [](auto row) { return row % 40; }),
       makeFlatVector<int32_t>(size, [](auto row) { return row; })});

  createDuckDbTable({data});

  const std::vector<std::string> kClauses = {
      "sum(c4) over (partition by c0, c2 order by c1, c3 range between unbounded preceding and unbounded following)"};

  auto plan = PlanBuilder()
                  .values({split(data, 10)})
                  .orderBy({"c0", "c2", "c1", "c3"}, false)
                  .streamingWindow(kClauses)
                  .planNode();

  std::atomic_bool isStreamCreated{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::RowsStreamingWindowBuild::RowsStreamingWindowBuild",
      std::function<void(RowsStreamingWindowBuild*)>(
          [&](RowsStreamingWindowBuild* windowBuild) {
            isStreamCreated.store(true);
          }));

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
      .assertResults(
          "SELECT *, sum(c4) over (partition by c0, c2 order by c1, c3 range between unbounded preceding and unbounded following) FROM tmp");

  ASSERT_FALSE(isStreamCreated.load());
}

DEBUG_ONLY_TEST_F(WindowTest, nonRowsStreamingWindow) {
  auto data = makeRowVector(
      {"c1"},
      {makeFlatVector<int64_t>(std::vector<int64_t>{1, 1, 1, 1, 1, 2, 2})});

  createDuckDbTable({data});

  const std::vector<std::string> kClauses = {
      "first_value(c1) over (order by c1 rows unbounded preceding)",
      "nth_value(c1, 1) over (order by c1 rows unbounded preceding)"};

  auto plan = PlanBuilder()
                  .values({data})
                  .orderBy({"c1"}, false)
                  .streamingWindow(kClauses)
                  .planNode();

  std::atomic_bool isStreamCreated{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::RowsStreamingWindowBuild::RowsStreamingWindowBuild",
      std::function<void(RowsStreamingWindowBuild*)>(
          [&](RowsStreamingWindowBuild* windowBuild) {
            isStreamCreated.store(true);
          }));

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
      .config(core::QueryConfig::kPreferredOutputBatchRows, "2")
      .config(core::QueryConfig::kMaxOutputBatchRows, "2")
      .assertResults(
          "SELECT *, first_value(c1) over (order by c1 rows unbounded preceding), nth_value(c1, 1) over (order by c1 rows unbounded preceding) FROM tmp");
  ASSERT_FALSE(isStreamCreated.load());
}

TEST_F(WindowTest, missingFunctionSignature) {
  auto input = {makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<std::string>({"A", "B", "C"}),
      makeFlatVector<int64_t>({10, 20, 30}),
  })};

  auto runWindow = [&](const core::CallTypedExprPtr& callExpr) {
    core::WindowNode::Frame frame{
        core::WindowNode::WindowType::kRows,
        core::WindowNode::BoundType::kUnboundedPreceding,
        nullptr,
        core::WindowNode::BoundType::kUnboundedFollowing,
        nullptr};

    core::WindowNode::Function windowFunction{callExpr, frame, false};

    CursorParameters params;
    params.planNode =
        PlanBuilder()
            .values(input)
            .addNode([&](auto nodeId, auto source) -> core::PlanNodePtr {
              return std::make_shared<core::WindowNode>(
                  nodeId,
                  std::vector<core::FieldAccessTypedExprPtr>{
                      std::make_shared<core::FieldAccessTypedExpr>(
                          BIGINT(), "c0")},
                  std::vector<core::FieldAccessTypedExprPtr>{}, // sortingKeys
                  std::vector<core::SortOrder>{}, // sortingOrders
                  std::vector<std::string>{"w"},
                  std::vector<core::WindowNode::Function>{windowFunction},
                  false,
                  source);
            })
            .planNode();

    readCursor(params, [](auto*) {});
  };

  auto callExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c1")},
      "sum");

  VELOX_ASSERT_THROW(
      runWindow(callExpr),
      "Window function signature is not supported: sum(VARCHAR). Supported signatures:");

  callExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c2")},
      "sum");

  VELOX_ASSERT_THROW(
      runWindow(callExpr),
      "Unexpected return type for window function sum(BIGINT). Expected BIGINT. Got VARCHAR.");
}

TEST_F(WindowTest, duplicateOrOverlappingKeys) {
  auto data = makeRowVector(
      ROW({"a", "b", "c", "d", "e"},
          {
              BIGINT(),
              BIGINT(),
              BIGINT(),
              BIGINT(),
              BIGINT(),
          }),
      10);

  auto plan = [&](const std::vector<std::string>& partitionKeys,
                  const std::vector<std::string>& sortingKeys) {
    std::ostringstream sql;
    sql << "row_number() over (";
    if (!partitionKeys.empty()) {
      sql << " partition by ";
      sql << folly::join(", ", partitionKeys);
    }
    if (!sortingKeys.empty()) {
      sql << " order by ";
      sql << folly::join(", ", sortingKeys);
    }
    sql << ")";

    PlanBuilder().values({data}).window({sql.str()}).planNode();
  };

  VELOX_ASSERT_THROW(
      plan({"a", "a"}, {"b"}),
      "Partitioning keys must be unique. Found duplicate key: a");

  VELOX_ASSERT_THROW(
      plan({"a", "b"}, {"c", "d", "c"}),
      "Sorting keys must be unique and not overlap with partitioning keys. Found duplicate key: c");

  VELOX_ASSERT_THROW(
      plan({"a", "b"}, {"c", "b"}),
      "Sorting keys must be unique and not overlap with partitioning keys. Found duplicate key: b");
}

} // namespace
} // namespace facebook::velox::exec
