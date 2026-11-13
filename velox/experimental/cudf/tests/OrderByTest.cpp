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
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Driver.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/VectorStream.h"

#include <fmt/format.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {

class OrderByTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    rng_.seed(123);

    rowType_ = ROW(
        {{"c0", INTEGER()},
         {"c1", INTEGER()},
         {"c2", VARCHAR()},
         {"c3", VARCHAR()}});
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key) {
    core::PlanNodeId orderById;
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);
    auto plan = PlanBuilder()
                    .values(input)
                    .orderBy({fmt::format("{} ASC NULLS LAST", key)}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    runTest(
        plan,
        orderById,
        fmt::format("SELECT * FROM tmp ORDER BY {} NULLS LAST", key),
        {keyIndex});

    plan = PlanBuilder()
               .values(input)
               .orderBy({fmt::format("{} DESC NULLS FIRST", key)}, false)
               .planNode();
    runTest(
        plan,
        orderById,
        fmt::format("SELECT * FROM tmp ORDER BY {} DESC NULLS FIRST", key),
        {keyIndex});
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key,
      const std::string& filter) {
    core::PlanNodeId orderById;
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);
    auto plan = PlanBuilder()
                    .values(input)
                    .filter(filter)
                    .orderBy({fmt::format("{} ASC NULLS LAST", key)}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    runTest(
        plan,
        orderById,
        fmt::format(
            "SELECT * FROM tmp WHERE {} ORDER BY {} NULLS LAST", filter, key),
        {keyIndex});

    plan = PlanBuilder()
               .values(input)
               .filter(filter)
               .orderBy({fmt::format("{} DESC NULLS FIRST", key)}, false)
               .capturePlanNodeId(orderById)
               .planNode();
    runTest(
        plan,
        orderById,
        fmt::format(
            "SELECT * FROM tmp WHERE {} ORDER BY {} DESC NULLS FIRST",
            filter,
            key),
        {keyIndex});
  }

  void testTwoKeys(
      const std::vector<RowVectorPtr>& input,
      const std::string& key1,
      const std::string& key2) {
    auto& rowType = input[0]->type()->asRow();
    auto keyIndices = {rowType.getChildIdx(key1), rowType.getChildIdx(key2)};

    std::vector<core::SortOrder> sortOrders = {
        core::kAscNullsLast, core::kDescNullsFirst};
    std::vector<std::string> sortOrderSqls = {"NULLS LAST", "DESC NULLS FIRST"};

    for (int i = 0; i < sortOrders.size(); i++) {
      for (int j = 0; j < sortOrders.size(); j++) {
        core::PlanNodeId orderById;
        auto plan = PlanBuilder()
                        .values(input)
                        .orderBy(
                            {fmt::format("{} {}", key1, sortOrderSqls[i]),
                             fmt::format("{} {}", key2, sortOrderSqls[j])},
                            false)
                        .capturePlanNodeId(orderById)
                        .planNode();
        runTest(
            plan,
            orderById,
            fmt::format(
                "SELECT * FROM tmp ORDER BY {} {}, {} {}",
                key1,
                sortOrderSqls[i],
                key2,
                sortOrderSqls[j]),
            keyIndices);
      }
    }
  }

  void runTest(
      core::PlanNodePtr planNode,
      const core::PlanNodeId& orderById,
      const std::string& duckDbSql,
      const std::vector<uint32_t>& sortingKeys) {
    {
      SCOPED_TRACE("run without spilling");
      assertQueryOrdered(planNode, duckDbSql, sortingKeys);
    }
  }

  std::vector<RowVectorPtr> makeVectors(
      const RowTypePtr& rowType,
      int32_t numVectors,
      int32_t rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          facebook::velox::test::BatchMaker::createBatch(
              rowType, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  // Sorts 'input' by 'keys', as the upstream task of a merge exchange does,
  // and cuts the result into 'numPages' consecutive pages, each sorted by
  // construction.
  std::vector<RowVectorPtr> sortedPages(
      const RowVectorPtr& input,
      const std::vector<std::string>& keys,
      int numPages) {
    auto sorted =
        AssertQueryBuilder(
            PlanBuilder().values({input}).orderBy(keys, false).planNode())
            .copyResults(pool());
    std::vector<RowVectorPtr> pages;
    const auto numRows = sorted->size();
    for (int i = 0; i < numPages; ++i) {
      const auto begin = numRows * i / numPages;
      const auto end = numRows * (i + 1) / numPages;
      auto page =
          BaseVector::create<RowVector>(sorted->type(), end - begin, pool());
      page->copy(sorted.get(), 0, begin, end - begin);
      pages.push_back(std::move(page));
    }
    return pages;
  }

  // Uploads 'page' on a stream of its own, as the exchange does for each
  // received page.
  cudf_velox::CudfVectorPtr toDevice(const RowVectorPtr& page) {
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    auto table = cudf_velox::with_arrow::toCudfTable(
        page, pool(), stream, cudf_velox::get_output_mr());
    return std::make_shared<cudf_velox::CudfVector>(
        pool(),
        asRowType(page->type()),
        page->size(),
        std::move(table),
        stream);
  }

  // The CudfOrderBy built for a MergeExchangeNode, with the task and driver
  // context it is bound to.
  struct MergeExchangeOrderBy {
    std::shared_ptr<Task> task;
    std::unique_ptr<DriverCtx> driverCtx;
    std::unique_ptr<cudf_velox::CudfOrderBy> orderBy;
  };

  // Builds the CudfOrderBy that follows a kept UcxExchange for a merge
  // exchange over 'rowType' ordered by 'keys'.
  MergeExchangeOrderBy makeMergeExchangeOrderBy(
      const RowTypePtr& rowType,
      const std::vector<std::string>& keys) {
    auto plan = PlanBuilder()
                    .mergeExchange(
                        rowType,
                        keys,
                        VectorSerde::kindName(VectorSerde::Kind::kPresto))
                    .planFragment();
    auto mergeExchangeNode =
        std::dynamic_pointer_cast<const core::MergeExchangeNode>(plan.planNode);
    VELOX_CHECK_NOT_NULL(mergeExchangeNode);
    MergeExchangeOrderBy result;
    result.task = Task::create(
        "test-merge-exchange-order-by",
        std::move(plan),
        /*destination=*/0,
        core::QueryCtx::create(executor_.get()),
        Task::ExecutionMode::kParallel);
    result.driverCtx = std::make_unique<DriverCtx>(
        result.task,
        /*driverId=*/0,
        /*pipelineId=*/0,
        kUngroupedGroupId,
        /*partitionId=*/0);
    result.orderBy = std::make_unique<cudf_velox::CudfOrderBy>(
        /*operatorId=*/0, result.driverCtx.get(), mergeExchangeNode);
    return result;
  }

  // What a run of the merge-exchange CudfOrderBy produced. The run comes
  // first so that its task, which owns the output's memory pool, outlives the
  // output.
  struct MergeExchangeResult {
    MergeExchangeOrderBy run;
    // The output on the host, or nullptr.
    RowVectorPtr rows;
    // The output vector itself.
    cudf_velox::CudfVectorPtr output;
    std::unordered_map<std::string, RuntimeMetric> runtimeStats;
  };

  // Drives the merge-exchange CudfOrderBy over 'rowType' ordered by 'keys' the
  // way the Driver does behind a kept UcxExchange: one addInput() per received
  // page, then noMoreInput() and getOutput().
  MergeExchangeResult runMergeExchangeOrderBy(
      const RowTypePtr& rowType,
      const std::vector<std::string>& keys,
      const std::vector<cudf_velox::CudfVectorPtr>& pages) {
    auto run = makeMergeExchangeOrderBy(rowType, keys);
    auto& orderBy = *run.orderBy;
    for (const auto& page : pages) {
      orderBy.addInput(page);
    }
    orderBy.noMoreInput();
    auto output = orderBy.getOutput();
    EXPECT_TRUE(orderBy.isFinished());
    EXPECT_EQ(orderBy.getOutput(), nullptr);

    MergeExchangeResult result;
    result.runtimeStats = orderBy.stats(false).runtimeStats;
    if (output != nullptr) {
      result.output = std::dynamic_pointer_cast<cudf_velox::CudfVector>(output);
      VELOX_CHECK_NOT_NULL(result.output);
      result.rows = cudf_velox::with_arrow::toVeloxColumn(
          result.output->getTableView(),
          pool(),
          rowType,
          result.output->stream(),
          cudf_velox::get_output_mr());
      result.output->stream().sync();
    }
    orderBy.close();
    result.run = std::move(run);
    return result;
  }

  // Unsorted rows of upstream stream 'streamId', distinguishable by c2.
  RowVectorPtr makeStreamInput(int streamId, int numRows) {
    return makeRowVector(
        {"c0", "c1", "c2"},
        {makeFlatVector<int64_t>(
             numRows,
             [&](auto row) { return (row * 7 + streamId) % 23; },
             nullEvery(5)),
         makeFlatVector<StringView>(
             numRows,
             [&](auto row) {
               return StringView::makeInline(
                   fmt::format("k{}", (row + streamId) % 11));
             },
             nullEvery(7, 1)),
         makeFlatVector<double>(
             numRows, [&](auto row) { return streamId * 1e4 + row; })});
  }

  // Runs the merge-exchange CudfOrderBy over 'numStreams' sorted streams, each
  // cut into 'pagesPerStream' pages that arrive interleaved across streams,
  // checks the output against DuckDB and returns the run.
  MergeExchangeResult
  testMergeExchange(int numStreams, int pagesPerStream, int rowsPerStream) {
    const auto& keys = mergeKeys_;
    const auto& rowType = mergeRowType_;

    std::vector<std::vector<RowVectorPtr>> streams;
    for (int s = 0; s < numStreams; ++s) {
      streams.push_back(
          sortedPages(makeStreamInput(s, rowsPerStream), keys, pagesPerStream));
    }

    std::vector<RowVectorPtr> hostPages;
    std::vector<cudf_velox::CudfVectorPtr> pages;
    for (int p = 0; p < pagesPerStream; ++p) {
      for (int s = 0; s < numStreams; ++s) {
        hostPages.push_back(streams[s][p]);
        pages.push_back(toDevice(streams[s][p]));
      }
    }
    // An empty page carries nothing to merge and is dropped on input.
    pages.insert(
        pages.begin() + 1,
        toDevice(BaseVector::create<RowVector>(rowType, 0, pool())));
    createDuckDbTable(hostPages);

    auto result = runMergeExchangeOrderBy(rowType, keys, pages);
    EXPECT_NE(result.rows, nullptr);
    if (result.rows != nullptr) {
      assertResultsOrdered(
          {result.rows}, rowType, mergeOrderBySql_, duckDbQueryRunner_, {0, 1});
    }
    return result;
  }

  folly::Random::DefaultGenerator rng_;
  RowTypePtr rowType_;

  // The keys and layout of the merge-exchange tests: keys tie across streams
  // and carry nulls, and the second key is descending with nulls first.
  const std::vector<std::string> mergeKeys_ = {
      "c0 ASC NULLS LAST",
      "c1 DESC NULLS FIRST"};
  const std::string mergeOrderBySql_ =
      "SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST, c1 DESC NULLS FIRST";
  const RowTypePtr mergeRowType_ =
      ROW({"c0", "c1", "c2"}, {BIGINT(), VARCHAR(), DOUBLE()});
};

TEST_F(OrderByTest, selectiveFilter) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 3; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  // c0 values are unique across batches
  testSingleKey(vectors, "c0", "c0 % 333 = 0");

  // c1 values are unique only within a batch
  testSingleKey(vectors, "c1", "c1 % 333 = 0");
}

TEST_F(OrderByTest, singleKey) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");

  // parser doesn't support "is not null" expression, hence, using c0 % 2 >= 0
  testSingleKey(vectors, "c0", "c0 % 2 >= 0");

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy({"c0 DESC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan, orderById, "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST", {0});

  plan = PlanBuilder()
             .values(vectors)
             .orderBy({"c0 ASC NULLS FIRST"}, false)
             .capturePlanNodeId(orderById)
             .planNode();
  runTest(plan, orderById, "SELECT * FROM tmp ORDER BY c0 NULLS FIRST", {0});
}

// A CudfOrderBy built for an OrderByNode is the node's only operator, so it
// counts both the node's input and its output.
TEST_F(OrderByTest, orderByNodeReportsBothPlanNodeBoundaries) {
  auto data = makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return 100 - row; })});
  createDuckDbTable({data});
  auto plan = PlanBuilder().values({data}).orderBy({"c0"}, false).planNode();

  auto task = assertQueryOrdered(plan, "SELECT * FROM tmp ORDER BY c0", {0});

  int numCudfOrderBy = 0;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& stats : pipeline.operatorStats) {
      if (stats.operatorType == "CudfOrderBy") {
        ++numCudfOrderBy;
        EXPECT_EQ(stats.planNodeBoundary, core::PlanNode::Boundary::kBoth);
      }
    }
  }
  EXPECT_EQ(numCudfOrderBy, 1);
}

// A CudfOrderBy built for a MergeExchangeNode shares that node with the
// exchange in front of it, which reports the node's input, so the sort reports
// only the node's output.
TEST_F(OrderByTest, mergeExchangeNodeReportsOutputPlanNodeBoundary) {
  auto run = makeMergeExchangeOrderBy(rowType_, {"c0"});
  EXPECT_EQ(run.orderBy->planNodeBoundary(), core::PlanNode::Boundary::kOutput);
  run.orderBy->close();
}

// The pages of a merge exchange are sorted slices of the upstream streams, so
// the CudfOrderBy behind the exchange merges them, here kMaxMergeInputs of
// them, the most it merges.
TEST_F(OrderByTest, mergeExchangeMergesSortedPages) {
  constexpr size_t kStreams = 4;
  constexpr size_t kPagesPerStream =
      cudf_velox::CudfOrderBy::kMaxMergeInputs / kStreams;
  static_assert(
      kStreams * kPagesPerStream == cudf_velox::CudfOrderBy::kMaxMergeInputs);
  auto result =
      testMergeExchange(kStreams, kPagesPerStream, /*rowsPerStream=*/1'000);
  EXPECT_EQ(
      result.runtimeStats.at(cudf_velox::CudfOrderBy::kMergedInputs).sum,
      static_cast<int64_t>(kStreams * kPagesPerStream));
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kSortFallbacks));
}

// One page past kMaxMergeInputs the operator sorts instead, with the same
// result.
TEST_F(OrderByTest, mergeExchangeSortsAboveTheMergeLimit) {
  constexpr size_t kStreams = 3;
  constexpr size_t kPagesPerStream = 11;
  static_assert(
      kStreams * kPagesPerStream ==
      cudf_velox::CudfOrderBy::kMaxMergeInputs + 1);
  auto result =
      testMergeExchange(kStreams, kPagesPerStream, /*rowsPerStream=*/500);
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kMergedInputs));
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kSortFallbacks));
}

// Nested keys are merged too: with a nested key cudf::is_sorted takes two
// adjacent rows of a page that tie on every key for a descent; the row index
// in the check breaks the tie, so such a page does not read as unsorted.
TEST_F(OrderByTest, mergeExchangeMergesNestedKeys) {
  const std::vector<std::string> keys = {"c0 DESC NULLS LAST"};
  const auto rowType = ROW({"c0", "c1"}, {ROW({"a"}, {BIGINT()}), DOUBLE()});
  std::vector<RowVectorPtr> hostPages;
  for (int s = 0; s < 2; ++s) {
    auto input = makeRowVector(
        {"c0", "c1"},
        {makeRowVector(
             {"a"},
             {makeFlatVector<int64_t>(
                 200, [&](auto row) { return (row * 3 + s) % 7; })}),
         makeFlatVector<double>(200, [&](auto row) { return s * 1e3 + row; })});
    for (auto& page : sortedPages(input, keys, 2)) {
      hostPages.push_back(std::move(page));
    }
  }
  std::vector<cudf_velox::CudfVectorPtr> pages;
  for (const auto& page : hostPages) {
    pages.push_back(toDevice(page));
  }
  createDuckDbTable(hostPages);

  auto result = runMergeExchangeOrderBy(rowType, keys, pages);
  ASSERT_NE(result.rows, nullptr);
  assertResultsOrdered(
      {result.rows},
      rowType,
      "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST",
      duckDbQueryRunner_,
      {0});
  EXPECT_EQ(
      result.runtimeStats.at(cudf_velox::CudfOrderBy::kMergedInputs).sum, 4);
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kSortFallbacks));
}

// An input that is not sorted (which a merge exchange's producers never send)
// still gives a sorted result: the operator checks every input before merging
// and sorts the rows instead.
TEST_F(OrderByTest, mergeExchangeSortsWhenAnInputIsNotSorted) {
  auto hostPages = sortedPages(makeStreamInput(0, 300), mergeKeys_, 2);
  hostPages.push_back(makeStreamInput(1, 300));
  std::vector<cudf_velox::CudfVectorPtr> pages;
  for (const auto& page : hostPages) {
    pages.push_back(toDevice(page));
  }
  createDuckDbTable(hostPages);

  auto result = runMergeExchangeOrderBy(mergeRowType_, mergeKeys_, pages);
  ASSERT_NE(result.rows, nullptr);
  assertResultsOrdered(
      {result.rows},
      mergeRowType_,
      mergeOrderBySql_,
      duckDbQueryRunner_,
      {0, 1});
  EXPECT_EQ(
      result.runtimeStats.at(cudf_velox::CudfOrderBy::kSortFallbacks).sum, 1);
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kMergedInputs));
}

// A single sorted page is the result and is handed on without a copy.
TEST_F(OrderByTest, mergeExchangeHandsOnASinglePage) {
  const std::vector<std::string> keys = {"c0 DESC NULLS LAST"};
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), VARCHAR()});
  auto input = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int32_t>(
           100, [](auto row) { return row % 13; }, nullEvery(9)),
       makeFlatVector<StringView>(100, [](auto row) {
         return StringView::makeInline(fmt::format("v{}", row));
       })});
  auto page = toDevice(sortedPages(input, keys, 1).front());
  createDuckDbTable({input});

  auto result = runMergeExchangeOrderBy(rowType, keys, {page});
  EXPECT_EQ(result.output, page);
  ASSERT_NE(result.rows, nullptr);
  assertResultsOrdered(
      {result.rows},
      rowType,
      "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST",
      duckDbQueryRunner_,
      {0});
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kMergedInputs));
  EXPECT_FALSE(
      result.runtimeStats.contains(cudf_velox::CudfOrderBy::kSortFallbacks));
}

TEST_F(OrderByTest, multipleKeys) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    // c0: half of rows are null, a quarter is 0 and remaining quarter is 1
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [](vector_size_t row) { return row % 4; }, nullEvery(2, 1));
    auto c1 = makeFlatVector<int32_t>(
        batchSize, [](vector_size_t row) { return row; }, nullEvery(7));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testTwoKeys(vectors, "c0", "c1");

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy({"c0 ASC NULLS FIRST", "c1 ASC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan,
      orderById,
      "SELECT * FROM tmp ORDER BY c0 NULLS FIRST, c1 NULLS LAST",
      {0, 1});

  plan = PlanBuilder()
             .values(vectors)
             .orderBy({"c0 DESC NULLS LAST", "c1 DESC NULLS FIRST"}, false)
             .capturePlanNodeId(orderById)
             .planNode();
  runTest(
      plan,
      orderById,
      "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST, c1 DESC NULLS FIRST",
      {0, 1});
}

TEST_F(OrderByTest, multiBatchResult) {
  vector_size_t batchSize = 5000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 10; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c1, c1, c1, c1}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");
}

TEST_F(OrderByTest, varfields) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    auto c2 = makeFlatVector<StringView>(
        batchSize,
        [](vector_size_t row) {
          return StringView::makeInline(std::to_string(row));
        },
        nullEvery(17));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c2");
}

/// Verifies output batch rows of OrderBy
TEST_F(OrderByTest, outputBatchRows) {
  struct {
    int numRowsPerBatch;
    int preferredOutBatchBytes;
    int maxOutBatchRows;
    int expectedOutputVectors;

    // TODO: add output size check with spilling enabled
    std::string debugString() const {
      return fmt::format(
          "numRowsPerBatch:{}, preferredOutBatchBytes:{}, maxOutBatchRows:{}, expectedOutputVectors:{}",
          numRowsPerBatch,
          preferredOutBatchBytes,
          maxOutBatchRows,
          expectedOutputVectors);
    }
  } testSettings[] = {
      {1024, 1, 100, 1024},
      // estimated size per row is ~2092, set preferredOutBatchBytes to 20920,
      // so each batch has 10 rows, so it would return 100 batches
      {1000, 20920, 100, 100},
      // same as above, but maxOutBatchRows is 1, so it would return 1000
      // batches
      {1000, 20920, 1, 1000}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const vector_size_t batchSize = testData.numRowsPerBatch;
    std::vector<RowVectorPtr> rowVectors;
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(11));
    std::vector<VectorPtr> vectors;
    vectors.push_back(c0);
    for (int i = 0; i < 256; ++i) {
      vectors.push_back(c1);
    }
    rowVectors.push_back(makeRowVector(vectors));
    createDuckDbTable(rowVectors);

    core::PlanNodeId orderById;
    auto plan = PlanBuilder()
                    .values(rowVectors)
                    .orderBy({fmt::format("{} ASC NULLS LAST", "c0")}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    queryCtx->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes,
          std::to_string(testData.preferredOutBatchBytes)},
         {core::QueryConfig::kMaxOutputBatchRows,
          std::to_string(testData.maxOutBatchRows)},
         {facebook::velox::cudf_velox::CudfToVelox::kPassthroughMode,
          "false"}});
    CursorParameters params;
    params.planNode = plan;
    params.queryCtx = queryCtx;
    auto task = assertQueryOrdered(
        params, "SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST", {0});

    EXPECT_EQ(
        testData.expectedOutputVectors,
        PlanNodeStats::extract(task->taskStats(), orderById)
            .operatorStats.at("CudfToVelox")
            ->outputVectors);
  }
}

} // namespace
