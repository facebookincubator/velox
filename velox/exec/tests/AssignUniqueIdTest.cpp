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
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

namespace facebook::velox::exec {
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

namespace {

class AssignUniqueIdTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
  }

  void verifyUniqueId(
      const std::shared_ptr<const core::PlanNode>& plan,
      const std::vector<RowVectorPtr>& input) {
    TaskPtr task;
    auto result = AssertQueryBuilder(plan).copyResults(pool(), task);
    verifyUniqueId(input, result);

    // Verify number of memory allocations. There should be exactly one
    // allocation (per thread of execution) for the values buffer of the
    // unique ID vector. Memory should be allocated when producing first
    // batch of output and re-used for subsequent batches.
    const auto uniqueIdStats =
        PlanNodeStats::extract(task->taskStats(), uniqueNodeId_);
    ASSERT_EQ(1, uniqueIdStats.numMemoryAllocations);
  }

  // Verifies that 'result' has the rows of 'input' in order, with a column of
  // unique ids added last.
  void verifyUniqueId(
      const std::vector<RowVectorPtr>& input,
      const RowVectorPtr& result) {
    const auto numColumns = input[0]->childrenSize();
    ASSERT_EQ(result->childrenSize(), numColumns + 1);

    vector_size_t offset = 0;
    for (const auto& batch : input) {
      for (auto i = 0; i < numColumns; ++i) {
        assertEqualVectors(
            batch->childAt(i),
            result->childAt(i)->slice(offset, batch->size()));
      }
      offset += batch->size();
    }
    ASSERT_EQ(result->size(), offset);

    const auto* ids =
        result->childAt(numColumns)->asFlatVector<int64_t>()->rawValues();
    ASSERT_EQ(
        std::set<int64_t>(ids, ids + result->size()).size(), result->size());
  }

  core::PlanNodeId uniqueNodeId_;
};

TEST_F(AssignUniqueIdTest, multiBatch) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> input;
  input.reserve(3);
  for (int i = 0; i < 3; ++i) {
    input.push_back(
        makeRowVector({makeFlatIdentityVector<int32_t>(batchSize)}));
  }

  auto plan = PlanBuilder()
                  .values(input)
                  .assignUniqueId()
                  .capturePlanNodeId(uniqueNodeId_)
                  .planNode();

  verifyUniqueId(plan, input);
}

TEST_F(AssignUniqueIdTest, exceedRequestLimit) {
  vector_size_t requestLimit = 1 << 20L;
  auto input = {
      makeRowVector({makeFlatIdentityVector<int32_t>(requestLimit - 10)}),
      makeRowVector({makeFlatIdentityVector<int32_t>(100)}),
      makeRowVector({makeFlatIdentityVector<int32_t>(100)}),
  };

  auto plan = PlanBuilder()
                  .values(input)
                  .assignUniqueId()
                  .capturePlanNodeId(uniqueNodeId_)
                  .planNode();

  verifyUniqueId(plan, input);
}

TEST_F(AssignUniqueIdTest, multiThread) {
  for (int i = 0; i < 3; i++) {
    vector_size_t batchSize = 1000;
    auto input = {makeRowVector({makeFlatIdentityVector<int32_t>(batchSize)})};
    auto plan = PlanBuilder()
                    .values(input, true)
                    .assignUniqueId()
                    .capturePlanNodeId(uniqueNodeId_)
                    .planNode();

    exec::TaskPtr task;
    auto result =
        AssertQueryBuilder(plan).maxDrivers(8).copyResults(pool(), task);
    ASSERT_EQ(batchSize * 8, result->size());

    std::set<int64_t> ids;
    auto idValues =
        result->children().back()->asFlatVector<int64_t>()->rawValues();
    std::copy(
        idValues, idValues + result->size(), std::inserter(ids, ids.end()));

    ASSERT_EQ(batchSize * 8, ids.size());

    // Verify number of memory allocations. There should be exactly one
    // allocation (per thread of execution) for the values buffer of the
    // unique ID vector. Memory should be allocated when producing first batch
    // of output and re-used for subsequent batches.
    const auto uniqueIdStats =
        PlanNodeStats::extract(task->taskStats(), uniqueNodeId_);
    ASSERT_EQ(8, uniqueIdStats.numMemoryAllocations);
  }
}

TEST_F(AssignUniqueIdTest, maxRowIdLimit) {
  auto input = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  auto plan = PlanBuilder().values(input).assignUniqueId().planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .beforeTaskStart([](Task& task) {
            // Advance the pool to the end of the 40-bit row id space so the
            // next request overflows.
            task.uniqueRowIdPool()->fetch_add(1L << 40);
          })
          .copyResults(pool()),
      "Ran out of unique IDs at 1099511627776");
}

TEST_F(AssignUniqueIdTest, sharedRowIdPool) {
  const vector_size_t size = 100;
  auto input = {makeRowVector({makeFlatIdentityVector<int32_t>(size)})};

  // The two nodes' generated ids are disjoint.
  auto plan = PlanBuilder()
                  .values(input)
                  .assignUniqueId("a")
                  .assignUniqueId("b")
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(result->size(), size);

  auto* a = result->childAt(1)->asFlatVector<int64_t>();
  auto* b = result->childAt(2)->asFlatVector<int64_t>();

  std::set<int64_t> ids;
  for (auto i = 0; i < size; ++i) {
    ASSERT_TRUE(ids.insert(a->valueAt(i)).second);
    ASSERT_TRUE(ids.insert(b->valueAt(i)).second);
  }
  ASSERT_EQ(ids.size(), 2 * size);
}

TEST_F(AssignUniqueIdTest, taskUniqueIdLimit) {
  auto input = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  auto plan = PlanBuilder().values(input).assignUniqueId().planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).taskUniqueId(1L << 24).copyResults(pool()),
      "Unique 24-bit ID specified for AssignUniqueId exceeds the limit");
}

TEST_F(AssignUniqueIdTest, barrier) {
  auto rowType{ROW({"c0", "c1"}, {BIGINT(), INTEGER()})};

  const int numSplits{5};

  std::vector<RowVectorPtr> vectors;

  const int numRowsPerSplit{100};
  for (int32_t i = 0; i < numSplits; ++i) {
    auto vector = makeRowVector(rowType, {.vectorSize = numRowsPerSplit});
    vectors.push_back(vector);
  }
  auto tempFiles = writeToFiles(vectors);

  auto plan = PlanBuilder()
                  .tableScan(rowType)
                  .assignUniqueId("row_number")
                  .project({"c0", "c1", "row_number"})
                  .planNode();
  struct {
    bool hasBarrier;
    bool serialExecution;

    std::string toString() const {
      return fmt::format(
          "hasBarrier: {}, serialExecution: {}", hasBarrier, serialExecution);
    }
  } testSettings[] = {
      {false, false}, {false, true}, {true, false}, {true, true}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.toString());
    TaskPtr task;
    auto result = AssertQueryBuilder(plan)
                      .splits(makeHiveConnectorSplits(tempFiles))
                      .serialExecution(testData.serialExecution)
                      .maxDrivers(testData.serialExecution ? 1 : 3)
                      .barrierExecution(testData.hasBarrier)
                      .copyResults(pool(), task);
    verifyUniqueId(vectors, result);

    const auto taskStats = task->taskStats();
    ASSERT_EQ(taskStats.numBarriers, testData.hasBarrier ? numSplits : 0);
    ASSERT_EQ(taskStats.numFinishedSplits, numSplits);
  }
}
} // namespace
} // namespace facebook::velox::exec
