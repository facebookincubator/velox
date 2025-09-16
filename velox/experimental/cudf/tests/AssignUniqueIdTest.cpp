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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

namespace facebook::velox::exec {

using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

namespace {

class AssignUniqueIdTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }

  void verifyUniqueId(
      const std::shared_ptr<const core::PlanNode>& plan,
      const std::vector<RowVectorPtr>& input) {
    CursorParameters params;
    params.planNode = plan;
    params.queryConfigs.insert(
        {cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1"});
    auto result = readCursor(params);
    ASSERT_EQ(result.second[0]->childrenSize(), input[0]->childrenSize() + 1);
    verifyUniqueId(input, result.second);

    auto task = result.first->task();
    // Verify number of memory allocations. As no memory is allocated for the
    // unique ID vector generation in GPU, the number of memory allocations
    // should be 0. cudf can generate new columns, and cannot reuse the memory.
    auto stats = toPlanStats(task->taskStats());
    ASSERT_EQ(0, stats.at(uniqueNodeId_).numMemoryAllocations);
    // check if CudfAssignUniqueId operator is executed with stats
    ASSERT_EQ(
        1, stats.at(uniqueNodeId_).operatorStats.count("CudfAssignUniqueId"));
  }

  void verifyUniqueId(
      const std::vector<RowVectorPtr>& input,
      const std::vector<facebook::velox::RowVectorPtr>& vectors) {
    auto numColumns = vectors[0]->childrenSize();
    ASSERT_EQ(numColumns, input[0]->childrenSize() + 1);

    std::set<int64_t> ids;
    for (int i = 0; i < numColumns; i++) {
      for (auto batch = 0; batch < vectors.size(); ++batch) {
        auto column = vectors[batch]->childAt(i);
        if (i < numColumns - 1) {
          assertEqualVectors(input[batch]->childAt(i), column);
        } else {
          auto idValues = column->asFlatVector<int64_t>()->rawValues();
          std::copy(
              idValues,
              idValues + column->size(),
              std::inserter(ids, ids.end()));
        }
      }
    }

    vector_size_t totalInputSize = 0;
    for (const auto& vector : input) {
      totalInputSize += vector->size();
    }

    ASSERT_EQ(totalInputSize, ids.size());
  }

  core::PlanNodeId uniqueNodeId_;
};

TEST_F(AssignUniqueIdTest, multiBatch) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> input;
  input.reserve(3);
  for (int i = 0; i < 3; ++i) {
    input.push_back(
        makeRowVector({makeFlatVector<int32_t>(batchSize, folly::identity)}));
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
      makeRowVector(
          {makeFlatVector<int32_t>(requestLimit - 10, folly::identity)}),
      makeRowVector({makeFlatVector<int32_t>(100, folly::identity)}),
      makeRowVector({makeFlatVector<int32_t>(100, folly::identity)}),
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
    auto input = {
        makeRowVector({makeFlatVector<int32_t>(batchSize, folly::identity)})};
    auto plan = PlanBuilder()
                    .values(input, true)
                    .assignUniqueId()
                    .capturePlanNodeId(uniqueNodeId_)
                    .planNode();

    std::shared_ptr<exec::Task> task;
    auto result = AssertQueryBuilder(plan)
                      .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1")
                      .maxDrivers(8)
                      .copyResults(pool(), task);
    ASSERT_EQ(batchSize * 8, result->size());

    std::set<int64_t> ids;
    auto idValues =
        result->children().back()->asFlatVector<int64_t>()->rawValues();
    std::copy(
        idValues, idValues + result->size(), std::inserter(ids, ids.end()));

    ASSERT_EQ(batchSize * 8, ids.size());

    // Verify number of memory allocations. As no memory is allocated for the
    // unique ID vector generation in GPU, the number of memory allocations
    // should be 0. cudf can generate new columns, and cannot reuse the memory.
    auto stats = toPlanStats(task->taskStats());
    ASSERT_EQ(0, stats.at(uniqueNodeId_).numMemoryAllocations);
  }
}

TEST_F(AssignUniqueIdTest, maxRowIdLimit) {
  auto input = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  auto plan = PlanBuilder().values(input).assignUniqueId().planNode();

  // Increase the counter to kMaxRowId.
  std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(plan)
      ->uniqueIdCounter()
      ->fetch_add(1L << 40);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Ran out of unique IDs at 1099511627776");
}

TEST_F(AssignUniqueIdTest, taskUniqueIdLimit) {
  auto input = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  auto plan =
      PlanBuilder().values(input).assignUniqueId("unique", 1L << 24).planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "(16777216 vs. 16777216) Unique 24-bit ID specified for AssignUniqueId exceeds the limit");
}

// TODO: Add test for barrier execution, other operators does not support
// draining yet.
} // namespace
} // namespace facebook::velox::exec
