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
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

class CudfBatchConcatTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    CudfConfig::getInstance().debugEnabled = true;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    CudfConfig::getInstance().concatOptimizationEnabled = false;
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void updateCudfConfig(int32_t min, std::optional<int32_t> max) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinThreshold = min;
    config.batchSizeMaxThreshold = max;
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, vector_size_t size) {
    return makeFlatVector<T>(size, [start](auto row) { return start + row; });
  }

  // Builds fragmented input via localPartitionRoundRobin to prevent Values
  // from coalescing small batches.
  core::PlanNodePtr createFragmentedSource(
      const std::vector<RowVectorPtr>& vectors,
      std::shared_ptr<core::PlanNodeIdGenerator> generator) {
    std::vector<core::PlanNodePtr> sources;
    for (const auto& vec : vectors) {
      sources.push_back(PlanBuilder(generator).values({vec}).planNode());
    }
    return PlanBuilder(generator)
        .localPartitionRoundRobin(sources)
        .planNode();
  }

  // Returns the per-operator-type stats for CudfBatchConcat within the given
  // plan node, or nullptr if CudfBatchConcat wasn't inserted for that node.
  const PlanNodeStats* getConcatStats(
      const std::shared_ptr<Task>& task,
      const core::PlanNodeId& aggNodeId) {
    auto planStats = toPlanStats(task->taskStats());
    auto nodeIt = planStats.find(aggNodeId);
    if (nodeIt == planStats.end()) {
      return nullptr;
    }
    auto opIt = nodeIt->second.operatorStats.find("CudfBatchConcat");
    if (opIt == nodeIt->second.operatorStats.end()) {
      return nullptr;
    }
    return opIt->second.get();
  }
};

// Verifies that CudfBatchConcat is inserted before aggregation and reduces
// the number of batches reaching the aggregation operator.
TEST_F(CudfBatchConcatTest, concatReducesBatchesBeforeAggregation) {
  // 6 batches of 10 rows each = 60 rows total.
  // With min threshold 30, concat should accumulate ~3 batches before flushing,
  // producing fewer output batches than the 6 it received.
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({}, {"sum(c0)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .maxDrivers(1)
          .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end())
      << "CudfBatchConcat should be present in operator stats";

  auto& concatStats = *concatIt->second;
  EXPECT_EQ(concatStats.inputVectors, 6)
      << "CudfBatchConcat should have received all 6 input batches";
  EXPECT_LT(concatStats.outputVectors, concatStats.inputVectors)
      << "CudfBatchConcat should produce fewer output batches than input";
}

// Verifies that CudfBatchConcat is not inserted when the optimization is
// disabled, even when aggregation is present.
TEST_F(CudfBatchConcatTest, concatNotInsertedWhenDisabled) {
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = false;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({}, {"sum(c0)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .maxDrivers(1)
          .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  EXPECT_EQ(nodeStats.operatorStats.count("CudfBatchConcat"), 0)
      << "CudfBatchConcat should not be present when optimization is disabled";
}

// When the threshold exceeds total input rows, concat accumulates all batches
// and flushes them as a single merged batch on noMoreInput.
TEST_F(CudfBatchConcatTest, concatMergesAllOnFlushWithHighThreshold) {
  updateCudfConfig(/*min=*/100000, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({}, {"sum(c0)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .maxDrivers(1)
          .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end())
      << "CudfBatchConcat should still be inserted even with high threshold";

  auto& concatStats = *concatIt->second;
  EXPECT_EQ(concatStats.inputVectors, 6);
  EXPECT_EQ(concatStats.outputVectors, 1)
      << "All batches should be merged into one on noMoreInput flush";
}

// Verifies correctness with grouped aggregation (non-global) and concat.
TEST_F(CudfBatchConcatTest, concatWithGroupedAggregation) {
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>(10, [](auto row) { return row % 3; }),
         makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .singleAggregation({"c0"}, {"sum(c1)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .maxDrivers(1)
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 6);
  EXPECT_LT(concatIt->second->outputVectors, 6);
}
