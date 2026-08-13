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
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/Driver.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <algorithm>
#include <limits>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

namespace {

class EstimatedSizeCudfVector final : public CudfVector {
 public:
  EstimatedSizeCudfVector(
      memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::table>&& table,
      rmm::cuda_stream_view stream,
      uint64_t estimatedSizeBytes)
      : CudfVector(pool, type, size, std::move(table), stream),
        estimatedSizeBytes_(estimatedSizeBytes) {}

  uint64_t estimateFlatSize() const override {
    return estimatedSizeBytes_;
  }

 private:
  const uint64_t estimatedSizeBytes_;
};

class CudfBatchConcatTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    CudfConfig::getInstance().debugEnabled = true;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    auto& config = CudfConfig::getInstance();
    config.concatOptimizationEnabled = false;
    config.batchSizeMinThreshold = 100'000;
    config.batchSizeMinBytes.reset();
    config.batchSizeMaxThreshold.reset();
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void updateCudfConfig(
      uint64_t minBytes,
      std::optional<int32_t> maxRows,
      int32_t zeroColumnMinRows = 100'000) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinBytes = minBytes;
    config.batchSizeMinThreshold = zeroColumnMinRows;
    config.batchSizeMaxThreshold = maxRows;
  }

  CudfVectorPtr toCudfVector(
      const RowVectorPtr& input,
      std::optional<uint64_t> estimatedSizeBytes = std::nullopt) {
    auto stream = cudfGlobalStreamPool().get_stream();
    std::unique_ptr<cudf::table> table;
    if (input->childrenSize() == 0) {
      table = std::make_unique<cudf::table>();
    } else {
      table =
          with_arrow::toCudfTable(input, pool_.get(), stream, get_output_mr());
    }

    if (estimatedSizeBytes.has_value()) {
      return std::make_shared<EstimatedSizeCudfVector>(
          pool_.get(),
          input->type(),
          input->size(),
          std::move(table),
          stream,
          estimatedSizeBytes.value());
    }
    return std::make_shared<CudfVector>(
        pool_.get(), input->type(), input->size(), std::move(table), stream);
  }

  core::PlanNodePtr createAggregationPlan(const RowVectorPtr& input) {
    return PlanBuilder()
        .values({input})
        .singleAggregation({}, {"count(*)"})
        .planNode();
  }

  std::shared_ptr<Task> createTask(const core::PlanNodePtr& planNode) {
    core::PlanFragment planFragment;
    planFragment.planNode = planNode->sources().front();
    return Task::create(
        "CudfBatchConcatTest",
        std::move(planFragment),
        0,
        core::QueryCtx::create(driverExecutor_.get()),
        Task::ExecutionMode::kParallel);
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
    return PlanBuilder(generator).localPartitionRoundRobin(sources).planNode();
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

} // namespace

TEST_F(CudfBatchConcatTest, singleColumnBearingInputPassesThrough) {
  updateCudfConfig(/*minBytes=*/1, /*maxRows=*/std::nullopt);

  auto input = makeRowVector({makeFlatSequence<int64_t>(0, 4)});
  auto plan = PlanBuilder()
                  .values({input})
                  .singleAggregation({}, {"sum(c0)"})
                  .planNode();

  core::PlanFragment planFragment;
  planFragment.planNode = plan;
  auto task = Task::create(
      "CudfBatchConcatTest_singleColumnBearingInputPassesThrough",
      std::move(planFragment),
      0,
      core::QueryCtx::create(executor_.get()),
      Task::ExecutionMode::kParallel);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  auto stream = cudfGlobalStreamPool().get_stream();
  auto table = with_arrow::toCudfTable(
      input, pool(), stream, cudf::get_current_device_resource_ref());
  auto cudfInput = std::make_shared<CudfVector>(
      pool(), input->type(), input->size(), std::move(table), stream);

  concat.addInput(cudfInput);
  auto output = concat.getOutput();

  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output.get(), cudfInput.get())
      << "A single column-bearing input must not be materialized by concat";
  concat.close();
}

TEST_F(
    CudfBatchConcatTest,
    flushesByEstimatedGpuBytesAndRetainsTailUntilNoMoreInput) {
  constexpr vector_size_t kRowsPerBatch = 10;
  auto smallInput = makeRowVector({makeFlatVector<std::string>(
      kRowsPerBatch, [](auto /*row*/) { return "x"; })});
  auto largeInput = makeRowVector({makeFlatVector<std::string>(
      kRowsPerBatch, [](auto /*row*/) { return std::string(4'096, 'y'); })});
  auto tailInput = makeRowVector({makeFlatVector<std::string>(
      kRowsPerBatch, [](auto /*row*/) { return "z"; })});

  auto small = toCudfVector(smallInput);
  auto large = toCudfVector(largeInput);
  auto tail = toCudfVector(tailInput);
  ASSERT_LT(small->estimateFlatSize(), large->estimateFlatSize());
  ASSERT_LT(tail->estimateFlatSize(), large->estimateFlatSize());

  const auto targetBytes =
      std::max(small->estimateFlatSize(), tail->estimateFlatSize()) + 1;
  ASSERT_LT(targetBytes, large->estimateFlatSize());
  updateCudfConfig(
      /*minBytes=*/targetBytes, /*maxRows=*/std::nullopt);
  auto plan = createAggregationPlan(smallInput);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  ASSERT_TRUE(concat.needsInput());
  concat.addInput(small);
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);

  concat.addInput(large);
  EXPECT_FALSE(concat.needsInput());
  auto fullBatch = concat.getOutput();
  ASSERT_NE(fullBatch, nullptr);
  EXPECT_EQ(fullBatch->size(), 2 * kRowsPerBatch);
  EXPECT_GE(
      fullBatch->estimateFlatSize(),
      CudfConfig::getInstance().batchSizeMinBytes.value());

  ASSERT_TRUE(concat.needsInput());
  concat.addInput(tail);
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);
  EXPECT_FALSE(concat.isFinished());

  concat.noMoreInput();
  auto tailBatch = concat.getOutput();
  ASSERT_NE(tailBatch, nullptr);
  EXPECT_EQ(tailBatch->size(), kRowsPerBatch);
  EXPECT_TRUE(concat.isFinished());
}

TEST_F(CudfBatchConcatTest, usesRowTargetWhenByteTargetIsNotConfigured) {
  constexpr vector_size_t kRowsPerBatch = 10;
  auto input = makeRowVector({makeFlatSequence<int64_t>(0, kRowsPerBatch)});
  auto first = toCudfVector(input, 1'000'000);
  auto second = toCudfVector(input, 1'000'000);

  auto& config = CudfConfig::getInstance();
  config.batchSizeMinBytes = {};
  config.batchSizeMinThreshold = 2 * kRowsPerBatch;
  config.batchSizeMaxThreshold.reset();

  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  concat.addInput(first);
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);

  concat.addInput(second);
  EXPECT_FALSE(concat.needsInput());
  auto output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), 2 * kRowsPerBatch);
}

TEST_F(CudfBatchConcatTest, rejectsZeroByteTarget) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  updateCudfConfig(/*minBytes=*/0, /*maxRows=*/std::nullopt);

  VELOX_ASSERT_THROW(
      CudfBatchConcat(0, &driverCtx, plan),
      "cuDF BatchConcat minimum byte target must be positive");
}

TEST_F(CudfBatchConcatTest, rejectsBufferedByteOverflow) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  auto first = toCudfVector(input, std::numeric_limits<uint64_t>::max() - 1);
  auto second = toCudfVector(input, 2);
  updateCudfConfig(
      /*minBytes=*/std::numeric_limits<uint64_t>::max(),
      /*maxRows=*/std::nullopt);

  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  concat.addInput(first);
  ASSERT_TRUE(concat.needsInput());
  VELOX_ASSERT_THROW(
      concat.addInput(second), "CudfBatchConcat buffered byte count overflow");
}

TEST_F(CudfBatchConcatTest, zeroColumnVectorsUseRowFallback) {
  constexpr vector_size_t kRowsPerBatch = 10;
  auto input = std::make_shared<RowVector>(
      pool_.get(),
      ROW({}, {}),
      BufferPtr(nullptr),
      kRowsPerBatch,
      std::vector<VectorPtr>{},
      std::nullopt);
  auto first = toCudfVector(input);
  auto second = toCudfVector(input);
  auto tail = toCudfVector(input);
  ASSERT_EQ(first->estimateFlatSize(), 0);

  updateCudfConfig(
      /*minBytes=*/1,
      /*maxRows=*/std::nullopt,
      /*zeroColumnMinRows=*/2 * kRowsPerBatch);
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  concat.addInput(first);
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);

  concat.addInput(second);
  EXPECT_FALSE(concat.needsInput());
  auto fullBatch = concat.getOutput();
  ASSERT_NE(fullBatch, nullptr);
  EXPECT_EQ(fullBatch->size(), 2 * kRowsPerBatch);
  EXPECT_EQ(fullBatch->estimateFlatSize(), 0);

  concat.addInput(tail);
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);
  concat.noMoreInput();
  auto tailBatch = concat.getOutput();
  ASSERT_NE(tailBatch, nullptr);
  EXPECT_EQ(tailBatch->size(), kRowsPerBatch);
  EXPECT_TRUE(concat.isFinished());
}

// Verifies that CudfBatchConcat is inserted before aggregation and reduces
// the number of batches reaching the aggregation operator.
TEST_F(CudfBatchConcatTest, concatReducesBatchesBeforeAggregation) {
  // 6 batches of 10 rows each = 60 rows total.
  // A byte target above their combined size merges them on noMoreInput.
  updateCudfConfig(/*minBytes=*/1'048'576, /*maxRows=*/std::nullopt);
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

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
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
  updateCudfConfig(/*minBytes=*/1'048'576, /*maxRows=*/std::nullopt);
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

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT sum(c0) FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  EXPECT_EQ(nodeStats.operatorStats.count("CudfBatchConcat"), 0)
      << "CudfBatchConcat should not be present when optimization is disabled";
}

// When the threshold exceeds total input bytes, concat accumulates all batches
// and flushes them as a single merged batch on noMoreInput.
TEST_F(CudfBatchConcatTest, concatMergesAllOnFlushWithHighThreshold) {
  updateCudfConfig(/*minBytes=*/1'048'576, /*maxRows=*/std::nullopt);
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

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
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
  updateCudfConfig(/*minBytes=*/1'048'576, /*maxRows=*/std::nullopt);
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

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
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

TEST_F(CudfBatchConcatTest, concatPreservesZeroColumnRowCountForCountStar) {
  updateCudfConfig(
      /*minBytes=*/1,
      /*maxRows=*/std::nullopt,
      /*zeroColumnMinRows=*/30);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .values({data})
                  .filter("c0 > 0")
                  .project({})
                  .singleAggregation({}, {"count(*)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 1);
  EXPECT_EQ(concatIt->second->outputVectors, 1);
}

// Verifies that CudfBatchConcat is inserted before the hash join probe and
// correctly handles the 2-source HashJoinNode plan node.
TEST_F(CudfBatchConcatTest, concatBeforeHashJoinProbe) {
  updateCudfConfig(/*minBytes=*/1'048'576, /*maxRows=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  // Probe side: 6 batches of 10 rows each.
  std::vector<RowVectorPtr> probeVectors;
  for (int i = 0; i < 6; ++i) {
    probeVectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(10, [i](auto row) { return row % 3; }),
         makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  // Build side: small dimension table.
  auto buildVector =
      makeRowVector({"u_c0"}, {makeFlatVector<int64_t>({0, 1, 2})});

  createDuckDbTable("probe", probeVectors);
  createDuckDbTable("build", {buildVector});

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId joinNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(probeVectors, generator);
                  })
                  .hashJoin(
                      {"c0"},
                      {"u_c0"},
                      PlanBuilder(generator).values({buildVector}).planNode(),
                      "",
                      {"c0", "c1"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinNodeId)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .maxDrivers(1)
          .assertResults(
              "SELECT p.c0, p.c1 FROM probe p INNER JOIN build b ON p.c0 = b.u_c0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(joinNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end())
      << "CudfBatchConcat should be present before hash join probe";

  auto& concatStats = *concatIt->second;
  EXPECT_EQ(concatStats.inputVectors, 6)
      << "CudfBatchConcat should have received all 6 probe batches";
  EXPECT_LT(concatStats.outputVectors, concatStats.inputVectors)
      << "CudfBatchConcat should produce fewer output batches than input";
}

TEST_F(CudfBatchConcatTest, rightJoinCollectsMatchedRowsFromPeerProbes) {
  updateCudfConfig(/*minBytes=*/1'048'576, /*maxRows=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> probeVectors;
  for (int i = 0; i < 6; ++i) {
    probeVectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeConstant<int64_t>(i, 10), makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  auto buildVector = makeRowVector(
      {"u_c0"}, {makeFlatSequence<int64_t>(0, probeVectors.size())});

  createDuckDbTable("probe", probeVectors);
  createDuckDbTable("build", {buildVector});

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId joinNodeId;
  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(probeVectors, generator);
                  })
                  .hashJoin(
                      {"c0"},
                      {"u_c0"},
                      PlanBuilder(generator).values({buildVector}).planNode(),
                      "",
                      {"c0", "c1", "u_c0"},
                      core::JoinType::kRight)
                  .capturePlanNodeId(joinNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(3)
                  .assertResults(
                      "SELECT p.c0, p.c1, b.u_c0 FROM probe p "
                      "RIGHT JOIN build b ON p.c0 = b.u_c0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(joinNodeId);
  ASSERT_NE(nodeStats.operatorStats.count("CudfBatchConcat"), 0);
  ASSERT_NE(nodeStats.operatorStats.count("CudfHashJoinProbe"), 0);
  ASSERT_EQ(nodeStats.operatorStats.at("CudfHashJoinProbe")->numDrivers, 3);
}

TEST_F(CudfBatchConcatTest, concatSplitsZeroColumnBatchesAtMaxThreshold) {
  updateCudfConfig(
      /*minBytes=*/1, /*maxRows=*/20, /*zeroColumnMinRows=*/30);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 3; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId aggNodeId;

  auto plan = PlanBuilder(generator)
                  .addNode([&](auto id, auto pool) {
                    return createFragmentedSource(vectors, generator);
                  })
                  .filter("c0 >= 0")
                  .project({})
                  .singleAggregation({}, {"count(*)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 >= 0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 3);
  EXPECT_EQ(concatIt->second->outputVectors, 2)
      << "30 zero-column rows should be split into 20-row and 10-row batches";
}

TEST_F(CudfBatchConcatTest, singleZeroColumnBatchSplitsAtMaxThreshold) {
  updateCudfConfig(
      /*minBytes=*/1, /*maxRows=*/20, /*zeroColumnMinRows=*/30);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

  auto data = makeRowVector({makeFlatSequence<int64_t>(0, 30)});
  createDuckDbTable({data});

  core::PlanNodeId aggNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .filter("c0 >= 0")
                  .project({})
                  .singleAggregation({}, {"count(*)"})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .maxDrivers(1)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 >= 0");

  auto planStats = toPlanStats(task->taskStats());
  auto& nodeStats = planStats.at(aggNodeId);
  auto concatIt = nodeStats.operatorStats.find("CudfBatchConcat");
  ASSERT_NE(concatIt, nodeStats.operatorStats.end());
  EXPECT_EQ(concatIt->second->inputVectors, 1);
  EXPECT_EQ(concatIt->second->outputVectors, 2)
      << "A 30-row zero-column input should be split into 20 and 10 rows";
}
