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

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

namespace {

// Reports a fixed byte estimate so tests can drive the byte target without
// large allocations.
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
    CudfConfig::getInstance().allowCpuFallback = false;
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

  void updateCudfConfig(int32_t min, std::optional<int32_t> max) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinBytes.reset();
    config.batchSizeMinThreshold = min;
    config.batchSizeMaxThreshold = max;
  }

  void updateCudfByteConfig(uint64_t minBytes, std::optional<int32_t> maxRows) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinBytes = minBytes;
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

  // Returns the CudfBatchConcat stats for the given plan node, or nullptr if
  // CudfBatchConcat wasn't inserted for that node.
  std::unique_ptr<PlanNodeStats> getConcatStats(
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
    // Move out: planStats is destroyed on return.
    return std::move(opIt->second);
  }
};

} // namespace

TEST_F(CudfBatchConcatTest, singleColumnBearingInputPassesThrough) {
  updateCudfConfig(/*min=*/4, /*max=*/std::nullopt);

  auto input = makeRowVector({makeFlatSequence<int64_t>(0, 4)});
  auto plan = PlanBuilder()
                  .values({input})
                  .singleAggregation({}, {"sum(c0)"})
                  .planNode();
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  auto cudfInput = toCudfVector(input);
  concat.addInput(cudfInput);
  auto output = concat.getOutput();

  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output.get(), cudfInput.get())
      << "A single column-bearing input must not be materialized by concat";
  concat.close();
}

// Verifies flushing at the byte target. The single concatenated batch measures
// below the target its inputs met, and must be emitted rather than re-buffered.
TEST_F(CudfBatchConcatTest, flushesAtByteTarget) {
  constexpr vector_size_t kRowsPerBatch = 10;
  constexpr uint64_t kTargetBytes = 1'000'000;
  auto input = makeRowVector({makeFlatSequence<int64_t>(0, kRowsPerBatch)});

  updateCudfByteConfig(/*minBytes=*/kTargetBytes, /*maxRows=*/std::nullopt);
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  concat.addInput(toCudfVector(input, kTargetBytes / 2));
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);

  concat.addInput(toCudfVector(input, kTargetBytes / 2));
  EXPECT_FALSE(concat.needsInput());
  auto output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), 2 * kRowsPerBatch);
  EXPECT_LT(output->estimateFlatSize(), kTargetBytes);

  // A below-target input waits for noMoreInput.
  concat.addInput(toCudfVector(input, kTargetBytes / 2));
  EXPECT_EQ(concat.getOutput(), nullptr);
  EXPECT_FALSE(concat.isFinished());

  concat.noMoreInput();
  output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), kRowsPerBatch);
  EXPECT_TRUE(concat.isFinished());
  concat.close();
}

// Verifies that a split's below-target tail stays buffered and its bytes count
// towards the next flush.
TEST_F(CudfBatchConcatTest, byteTargetRetainsSplitTailBytes) {
  constexpr vector_size_t kRowsPerBatch = 20;
  // Inputs are grouped whole, so two 20-row inputs cannot share a 25-row batch
  // and every flush of two inputs splits into two outputs.
  constexpr int32_t kMaxRows = 25;
  constexpr uint64_t kTargetBytes = 1'000'000;
  auto input = makeRowVector({makeFlatSequence<int64_t>(0, kRowsPerBatch)});

  updateCudfByteConfig(/*minBytes=*/kTargetBytes, /*maxRows=*/kMaxRows);
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  concat.addInput(toCudfVector(input, kTargetBytes / 2));
  concat.addInput(toCudfVector(input, kTargetBytes / 2));
  auto output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), kRowsPerBatch);
  EXPECT_EQ(concat.getOutput(), nullptr) << "The tail must stay buffered";

  // One byte short on its own, so this flushes only if the tail is counted.
  ASSERT_TRUE(concat.needsInput());
  concat.addInput(toCudfVector(input, kTargetBytes - 1));
  EXPECT_FALSE(concat.needsInput());
  output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), kRowsPerBatch);

  concat.noMoreInput();
  output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), kRowsPerBatch);
  EXPECT_TRUE(concat.isFinished());
  concat.close();
}

TEST_F(CudfBatchConcatTest, usesRowTargetWhenByteTargetIsNotConfigured) {
  constexpr vector_size_t kRowsPerBatch = 10;
  auto input = makeRowVector({makeFlatSequence<int64_t>(0, kRowsPerBatch)});

  updateCudfConfig(/*min=*/2 * kRowsPerBatch, /*max=*/std::nullopt);
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  // Large byte estimates must not trigger a flush without a byte target.
  concat.addInput(toCudfVector(input, 1'000'000));
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);

  concat.addInput(toCudfVector(input, 1'000'000));
  EXPECT_FALSE(concat.needsInput());
  auto output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), 2 * kRowsPerBatch);
  concat.close();
}

TEST_F(CudfBatchConcatTest, rejectsZeroByteTarget) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  updateCudfByteConfig(/*minBytes=*/0, /*maxRows=*/std::nullopt);

  VELOX_ASSERT_THROW(
      CudfBatchConcat(0, &driverCtx, plan),
      "cuDF BatchConcat minimum byte target must be positive");
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

  // A one-byte target would flush on the first input if bytes were counted.
  updateCudfByteConfig(/*minBytes=*/1, /*maxRows=*/std::nullopt);
  CudfConfig::getInstance().batchSizeMinThreshold = 2 * kRowsPerBatch;
  auto plan = createAggregationPlan(input);
  auto task = createTask(plan);
  DriverCtx driverCtx(task, 0, 0, 0, 0);
  CudfBatchConcat concat(0, &driverCtx, plan);

  concat.addInput(toCudfVector(input));
  EXPECT_TRUE(concat.needsInput());
  EXPECT_EQ(concat.getOutput(), nullptr);

  concat.addInput(toCudfVector(input));
  EXPECT_FALSE(concat.needsInput());
  auto output = concat.getOutput();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), 2 * kRowsPerBatch);
  concat.close();
}

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

// Verifies that a byte target below the total input size flushes mid-stream
// rather than holding everything until noMoreInput.
TEST_F(CudfBatchConcatTest, concatFlushesMidStreamAtByteTarget) {
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }
  createDuckDbTable(vectors);

  // Measure rather than hard-code, so the target tracks cuDF's layout.
  const auto batchBytes = toCudfVector(vectors[0])->estimateFlatSize();
  ASSERT_GT(batchBytes, 0u);
  updateCudfByteConfig(/*minBytes=*/3 * batchBytes, /*maxRows=*/std::nullopt);
  CudfConfig::getInstance().concatOptimizationEnabled = true;

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

  auto concatStats = getConcatStats(task, aggNodeId);
  ASSERT_NE(concatStats, nullptr);
  EXPECT_EQ(concatStats->inputVectors, 6);
  EXPECT_GT(concatStats->outputVectors, 1)
      << "A byte target below the total input size should flush mid-stream";
  EXPECT_LT(concatStats->outputVectors, concatStats->inputVectors)
      << "CudfBatchConcat should still reduce the number of batches";
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

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
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
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
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
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
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
  updateCudfConfig(/*min=*/30, /*max=*/std::nullopt);
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
  updateCudfConfig(/*min=*/30, /*max=*/20);
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
  updateCudfConfig(/*min=*/30, /*max=*/20);
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
