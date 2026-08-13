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
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <rmm/device_buffer.hpp>

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <map>
#include <numeric>

namespace facebook::velox::cudf_velox {
namespace {

class TestCudfOperator final : public CudfOperatorBase {
 public:
  TestCudfOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId)
      : CudfOperatorBase(
            operatorId,
            driverCtx,
            std::move(outputType),
            planNodeId,
            "TestCudfOperator") {}

  ~TestCudfOperator() override {
    if (trackedGpuBytes_ != 0) {
      gpuPool()->reportExternalFree(trackedGpuBytes_);
      trackedGpuBytes_ = 0;
    }
  }

  bool needsInput() const override {
    return false;
  }

  bool isFinished() override {
    return true;
  }

  memory::MemoryPool* gpuPool() const {
    return customPool(kCudfMemoryResourceTag);
  }

  void trackGpuBytes(uint64_t bytes) {
    VELOX_CHECK_EQ(trackedGpuBytes_, 0);
    gpuPool()->reportExternalAllocation(bytes);
    trackedGpuBytes_ = bytes;
  }

  uint64_t gpuReclaimCalls() const {
    return gpuReclaimCalls_;
  }

  uint64_t lastGpuReclaimTarget() const {
    return lastGpuReclaimTarget_;
  }

 protected:
  void doAddInput(RowVectorPtr /*input*/) override {}

  RowVectorPtr doGetOutput() override {
    return nullptr;
  }

  bool gpuReclaimableBytes(uint64_t& reclaimableBytes) const override {
    reclaimableBytes = trackedGpuBytes_;
    return reclaimableBytes != 0;
  }

  void reclaimGpu(
      uint64_t targetBytes,
      memory::MemoryReclaimer::Stats& /*stats*/) override {
    ++gpuReclaimCalls_;
    lastGpuReclaimTarget_ = targetBytes;
    const auto bytesToFree = targetBytes == 0
        ? trackedGpuBytes_
        : std::min(targetBytes, trackedGpuBytes_);
    if (bytesToFree != 0) {
      gpuPool()->reportExternalFree(bytesToFree);
      trackedGpuBytes_ -= bytesToFree;
    }
  }

 private:
  uint64_t trackedGpuBytes_{0};
  uint64_t gpuReclaimCalls_{0};
  uint64_t lastGpuReclaimTarget_{0};
};

class CudfOperatorTest : public exec::test::OperatorTestBase {
 protected:
  struct TestOperatorHarness {
    std::shared_ptr<core::QueryCtx> queryCtx;
    std::shared_ptr<exec::Task> task;
    std::shared_ptr<exec::Driver> driver;
    std::unique_ptr<TestCudfOperator> operatorInstance;
  };

  void SetUp() override {
    OperatorTestBase::SetUp();
    registerCudf();
    registerPrestoAggregateFunctions("");
  }

  void TearDown() override {
    unregisterCudf();
    unregisterAggregateFunctions();
    OperatorTestBase::TearDown();
  }

  TestOperatorHarness makeOperatorHarness(const std::string& queryId) {
    auto resource = cudfCustomMemoryResource();
    VELOX_CHECK_NOT_NULL(resource);
    auto queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId(queryId)
                        .customMemoryResource(resource)
                        .build();

    const core::PlanNodeId planNodeId{"values"};
    auto input = makeRowVector({makeFlatVector<int64_t>({1})});
    core::PlanFragment planFragment;
    planFragment.planNode =
        std::make_shared<core::ValuesNode>(planNodeId, std::vector{input});
    auto task = exec::Task::create(
        "CudfOperatorTest",
        std::move(planFragment),
        0,
        queryCtx,
        exec::Task::ExecutionMode::kParallel);
    auto driver = exec::Driver::testingCreate(
        std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));
    auto operatorInstance = std::make_unique<TestCudfOperator>(
        0,
        driver->driverCtx(),
        std::dynamic_pointer_cast<const RowType>(input->type()),
        planNodeId);
    return TestOperatorHarness{
        std::move(queryCtx),
        std::move(task),
        std::move(driver),
        std::move(operatorInstance)};
  }
};

TEST_F(CudfOperatorTest, installsGpuLeafReclaimerDuringInitialize) {
  auto resource = cudfCustomMemoryResource();
  ASSERT_NE(resource, nullptr);
  auto queryCtx = core::QueryCtx::Builder()
                      .executor(driverExecutor_.get())
                      .queryId("cudf-operator-reclaimer")
                      .customMemoryResource(resource)
                      .build();

  const core::PlanNodeId planNodeId{"values"};
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  core::PlanFragment planFragment;
  planFragment.planNode =
      std::make_shared<core::ValuesNode>(planNodeId, std::vector{input});
  auto task = exec::Task::create(
      "CudfOperatorTest",
      std::move(planFragment),
      0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel);
  auto driver = exec::Driver::testingCreate(
      std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));
  TestCudfOperator testOperator{
      0,
      driver->driverCtx(),
      std::dynamic_pointer_cast<const RowType>(input->type()),
      planNodeId};

  auto* gpuPool = testOperator.gpuPool();
  ASSERT_NE(gpuPool, nullptr);
  EXPECT_EQ(gpuPool->reclaimer(), nullptr);

  testOperator.initialize();

  ASSERT_NE(gpuPool->reclaimer(), nullptr);
  uint64_t reclaimableBytes{1};
  EXPECT_FALSE(
      gpuPool->reclaimer()->reclaimableBytes(*gpuPool, reclaimableBytes));
  EXPECT_EQ(reclaimableBytes, 0);
}

TEST_F(CudfOperatorTest, routesOnlyGpuPoolReclaimToOperatorGpuHooks) {
  constexpr uint64_t kTrackedBytes = 4UL << 20;
  constexpr uint64_t kTargetBytes = 1UL << 20;
  auto harness = makeOperatorHarness("cudf-operator-gpu-callback");
  auto& op = *harness.operatorInstance;
  op.initialize();
  op.trackGpuBytes(kTrackedBytes);

  EXPECT_FALSE(op.canReclaim());
  uint64_t cpuReclaimableBytes{1};
  EXPECT_FALSE(op.reclaimableBytes(cpuReclaimableBytes));
  EXPECT_EQ(cpuReclaimableBytes, 0);

  auto* gpuPool = op.gpuPool();
  ASSERT_NE(gpuPool->reclaimer(), nullptr);
  uint64_t gpuReclaimableBytes{0};
  EXPECT_TRUE(
      gpuPool->reclaimer()->reclaimableBytes(*gpuPool, gpuReclaimableBytes));
  EXPECT_EQ(gpuReclaimableBytes, kTrackedBytes);

  harness.task->requestPause().wait();
  auto resumeTask =
      folly::makeGuard([&]() { exec::Task::resume(harness.task); });
  memory::MemoryReclaimer::Stats stats;
  {
    memory::ScopedMemoryArbitrationContext context(gpuPool);
    EXPECT_EQ(gpuPool->reclaim(kTargetBytes, 0, stats), kTargetBytes);
  }

  EXPECT_EQ(op.gpuReclaimCalls(), 1);
  EXPECT_EQ(op.lastGpuReclaimTarget(), kTargetBytes);
  EXPECT_EQ(gpuPool->usedBytes(), kTrackedBytes - kTargetBytes);
  EXPECT_EQ(stats.reclaimedBytes, kTargetBytes);

  {
    memory::ScopedMemoryArbitrationContext context(gpuPool);
    EXPECT_EQ(gpuPool->reclaim(0, 0, stats), kTrackedBytes - kTargetBytes);
  }
  EXPECT_EQ(op.gpuReclaimCalls(), 2);
  EXPECT_EQ(op.lastGpuReclaimTarget(), 0);
  EXPECT_EQ(gpuPool->usedBytes(), 0);
  EXPECT_EQ(stats.reclaimedBytes, kTrackedBytes);
}

TEST_F(CudfOperatorTest, groupbySpillsAndRestoresThroughSharedArbitrator) {
  constexpr int64_t kGpuCapacity = 64L << 20;
  constexpr int64_t kNumInitialGroups = 128L << 10;
  auto& config = CudfConfig::getInstance();
  const auto savedBatchSizeMaxThreshold = config.batchSizeMaxThreshold;
  const auto savedConcatOptimizationEnabled = config.concatOptimizationEnabled;
  const auto savedGroupbySpillStrategy = config.groupbySpillStrategy;
  auto restoreConfig = folly::makeGuard([&]() {
    config.batchSizeMaxThreshold = savedBatchSizeMaxThreshold;
    config.concatOptimizationEnabled = savedConcatOptimizationEnabled;
    config.groupbySpillStrategy = savedGroupbySpillStrategy;
  });
  config.batchSizeMaxThreshold = kNumInitialGroups + 1;
  config.concatOptimizationEnabled = false;
  config.groupbySpillStrategy = GroupbySpillStrategy::kStaySpilled;

  auto rawInput = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({1, 1})});
  auto plan = exec::test::PlanBuilder()
                  .values({rawInput})
                  .partialAggregation({"c0"}, {"count(1)"})
                  .finalAggregation()
                  .planNode();
  auto aggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  ASSERT_NE(aggregationNode, nullptr);

  auto resource = createCudfCustomMemoryResource(kGpuCapacity);
  ASSERT_NE(resource, nullptr);
  auto queryCtx = core::QueryCtx::Builder()
                      .executor(driverExecutor_.get())
                      .queryId("cudf-groupby-shared-arbitrator")
                      .customMemoryResource(resource)
                      .build();
  core::PlanFragment planFragment;
  planFragment.planNode = plan;
  auto task = exec::Task::create(
      "CudfGroupbyReclaimerTest",
      std::move(planFragment),
      0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel);
  auto driver = exec::Driver::testingCreate(
      std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));
  CudfGroupby groupby{0, driver->driverCtx(), aggregationNode};
  groupby.initialize();

  auto makePartialInput = [&](std::vector<int64_t> keys,
                              std::vector<int64_t> counts) {
    auto row = makeRowVector(
        {"c0", "a0"},
        {makeFlatVector<int64_t>(std::move(keys)),
         makeFlatVector<int64_t>(std::move(counts))});
    auto stream = cudfGlobalStreamPool().get_stream();
    auto table =
        with_arrow::toCudfTable(row, groupby.pool(), stream, get_output_mr());
    stream.synchronize();
    return std::make_shared<CudfVector>(
        groupby.pool(), row->type(), row->size(), std::move(table), stream);
  };

  std::vector<int64_t> initialKeys(kNumInitialGroups);
  std::iota(initialKeys.begin(), initialKeys.end(), 0);
  groupby.addInput(makePartialInput(
      std::move(initialKeys), std::vector<int64_t>(kNumInitialGroups, 1)));

  auto* gpuPool = groupby.customPool(kCudfMemoryResourceTag);
  ASSERT_NE(gpuPool, nullptr);
  ASSERT_NE(gpuPool->reclaimer(), nullptr);
  uint64_t reclaimableBytes{0};
  ASSERT_TRUE(
      gpuPool->reclaimer()->reclaimableBytes(*gpuPool, reclaimableBytes));
  ASSERT_GT(reclaimableBytes, 0);
  const auto usedBytesBefore = gpuPool->usedBytes();

  const auto statsBefore = resource->arbitrator()->stats();
  {
    auto pressureQueryCtx = core::QueryCtx::Builder()
                                .executor(driverExecutor_.get())
                                .queryId("cudf-groupby-arbitration-pressure")
                                .customMemoryResource(resource)
                                .build();
    auto pressurePool =
        pressureQueryCtx->customPool(std::string{kCudfMemoryResourceTag})
            ->addLeafChild("arbitrationPressure");
    CudfMemoryResource pressureResource{*output_mr_, pressurePool, resource};
    exec::Operator::NonReclaimableSectionGuard nonReclaimableGuard(&groupby);
    rmm::device_buffer pressure{
        kGpuCapacity, rmm::cuda_stream_default, pressureResource};
    EXPECT_EQ(pressure.size(), kGpuCapacity);
  }

  const auto statsAfter = resource->arbitrator()->stats();
  EXPECT_GT(statsAfter.numRequests, statsBefore.numRequests);
  EXPECT_GE(
      statsAfter.reclaimedUsedBytes - statsBefore.reclaimedUsedBytes,
      usedBytesBefore);
  EXPECT_EQ(gpuPool->usedBytes(), 0);
  reclaimableBytes = 1;
  EXPECT_FALSE(
      gpuPool->reclaimer()->reclaimableBytes(*gpuPool, reclaimableBytes));
  EXPECT_EQ(reclaimableBytes, 0);

  groupby.addInput(makePartialInput({1, kNumInitialGroups}, {2, 1}));
  groupby.addInput(makePartialInput({1, kNumInitialGroups + 1}, {3, 1}));
  EXPECT_EQ(gpuPool->usedBytes(), 0);
  const auto statsBeforeDrain = groupby.stats(false).runtimeStats;
  EXPECT_EQ(
      statsBeforeDrain.at(std::string(CudfGroupby::kSpilledInputChunks)).sum,
      2);
  EXPECT_EQ(
      statsBeforeDrain.count(std::string(CudfGroupby::kRestoredLeaves)), 0);
  EXPECT_EQ(
      statsBeforeDrain.count(std::string(CudfGroupby::kRestoredInputChunks)),
      0);
  groupby.noMoreInput();

  std::map<int64_t, int64_t> actual;
  size_t outputBatches = 0;
  while (!groupby.isFinished()) {
    auto output = std::dynamic_pointer_cast<CudfVector>(groupby.getOutput());
    if (!output) {
      continue;
    }
    ++outputBatches;
    auto stream = output->stream();
    auto row = with_arrow::toVeloxColumn(
        output->getTableView(),
        groupby.pool(),
        aggregationNode->outputType(),
        "",
        stream,
        get_output_mr());
    stream.synchronize();
    auto* keys = row->childAt(0)->as<FlatVector<int64_t>>();
    auto* counts = row->childAt(1)->as<FlatVector<int64_t>>();
    for (vector_size_t i = 0; i < row->size(); ++i) {
      actual.emplace(keys->valueAt(i), counts->valueAt(i));
    }
  }

  ASSERT_EQ(actual.size(), kNumInitialGroups + 2);
  EXPECT_GT(outputBatches, 1);
  EXPECT_EQ(actual.at(0), 1);
  EXPECT_EQ(actual.at(1), 6);
  EXPECT_EQ(actual.at(kNumInitialGroups - 1), 1);
  EXPECT_EQ(actual.at(kNumInitialGroups), 1);
  EXPECT_EQ(actual.at(kNumInitialGroups + 1), 1);

  const auto runtimeStats = groupby.stats(false).runtimeStats;
  EXPECT_GE(
      runtimeStats.at(std::string(CudfGroupby::kSpilledBytes)).sum,
      usedBytesBefore);
  EXPECT_EQ(
      runtimeStats.at(std::string(CudfGroupby::kSpilledRows)).sum,
      kNumInitialGroups + 4);
  EXPECT_GT(runtimeStats.at(std::string(CudfGroupby::kSpilledLeaves)).sum, 0);
  EXPECT_EQ(
      runtimeStats.at(std::string(CudfGroupby::kSpilledInputChunks)).sum, 2);
  EXPECT_GE(
      runtimeStats.at(std::string(CudfGroupby::kRestoredBytes)).sum,
      usedBytesBefore);
  EXPECT_EQ(
      runtimeStats.at(std::string(CudfGroupby::kRestoredRows)).sum,
      kNumInitialGroups + 4);
  EXPECT_GT(runtimeStats.at(std::string(CudfGroupby::kRestoredLeaves)).sum, 0);
  EXPECT_EQ(
      runtimeStats.at(std::string(CudfGroupby::kRestoredInputChunks)).sum, 2);
}

TEST_F(CudfOperatorTest, stickyGroupbyDrainCompactsOverlappingKeys) {
  constexpr int64_t kGpuCapacity = 64L << 20;
  auto& config = CudfConfig::getInstance();
  const auto savedBatchSizeMaxThreshold = config.batchSizeMaxThreshold;
  const auto savedConcatOptimizationEnabled = config.concatOptimizationEnabled;
  const auto savedGroupbySpillStrategy = config.groupbySpillStrategy;
  auto restoreConfig = folly::makeGuard([&]() {
    config.batchSizeMaxThreshold = savedBatchSizeMaxThreshold;
    config.concatOptimizationEnabled = savedConcatOptimizationEnabled;
    config.groupbySpillStrategy = savedGroupbySpillStrategy;
  });
  config.batchSizeMaxThreshold = 1;
  config.concatOptimizationEnabled = false;
  config.groupbySpillStrategy = GroupbySpillStrategy::kStaySpilled;

  auto rawInput = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({7}), makeFlatVector<int64_t>({1})});
  auto plan = exec::test::PlanBuilder()
                  .values({rawInput})
                  .partialAggregation({"c0"}, {"count(1)"})
                  .finalAggregation()
                  .planNode();
  auto aggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  ASSERT_NE(aggregationNode, nullptr);

  auto resource = createCudfCustomMemoryResource(kGpuCapacity);
  ASSERT_NE(resource, nullptr);
  auto queryCtx = core::QueryCtx::Builder()
                      .executor(driverExecutor_.get())
                      .queryId("cudf-groupby-sticky-overlap")
                      .customMemoryResource(resource)
                      .build();
  core::PlanFragment planFragment;
  planFragment.planNode = plan;
  auto task = exec::Task::create(
      "CudfGroupbyStickyOverlapTest",
      std::move(planFragment),
      0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel);
  auto driver = exec::Driver::testingCreate(
      std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));
  CudfGroupby groupby{0, driver->driverCtx(), aggregationNode};
  groupby.initialize();

  auto makePartialInput = [&](int64_t count) {
    auto row = makeRowVector(
        {"c0", "a0"},
        {makeFlatVector<int64_t>({7}), makeFlatVector<int64_t>({count})});
    auto stream = cudfGlobalStreamPool().get_stream();
    auto table =
        with_arrow::toCudfTable(row, groupby.pool(), stream, get_output_mr());
    stream.synchronize();
    return std::make_shared<CudfVector>(
        groupby.pool(), row->type(), row->size(), std::move(table), stream);
  };

  groupby.addInput(makePartialInput(1));
  auto* gpuPool = groupby.customPool(kCudfMemoryResourceTag);
  ASSERT_NE(gpuPool, nullptr);
  ASSERT_NE(gpuPool->reclaimer(), nullptr);
  {
    task->requestPause().wait();
    auto resumeTask = folly::makeGuard([&]() { exec::Task::resume(task); });
    memory::MemoryReclaimer::Stats stats;
    memory::ScopedMemoryArbitrationContext context(gpuPool);
    EXPECT_GT(gpuPool->reclaim(0, 0, stats), 0);
  }
  EXPECT_EQ(gpuPool->usedBytes(), 0);

  groupby.addInput(makePartialInput(2));
  groupby.addInput(makePartialInput(3));
  EXPECT_EQ(gpuPool->usedBytes(), 0);
  groupby.noMoreInput();

  int64_t resultCount = 0;
  size_t outputBatches = 0;
  while (!groupby.isFinished()) {
    auto output = std::dynamic_pointer_cast<CudfVector>(groupby.getOutput());
    if (!output) {
      continue;
    }
    ++outputBatches;
    auto stream = output->stream();
    auto row = with_arrow::toVeloxColumn(
        output->getTableView(),
        groupby.pool(),
        aggregationNode->outputType(),
        "",
        stream,
        get_output_mr());
    stream.synchronize();
    ASSERT_EQ(row->size(), 1);
    EXPECT_EQ(row->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0), 7);
    resultCount = row->childAt(1)->as<FlatVector<int64_t>>()->valueAt(0);
  }

  EXPECT_EQ(outputBatches, 1);
  EXPECT_EQ(resultCount, 6);
  const auto runtimeStats = groupby.stats(false).runtimeStats;
  EXPECT_EQ(runtimeStats.at(std::string(CudfGroupby::kRestoredRows)).sum, 3);
  EXPECT_EQ(
      runtimeStats.at(std::string(CudfGroupby::kRestoredInputChunks)).sum, 2);
}

TEST_F(CudfOperatorTest, skipsGpuReclaimInNonReclaimableSection) {
  constexpr uint64_t kTrackedBytes = 2UL << 20;
  auto harness = makeOperatorHarness("cudf-operator-non-reclaimable");
  auto& op = *harness.operatorInstance;
  op.initialize();
  op.trackGpuBytes(kTrackedBytes);
  auto* gpuPool = op.gpuPool();

  harness.task->requestPause().wait();
  auto resumeTask =
      folly::makeGuard([&]() { exec::Task::resume(harness.task); });
  memory::MemoryReclaimer::Stats stats;
  {
    exec::Operator::NonReclaimableSectionGuard guard(&op);
    memory::ScopedMemoryArbitrationContext context(gpuPool);
    EXPECT_EQ(gpuPool->reclaim(0, 0, stats), 0);
  }
  EXPECT_EQ(stats.numNonReclaimableAttempts, 1);
  EXPECT_EQ(op.gpuReclaimCalls(), 0);
  EXPECT_EQ(gpuPool->usedBytes(), kTrackedBytes);

  {
    memory::ScopedMemoryArbitrationContext context(gpuPool);
    EXPECT_EQ(gpuPool->reclaim(0, 0, stats), kTrackedBytes);
  }
  EXPECT_EQ(op.gpuReclaimCalls(), 1);
  EXPECT_EQ(gpuPool->usedBytes(), 0);
}

TEST_F(CudfOperatorTest, gpuReclaimerStopsUsingOperatorAfterDriverExpires) {
  auto harness = makeOperatorHarness("cudf-operator-driver-lifetime");
  harness.operatorInstance->initialize();
  auto* gpuPool = harness.operatorInstance->gpuPool();
  ASSERT_NE(gpuPool->reclaimer(), nullptr);

  harness.driver.reset();

  uint64_t reclaimableBytes{1};
  EXPECT_FALSE(
      gpuPool->reclaimer()->reclaimableBytes(*gpuPool, reclaimableBytes));
  EXPECT_EQ(reclaimableBytes, 0);

  memory::MemoryReclaimer::Stats stats;
  memory::ScopedMemoryArbitrationContext context(gpuPool);
  EXPECT_EQ(gpuPool->reclaim(0, 0, stats), 0);
}

} // namespace
} // namespace facebook::velox::cudf_velox
