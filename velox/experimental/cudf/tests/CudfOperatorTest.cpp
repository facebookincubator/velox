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

#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <folly/ScopeGuard.h>

#include <algorithm>

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
  }

  void TearDown() override {
    unregisterCudf();
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
