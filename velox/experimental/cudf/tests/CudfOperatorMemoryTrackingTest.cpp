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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TaskMonitor.h"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <folly/ScopeGuard.h>
#include <gmock/gmock.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

std::shared_ptr<memory::CustomMemoryResource> makeCustomResource(
    int64_t capacity) {
  memory::MemoryAllocator::Options options;
  options.capacity = capacity;
  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCudfMemoryResourceTag},
      std::make_shared<memory::MallocAllocator>(options),
      memory::MemoryArbitrator::create({}),
      []() { return memory::MemoryReclaimer::create(0); },
      capacity);
}

// Allocates and frees one temporary through cuDF's current device resource,
// the path cuDF APIs take when they are not given a resource.
void exerciseLegacyTemporaryAllocation() {
  constexpr std::size_t kTempBytes = 256;
  constexpr auto kAlignment = alignof(std::max_align_t);
  auto currentResource = cudf::get_current_device_resource_ref();
  auto* temporary = currentResource.allocate(
      cudf::get_default_stream(), kTempBytes, kAlignment);
  currentResource.deallocate(
      cudf::get_default_stream(), temporary, kTempBytes, kAlignment);
}

// Calls 'visitor' for every leaf pool under 'pool'.
void forEachLeafPool(
    memory::MemoryPool& pool,
    const std::function<void(memory::MemoryPool&)>& visitor) {
  pool.visitChildren([&](memory::MemoryPool* child) {
    if (child->isLeaf()) {
      visitor(*child);
    } else {
      forEachLeafPool(*child, visitor);
    }
    return true;
  });
}

class TestCudfOperator final : public CudfOperatorBase {
 public:
  // Owns a copy of the resource, as device buffers do, so the allocation can
  // be freed after the operator is destroyed.
  struct OutputAllocation {
    void* pointer;
    std::size_t bytes;
    cuda::mr::any_resource<cuda::mr::device_accessible> resource;
  };

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

  bool needsInput() const override {
    return true;
  }

  bool isFinished() override {
    return false;
  }

  memory::MemoryPool* gpuPool() const {
    return customPool(kCudfMemoryResourceTag);
  }

  OutputAllocation releaseOutputAllocation() {
    VELOX_CHECK(outputAllocation_.has_value());
    return std::exchange(outputAllocation_, std::nullopt).value();
  }

 protected:
  void doInitialize() override {
    exerciseLegacyTemporaryAllocation();
  }

  exec::BlockingReason doIsBlocked(ContinueFuture* /*future*/) override {
    exerciseLegacyTemporaryAllocation();
    return exec::BlockingReason::kNotBlocked;
  }

  void doAddInput(RowVectorPtr /*input*/) override {
    exerciseLegacyTemporaryAllocation();
  }

  RowVectorPtr doGetOutput() override {
    exerciseLegacyTemporaryAllocation();
    constexpr std::size_t kOutputBytes = 512;
    constexpr auto kAlignment = alignof(std::max_align_t);
    auto outputResource = get_output_mr();
    auto* output = outputResource.allocate(
        cudf::get_default_stream(), kOutputBytes, kAlignment);
    outputAllocation_.emplace(
        OutputAllocation{output, kOutputBytes, outputResource});
    return nullptr;
  }

  void doNoMoreInput() override {
    exerciseLegacyTemporaryAllocation();
    Operator::noMoreInput();
  }

  void doClose() override {
    exerciseLegacyTemporaryAllocation();
    Operator::close();
  }

 private:
  std::optional<OutputAllocation> outputAllocation_;
};

class TestCudfSourceOperator final : public CudfSourceOperatorBase {
 public:
  TestCudfSourceOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId)
      : CudfSourceOperatorBase(
            operatorId,
            driverCtx,
            std::move(outputType),
            planNodeId,
            "TestCudfSourceOperator") {}

  bool isFinished() override {
    return false;
  }

  memory::MemoryPool* gpuPool() const {
    return customPool(kCudfMemoryResourceTag);
  }

 protected:
  void doInitialize() override {
    exerciseLegacyTemporaryAllocation();
  }

  exec::BlockingReason doIsBlocked(ContinueFuture* /*future*/) override {
    exerciseLegacyTemporaryAllocation();
    return exec::BlockingReason::kNotBlocked;
  }

  RowVectorPtr doGetOutput() override {
    exerciseLegacyTemporaryAllocation();
    return nullptr;
  }

  void doClose() override {
    exerciseLegacyTemporaryAllocation();
    SourceOperator::close();
  }
};

class CudfOperatorMemoryTrackingTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    // The operator lifecycle methods bind the cuDF context to the calling
    // thread, which requires the registration-time device to be recorded.
    // Registration also installs the resources the operators charge through.
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Returns a query context whose `gpu` custom root is 'gpuRoot', backed by
  // 'resourceOwner'.
  std::shared_ptr<core::QueryCtx> makeTrackedQueryCtx(
      const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner,
      std::shared_ptr<memory::MemoryPool> gpuRoot,
      const std::string& queryId) {
    auto queryCtx =
        core::QueryCtx::Builder()
            .executor(driverExecutor_.get())
            .queryId(queryId)
            .customPool(std::string{kCudfMemoryResourceTag}, std::move(gpuRoot))
            .build();
    auto customResources =
        memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
    customResources->insert(std::string{kCudfMemoryResourceTag}, resourceOwner);
    queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
        memory::kCustomMemoryResourceRegistryKey, customResources);
    return queryCtx;
  }

  // Creates a driver of a single-node task, for driving operators directly.
  std::shared_ptr<exec::Driver> makeDriver(
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      const RowVectorPtr& input,
      const core::PlanNodeId& planNodeId) {
    core::PlanFragment planFragment;
    planFragment.planNode =
        std::make_shared<core::ValuesNode>(planNodeId, std::vector{input});
    auto task = exec::Task::create(
        "CudfOperatorMemoryTrackingTest",
        std::move(planFragment),
        0,
        queryCtx,
        exec::Task::ExecutionMode::kParallel);
    return exec::Driver::testingCreate(
        std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));
  }

  // A plan that runs filter/project and order by on the GPU over each of
  // 'inputs', and merges the sorted results there. Two or more inputs make the
  // merge allocate; it forwards the batch of a single input.
  core::PlanNodePtr makeFilterProjectOrderByMergePlan(
      const std::vector<RowVectorPtr>& inputs) {
    exec::test::PlanBuilder builder;
    std::vector<core::PlanNodePtr> sources;
    for (const auto& input : inputs) {
      sources.push_back(builder.newPlan()
                            .values({input})
                            .filter("c1 > 3")
                            .project({"c0", "c1 * 2 AS c2"})
                            .orderBy({"c0"}, true)
                            .planNode());
    }
    return builder.newPlan().localMerge({"c0"}, std::move(sources)).planNode();
  }
};

TEST_F(
    CudfOperatorMemoryTrackingTest,
    scopesEveryLifecycleMethodAndKeepsOutputsTracked) {
  auto resourceOwner = makeCustomResource(1L << 30);
  auto queryCtx = makeTrackedQueryCtx(
      resourceOwner,
      memory::memoryManager()->addCustomRootPool(
          "cudfOperatorMemoryTracking.gpu", resourceOwner),
      "cudf-operator-memory-tracking");
  const core::PlanNodeId planNodeId{"values"};
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  auto driver = makeDriver(queryCtx, input, planNodeId);

  std::optional<TestCudfOperator::OutputAllocation> outputAllocation;
  std::weak_ptr<memory::MemoryPool> gpuPool;
  {
    TestCudfOperator testOperator{
        0,
        driver->driverCtx(),
        std::dynamic_pointer_cast<const RowType>(input->type()),
        planNodeId};
    ASSERT_NE(testOperator.gpuPool(), nullptr);
    gpuPool = testOperator.gpuPool()->shared_from_this();

    testOperator.initialize();
    ContinueFuture future;
    EXPECT_EQ(
        testOperator.isBlocked(&future), exec::BlockingReason::kNotBlocked);
    testOperator.addInput(nullptr);
    EXPECT_EQ(testOperator.getOutput(), nullptr);
    testOperator.noMoreInput();
    testOperator.close();

    outputAllocation = testOperator.releaseOutputAllocation();
    EXPECT_EQ(gpuPool.lock()->usedBytes(), outputAllocation->bytes);
    // Each of initialize, isBlocked, addInput, getOutput, noMoreInput and
    // close makes one temporary allocation and frees it; getOutput makes one
    // more that the test holds past close().
    constexpr int32_t kLifecycleTemporaries = 6;
    const auto stats = gpuPool.lock()->stats();
    EXPECT_EQ(stats.numExternalAllocs, kLifecycleTemporaries + 1);
    EXPECT_EQ(stats.numExternalFrees, kLifecycleTemporaries);
  }

  outputAllocation->resource.deallocate(
      cudf::get_default_stream(),
      outputAllocation->pointer,
      outputAllocation->bytes,
      alignof(std::max_align_t));
  EXPECT_EQ(gpuPool.lock()->usedBytes(), 0);
}

TEST_F(CudfOperatorMemoryTrackingTest, scopesEverySourceLifecycleMethod) {
  auto resourceOwner = makeCustomResource(1L << 30);
  auto queryCtx = makeTrackedQueryCtx(
      resourceOwner,
      memory::memoryManager()->addCustomRootPool(
          "cudfSourceOperatorMemoryTracking.gpu", resourceOwner),
      "cudf-source-operator-memory-tracking");
  const core::PlanNodeId planNodeId{"values"};
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  auto driver = makeDriver(queryCtx, input, planNodeId);

  TestCudfSourceOperator testOperator{
      0,
      driver->driverCtx(),
      std::dynamic_pointer_cast<const RowType>(input->type()),
      planNodeId};
  auto* gpuPool = testOperator.gpuPool();
  ASSERT_NE(gpuPool, nullptr);

  testOperator.initialize();
  ContinueFuture future;
  EXPECT_EQ(testOperator.isBlocked(&future), exec::BlockingReason::kNotBlocked);
  EXPECT_EQ(testOperator.getOutput(), nullptr);
  testOperator.close();

  // initialize, isBlocked, getOutput and close each make one temporary.
  constexpr int32_t kLifecycleTemporaries = 4;
  const auto stats = gpuPool->stats();
  EXPECT_EQ(stats.numExternalAllocs, kLifecycleTemporaries);
  EXPECT_EQ(stats.numExternalFrees, kLifecycleTemporaries);
  EXPECT_EQ(gpuPool->usedBytes(), 0);
}

TEST_F(CudfOperatorMemoryTrackingTest, chargesEveryCudfOperatorOfAPlan) {
  auto& config = CudfConfig::getInstance();
  const auto savedAllowCpuFallback = config.allowCpuFallback;
  config.allowCpuFallback = false;
  SCOPE_EXIT {
    config.allowCpuFallback = savedAllowCpuFallback;
  };

  constexpr vector_size_t kNumRows = 1'000;
  std::vector<RowVectorPtr> data;
  for (int32_t source = 0; source < 2; ++source) {
    data.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            kNumRows, [&](auto row) { return (row * 7 + source) % kNumRows; }),
        makeFlatVector<int64_t>(kNumRows, [](auto row) { return row % 13; }),
    }));
  }
  createDuckDbTable(data);

  auto resourceOwner = makeCustomResource(1L << 30);
  auto gpuRoot = memory::memoryManager()->addCustomRootPool(
      "cudfPlanMemoryTracking.gpu", resourceOwner);
  std::weak_ptr<memory::MemoryPool> weakRoot = gpuRoot;
  auto queryCtx = makeTrackedQueryCtx(
      resourceOwner, std::move(gpuRoot), "cudf-plan-memory-tracking");

  auto task = exec::test::AssertQueryBuilder(
                  makeFilterProjectOrderByMergePlan(data), duckDbQueryRunner_)
                  .queryCtx(queryCtx)
                  .assertResults(
                      "SELECT c0, c1 * 2 FROM tmp WHERE c1 > 3 ORDER BY c0",
                      std::vector<uint32_t>{0});

  // Leaf pools are named op.<node>.<pipeline>.<driver>.<operator type>.gpu.
  std::set<std::string> chargedOperators;
  forEachLeafPool(*weakRoot.lock(), [&](memory::MemoryPool& leaf) {
    EXPECT_EQ(leaf.usedBytes(), 0) << leaf.name();
    if (leaf.stats().numExternalAllocs > 0) {
      const auto& name = leaf.name();
      const auto typeEnd = name.rfind('.');
      const auto typeBegin = name.rfind('.', typeEnd - 1) + 1;
      chargedOperators.insert(name.substr(typeBegin, typeEnd - typeBegin));
    }
  });
  EXPECT_THAT(
      chargedOperators,
      testing::IsSupersetOf(
          {"CudfFromVelox",
           "CudfFilterProject",
           "CudfOrderBy",
           "CudfLocalMerge"}));
  EXPECT_EQ(weakRoot.lock()->usedBytes(), 0);

  task.reset();
  queryCtx.reset();
  exec::test::TaskMonitor::waitForAllTasksDeleted();
  EXPECT_TRUE(weakRoot.expired());
}

TEST_F(CudfOperatorMemoryTrackingTest, failedPlanReleasesEverything) {
  auto& config = CudfConfig::getInstance();
  const auto savedAllowCpuFallback = config.allowCpuFallback;
  config.allowCpuFallback = false;
  SCOPE_EXIT {
    config.allowCpuFallback = savedAllowCpuFallback;
  };

  constexpr vector_size_t kNumRows = 1'000;
  std::vector<RowVectorPtr> data;
  for (int32_t source = 0; source < 2; ++source) {
    data.push_back(makeRowVector({
        makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; }),
        makeFlatVector<int64_t>(kNumRows, [](auto row) { return row % 13; }),
    }));
  }

  // Smaller than the reservation of the first allocation.
  constexpr int64_t kCapacity = 1'024;
  auto resourceOwner = makeCustomResource(kCapacity);
  auto gpuRoot = memory::memoryManager()->addCustomRootPool(
      "cudfFailedPlanMemoryTracking.gpu", resourceOwner);
  std::weak_ptr<memory::MemoryPool> weakRoot = gpuRoot;
  auto queryCtx = makeTrackedQueryCtx(
      resourceOwner, std::move(gpuRoot), "cudf-failed-plan-memory-tracking");

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(makeFilterProjectOrderByMergePlan(data))
          .queryCtx(queryCtx)
          .copyResults(pool()),
      "Exceeded memory pool capacity");

  exec::test::TaskMonitor::waitForAllTasksDeleted();
  EXPECT_EQ(weakRoot.lock()->usedBytes(), 0);
  queryCtx.reset();
  EXPECT_TRUE(weakRoot.expired());
}

} // namespace
} // namespace facebook::velox::cudf_velox
