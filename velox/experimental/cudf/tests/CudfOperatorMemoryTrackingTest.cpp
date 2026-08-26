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

#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <cudf/utilities/memory_resource.hpp>

#include <folly/ScopeGuard.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

std::shared_ptr<memory::CustomMemoryResource> makeCustomResource() {
  constexpr int64_t kCapacity = 1L << 30;
  memory::MemoryAllocator::Options options;
  options.capacity = kCapacity;
  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCudfMemoryResourceTag},
      std::make_shared<memory::MallocAllocator>(options),
      memory::MemoryArbitrator::create({}),
      []() { return memory::MemoryReclaimer::create(0); },
      kCapacity);
}

class TestCudfOperator final : public CudfOperatorBase {
 public:
  struct OutputAllocation {
    void* pointer;
    std::size_t bytes;
    rmm::device_async_resource_ref resource;
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
        rmm::cuda_stream_default, kOutputBytes, kAlignment);
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
  void exerciseLegacyTemporaryAllocation() {
    constexpr std::size_t kTempBytes = 256;
    constexpr auto kAlignment = alignof(std::max_align_t);
    auto currentResource = cudf::get_current_device_resource_ref();
    auto* temporary = currentResource.allocate(
        rmm::cuda_stream_default, kTempBytes, kAlignment);
    currentResource.deallocate(
        rmm::cuda_stream_default, temporary, kTempBytes, kAlignment);
  }

  std::optional<OutputAllocation> outputAllocation_;
};

class CudfOperatorMemoryTrackingTest : public exec::test::OperatorTestBase {};

TEST_F(
    CudfOperatorMemoryTrackingTest,
    scopesEveryLifecycleMethodAndKeepsOutputsTracked) {
  auto resourceOwner = makeCustomResource();
  auto gpuRoot = memory::memoryManager()->addCustomRootPool(
      "cudfOperatorMemoryTracking.gpu", resourceOwner);
  auto queryCtx = core::QueryCtx::Builder()
                      .executor(driverExecutor_.get())
                      .queryId("cudf-operator-memory-tracking")
                      .customPool(std::string{kCudfMemoryResourceTag}, gpuRoot)
                      .build();
  auto customResources =
      memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
  customResources->insert(std::string{kCudfMemoryResourceTag}, resourceOwner);
  queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
      memory::kCustomMemoryResourceRegistryKey, customResources);

  const core::PlanNodeId planNodeId{"values"};
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  core::PlanFragment planFragment;
  planFragment.planNode =
      std::make_shared<core::ValuesNode>(planNodeId, std::vector{input});
  auto task = exec::Task::create(
      "CudfOperatorMemoryTrackingTest",
      std::move(planFragment),
      0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel);
  auto driver = exec::Driver::testingCreate(
      std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));

  auto upstream = createMemoryResource("cuda", 0);
  auto savedTempMr = mr_;
  auto savedOutputMr = output_mr_;
  auto previousCurrentResource = cudf::set_current_device_resource(
      createThreadLocalTemporaryMemoryResource(upstream));
  mr_ = upstream;
  output_mr_ = upstream;
  SCOPE_EXIT {
    output_mr_ = std::move(savedOutputMr);
    mr_ = std::move(savedTempMr);
    cudf::set_current_device_resource(std::move(previousCurrentResource));
  };

  std::optional<TestCudfOperator::OutputAllocation> outputAllocation;
  memory::MemoryPool* gpuPool{nullptr};
  {
    TestCudfOperator testOperator{
        0,
        driver->driverCtx(),
        std::dynamic_pointer_cast<const RowType>(input->type()),
        planNodeId};
    gpuPool = testOperator.gpuPool();
    ASSERT_NE(gpuPool, nullptr);

    testOperator.initialize();
    ContinueFuture future;
    EXPECT_EQ(
        testOperator.isBlocked(&future), exec::BlockingReason::kNotBlocked);
    testOperator.addInput(nullptr);
    EXPECT_EQ(testOperator.getOutput(), nullptr);
    testOperator.noMoreInput();
    testOperator.close();

    outputAllocation = testOperator.releaseOutputAllocation();
    EXPECT_EQ(gpuPool->usedBytes(), outputAllocation->bytes);
    const auto stats = gpuPool->stats();
    EXPECT_EQ(stats.numExternalAllocs, 7);
    EXPECT_EQ(stats.numExternalFrees, 6);
  }

  outputAllocation->resource.deallocate(
      rmm::cuda_stream_default,
      outputAllocation->pointer,
      outputAllocation->bytes,
      alignof(std::max_align_t));
  EXPECT_EQ(gpuPool->usedBytes(), 0);
}

} // namespace
} // namespace facebook::velox::cudf_velox
