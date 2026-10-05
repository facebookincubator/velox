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
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda/stream>

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <new>
#include <thread>

namespace facebook::velox::cudf_velox {
namespace {
struct HostBackedResourceState {
  std::atomic<bool> failNextAllocation{false};
  std::atomic<bool> failNextCopy{false};
  std::atomic<std::uintptr_t> allocationStream{0};
  std::atomic<std::uintptr_t> deallocationStream{0};
  std::atomic<int64_t> liveBytes{0};
};

/// Host-backed test resource satisfying the device-accessible resource
/// concept. Tests only exercise resource delegation and never dereference the
/// allocation from device code.
class HostBackedDeviceResource {
 public:
  explicit HostBackedDeviceResource(
      std::shared_ptr<HostBackedResourceState> state)
      : state_(std::move(state)) {}

  HostBackedDeviceResource(const HostBackedDeviceResource& other)
      : state_(other.state_) {
    if (state_->failNextCopy.exchange(false)) {
      throw std::runtime_error{"injected resource copy failure"};
    }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    if (state_->failNextAllocation.exchange(false)) {
      throw std::bad_alloc{};
    }
    auto* allocation = ::operator new(bytes, std::align_val_t(alignment));
    state_->liveBytes.fetch_add(bytes);
    return allocation;
  }

  void deallocate_sync(
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    state_->liveBytes.fetch_sub(bytes);
    ::operator delete(pointer, std::align_val_t(alignment));
  }

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    state_->allocationStream = reinterpret_cast<std::uintptr_t>(stream.get());
    return allocate_sync(bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    state_->deallocationStream = reinterpret_cast<std::uintptr_t>(stream.get());
    deallocate_sync(pointer, bytes, alignment);
  }

  bool operator==(const HostBackedDeviceResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const HostBackedDeviceResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const HostBackedDeviceResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  std::shared_ptr<HostBackedResourceState> state_;
};

cuda::mr::any_resource<cuda::mr::device_accessible> makeTestUpstream(
    const std::shared_ptr<HostBackedResourceState>& state) {
  return HostBackedDeviceResource{state};
}

std::shared_ptr<memory::CustomMemoryResource> makeCustomResource(
    int64_t capacity = 1L << 30) {
  memory::MemoryAllocator::Options options;
  options.capacity = capacity;
  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCudfMemoryResourceTag},
      std::make_shared<memory::MallocAllocator>(options),
      memory::MemoryArbitrator::create({}),
      []() { return memory::MemoryReclaimer::create(0); },
      capacity);
}

class CudfMemoryResourceTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize({});
  }
};
TEST_F(CudfMemoryResourceTest, ReportsAcrossThreadsAndRollsBackFailure) {
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceTest");
  auto state = std::make_shared<HostBackedResourceState>();
  CudfMemoryResource reportingResource{makeTestUpstream(state), pool};
  cuda::mr::any_resource<cuda::mr::device_accessible> ownedResource{
      rmm::device_async_resource_ref{reportingResource}};

  constexpr std::size_t kBytes = 256;
  cuda::stream allocationStream{
      cuda::device_ref{rmm::get_current_cuda_device().value()}};
  cuda::stream deallocationStream{
      cuda::device_ref{rmm::get_current_cuda_device().value()}};
  void* allocation = nullptr;
  std::thread allocateThread([&] {
    allocation = ownedResource.allocate(
        allocationStream.get(), kBytes, alignof(std::max_align_t));
  });
  allocateThread.join();

  EXPECT_NE(allocation, nullptr);
  EXPECT_EQ(pool->usedBytes(), kBytes);

  std::thread deallocateThread([&] {
    ownedResource.deallocate(
        deallocationStream.get(),
        allocation,
        kBytes,
        alignof(std::max_align_t));
  });
  deallocateThread.join();
  EXPECT_EQ(pool->usedBytes(), 0);
  EXPECT_EQ(
      state->allocationStream,
      reinterpret_cast<std::uintptr_t>(allocationStream.get()));
  EXPECT_EQ(
      state->deallocationStream,
      reinterpret_cast<std::uintptr_t>(deallocationStream.get()));
  EXPECT_NE(state->allocationStream, state->deallocationStream);

  auto* synchronousAllocation =
      ownedResource.allocate_sync(kBytes, alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), kBytes);
  ownedResource.deallocate_sync(
      synchronousAllocation, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), 0);

  state->failNextAllocation = true;
  EXPECT_THROW(
      ownedResource.allocate(
          cuda::stream_ref{cudaStream_t{}}, kBytes, alignof(std::max_align_t)),
      std::bad_alloc);
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(CudfMemoryResourceTest, RetainsCompleteCustomPoolHierarchy) {
  auto state = std::make_shared<HostBackedResourceState>();
  auto owner = makeCustomResource();
  auto root = memory::memoryManager()->addCustomRootPool(
      "cudfMemoryResourceOwnershipRoot", owner);
  auto taskPool = root->addAggregateChild("task");
  auto nodePool = taskPool->addAggregateChild("node");
  auto operatorPool = nodePool->addLeafChild("operator");

  std::weak_ptr<memory::CustomMemoryResource> weakOwner = owner;
  std::weak_ptr<memory::MemoryPool> weakRoot = root;
  std::weak_ptr<memory::MemoryPool> weakOperatorPool = operatorPool;
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
      retainedResource;
  {
    CudfMemoryResource reportingResource{makeTestUpstream(state), operatorPool};
    retainedResource.emplace(reportingResource);
  }

  constexpr std::size_t kBytes = 192;
  auto* allocation = retainedResource->allocate(
      cuda::stream_ref{cudaStream_t{}}, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(operatorPool->usedBytes(), kBytes);

  operatorPool.reset();
  nodePool.reset();
  taskPool.reset();
  root.reset();
  EXPECT_FALSE(weakOwner.expired());
  EXPECT_FALSE(weakRoot.expired());
  EXPECT_FALSE(weakOperatorPool.expired());

  retainedResource->deallocate(
      cuda::stream_ref{cudaStream_t{}},
      allocation,
      kBytes,
      alignof(std::max_align_t));
  EXPECT_EQ(weakOperatorPool.lock()->usedBytes(), 0);
  retainedResource.reset();

  EXPECT_TRUE(weakOperatorPool.expired());
  EXPECT_TRUE(weakRoot.expired());
  // Isolated test backends, unlike the process backend, are caller-owned.
  owner.reset();
  EXPECT_TRUE(weakOwner.expired());
}

TEST_F(CudfMemoryResourceTest, RejectsAggregateAccountingPool) {
  auto root = memory::memoryManager()->addRootPool();
  auto state = std::make_shared<HostBackedResourceState>();
  EXPECT_THROW(
      (CudfMemoryResource{makeTestUpstream(state), root}), VeloxRuntimeError);
}

TEST_F(CudfMemoryResourceTest, ReservationAndExternalAccountingCoexist) {
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceReservationTest");
  auto state = std::make_shared<HostBackedResourceState>();
  CudfMemoryResource resource{makeTestUpstream(state), pool};

  constexpr std::size_t kBytes = 256;
  ASSERT_TRUE(pool->maybeReserve(kBytes * 2));
  auto* allocation = resource.allocate(
      cuda::stream_ref{cudaStream_t{}}, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), kBytes);
  resource.deallocate(
      cuda::stream_ref{cudaStream_t{}},
      allocation,
      kBytes,
      alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), 0);
  pool->release();
  EXPECT_EQ(pool->reservedBytes(), 0);
}

TEST_F(CudfMemoryResourceTest, RegistryConstructionFailureIsRetryable) {
  CudfMemoryResourceRegistry registry;
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceRegistryRetryTest");
  auto state = std::make_shared<HostBackedResourceState>();
  auto upstream = makeTestUpstream(state);

  state->failNextCopy = true;
  EXPECT_THROW(
      registry.resourcesFor(upstream, upstream, pool), std::runtime_error);

  auto resources = registry.resourcesFor(upstream, upstream, pool);
  EXPECT_EQ(resources.temp, resources.output);
}

TEST_F(CudfMemoryResourceTest, RegistryValidatesStableInputs) {
  auto owner = makeCustomResource();
  CudfMemoryResourceRegistry registry;
  auto root = memory::memoryManager()->addCustomRootPool(
      "cudfMemoryResourceRegistryStableRoot", owner);
  auto pool = root->addLeafChild("operator");
  auto stateA = std::make_shared<HostBackedResourceState>();
  auto stateB = std::make_shared<HostBackedResourceState>();
  auto upstreamA = makeTestUpstream(stateA);
  auto upstreamB = makeTestUpstream(stateB);

  auto resources = registry.resourcesFor(upstreamA, upstreamA, pool);
  EXPECT_EQ(resources.temp, resources.output);
  auto repeated = registry.resourcesFor(upstreamA, upstreamA, pool);
  EXPECT_EQ(resources.temp, repeated.temp);

  EXPECT_THROW(
      registry.resourcesFor(upstreamB, upstreamA, pool), VeloxRuntimeError);
}

TEST_F(CudfMemoryResourceTest, RegistryUsesDistinctOutputWrapperWhenNeeded) {
  CudfMemoryResourceRegistry registry;
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceDistinctWrappersTest");
  auto stateA = std::make_shared<HostBackedResourceState>();
  auto stateB = std::make_shared<HostBackedResourceState>();
  auto resources = registry.resourcesFor(
      makeTestUpstream(stateA), makeTestUpstream(stateB), pool);
  EXPECT_NE(resources.temp, resources.output);
}

TEST_F(CudfMemoryResourceTest, RegistryKeepsReferencedResourceAlive) {
  auto registry = std::make_unique<CudfMemoryResourceRegistry>();
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceRegistryTest");
  std::weak_ptr<memory::MemoryPool> weakPool = pool;
  auto state = std::make_shared<HostBackedResourceState>();
  auto upstream = makeTestUpstream(state);
  auto resources = registry->resourcesFor(upstream, upstream, std::move(pool));

  constexpr std::size_t kBytes = 128;
  auto* allocation = resources.output.allocate(
      cuda::stream_ref{cudaStream_t{}}, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(weakPool.lock()->usedBytes(), kBytes);
  resources.output.deallocate(
      cuda::stream_ref{cudaStream_t{}},
      allocation,
      kBytes,
      alignof(std::max_align_t));

  EXPECT_FALSE(weakPool.expired());
  registry.reset();
  EXPECT_TRUE(weakPool.expired());
}

TEST_F(CudfMemoryResourceTest, ScopedSelectionSupportsNesting) {
  auto stateA = std::make_shared<HostBackedResourceState>();
  auto stateB = std::make_shared<HostBackedResourceState>();
  cuda::mr::any_resource<cuda::mr::device_accessible> resourceA{
      makeTestUpstream(stateA)};
  cuda::mr::any_resource<cuda::mr::device_accessible> resourceB{
      makeTestUpstream(stateB)};
  rmm::device_async_resource_ref refA{resourceA};
  rmm::device_async_resource_ref refB{resourceB};
  auto ownerA = std::make_shared<int>(1);
  auto ownerB = std::make_shared<int>(2);

  ScopedCudfMemoryResources outer{refA, refB, ownerA};
  EXPECT_EQ(get_temp_mr(), refA);
  EXPECT_EQ(get_output_mr(), refB);
  EXPECT_EQ(get_memory_resource_owner(), ownerA);

  {
    ScopedCudfMemoryResources inner{refB, refA};
    EXPECT_EQ(get_temp_mr(), refB);
    EXPECT_EQ(get_output_mr(), refA);
    EXPECT_EQ(get_memory_resource_owner(), ownerA);
  }

  {
    ScopedCudfMemoryResources inner{refB, refA, ownerB};
    EXPECT_EQ(get_memory_resource_owner(), ownerB);
  }

  EXPECT_EQ(get_temp_mr(), refA);
  EXPECT_EQ(get_output_mr(), refB);
  EXPECT_EQ(get_memory_resource_owner(), ownerA);
}

TEST_F(CudfMemoryResourceTest, ScopedOwnerDoesNotLeakPastOutermostScope) {
  EXPECT_EQ(get_memory_resource_owner(), nullptr);
}

TEST_F(
    CudfMemoryResourceTest,
    CurrentResourceRoutesOperationalAllocationsButNotOutputs) {
  constexpr std::size_t kTempBytes = 256;
  constexpr std::size_t kOutputBytes = 512;
  constexpr auto kAlignment = alignof(std::max_align_t);

  auto fallbackState = std::make_shared<HostBackedResourceState>();
  auto fallbackResource = makeTestUpstream(fallbackState);
  rmm::device_async_resource_ref fallbackRef{fallbackResource};
  auto globalTemporaryResource =
      createThreadLocalTemporaryMemoryResource(fallbackResource);
  auto previousResource =
      cudf::set_current_device_resource(globalTemporaryResource);
  SCOPE_EXIT {
    cudf::set_current_device_resource(std::move(previousResource));
  };

  auto root = memory::memoryManager()->addRootPool();
  auto tempPool = root->addLeafChild("tlsTemp");
  auto outputPool = root->addLeafChild("explicitOutput");
  auto tempState = std::make_shared<HostBackedResourceState>();
  auto outputState = std::make_shared<HostBackedResourceState>();
  CudfMemoryResource tempResource{makeTestUpstream(tempState), tempPool};
  CudfMemoryResource outputResource{makeTestUpstream(outputState), outputPool};
  rmm::device_async_resource_ref tempRef{tempResource};
  rmm::device_async_resource_ref outputRef{outputResource};

  auto currentResource = cudf::get_current_device_resource_ref();
  auto* fallbackAllocation =
      currentResource.allocate_sync(kTempBytes, kAlignment);
  EXPECT_EQ(fallbackState->liveBytes, kTempBytes);
  EXPECT_EQ(tempPool->usedBytes(), 0);
  currentResource.deallocate_sync(fallbackAllocation, kTempBytes, kAlignment);
  EXPECT_EQ(fallbackState->liveBytes, 0);

  // This is the path used by operators without a custom GPU pool. Their temp
  // scope must use the underlying configured resource, not the process-wide
  // dispatcher itself, to avoid recursive dispatch.
  {
    ScopedCudfMemoryResources untrackedScope{fallbackRef, outputRef};
    auto* untrackedAllocation = currentResource.allocate(
        cuda::stream_ref{cudaStream_t{}}, kTempBytes, kAlignment);
    EXPECT_EQ(fallbackState->liveBytes, kTempBytes);
    EXPECT_EQ(tempPool->usedBytes(), 0);
    currentResource.deallocate(
        cuda::stream_ref{cudaStream_t{}},
        untrackedAllocation,
        kTempBytes,
        kAlignment);
  }
  EXPECT_EQ(fallbackState->liveBytes, 0);

  void* tempAllocation = nullptr;
  void* outputAllocation = nullptr;
  {
    ScopedCudfMemoryResources scope{tempRef, outputRef};
    tempAllocation = currentResource.allocate(
        cuda::stream_ref{cudaStream_t{}}, kTempBytes, kAlignment);
    EXPECT_EQ(tempPool->usedBytes(), kTempBytes);
    EXPECT_EQ(outputPool->usedBytes(), 0);
    EXPECT_EQ(fallbackState->liveBytes, 0);

    outputAllocation = outputRef.allocate(
        cuda::stream_ref{cudaStream_t{}}, kOutputBytes, kAlignment);
    EXPECT_EQ(tempPool->usedBytes(), kTempBytes);
    EXPECT_EQ(outputPool->usedBytes(), kOutputBytes);
  }

  // The dispatcher retains the selected resource for the pointer, so the
  // deallocation does not depend on the current TLS scope.
  EXPECT_EQ(tempPool->usedBytes(), kTempBytes);
  currentResource.deallocate(
      cuda::stream_ref{cudaStream_t{}}, tempAllocation, kTempBytes, kAlignment);
  EXPECT_EQ(tempPool->usedBytes(), 0);

  // Explicit output resources independently carry their accounting identity
  // and can likewise be deallocated after the operational TLS scope has ended.
  EXPECT_EQ(outputPool->usedBytes(), kOutputBytes);
  outputRef.deallocate(
      cuda::stream_ref{cudaStream_t{}},
      outputAllocation,
      kOutputBytes,
      kAlignment);
  EXPECT_EQ(outputPool->usedBytes(), 0);
}

TEST_F(CudfMemoryResourceTest, DispatcherOwnsWrapperUntilFinalDeallocation) {
  for (const bool async : {false, true}) {
    auto state = std::make_shared<HostBackedResourceState>();
    auto dispatcher = createThreadLocalTemporaryMemoryResource(
        makeTestUpstream(std::make_shared<HostBackedResourceState>()));
    rmm::device_async_resource_ref current{dispatcher};
    // System leaves follow trackDefaultUsage (off in this fixture). Use an
    // explicitly tracked root so byte accounting is actually exercised.
    auto pool =
        memory::memoryManager()->addRootPool()->addLeafChild("dispatcher");
    std::weak_ptr<memory::MemoryPool> weakPool = pool;
    constexpr std::size_t kBytes = 192;
    constexpr auto kAlignment = alignof(std::max_align_t);
    void* pointer = nullptr;
    {
      CudfMemoryResource wrapper{makeTestUpstream(state), pool};
      ScopedCudfMemoryResources scope{wrapper, wrapper};
      pointer = async
          ? current.allocate(
                cuda::stream_ref{cudaStream_t{}}, kBytes, kAlignment)
          : current.allocate_sync(kBytes, kAlignment);
      pool.reset();
    }
    // Neither a QueryCtx registry nor the original wrapper remains alive.
    ASSERT_FALSE(weakPool.expired());
    EXPECT_EQ(weakPool.lock()->usedBytes(), kBytes);
    std::thread freeOnAnotherThread([&]() {
      if (async) {
        current.deallocate(
            cuda::stream_ref{cudaStream_t{}}, pointer, kBytes, kAlignment);
      } else {
        current.deallocate_sync(pointer, kBytes, kAlignment);
      }
    });
    freeOnAnotherThread.join();
    EXPECT_TRUE(weakPool.expired());
    EXPECT_EQ(state->liveBytes, 0);
  }
}

TEST_F(CudfMemoryResourceTest, CurrentResourceUsesCallingThreadScope) {
  constexpr std::size_t kBytes = 384;
  constexpr auto kAlignment = alignof(std::max_align_t);

  auto fallbackState = std::make_shared<HostBackedResourceState>();
  auto globalTemporaryResource =
      createThreadLocalTemporaryMemoryResource(makeTestUpstream(fallbackState));
  rmm::device_async_resource_ref currentResource{globalTemporaryResource};

  auto root = memory::memoryManager()->addRootPool();
  auto mainPool = root->addLeafChild("mainThreadTemp");
  auto childPool = root->addLeafChild("childThreadTemp");
  auto mainState = std::make_shared<HostBackedResourceState>();
  auto childState = std::make_shared<HostBackedResourceState>();
  CudfMemoryResource mainResource{makeTestUpstream(mainState), mainPool};
  CudfMemoryResource childResource{makeTestUpstream(childState), childPool};
  rmm::device_async_resource_ref mainRef{mainResource};
  rmm::device_async_resource_ref childRef{childResource};

  std::promise<void*> childAllocation;
  auto childAllocationFuture = childAllocation.get_future();
  std::thread child([&] {
    ScopedCudfMemoryResources childScope{childRef, childRef};
    auto* allocation = currentResource.allocate(
        cuda::stream_ref{cudaStream_t{}}, kBytes, kAlignment);
    childAllocation.set_value(allocation);
  });

  auto* allocationFromChild = childAllocationFuture.get();
  child.join();
  {
    ScopedCudfMemoryResources mainScope{mainRef, mainRef};
    auto* allocation = currentResource.allocate(
        cuda::stream_ref{cudaStream_t{}}, kBytes, kAlignment);
    EXPECT_EQ(mainPool->usedBytes(), kBytes);
    EXPECT_EQ(childPool->usedBytes(), kBytes);
    EXPECT_EQ(fallbackState->liveBytes, 0);
    currentResource.deallocate(
        cuda::stream_ref{cudaStream_t{}}, allocation, kBytes, kAlignment);
  }

  EXPECT_EQ(mainPool->usedBytes(), 0);
  EXPECT_EQ(childPool->usedBytes(), kBytes);
  currentResource.deallocate(
      cuda::stream_ref{cudaStream_t{}},
      allocationFromChild,
      kBytes,
      kAlignment);
  EXPECT_EQ(childPool->usedBytes(), 0);
  EXPECT_EQ(fallbackState->liveBytes, 0);
}

TEST_F(CudfMemoryResourceTest, ScopedSelectionIsThreadLocal) {
  auto stateA = std::make_shared<HostBackedResourceState>();
  auto stateB = std::make_shared<HostBackedResourceState>();
  cuda::mr::any_resource<cuda::mr::device_accessible> resourceA{
      makeTestUpstream(stateA)};
  cuda::mr::any_resource<cuda::mr::device_accessible> resourceB{
      makeTestUpstream(stateB)};
  rmm::device_async_resource_ref refA{resourceA};
  rmm::device_async_resource_ref refB{resourceB};

  std::promise<void> childReady;
  std::promise<void> releaseChild;
  auto releaseFuture = releaseChild.get_future().share();
  std::atomic<bool> childSawOwnResources{false};

  ScopedCudfMemoryResources mainScope{refA, refA};
  std::thread child([&] {
    ScopedCudfMemoryResources childScope{refB, refB};
    childSawOwnResources = get_temp_mr() == refB && get_output_mr() == refB;
    childReady.set_value();
    releaseFuture.wait();
    childSawOwnResources = childSawOwnResources && get_temp_mr() == refB &&
        get_output_mr() == refB;
  });

  childReady.get_future().wait();
  EXPECT_EQ(get_temp_mr(), refA);
  EXPECT_EQ(get_output_mr(), refA);
  releaseChild.set_value();
  child.join();
  EXPECT_TRUE(childSawOwnResources);
}

} // namespace
} // namespace facebook::velox::cudf_velox
