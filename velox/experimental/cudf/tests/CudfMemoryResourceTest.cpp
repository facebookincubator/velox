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
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/core/QueryCtx.h"

#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

struct HostBackedResourceState {
  std::atomic<bool> failNextAllocation{false};
  std::atomic<bool> failNextCopy{false};
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
    return ::operator new(bytes, std::align_val_t(alignment));
  }

  void deallocate_sync(
      void* pointer,
      std::size_t /*bytes*/,
      std::size_t alignment) noexcept {
    ::operator delete(pointer, std::align_val_t(alignment));
  }

  void* allocate(
      cuda::stream_ref /*stream*/,
      std::size_t bytes,
      std::size_t alignment) {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref /*stream*/,
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
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

std::shared_ptr<core::QueryCtx> makeTrackedQueryCtx(
    const std::shared_ptr<memory::CustomMemoryResource>& resource,
    const std::string& queryId,
    const std::string& rootName = "") {
  auto root = memory::memoryManager()->addCustomRootPool(
      (rootName.empty() ? queryId : rootName) + ".gpu", resource);
  auto queryCtx = core::QueryCtx::Builder()
                      .queryId(queryId)
                      .customPool(std::string{kCudfMemoryResourceTag}, root)
                      .build();
  auto registry = memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
  registry->insert(std::string{kCudfMemoryResourceTag}, resource);
  queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
      memory::kCustomMemoryResourceRegistryKey, registry);
  return queryCtx;
}

std::unique_ptr<cudf::table> makeDeviceTable(rmm::cuda_stream_view stream) {
  std::array<int32_t, 4> values{{1, 2, 3, 4}};
  rmm::device_buffer data(
      values.size() * sizeof(int32_t),
      stream,
      cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      data.data(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(values.size()),
          std::move(data),
          rmm::device_buffer{},
          0));
  return std::make_unique<cudf::table>(std::move(columns));
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
  void* allocation = nullptr;
  std::thread allocateThread([&] {
    allocation = ownedResource.allocate(
        rmm::cuda_stream_default, kBytes, alignof(std::max_align_t));
  });
  allocateThread.join();

  EXPECT_NE(allocation, nullptr);
  EXPECT_EQ(pool->usedBytes(), kBytes);

  std::thread deallocateThread([&] {
    ownedResource.deallocate(
        rmm::cuda_stream_default,
        allocation,
        kBytes,
        alignof(std::max_align_t));
  });
  deallocateThread.join();
  EXPECT_EQ(pool->usedBytes(), 0);

  auto* synchronousAllocation =
      ownedResource.allocate_sync(kBytes, alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), kBytes);
  ownedResource.deallocate_sync(
      synchronousAllocation, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), 0);

  state->failNextAllocation = true;
  EXPECT_THROW(
      ownedResource.allocate(
          rmm::cuda_stream_default, kBytes, alignof(std::max_align_t)),
      std::bad_alloc);
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(CudfMemoryResourceTest, RetainsCompleteCustomPoolHierarchyAndOwner) {
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
    CudfMemoryResource reportingResource{
        makeTestUpstream(state), operatorPool, owner};
    retainedResource.emplace(reportingResource);
  }

  constexpr std::size_t kBytes = 192;
  auto* allocation = retainedResource->allocate(
      rmm::cuda_stream_default, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(operatorPool->usedBytes(), kBytes);

  operatorPool.reset();
  nodePool.reset();
  taskPool.reset();
  root.reset();
  owner.reset();

  EXPECT_FALSE(weakOwner.expired());
  EXPECT_FALSE(weakRoot.expired());
  EXPECT_FALSE(weakOperatorPool.expired());

  retainedResource->deallocate(
      rmm::cuda_stream_default, allocation, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(weakOperatorPool.lock()->usedBytes(), 0);
  retainedResource.reset();

  EXPECT_TRUE(weakOperatorPool.expired());
  EXPECT_TRUE(weakRoot.expired());
  EXPECT_TRUE(weakOwner.expired());
}

TEST_F(CudfMemoryResourceTest, RejectsAggregateAccountingPool) {
  auto root = memory::memoryManager()->addRootPool();
  auto state = std::make_shared<HostBackedResourceState>();
  EXPECT_THROW(
      (CudfMemoryResource{makeTestUpstream(state), root}), VeloxRuntimeError);
}

TEST_F(CudfMemoryResourceTest, RejectsNonThreadSafeAccountingPool) {
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("nonThreadSafe", false);
  auto state = std::make_shared<HostBackedResourceState>();
  EXPECT_THROW(
      (CudfMemoryResource{makeTestUpstream(state), pool}), VeloxRuntimeError);
}

TEST_F(CudfMemoryResourceTest, ReservationAndExternalAccountingCoexist) {
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceReservationTest");
  auto state = std::make_shared<HostBackedResourceState>();
  CudfMemoryResource resource{makeTestUpstream(state), pool};

  constexpr std::size_t kBytes = 256;
  ASSERT_TRUE(pool->maybeReserve(kBytes * 2));
  auto* allocation = resource.allocate(
      rmm::cuda_stream_default, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(pool->usedBytes(), kBytes);
  resource.deallocate(
      rmm::cuda_stream_default, allocation, kBytes, alignof(std::max_align_t));
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
      registry.resourcesFor(upstream, upstream, pool, nullptr),
      std::runtime_error);

  auto resources = registry.resourcesFor(upstream, upstream, pool, nullptr);
  EXPECT_EQ(resources.temp, resources.output);
}

TEST_F(CudfMemoryResourceTest, RegistryValidatesStableInputs) {
  CudfMemoryResourceRegistry registry;
  auto owner = makeCustomResource();
  auto otherOwner = makeCustomResource();
  auto root = memory::memoryManager()->addCustomRootPool(
      "cudfMemoryResourceRegistryStableRoot", owner);
  auto pool = root->addLeafChild("operator");
  auto stateA = std::make_shared<HostBackedResourceState>();
  auto stateB = std::make_shared<HostBackedResourceState>();
  auto upstreamA = makeTestUpstream(stateA);
  auto upstreamB = makeTestUpstream(stateB);

  auto resources = registry.resourcesFor(upstreamA, upstreamA, pool, owner);
  EXPECT_EQ(resources.temp, resources.output);
  auto repeated = registry.resourcesFor(upstreamA, upstreamA, pool, owner);
  EXPECT_EQ(resources.temp, repeated.temp);

  EXPECT_THROW(
      registry.resourcesFor(upstreamB, upstreamA, pool, owner),
      VeloxRuntimeError);
  EXPECT_THROW(
      registry.resourcesFor(upstreamA, upstreamA, pool, otherOwner),
      VeloxRuntimeError);
}

TEST_F(CudfMemoryResourceTest, RegistryUsesDistinctOutputWrapperWhenNeeded) {
  CudfMemoryResourceRegistry registry;
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceDistinctWrappersTest");
  auto stateA = std::make_shared<HostBackedResourceState>();
  auto stateB = std::make_shared<HostBackedResourceState>();
  auto resources = registry.resourcesFor(
      makeTestUpstream(stateA), makeTestUpstream(stateB), pool, nullptr);
  EXPECT_NE(resources.temp, resources.output);
}

TEST_F(CudfMemoryResourceTest, RegistryKeepsReferencedResourceAlive) {
  auto registry = std::make_unique<CudfMemoryResourceRegistry>();
  auto root = memory::memoryManager()->addRootPool();
  auto pool = root->addLeafChild("cudfMemoryResourceRegistryTest");
  std::weak_ptr<memory::MemoryPool> weakPool = pool;
  auto state = std::make_shared<HostBackedResourceState>();
  auto upstream = makeTestUpstream(state);
  auto resources =
      registry->resourcesFor(upstream, upstream, std::move(pool), nullptr);

  constexpr std::size_t kBytes = 128;
  auto* allocation = resources.output.allocate(
      rmm::cuda_stream_default, kBytes, alignof(std::max_align_t));
  EXPECT_EQ(weakPool.lock()->usedBytes(), kBytes);
  resources.output.deallocate(
      rmm::cuda_stream_default, allocation, kBytes, alignof(std::max_align_t));

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

  ScopedCudfMemoryResources outer{refA, refB};
  EXPECT_EQ(get_temp_mr(), refA);
  EXPECT_EQ(get_output_mr(), refB);

  {
    ScopedCudfMemoryResources inner{refB, refA};
    EXPECT_EQ(get_temp_mr(), refB);
    EXPECT_EQ(get_output_mr(), refA);
  }

  EXPECT_EQ(get_temp_mr(), refA);
  EXPECT_EQ(get_output_mr(), refB);
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

TEST_F(CudfMemoryResourceTest, ExchangeResourceRequiresTrackedQuery) {
  auto queryCtx = core::QueryCtx::Builder().queryId("untracked").build();
  EXPECT_FALSE(cudfExchangeOutputMemoryResource(*queryCtx).has_value());
  EXPECT_EQ(cudfExchangeMemoryPool(*queryCtx), nullptr);
}

TEST_F(CudfMemoryResourceTest, ExchangeResourcesAreScopedByQueryRoot) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  resetCudfExchangeMemoryResource();
  auto savedOutputMr = output_mr_;
  output_mr_ = createMemoryResource("cuda", 0);
  SCOPE_EXIT {
    resetCudfExchangeMemoryResource();
    output_mr_ = std::move(savedOutputMr);
  };

  auto ownerA = makeCustomResource();
  auto ownerB = makeCustomResource();
  auto queryCtxA = makeTrackedQueryCtx(
      ownerA, "same-exchange-query", "exchange-query-root-a");
  auto queryCtxB = makeTrackedQueryCtx(
      ownerB, "same-exchange-query", "exchange-query-root-b");

  auto exchangeMrA = cudfExchangeOutputMemoryResource(*queryCtxA);
  auto exchangeMrB = cudfExchangeOutputMemoryResource(*queryCtxB);
  ASSERT_TRUE(exchangeMrA.has_value());
  ASSERT_TRUE(exchangeMrB.has_value());
  EXPECT_NE(*exchangeMrA, *exchangeMrB);

  auto exchangePoolA = cudfExchangeMemoryPool(*queryCtxA);
  auto exchangePoolB = cudfExchangeMemoryPool(*queryCtxB);
  ASSERT_NE(exchangePoolA, nullptr);
  ASSERT_NE(exchangePoolB, nullptr);
  EXPECT_NE(exchangePoolA, exchangePoolB);

  int originalDevice = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&originalDevice));
  SCOPE_EXIT {
    cudaSetDevice(originalDevice);
  };
  if (deviceCount > 1) {
    const auto otherDevice = (originalDevice + 1) % deviceCount;
    CUDF_CUDA_TRY(cudaSetDevice(otherDevice));
    auto exchangeMrOtherDevice = cudfExchangeOutputMemoryResource(*queryCtxA);
    ASSERT_TRUE(exchangeMrOtherDevice.has_value());
    EXPECT_NE(*exchangeMrA, *exchangeMrOtherDevice);
    auto exchangePoolOtherDevice = cudfExchangeMemoryPool(*queryCtxA);
    ASSERT_NE(exchangePoolOtherDevice, nullptr);
    EXPECT_NE(exchangePoolA, exchangePoolOtherDevice);
    CUDF_CUDA_TRY(cudaSetDevice(originalDevice));
  }

  EXPECT_THROW(resetCudfExchangeMemoryResource(), VeloxRuntimeError);
  EXPECT_FALSE(tryResetCudfExchangeMemoryResource());

  queryCtxA.reset();
  queryCtxB.reset();
  exchangePoolA.reset();
  exchangePoolB.reset();
  EXPECT_TRUE(tryResetCudfExchangeMemoryResource());
}

TEST_F(CudfMemoryResourceTest, PackedGpuMemoryIsChargedToExchangePool) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  resetCudfExchangeMemoryResource();
  auto savedOutputMr = output_mr_;
  output_mr_ = createMemoryResource("cuda", 0);
  SCOPE_EXIT {
    resetCudfExchangeMemoryResource();
    output_mr_ = std::move(savedOutputMr);
  };

  auto owner = makeCustomResource();
  std::weak_ptr<memory::CustomMemoryResource> weakOwner = owner;
  auto queryCtx = makeTrackedQueryCtx(owner, "exchange-accounting");
  std::weak_ptr<memory::MemoryPool> weakQueryPool =
      queryCtx->customPool(std::string{kCudfMemoryResourceTag});
  auto exchangeMr = cudfExchangeOutputMemoryResource(*queryCtx);
  ASSERT_TRUE(exchangeMr.has_value());
  auto exchangePool = cudfExchangeMemoryPool(*queryCtx);
  ASSERT_NE(exchangePool, nullptr);

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  auto table = makeDeviceTable(stream);
  auto packed = cudf::pack(table->view(), stream, *exchangeMr);
  stream.synchronize();

  ASSERT_NE(packed.gpu_data, nullptr);
  const auto packedBytes = packed.gpu_data->size();
  ASSERT_GT(packedBytes, 0);
  EXPECT_EQ(exchangePool->usedBytes(), packedBytes);
  EXPECT_EQ(weakQueryPool.lock()->usedBytes(), packedBytes);

  queryCtx.reset();
  owner.reset();
  EXPECT_FALSE(weakQueryPool.expired());
  EXPECT_FALSE(weakOwner.expired());
  EXPECT_EQ(exchangePool->usedBytes(), packedBytes);
  EXPECT_THROW(resetCudfExchangeMemoryResource(), VeloxRuntimeError);
  EXPECT_FALSE(tryResetCudfExchangeMemoryResource());

  packed.gpu_data.reset();
  EXPECT_EQ(exchangePool->usedBytes(), 0);
  EXPECT_EQ(weakQueryPool.lock()->usedBytes(), 0);
  exchangePool.reset();
  EXPECT_TRUE(tryResetCudfExchangeMemoryResource());
  EXPECT_TRUE(weakQueryPool.expired());
  EXPECT_TRUE(weakOwner.expired());
}

TEST_F(CudfMemoryResourceTest, UnregisterRetainsBusyExchangeResourceSafely) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  ASSERT_FALSE(cudfIsRegistered());
  resetCudfExchangeMemoryResource();
  auto savedTempMr = mr_;
  auto savedOutputMr = output_mr_;
  mr_ = createMemoryResource("cuda", 0);
  output_mr_ = createMemoryResource("cuda", 0);
  SCOPE_EXIT {
    tryResetCudfExchangeMemoryResource();
    mr_ = std::move(savedTempMr);
    output_mr_ = std::move(savedOutputMr);
  };

  auto owner = makeCustomResource();
  auto queryCtx = makeTrackedQueryCtx(owner, "busy-unregister");
  auto exchangeMr = cudfExchangeOutputMemoryResource(*queryCtx);
  ASSERT_TRUE(exchangeMr.has_value());
  auto exchangePool = cudfExchangeMemoryPool(*queryCtx);
  ASSERT_NE(exchangePool, nullptr);

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  auto table = makeDeviceTable(stream);
  auto packed = cudf::pack(table->view(), stream, *exchangeMr);
  stream.synchronize();
  ASSERT_NE(packed.gpu_data, nullptr);
  ASSERT_GT(packed.gpu_data->size(), 0);

  queryCtx.reset();
  owner.reset();
  EXPECT_FALSE(tryResetCudfExchangeMemoryResource());
  EXPECT_NO_THROW(unregisterCudf());
  EXPECT_FALSE(cudfIsRegistered());
  EXPECT_FALSE(mr_.has_value());
  EXPECT_FALSE(output_mr_.has_value());
  EXPECT_GT(exchangePool->usedBytes(), 0);

  packed.gpu_data.reset();
  EXPECT_EQ(exchangePool->usedBytes(), 0);
  exchangePool.reset();
  EXPECT_TRUE(tryResetCudfExchangeMemoryResource());
}

} // namespace
} // namespace facebook::velox::cudf_velox
