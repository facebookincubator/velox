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

#include "velox/experimental/cudf/exec/GpuMemoryTracker.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic pop

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::test {
namespace {

struct RecordingResourceState {
  std::mutex mutex;
  std::vector<std::size_t> allocationAlignments;
  std::vector<std::size_t> deallocationAlignments;
  bool throwOnAllocation{false};
};

class RecordingResource {
 public:
  explicit RecordingResource(std::shared_ptr<RecordingResourceState> state)
      : state_(std::move(state)) {}

  void* allocate(
      cuda::stream_ref /*stream*/,
      std::size_t bytes,
      std::size_t alignment) {
    return allocateImpl(bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref /*stream*/,
      void* address,
      std::size_t /*bytes*/,
      std::size_t alignment) noexcept {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->deallocationAlignments.push_back(alignment);
    }
    std::free(address);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return allocateImpl(bytes, alignment);
  }

  void deallocate_sync(
      void* address,
      std::size_t /*bytes*/,
      std::size_t alignment) noexcept {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->deallocationAlignments.push_back(alignment);
    }
    std::free(address);
  }

  bool operator==(const RecordingResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const RecordingResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const RecordingResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  void* allocateImpl(std::size_t bytes, std::size_t alignment) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->allocationAlignments.push_back(alignment);
      if (state_->throwOnAllocation) {
        throw rmm::out_of_memory("deterministic test allocation failure");
      }
    }

    const auto allocationSize =
        std::max(alignment, ((bytes + alignment - 1) / alignment) * alignment);
    auto* address = std::aligned_alloc(alignment, allocationSize);
    if (address == nullptr) {
      throw rmm::out_of_memory("host allocation failed in test resource");
    }
    return address;
  }

  std::shared_ptr<RecordingResourceState> state_;
};

static_assert(
    cuda::mr::resource_with<RecordingResource, cuda::mr::device_accessible>);

struct AddressReusingResourceState {
  std::mutex mutex;
  std::condition_variable condition;
  bool allocated{false};
  bool blockNextDeallocation{true};
  bool addressFreed{false};
  bool releaseDeallocation{false};
  alignas(256) std::array<std::byte, 256> storage;
};

class AddressReusingResource {
 public:
  explicit AddressReusingResource(
      std::shared_ptr<AddressReusingResourceState> state)
      : state_(std::move(state)) {}

  void* allocate(
      cuda::stream_ref /*stream*/,
      std::size_t /*bytes*/,
      std::size_t /*alignment*/) {
    return allocateImpl();
  }

  void deallocate(
      cuda::stream_ref /*stream*/,
      void* address,
      std::size_t /*bytes*/,
      std::size_t /*alignment*/) noexcept {
    deallocateImpl(address);
  }

  void* allocate_sync(std::size_t /*bytes*/, std::size_t /*alignment*/) {
    return allocateImpl();
  }

  void deallocate_sync(
      void* address,
      std::size_t /*bytes*/,
      std::size_t /*alignment*/) noexcept {
    deallocateImpl(address);
  }

  bool operator==(const AddressReusingResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const AddressReusingResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const AddressReusingResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  void* allocateImpl() {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->condition.wait(lock, [&] { return !state_->allocated; });
    state_->allocated = true;
    return state_->storage.data();
  }

  void deallocateImpl(void* address) noexcept {
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (address != state_->storage.data()) {
      std::terminate();
    }
    state_->allocated = false;
    if (state_->blockNextDeallocation) {
      state_->blockNextDeallocation = false;
      state_->addressFreed = true;
      state_->condition.notify_all();
      state_->condition.wait(lock, [&] { return state_->releaseDeallocation; });
    }
    state_->condition.notify_all();
  }

  std::shared_ptr<AddressReusingResourceState> state_;
};

static_assert(cuda::mr::resource_with<
              AddressReusingResource,
              cuda::mr::device_accessible>);

GpuMemoryOwner makeOwner(
    std::string taskSuffix,
    std::string planNodeId,
    int32_t pipelineId,
    int32_t driverId,
    int32_t operatorId) {
  return GpuMemoryOwner{
      .taskUuid = "task-" + taskSuffix + "-uuid",
      .taskId = "task-" + taskSuffix,
      .queryId = "query-" + taskSuffix,
      .planNodeId = std::move(planNodeId),
      .planNodeType = "TestPlanNode",
      .pipelineId = pipelineId,
      .driverId = driverId,
      .operatorId = operatorId,
      .operatorType = "TestOperator"};
}

const GpuMemoryOwnerSnapshot* findOwner(
    const GpuMemorySnapshot& snapshot,
    GpuMemoryOwnerHandle handle) {
  const auto it = std::find_if(
      snapshot.owners.begin(), snapshot.owners.end(), [&](const auto& owner) {
        return owner.handle == handle;
      });
  return it == snapshot.owners.end() ? nullptr : &*it;
}

/// Holds once activity quiesces. The allocation path takes no shared lock, so
/// it does not hold instant by instant while allocations are in flight.
bool isCoherent(const GpuMemorySnapshot& snapshot) {
  uint64_t ownerBytes{0};
  for (const auto& owner : snapshot.owners) {
    ownerBytes += owner.currentBytes;
  }
  return snapshot.currentBytes == ownerBytes;
}

} // namespace

TEST(GpuMemoryTrackerTest, TracksAllocationOriginAndOrderedTransitions) {
  GpuMemoryLedger tracker;
  const auto ownerA = makeOwner("shared", "plan-a", 1, 7, 2);
  auto ownerB = ownerA;
  ownerB.driverId = 8;

  const auto handleA = tracker.registerOwner(ownerA);
  const auto duplicateHandleA = tracker.registerOwner(ownerA);
  const auto handleB = tracker.registerOwner(ownerB);
  EXPECT_EQ(duplicateHandleA, handleA);
  EXPECT_NE(handleB.ownerId, handleA.ownerId);
  auto relabeledOwnerA = ownerA;
  relabeledOwnerA.planNodeType = "RelabeledPlanNode";
  EXPECT_EQ(tracker.registerOwner(relabeledOwnerA), handleA);
  EXPECT_EQ(handleB.planNodeKey, handleA.planNodeKey);

  auto otherTaskOwner = ownerA;
  otherTaskOwner.taskUuid = "other-task-uuid";
  otherTaskOwner.taskId = "other-task";
  const auto otherTaskHandle = tracker.registerOwner(otherTaskOwner);
  EXPECT_NE(otherTaskHandle.planNodeKey, handleA.planNodeKey);

  int allocationA;
  int allocationB;
  const auto first = tracker.recordAllocation(&allocationA, 100, handleA);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->ownerId, handleA.ownerId);
  EXPECT_EQ(first->planNodeKey, handleA.planNodeKey);
  EXPECT_EQ(first->globalCurrentBytes, 100);
  EXPECT_EQ(first->globalPeakBytes, 100);
  EXPECT_EQ(first->queryCurrentBytes, 100);
  EXPECT_EQ(first->taskCurrentBytes, 100);
  EXPECT_EQ(first->planNodeCurrentBytes, 100);
  EXPECT_EQ(first->ownerCurrentBytes, 100);

  const auto second = tracker.recordAllocation(&allocationB, 60, handleB);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->globalCurrentBytes, 160);
  EXPECT_EQ(second->globalPeakBytes, 160);
  EXPECT_EQ(second->queryCurrentBytes, 160);
  EXPECT_EQ(second->taskCurrentBytes, 160);
  EXPECT_EQ(second->planNodeCurrentBytes, 160);
  EXPECT_EQ(second->ownerCurrentBytes, 60);

  std::optional<GpuMemoryUpdate> deallocation;
  std::thread orphanDeallocator(
      [&] { deallocation = tracker.recordDeallocation(&allocationA, 100); });
  orphanDeallocator.join();

  ASSERT_TRUE(deallocation.has_value());
  EXPECT_EQ(deallocation->ownerId, handleA.ownerId);
  EXPECT_EQ(deallocation->planNodeKey, handleA.planNodeKey);
  EXPECT_EQ(deallocation->globalCurrentBytes, 60);
  EXPECT_EQ(deallocation->globalPeakBytes, 160);
  EXPECT_EQ(deallocation->queryCurrentBytes, 60);
  EXPECT_EQ(deallocation->taskCurrentBytes, 60);
  EXPECT_EQ(deallocation->planNodeCurrentBytes, 60);
  EXPECT_EQ(deallocation->ownerCurrentBytes, 0);

  const auto snapshot = tracker.snapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 60);
  EXPECT_EQ(snapshot.peakBytes, 160);
  EXPECT_EQ(snapshot.totalBytes, 160);
  EXPECT_EQ(snapshot.dataLossEvents, 0);

  const auto* ownerASnapshot = findOwner(snapshot, handleA);
  ASSERT_NE(ownerASnapshot, nullptr);
  EXPECT_EQ(ownerASnapshot->currentBytes, 0);
  EXPECT_EQ(ownerASnapshot->peakBytes, 100);
  EXPECT_EQ(ownerASnapshot->totalBytes, 100);

  const auto* ownerBSnapshot = findOwner(snapshot, handleB);
  ASSERT_NE(ownerBSnapshot, nullptr);
  EXPECT_EQ(ownerBSnapshot->currentBytes, 60);
  EXPECT_EQ(ownerBSnapshot->peakBytes, 60);
  EXPECT_EQ(ownerBSnapshot->totalBytes, 60);

  const auto final = tracker.recordDeallocation(&allocationB, 60);
  ASSERT_TRUE(final.has_value());
  EXPECT_EQ(final->globalCurrentBytes, 0);
  EXPECT_EQ(final->queryCurrentBytes, 0);
  EXPECT_EQ(final->taskCurrentBytes, 0);
  EXPECT_EQ(final->planNodeCurrentBytes, 0);
  EXPECT_EQ(final->ownerCurrentBytes, 0);
}

/// Why no retirement list is needed: rmm::device_buffer stores an owning
/// any_resource, so the wrapper, ledger and upstream all outlive unregistration
/// for as long as something holds memory from them. Freeing through the buffer
/// after the reset is what faults if that ever stops being true.
TEST(GpuMemoryTrackerTest, BufferOutlivesTheResourceThatIsReset) {
  auto state = std::make_shared<RecordingResourceState>();
  std::optional<rmm::device_buffer> buffer;
  {
    auto resources = createGpuMemoryTrackingResources(
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            RecordingResource{state}},
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            RecordingResource{state}});
    buffer.emplace(256, rmm::cuda_stream_default, resources.main);
    EXPECT_EQ(getGpuMemorySnapshot().currentBytes, 256);

    resetGpuMemoryTracking();
    // Both wrappers go out of scope here while the buffer still holds memory.
  }

  EXPECT_EQ(getGpuMemorySnapshot().currentBytes, 0)
      << "the reset ledger is no longer the process-wide one";
  // Frees through the resource the buffer still owns.
  buffer.reset();

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->allocationAlignments.size(), 1);
  EXPECT_EQ(state->deallocationAlignments.size(), 1)
      << "the upstream must still have been reachable to free through";
}

/// Velox permits an empty query identifier, and two unrelated queries would
/// otherwise share one set of query counters whose total means neither.
TEST(GpuMemoryTrackerTest, TasksWithNoQueryIdKeepSeparateQueryCounters) {
  GpuMemoryLedger ledger;
  auto ownerA = makeOwner("a", "plan-a", 1, 0, 2);
  ownerA.queryId = "";
  auto ownerB = makeOwner("b", "plan-b", 1, 0, 2);
  ownerB.queryId = "";

  const auto handleA = ledger.registerOwner(ownerA);
  const auto handleB = ledger.registerOwner(ownerB);

  int allocationA;
  int allocationB;
  const auto first = ledger.recordAllocation(&allocationA, 100, handleA);
  ASSERT_TRUE(first.has_value());
  const auto second = ledger.recordAllocation(&allocationB, 60, handleB);
  ASSERT_TRUE(second.has_value());

  EXPECT_EQ(first->queryCurrentBytes, 100);
  EXPECT_EQ(second->queryCurrentBytes, 60)
      << "the second task must not accumulate onto the first task's query";
}

TEST(GpuMemoryTrackerTest, TracksQueryAndTaskAggregates) {
  GpuMemoryLedger tracker;
  const auto ownerA = makeOwner("shared", "plan-a", 1, 7, 2);
  auto ownerB = ownerA;
  ownerB.taskUuid = "task-other-uuid";
  ownerB.taskId = "task-other";
  ownerB.planNodeId = "plan-b";

  const auto handleA = tracker.registerOwner(ownerA);
  const auto handleB = tracker.registerOwner(ownerB);
  int allocationA;
  int allocationB;

  const auto first = tracker.recordAllocation(&allocationA, 100, handleA);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->queryCurrentBytes, 100);
  EXPECT_EQ(first->taskCurrentBytes, 100);

  const auto second = tracker.recordAllocation(&allocationB, 60, handleB);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->queryCurrentBytes, 160);
  EXPECT_EQ(second->taskCurrentBytes, 60);
  EXPECT_EQ(second->planNodeCurrentBytes, 60);

  const auto firstDeallocation = tracker.recordDeallocation(&allocationA, 100);
  ASSERT_TRUE(firstDeallocation.has_value());
  EXPECT_EQ(firstDeallocation->queryCurrentBytes, 60);
  EXPECT_EQ(firstDeallocation->taskCurrentBytes, 0);

  const auto secondDeallocation = tracker.recordDeallocation(&allocationB, 60);
  ASSERT_TRUE(secondDeallocation.has_value());
  EXPECT_EQ(secondDeallocation->queryCurrentBytes, 0);
  EXPECT_EQ(secondDeallocation->taskCurrentBytes, 0);
}

/// Every counter must be exact as its own sequence of transitions: once
/// activity quiesces, live bytes return to zero and totals match what was
/// allocated.
TEST(GpuMemoryTrackerTest, ConcurrentAllocationsReconcileAtQuiescence) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 2'000;
  constexpr uint64_t kBytes = 64;

  GpuMemoryLedger ledger;
  std::vector<GpuMemoryOwnerHandle> handles;
  for (int driver = 0; driver < kThreads; ++driver) {
    handles.push_back(
        ledger.registerOwner(makeOwner("concurrent", "plan-a", 1, driver, 2)));
  }

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      std::vector<std::unique_ptr<char[]>> storage;
      storage.reserve(kPerThread);
      for (int i = 0; i < kPerThread; ++i) {
        storage.push_back(std::make_unique<char[]>(kBytes));
        EXPECT_TRUE(
            ledger
                .recordAllocation(storage.back().get(), kBytes, handles[thread])
                .has_value());
      }
      for (auto& allocation : storage) {
        EXPECT_TRUE(
            ledger.recordDeallocation(allocation.get(), kBytes).has_value());
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  const auto snapshot = ledger.snapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.dataLossEvents, 0);
  EXPECT_EQ(snapshot.totalBytes, kThreads * kPerThread * kBytes);
  EXPECT_LE(snapshot.peakBytes, snapshot.totalBytes);
  EXPECT_GT(snapshot.peakBytes, 0);

  for (const auto& handle : handles) {
    const auto* owner = findOwner(snapshot, handle);
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(owner->currentBytes, 0);
    EXPECT_EQ(owner->totalBytes, kPerThread * kBytes);
  }
}

TEST(GpuMemoryTrackerTest, DeallocationSizeMismatchIsReportedAsDataLoss) {
  GpuMemoryLedger ledger;
  const auto handle =
      ledger.registerOwner(makeOwner("mismatch", "plan-a", 1, 2, 3));

  int allocation;
  ASSERT_TRUE(ledger.recordAllocation(&allocation, 1'024, handle).has_value());
  ASSERT_TRUE(ledger.recordDeallocation(&allocation, 2'048).has_value());

  const auto snapshot = ledger.snapshot();
  EXPECT_EQ(snapshot.dataLossEvents, 1);
  // The recorded size is used for the debit, so the ledger still balances.
  EXPECT_EQ(snapshot.currentBytes, 0);
}

TEST(GpuMemoryTrackerTest, InvalidPointerEventsDoNotCorruptLedger) {
  GpuMemoryLedger tracker;
  const auto handle =
      tracker.registerOwner(makeOwner("invalid", "plan-a", 1, 2, 3));

  int allocation;
  ASSERT_TRUE(tracker.recordAllocation(&allocation, 64, handle).has_value());
  EXPECT_FALSE(tracker.recordAllocation(&allocation, 128, handle).has_value());

  int unknownAllocation;
  EXPECT_FALSE(tracker.recordDeallocation(&unknownAllocation, 32).has_value());

  auto snapshot = tracker.snapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 64);
  EXPECT_EQ(snapshot.dataLossEvents, 2);

  ASSERT_TRUE(tracker.recordDeallocation(&allocation, 64).has_value());
  snapshot = tracker.snapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.dataLossEvents, 2);
}

TEST(GpuMemoryTrackerTest, TrackedResourcesShareCombinedSnapshot) {
  auto mainState = std::make_shared<RecordingResourceState>();
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{mainState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto mainResource = rmm::device_async_resource_ref{resources.main};
  auto outputResource = rmm::device_async_resource_ref{resources.output};
  auto* mainAddress = mainResource.allocate(rmm::cuda_stream_default, 256, 256);
  auto* outputAddress =
      outputResource.allocate(rmm::cuda_stream_default, 512, 256);

  auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 768);
  EXPECT_EQ(snapshot.peakBytes, 768);
  EXPECT_EQ(snapshot.totalBytes, 768);

  mainResource.deallocate(rmm::cuda_stream_default, mainAddress, 256, 256);
  outputResource.deallocate(rmm::cuda_stream_default, outputAddress, 512, 256);

  snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.peakBytes, 768);
}

TEST(GpuMemoryTrackerTest, PreservesAllocationAlignment) {
  auto state = std::make_shared<RecordingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      RecordingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  constexpr std::size_t kAlignment = 4'096;
  auto resource = rmm::device_async_resource_ref{resources.main};
  auto* address = resource.allocate(rmm::cuda_stream_default, 64, kAlignment);
  resource.deallocate(rmm::cuda_stream_default, address, 64, kAlignment);

  std::lock_guard<std::mutex> lock(state->mutex);
  ASSERT_EQ(state->allocationAlignments.size(), 1);
  EXPECT_EQ(state->allocationAlignments.front(), kAlignment);
  ASSERT_EQ(state->deallocationAlignments.size(), 1);
  EXPECT_EQ(state->deallocationAlignments.front(), kAlignment);
}

TEST(GpuMemoryTrackerTest, ZeroSizeAllocationIsUntracked) {
  auto state = std::make_shared<RecordingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      RecordingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto resource = rmm::device_async_resource_ref{resources.main};
  auto* address = resource.allocate(rmm::cuda_stream_default, 0, 256);
  EXPECT_EQ(address, nullptr);
  resource.deallocate(rmm::cuda_stream_default, address, 0, 256);

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.dataLossEvents, 0);

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_TRUE(state->allocationAlignments.empty());
  EXPECT_TRUE(state->deallocationAlignments.empty());
}

/// Exercises the wrapper rather than the ledger: this is the path that resolves
/// an owner from the thread-local cache.
TEST(GpuMemoryTrackerTest, ConcurrentWrapperUseReconciles) {
  auto mainState = std::make_shared<RecordingResourceState>();
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{mainState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto mainResource = rmm::device_async_resource_ref{resources.main};
  auto outputResource = rmm::device_async_resource_ref{resources.output};

  constexpr int kWorkerCount = 4;
  constexpr int kIterations = 2'000;
  std::vector<std::thread> workers;
  workers.reserve(kWorkerCount);
  for (int worker = 0; worker < kWorkerCount; ++worker) {
    workers.emplace_back([&, worker] {
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        auto resource =
            ((worker + iteration) % 2 == 0) ? mainResource : outputResource;
        const std::size_t bytes = 64 + (iteration % 4) * 64;
        auto* address = resource.allocate(rmm::cuda_stream_default, bytes, 256);
        resource.deallocate(rmm::cuda_stream_default, address, bytes, 256);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.liveAllocations, 0);
  EXPECT_EQ(snapshot.dataLossEvents, 0);
}

TEST(GpuMemoryTrackerTest, ReusedAddressKeepsReplacementAllocation) {
  auto state = std::make_shared<AddressReusingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      AddressReusingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto mainResource = rmm::device_async_resource_ref{resources.main};
  auto outputResource = rmm::device_async_resource_ref{resources.output};
  auto* originalAddress =
      mainResource.allocate(rmm::cuda_stream_default, 64, 256);

  std::thread deallocator([&] {
    mainResource.deallocate(rmm::cuda_stream_default, originalAddress, 64, 256);
  });
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condition.wait(lock, [&] { return state->addressFreed; });
  }

  auto* replacementAddress =
      outputResource.allocate(rmm::cuda_stream_default, 128, 256);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->releaseDeallocation = true;
  }
  state->condition.notify_all();
  deallocator.join();

  ASSERT_EQ(replacementAddress, originalAddress);
  const auto replacementSnapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(replacementSnapshot));
  EXPECT_EQ(replacementSnapshot.currentBytes, 128);
  EXPECT_EQ(replacementSnapshot.dataLossEvents, 0);

  outputResource.deallocate(
      rmm::cuda_stream_default, replacementAddress, 128, 256);
  const auto finalSnapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(finalSnapshot));
  EXPECT_EQ(finalSnapshot.currentBytes, 0);
}

TEST(GpuMemoryTrackerTest, AllocationFailureRethrowsWithoutCounting) {
  auto failingState = std::make_shared<RecordingResourceState>();
  failingState->throwOnAllocation = true;
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{failingState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  constexpr std::size_t kAlignment = 4'096;
  auto resource = rmm::device_async_resource_ref{resources.main};
  EXPECT_THROW(
      resource.allocate(rmm::cuda_stream_default, 1'234, kAlignment),
      rmm::out_of_memory);

  {
    std::lock_guard<std::mutex> lock(failingState->mutex);
    ASSERT_EQ(failingState->allocationAlignments.size(), 1);
    EXPECT_EQ(failingState->allocationAlignments.front(), kAlignment);
  }

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.peakBytes, 0);
  EXPECT_EQ(snapshot.totalBytes, 0);
}

} // namespace facebook::velox::cudf_velox::test
