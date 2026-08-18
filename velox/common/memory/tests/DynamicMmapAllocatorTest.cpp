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

#include "velox/common/memory/MmapAllocator.h"

#include <chrono>
#include <exception>
#include <thread>

#include <folly/ScopeGuard.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/testutil/TestValue.h"

using namespace facebook::velox::common::testutil;
using namespace std::chrono_literals;

namespace facebook::velox::memory {
namespace {

constexpr MachinePageCount kConfiguredCapacityPages = 64 * 256;

MemoryAllocator::Options makeAllocatorOptions() {
  MemoryAllocator::Options options;
  options.capacity = AllocationTraits::pageBytes(kConfiguredCapacityPages);
  options.maxMallocBytes = 0;
  return options;
}

// Exercises the extension points used by allocators with a runtime admission
// capacity while keeping MmapAllocator's configured capacity unchanged.
class TestingAdmissionCapacityMmapAllocator : public MmapAllocator {
 public:
  explicit TestingAdmissionCapacityMmapAllocator(
      MachinePageCount admissionCapacity)
      : MmapAllocator(makeAllocatorOptions()),
        admissionCapacity_(admissionCapacity) {}

  void setAdmissionCapacity(MachinePageCount admissionCapacity) {
    admissionCapacity_.store(admissionCapacity);
  }

 private:
  bool allocateNonContiguousWithoutRetry(
      const SizeMix& sizeMix,
      Allocation& out) override {
    return allocateNonContiguousWithCapacity(
        sizeMix, out, admissionCapacity_.load());
  }

  bool allocateContiguousWithoutRetry(
      MachinePageCount numPages,
      Allocation* collateral,
      ContiguousAllocation& allocation,
      MachinePageCount maxPages) override {
    return allocateContiguousWithCapacity(
        numPages, collateral, allocation, maxPages, admissionCapacity_.load());
  }

  bool growContiguousWithoutRetry(
      MachinePageCount increment,
      ContiguousAllocation& allocation) override {
    return growContiguousWithCapacity(
        increment, allocation, admissionCapacity_.load());
  }

  std::atomic<MachinePageCount> admissionCapacity_;
};

TEST(DynamicMmapAllocatorTest, enforcesAdmissionCapacity) {
  constexpr MachinePageCount kAdmissionCapacityPages = 8;
  constexpr MachinePageCount kReducedCapacityPages = 4;
  TestingAdmissionCapacityMmapAllocator allocator(kAdmissionCapacityPages);

  EXPECT_EQ(
      allocator.capacity(),
      AllocationTraits::pageBytes(kConfiguredCapacityPages));

  Allocation nonContiguous;
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kAdmissionCapacityPages, nonContiguous));
  Allocation extraNonContiguous;
  EXPECT_FALSE(allocator.allocateNonContiguous(1, extraNonContiguous));
  allocator.freeNonContiguous(nonContiguous);

  ASSERT_TRUE(allocator.allocateNonContiguous(4, nonContiguous));
  allocator.freeNonContiguous(nonContiguous);
  ASSERT_TRUE(allocator.allocateNonContiguous(1, nonContiguous));
  allocator.freeNonContiguous(nonContiguous);
  ASSERT_GT(allocator.numMapped(), kReducedCapacityPages);

  allocator.setAdmissionCapacity(kReducedCapacityPages);
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kReducedCapacityPages, nonContiguous));
  EXPECT_LE(allocator.numMapped(), kReducedCapacityPages);
  allocator.freeNonContiguous(nonContiguous);

  allocator.setAdmissionCapacity(kAdmissionCapacityPages);
  ContiguousAllocation contiguous;
  ASSERT_TRUE(allocator.allocateContiguous(
      kAdmissionCapacityPages, nullptr, contiguous));
  ContiguousAllocation extraContiguous;
  EXPECT_FALSE(allocator.allocateContiguous(1, nullptr, extraContiguous));
  allocator.freeContiguous(contiguous);

  constexpr MachinePageCount kGrownCapacityPages = 12;
  allocator.setAdmissionCapacity(kGrownCapacityPages);
  ASSERT_TRUE(allocator.allocateContiguous(
      kAdmissionCapacityPages,
      nullptr,
      contiguous,
      nullptr,
      kGrownCapacityPages + 1));
  EXPECT_TRUE(allocator.growContiguous(
      kGrownCapacityPages - kAdmissionCapacityPages, contiguous));
  EXPECT_FALSE(allocator.growContiguous(1, contiguous));
  allocator.freeContiguous(contiguous);
}

TEST(DynamicMmapAllocatorTest, allowsNoGrowthAboveReducedCapacity) {
  constexpr MachinePageCount kInitialCapacityPages = 8;
  constexpr MachinePageCount kReducedCapacityPages = 4;
  TestingAdmissionCapacityMmapAllocator allocator(kInitialCapacityPages);

  Allocation nonContiguous;
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kInitialCapacityPages, nonContiguous));
  allocator.setAdmissionCapacity(kReducedCapacityPages);
  EXPECT_TRUE(
      allocator.allocateNonContiguous(kInitialCapacityPages, nonContiguous));
  allocator.freeNonContiguous(nonContiguous);

  allocator.setAdmissionCapacity(kInitialCapacityPages);
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kInitialCapacityPages, nonContiguous));
  allocator.setAdmissionCapacity(kReducedCapacityPages);
  ContiguousAllocation contiguous;
  EXPECT_TRUE(allocator.allocateContiguous(
      kInitialCapacityPages, &nonContiguous, contiguous));
  EXPECT_TRUE(nonContiguous.empty());
  allocator.freeContiguous(contiguous);
}

DEBUG_ONLY_TEST(
    DynamicMmapAllocatorTest,
    preservesConfiguredMappedCapacityDuringConcurrentFailure) {
  MmapAllocator allocator(makeAllocatorOptions());

  Allocation mapped;
  Allocation mappedFree;
  Allocation additionalMappedFree;
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kConfiguredCapacityPages - 2, mapped));
  ASSERT_TRUE(allocator.allocateNonContiguous(1, mappedFree));
  ASSERT_TRUE(allocator.allocateNonContiguous(1, additionalMappedFree));
  allocator.freeNonContiguous(mappedFree);
  allocator.freeNonContiguous(additionalMappedFree);
  ASSERT_EQ(allocator.numAllocated(), kConfiguredCapacityPages - 2);
  ASSERT_EQ(allocator.numMapped(), kConfiguredCapacityPages);
  auto mappedGuard =
      folly::makeGuard([&]() { allocator.freeNonContiguous(mapped); });

  folly::Baton<> firstReservation;
  folly::Baton<> secondReservation;
  folly::Baton<> releaseFirstReservation;
  folly::Baton<> releaseSecondReservation;
  std::atomic<int32_t> numReservations{0};
  TestValue::enable();
  SCOPED_TESTVALUE_SET(
      "facebook::velox::memory::MmapAllocator::allocateContiguousImpl",
      std::function<void(MmapAllocator*)>([&](MmapAllocator* /*unused*/) {
        if (numReservations.fetch_add(1) == 0) {
          firstReservation.post();
          releaseFirstReservation.wait();
        } else {
          secondReservation.post();
          releaseSecondReservation.wait();
        }
      }));

  ContiguousAllocation first;
  bool firstSucceeded{false};
  std::exception_ptr firstError;
  std::thread firstThread([&]() {
    try {
      firstSucceeded = allocator.allocateContiguous(2, nullptr, first);
    } catch (...) {
      firstError = std::current_exception();
    }
  });
  auto firstThreadGuard = folly::makeGuard([&]() {
    releaseFirstReservation.post();
    releaseSecondReservation.post();
    if (firstThread.joinable()) {
      firstThread.join();
    }
    allocator.freeContiguous(first);
  });
  ASSERT_TRUE(firstReservation.try_wait_for(5s));

  ContiguousAllocation second;
  bool secondSucceeded{false};
  std::exception_ptr secondError;
  std::thread secondThread([&]() {
    try {
      secondSucceeded = allocator.allocateContiguous(1, nullptr, second);
    } catch (...) {
      secondError = std::current_exception();
    }
  });
  auto secondThreadGuard = folly::makeGuard([&]() {
    releaseFirstReservation.post();
    releaseSecondReservation.post();
    if (secondThread.joinable()) {
      secondThread.join();
    }
    allocator.freeContiguous(second);
  });
  ASSERT_TRUE(secondReservation.try_wait_for(5s));
  ASSERT_EQ(allocator.numAllocated(), kConfiguredCapacityPages + 1);

  releaseFirstReservation.post();
  firstThread.join();
  EXPECT_LE(allocator.numMapped(), kConfiguredCapacityPages);
  releaseSecondReservation.post();
  secondThread.join();
  if (firstError) {
    std::rethrow_exception(firstError);
  }
  if (secondError) {
    std::rethrow_exception(secondError);
  }

  EXPECT_TRUE(firstSucceeded);
  EXPECT_FALSE(secondSucceeded);
  EXPECT_TRUE(allocator.checkConsistency());

  allocator.freeNonContiguous(mapped);
  mappedGuard.dismiss();
  allocator.freeContiguous(first);
  firstThreadGuard.dismiss();
  allocator.freeContiguous(second);
  secondThreadGuard.dismiss();
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_TRUE(allocator.checkConsistency());
}

TEST(DynamicMmapAllocatorTest, allowsMappedAllocationWhenBestEffortTrimFails) {
  constexpr MachinePageCount kInitialCapacityPages = 8;
  constexpr MachinePageCount kReducedCapacityPages = 4;
  TestingAdmissionCapacityMmapAllocator allocator(kInitialCapacityPages);

  Allocation mapped;
  Allocation additionalMapped;
  ASSERT_TRUE(allocator.allocateNonContiguous(kReducedCapacityPages, mapped));
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kReducedCapacityPages, additionalMapped));
  allocator.freeNonContiguous(mapped);
  allocator.freeNonContiguous(additionalMapped);
  ASSERT_EQ(allocator.numMapped(), kInitialCapacityPages);

  allocator.setAdmissionCapacity(kReducedCapacityPages);
  allocator.testingSetFailureInjection(
      MemoryAllocator::InjectedFailure::kMadvise);
  EXPECT_TRUE(allocator.allocateNonContiguous(kReducedCapacityPages, mapped));
  EXPECT_EQ(allocator.numAllocated(), kReducedCapacityPages);
  EXPECT_TRUE(allocator.checkConsistency());

  allocator.freeNonContiguous(mapped);
}

DEBUG_ONLY_TEST(
    DynamicMmapAllocatorTest,
    rechecksMappedCapacityAfterConcurrentAllocation) {
  constexpr MachinePageCount kInitialCapacityPages = 16;
  constexpr MachinePageCount kReducedCapacityPages = 2;
  TestingAdmissionCapacityMmapAllocator allocator(kInitialCapacityPages);

  Allocation mappedFree;
  ASSERT_TRUE(allocator.allocateNonContiguous(8, mappedFree));
  allocator.freeNonContiguous(mappedFree);

  Allocation replacement;
  ASSERT_TRUE(allocator.allocateNonContiguous(4, replacement));
  ASSERT_EQ(allocator.numAllocated(), 4);
  ASSERT_EQ(allocator.numMapped(), 12);

  allocator.setAdmissionCapacity(kReducedCapacityPages);
  folly::Baton<> targetComputed;
  folly::Baton<> resumeRebalance;
  TestValue::enable();
  SCOPED_TESTVALUE_SET(
      "facebook::velox::memory::MmapAllocator::ensureEnoughMappedPages",
      std::function<void(MmapAllocator*)>([&](MmapAllocator* /*unused*/) {
        targetComputed.post();
        resumeRebalance.wait();
      }));

  bool replacementSucceeded{false};
  std::exception_ptr replacementError;
  std::thread replacementThread([&]() {
    try {
      replacementSucceeded = allocator.allocateNonContiguous(2, replacement);
    } catch (...) {
      replacementError = std::current_exception();
    }
  });
  auto threadGuard = folly::makeGuard([&]() {
    resumeRebalance.post();
    if (replacementThread.joinable()) {
      replacementThread.join();
    }
  });
  ASSERT_TRUE(targetComputed.try_wait_for(5s));

  Allocation concurrent;
  allocator.setAdmissionCapacity(kInitialCapacityPages);
  EXPECT_TRUE(allocator.allocateNonContiguous(8, concurrent));
  allocator.setAdmissionCapacity(kReducedCapacityPages);

  resumeRebalance.post();
  replacementThread.join();
  threadGuard.dismiss();
  if (replacementError) {
    std::rethrow_exception(replacementError);
  }

  EXPECT_TRUE(replacementSucceeded);
  if (!replacementSucceeded) {
    allocator.freeNonContiguous(concurrent);
    return;
  }
  ASSERT_EQ(replacement.numPages(), 2);
  ASSERT_EQ(concurrent.numPages(), 8);
  replacement.runAt(0).data<uint8_t>()[0] = 0x11;
  concurrent.runAt(0).data<uint8_t>()[0] = 0x22;
  EXPECT_EQ(replacement.runAt(0).data<uint8_t>()[0], 0x11);
  EXPECT_EQ(concurrent.runAt(0).data<uint8_t>()[0], 0x22);
  EXPECT_EQ(allocator.numAllocated(), 10);
  EXPECT_EQ(allocator.numMapped(), 10);
  EXPECT_TRUE(allocator.checkConsistency());

  allocator.freeNonContiguous(replacement);
  allocator.freeNonContiguous(concurrent);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_TRUE(allocator.checkConsistency());
}

DEBUG_ONLY_TEST(
    DynamicMmapAllocatorTest,
    usesCurrentMappedCountAfterConcurrentFree) {
  constexpr MachinePageCount kInitialCapacityPages = 110;
  constexpr MachinePageCount kReducedCapacityPages = 80;
  TestingAdmissionCapacityMmapAllocator allocator(kInitialCapacityPages);

  ContiguousAllocation contiguous;
  ASSERT_TRUE(allocator.allocateContiguous(40, nullptr, contiguous));
  Allocation replacement;
  ASSERT_TRUE(allocator.allocateNonContiguous(20, replacement));

  Allocation thirtyPages;
  Allocation eighteenPages;
  ASSERT_TRUE(allocator.allocateNonContiguous(30, thirtyPages));
  ASSERT_TRUE(allocator.allocateNonContiguous(18, eighteenPages));
  allocator.freeNonContiguous(thirtyPages);
  allocator.freeNonContiguous(eighteenPages);
  ASSERT_EQ(allocator.numAllocated(), 60);
  ASSERT_EQ(allocator.numMapped(), 110);

  allocator.setAdmissionCapacity(kReducedCapacityPages);
  folly::Baton<> targetComputed;
  folly::Baton<> resumeRebalance;
  TestValue::enable();
  SCOPED_TESTVALUE_SET(
      "facebook::velox::memory::MmapAllocator::ensureEnoughMappedPages",
      std::function<void(MmapAllocator*)>([&](MmapAllocator* /*unused*/) {
        targetComputed.post();
        resumeRebalance.wait();
      }));

  bool replacementSucceeded{false};
  std::exception_ptr replacementError;
  std::thread replacementThread([&]() {
    try {
      replacementSucceeded = allocator.allocateNonContiguous(20, replacement);
    } catch (...) {
      replacementError = std::current_exception();
    }
  });
  auto threadGuard = folly::makeGuard([&]() {
    resumeRebalance.post();
    if (replacementThread.joinable()) {
      replacementThread.join();
    }
  });
  ASSERT_TRUE(targetComputed.try_wait_for(5s));

  allocator.freeContiguous(contiguous);
  Allocation concurrent;
  allocator.setAdmissionCapacity(kInitialCapacityPages);
  ASSERT_TRUE(allocator.allocateNonContiguous(30, concurrent));
  allocator.setAdmissionCapacity(kReducedCapacityPages);

  resumeRebalance.post();
  replacementThread.join();
  threadGuard.dismiss();
  if (replacementError) {
    std::rethrow_exception(replacementError);
  }

  EXPECT_TRUE(replacementSucceeded);
  if (!replacementSucceeded) {
    allocator.freeNonContiguous(concurrent);
    return;
  }
  EXPECT_EQ(
      allocator.numAllocated(), replacement.numPages() + concurrent.numPages());
  EXPECT_LE(
      allocator.numMapped(),
      std::max(kReducedCapacityPages, allocator.numAllocated()));
  EXPECT_TRUE(allocator.checkConsistency());

  allocator.freeNonContiguous(replacement);
  allocator.freeNonContiguous(concurrent);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_TRUE(allocator.checkConsistency());
}

TEST(DynamicMmapAllocatorTest, clampsCapacityToConfiguredCapacity) {
  TestingAdmissionCapacityMmapAllocator allocator(kConfiguredCapacityPages + 1);

  Allocation nonContiguous;
  ASSERT_TRUE(
      allocator.allocateNonContiguous(kConfiguredCapacityPages, nonContiguous));
  Allocation extra;
  EXPECT_FALSE(allocator.allocateNonContiguous(1, extra));
  allocator.freeNonContiguous(nonContiguous);

  ContiguousAllocation contiguous;
  ASSERT_TRUE(allocator.allocateContiguous(
      kConfiguredCapacityPages, nullptr, contiguous));
  ContiguousAllocation extraContiguous;
  EXPECT_FALSE(allocator.allocateContiguous(1, nullptr, extraContiguous));
  allocator.freeContiguous(contiguous);

  ASSERT_TRUE(allocator.allocateContiguous(
      kConfiguredCapacityPages - 1,
      nullptr,
      contiguous,
      nullptr,
      kConfiguredCapacityPages + 1));
  EXPECT_TRUE(allocator.growContiguous(1, contiguous));
  EXPECT_FALSE(allocator.growContiguous(1, contiguous));
  allocator.freeContiguous(contiguous);
}

TEST(DynamicMmapAllocatorTest, cacheEvictsAtAdmissionCapacity) {
  constexpr MachinePageCount kInitialCapacityPages = 8;
  constexpr MachinePageCount kReducedCapacityPages = 4;
  constexpr uint64_t kEntryBytes =
      kReducedCapacityPages * AllocationTraits::kPageSize;
  auto allocator = std::make_shared<TestingAdmissionCapacityMmapAllocator>(
      kInitialCapacityPages);
  auto dataCache = cache::AsyncDataCache::create(allocator.get());

  {
    StringIdLease file{
        fileIds(),
        std::string_view{"dynamic_capacity_cache_test"},
    };
    for (uint64_t i = 0; i < 2; ++i) {
      auto pin = dataCache->findOrCreate(
          cache::RawFileCacheKey{file.id(), i * kEntryBytes}, kEntryBytes);
      ASSERT_FALSE(pin.empty());
      pin.checkedEntry()->setExclusiveToShared();
    }
    ASSERT_EQ(dataCache->cachedPages(), kInitialCapacityPages);

    allocator->setAdmissionCapacity(kReducedCapacityPages);
    auto pin = dataCache->findOrCreate(
        cache::RawFileCacheKey{file.id(), 2 * kEntryBytes}, kEntryBytes);
    EXPECT_FALSE(pin.empty());
    EXPECT_LE(dataCache->cachedPages(), kReducedCapacityPages);
  }

  dataCache->shutdown();
  dataCache.reset();
}

} // namespace
} // namespace facebook::velox::memory
