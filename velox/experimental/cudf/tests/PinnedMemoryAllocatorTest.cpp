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

#include "velox/experimental/cudf/memory/PinnedMemoryAllocator.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TestValue.h"

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <folly/ScopeGuard.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <barrier>
#include <cstring>
#include <filesystem>
#include <latch>
#include <thread>

DECLARE_bool(velox_ssd_odirect);

namespace facebook::velox::cudf_velox {
namespace {
using namespace memory;
constexpr uint64_t kMiB = 1 << 20;
class PinnedMemoryAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "CUDA device required";
    }
    int supported = 0;
    ASSERT_EQ(
        cudaSuccess,
        cudaDeviceGetAttribute(
            &supported, cudaDevAttrHostNumaMemoryPoolsSupported, 0));
    if (!supported) {
      GTEST_SKIP() << "Pinned CUDA memory pools unsupported";
    }
  }
  std::shared_ptr<PinnedMemoryAllocator> allocator(
      uint64_t capacity = 16 * kMiB) {
    return std::make_shared<PinnedMemoryAllocator>(
        PinnedMemoryAllocator::Options{capacity, 0});
  }
};

TEST_F(PinnedMemoryAllocatorTest, alignmentAndCpuReadiness) {
  auto mr = allocator();
  EXPECT_EQ(mr->kind(), MemoryAllocator::Kind::kCustom);
  EXPECT_EQ(MemoryAllocator::kindString(mr->kind()), "CUSTOM");
  for (auto alignment : {16, 64, 256, 4096, 32768}) {
    const uint64_t bytes = 65539;
    auto* ptr = mr->allocateBytes(bytes, alignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0);
    std::memset(ptr, 0x6b, bytes);
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[bytes - 1], 0x6b);
    EXPECT_EQ(mr->totalUsedBytes(), bytes);
    EXPECT_GE(AllocationTraits::pageBytes(mr->numAllocated()), bytes);
    mr->freeBytes(ptr, bytes);
    EXPECT_EQ(mr->numAllocated(), 0);
  }
  EXPECT_THROW(mr->allocateBytes(128, 3), VeloxException);
  EXPECT_EQ(mr->allocateBytes(std::numeric_limits<uint64_t>::max()), nullptr);
  EXPECT_EQ(mr->allocateBytes(0), nullptr);
  EXPECT_TRUE(mr->checkConsistency());
}

TEST_F(PinnedMemoryAllocatorTest, retainedBackingAndReuse) {
  auto mr = allocator();
  auto* p = mr->allocateBytes(kMiB);
  ASSERT_NE(p, nullptr);
  const auto retained = mr->retainedBytes();
  EXPECT_GE(retained, kMiB);
  mr->freeBytes(p, kMiB);
  EXPECT_EQ(mr->totalUsedBytes(), 0);
  EXPECT_EQ(mr->unmap(256), 0);
  auto* q = mr->allocateBytes(kMiB);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(mr->retainedBytes(), retained);
  // CUDA rounds its physical pool ceiling (32 MiB on the tested platform).
  // This allowance does not relax the allocator's live-allocation budget.
  EXPECT_LE(mr->retainedBytes(), mr->capacity() + 32 * kMiB);
  EXPECT_LE(AllocationTraits::pageBytes(mr->numAllocated()), mr->capacity());
  mr->freeBytes(q, kMiB);
}

TEST_F(PinnedMemoryAllocatorTest, sharedCapacityAndRollback) {
  auto mr = allocator(8 * kMiB);
  Allocation pages;
  ASSERT_TRUE(mr->allocateNonContiguous(257, pages));
  EXPECT_GE(pages.numRuns(), 2);
  for (uint32_t i = 0; i < pages.numRuns(); ++i) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(pages.runAt(i).data()) % 4096, 0);
    std::memset(pages.runAt(i).data(), 0xac, pages.runAt(i).numBytes());
  }
  const auto free =
      mr->capacity() - AllocationTraits::pageBytes(mr->numAllocated());
  auto* bytes = mr->allocateBytes(free);
  ASSERT_NE(bytes, nullptr);
  EXPECT_EQ(mr->allocateBytes(1), nullptr);
  mr->freeBytes(bytes, free);
  // Replacement fails after releasing the old allocation.
  EXPECT_FALSE(mr->allocateNonContiguous(4096, pages));
  EXPECT_TRUE(pages.empty());
  EXPECT_EQ(mr->numAllocated(), 0);
  EXPECT_EQ(mr->totalUsedBytes(), 0);
  EXPECT_TRUE(mr->checkConsistency());
}

DEBUG_ONLY_TEST_F(PinnedMemoryAllocatorTest, partialRunFailure) {
  common::testutil::TestValue::enable();
  auto mr = allocator();
  Allocation pages;
  int calls = 0;
  {
    SCOPED_TESTVALUE_SET(
        "PinnedMemoryAllocator::allocate",
        std::function<void(bool*)>([&](bool* fail) { *fail = ++calls == 2; }));
    EXPECT_FALSE(mr->allocateNonContiguous(257, pages));
  }
  EXPECT_EQ(calls, 2);
  EXPECT_TRUE(pages.empty());
  EXPECT_EQ(mr->numAllocated(), 0);
  ASSERT_TRUE(mr->allocateNonContiguous(257, pages));
  mr->freeNonContiguous(pages);
  EXPECT_TRUE(mr->checkConsistency());
}

DEBUG_ONLY_TEST_F(
    PinnedMemoryAllocatorTest,
    fragmentedPoolReclaimsBeyondRequestSize) {
  common::testutil::TestValue::enable();
  auto mr = allocator(64 * kMiB);
  cache::AsyncDataCache::Options options;
  options.numShards = 1;
  auto cache = cache::AsyncDataCache::create(mr.get(), nullptr, options);
  auto shutdown = folly::makeGuard([&] { cache->shutdown(); });
  StringIdLease file(fileIds(), std::string_view("fragmented-pinned-pool"));
  for (int i = 0; i < 48; ++i) {
    auto pin = cache->findOrCreate({file.id(), i * kMiB}, kMiB, true, nullptr);
    pin.entry()->setExclusiveToShared();
  }
  // Reproduce a pool that has enough logical headroom but cannot satisfy a
  // small contiguous request until substantially more cache has been evicted.
  // This is the condition observed with real CUDA host-pool fragmentation.
  {
    SCOPED_TESTVALUE_SET(
        "PinnedMemoryAllocator::poolAllocate",
        std::function<void(uint64_t*)>([](uint64_t* charged) {
          if (*charged > 8 * kMiB) {
            throw cuda::cuda_error(
                cudaErrorMemoryAllocation, "fragmented test pool");
          }
        }));
    auto* spill = mr->allocateBytes(kMiB);
    ASSERT_NE(spill, nullptr);
    EXPECT_GT(cache->refreshStats().numEvict, 4);
    mr->freeBytes(spill, kMiB);
  }
  cache->shutdown();
  shutdown.dismiss();
  EXPECT_EQ(mr->numAllocated(), 0);
  EXPECT_TRUE(mr->checkConsistency());
}

TEST_F(PinnedMemoryAllocatorTest, unsupportedOperations) {
  auto mr = allocator();
  ContiguousAllocation allocation;
  uint64_t reserved = 0;
  EXPECT_FALSE(mr->allocateContiguous(
      256, nullptr, allocation, [&](uint64_t bytes, bool pre) {
        if (pre) {
          reserved += bytes;
        } else {
          reserved -= bytes;
        }
      }));
  EXPECT_TRUE(allocation.empty());
  EXPECT_EQ(reserved, 0);
  EXPECT_NE(
      mr->getAndClearFailureMessage().find("does not support"),
      std::string::npos);
  EXPECT_EQ(mr->numAllocated(), 0);
}

TEST_F(PinnedMemoryAllocatorTest, concurrentCapacity) {
  auto mr = allocator(4 * kMiB);
  std::barrier gate(8);
  std::atomic<int> succeeded{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      auto* ptr = mr->allocateBytes(kMiB);
      if (ptr) {
        ++succeeded;
      }
      gate.arrive_and_wait();
      EXPECT_LE(mr->numAllocated(), 1024);
      gate.arrive_and_wait();
      if (ptr) {
        mr->freeBytes(ptr, kMiB);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(succeeded, 4);
  EXPECT_EQ(mr->numAllocated(), 0);
  EXPECT_TRUE(mr->checkConsistency());
}

TEST_F(PinnedMemoryAllocatorTest, cachePinsEvictionAndRefill) {
  auto mr = allocator(8 * kMiB);
  cache::AsyncDataCache::Options options;
  options.numShards = 1;
  auto cache = cache::AsyncDataCache::create(mr.get(), nullptr, options);
  auto shutdown = folly::makeGuard([&] { cache->shutdown(); });
  StringIdLease file(fileIds(), std::string_view("pinned-cache-test"));
  std::vector<cache::CachePin> pins;
  for (int i = 0; i < 8; ++i) {
    auto pin = cache->findOrCreate({file.id(), i * kMiB}, kMiB, true, nullptr);
    std::memset(pin.checkedEntry()->contiguousData(), i + 1, kMiB);
    pin.checkedEntry()->setExclusiveToShared();
    pins.push_back(std::move(pin));
  }
  EXPECT_EQ(mr->allocateBytes(kMiB), nullptr);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(
        reinterpret_cast<uint8_t*>(pins[i].checkedEntry()->contiguousData())[0],
        i + 1);
  }
  pins.clear();
  auto* spill = mr->allocateBytes(6 * kMiB);
  ASSERT_NE(spill, nullptr);
  std::memset(spill, 0x52, 6 * kMiB);
  EXPECT_LE(mr->numAllocated(), mr->capacity() / 4096);
  EXPECT_GT(cache->refreshStats().numEvict, 0);
  mr->freeBytes(spill, 6 * kMiB);
  auto pin = cache->findOrCreate({file.id(), 20 * kMiB}, kMiB, false, nullptr);
  ASSERT_FALSE(pin.checkedEntry()->nonContiguousData().empty());
  for (uint32_t i = 0; i < pin.checkedEntry()->nonContiguousData().numRuns();
       ++i) {
    auto run = pin.checkedEntry()->nonContiguousData().runAt(i);
    std::memset(run.data(), 0x74, run.numBytes());
  }
  pin.checkedEntry()->setExclusiveToShared();
  pin.clear();
  cache->shutdown();
  EXPECT_EQ(mr->numAllocated(), 0);
  EXPECT_TRUE(mr->checkConsistency());
  shutdown.dismiss();
}

TEST_F(PinnedMemoryAllocatorTest, cachePinHeldThroughGpuRead) {
  auto mr = allocator(8 * kMiB);
  cache::AsyncDataCache::Options options;
  options.numShards = 1;
  auto cache = cache::AsyncDataCache::create(mr.get(), nullptr, options);
  auto shutdown = folly::makeGuard([&] { cache->shutdown(); });
  StringIdLease file(fileIds(), std::string_view("pinned-cache-gpu-read"));
  auto pin = cache->findOrCreate({file.id(), 0}, 8 * kMiB, true, nullptr);
  std::memset(pin.checkedEntry()->contiguousData(), 0x28, 8 * kMiB);
  pin.checkedEntry()->setExclusiveToShared();
  void* device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device, 8 * kMiB));
  auto deviceGuard = folly::makeGuard([&] { cudaFree(device); });
  cudaStream_t stream;
  ASSERT_EQ(
      cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto streamGuard = folly::makeGuard([&] { cudaStreamDestroy(stream); });
  std::latch gate(1);
  ASSERT_EQ(
      cudaSuccess,
      cudaLaunchHostFunc(
          stream, [](void* p) { static_cast<std::latch*>(p)->wait(); }, &gate));
  auto unblock = folly::makeGuard([&] {
    gate.count_down();
    cudaStreamSynchronize(stream);
  });
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpyAsync(
          device,
          pin.checkedEntry()->contiguousData(),
          8 * kMiB,
          cudaMemcpyHostToDevice,
          stream));
  // The source is pinned while a GPU read is outstanding. Cache eviction must
  // fail rather than recycle the source, even though the allocator is
  // exhausted.
  EXPECT_EQ(mr->allocateBytes(kMiB), nullptr);
  gate.count_down();
  unblock.dismiss();
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  pin.clear();
  auto* spill = mr->allocateBytes(8 * kMiB);
  ASSERT_NE(spill, nullptr);
  auto freeSpill = folly::makeGuard([&] { mr->freeBytes(spill, 8 * kMiB); });
  std::memset(spill, 0x73, 8 * kMiB);
  std::vector<uint8_t> result(8 * kMiB);
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(result.data(), device, result.size(), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](uint8_t v) {
    return v == 0x28;
  }));
}

TEST_F(PinnedMemoryAllocatorTest, ssdCacheRoundTrip) {
  filesystems::registerLocalFileSystem();
  if (!MemoryManager::testInstance()) {
    MemoryManager::initialize({});
  }
  const bool oldDirect = FLAGS_velox_ssd_odirect;
  FLAGS_velox_ssd_odirect = false;
  auto restoreFlag =
      folly::makeGuard([&] { FLAGS_velox_ssd_odirect = oldDirect; });
  char directory[] = "/tmp/velox-pinned-ssd-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  auto removeFiles =
      folly::makeGuard([&] { std::filesystem::remove_all(directory); });
  folly::IOThreadPoolExecutor executor(1);
  auto mr = allocator(8 * kMiB);
  auto ssd = std::make_unique<cache::SsdCache>(cache::SsdCache::Config(
      std::string(directory) + "/cache",
      64 * kMiB,
      1,
      &executor,
      0,
      false,
      true,
      true));
  auto cache = cache::AsyncDataCache::create(mr.get(), std::move(ssd));
  auto shutdown = folly::makeGuard([&] { cache->shutdown(); });
  StringIdLease file(fileIds(), std::string_view("pinned-ssd-roundtrip"));
  for (bool contiguous : {false, true}) {
    cache::RawFileCacheKey key{file.id(), contiguous ? kMiB : 0};
    std::vector<cache::CachePin> pins;
    pins.push_back(cache->findOrCreate(key, kMiB, contiguous, nullptr));
    auto* entry = pins[0].checkedEntry();
    if (contiguous) {
      std::memset(entry->contiguousData(), 0x5c, kMiB);
    } else {
      for (uint32_t i = 0; i < entry->nonContiguousData().numRuns(); ++i) {
        auto run = entry->nonContiguousData().runAt(i);
        std::memset(run.data(), 0x5c, run.numBytes());
      }
    }
    entry->setExclusiveToShared();
    auto& ssdFile = cache->ssdCache()->file(file.id());
    ssdFile.write(pins);
    std::vector<cache::SsdPin> diskPins;
    diskPins.push_back(ssdFile.find(key));
    ASSERT_FALSE(diskPins[0].empty());
    pins.clear();
    cache->shrink(8 * kMiB);
    pins.push_back(cache->findOrCreate(key, kMiB, contiguous, nullptr));
    ASSERT_TRUE(pins[0].checkedEntry()->isExclusive());
    ssdFile.load(diskPins, pins);
    entry = pins[0].checkedEntry();
    if (contiguous) {
      EXPECT_TRUE(
          std::all_of(
              entry->contiguousData(),
              entry->contiguousData() + kMiB,
              [](char v) { return v == 0x5c; }));
    } else {
      for (uint32_t i = 0; i < entry->nonContiguousData().numRuns(); ++i) {
        auto run = entry->nonContiguousData().runAt(i);
        EXPECT_TRUE(
            std::all_of(run.data(), run.data() + run.numBytes(), [](uint8_t v) {
              return v == 0x5c;
            }));
      }
    }
    entry->setExclusiveToShared();
  }
  cache->shutdown();
  shutdown.dismiss();
  EXPECT_EQ(mr->numAllocated(), 0);
}

TEST_F(PinnedMemoryAllocatorTest, spillRestoreAcrossStreams) {
  auto mr = allocator();
  const size_t bytes = kMiB;
  void* device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device, bytes));
  auto freeDevice = folly::makeGuard([&] { cudaFree(device); });
  cudaStream_t producer, restore;
  ASSERT_EQ(
      cudaSuccess, cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
  ASSERT_EQ(
      cudaSuccess, cudaStreamCreateWithFlags(&restore, cudaStreamNonBlocking));
  auto streams = folly::makeGuard([&] {
    cudaStreamDestroy(producer);
    cudaStreamDestroy(restore);
  });
  auto* host = mr->allocateBytes(bytes);
  ASSERT_NE(host, nullptr);
  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(device, 0x37, bytes, producer));
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, producer));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(producer));
  EXPECT_EQ(static_cast<uint8_t*>(host)[bytes - 1], 0x37);
  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(device, 0, bytes, restore));
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, restore));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(restore));
  mr->freeBytes(host, bytes);
  auto* reused = mr->allocateBytes(bytes);
  ASSERT_NE(reused, nullptr);
  std::memset(reused, 0x99, bytes);
  std::vector<uint8_t> result(bytes);
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](auto byte) {
    return byte == 0x37;
  }));
  mr->freeBytes(reused, bytes);
}
} // namespace
} // namespace facebook::velox::cudf_velox
