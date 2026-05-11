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

#include "folly/synchronization/EventCount.h"
#include "velox/common/base/Semaphore.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/caching/tests/CacheTestUtil.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/testutil/ScopedTestTime.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/types.h>

using namespace facebook::velox;
using namespace facebook::velox::cache;
using namespace facebook::velox::common::testutil;

using facebook::velox::memory::MemoryAllocator;

DECLARE_bool(velox_ssd_odirect);
DECLARE_bool(velox_ssd_verify_write);

// Represents a planned load from a file. Many of these constitute a load plan.
struct Request {
  Request(uint64_t _offset, uint64_t _size) : offset(_offset), size(_size) {}

  uint64_t offset;
  uint64_t size;
  SsdPin ssdPin;
};

struct TestParam {
  bool checksumEnabled;
  bool checksumVerificationEnabled;
};

class AsyncDataCacheTest : public ::testing::TestWithParam<TestParam> {
 public:
  static std::vector<TestParam> getTestParams() {
    static std::vector<TestParam> testParams = {
        {false, false}, {true, false}, {true, true}};
    return testParams;
  }

  // Deterministically fills 'allocation' based on 'sequence'
  static void initializeContents(int64_t sequence, memory::Allocation& alloc) {
    for (int32_t i = 0; i < alloc.numRuns(); ++i) {
      memory::Allocation::PageRun run = alloc.runAt(i);
      int64_t* ptr = reinterpret_cast<int64_t*>(run.data());
      const int32_t numWords =
          memory::AllocationTraits::pageBytes(run.numPages()) / sizeof(void*);
      for (int32_t offset = 0; offset < numWords; ++offset) {
        ptr[offset] = sequence + offset;
      }
    }
  }

 protected:
  static constexpr int32_t kNumFiles = 100;
  static constexpr int32_t kNumSsdShards = 4;

  static void SetUpTestCase() {
    TestValue::enable();
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
  }

  void TearDown() override {
    if (cache_) {
      cache_->shutdown();
      auto* ssdCache = cache_->ssdCache();
      if (ssdCache) {
        ssdCacheHelper_->deleteFiles();
      }
    }
    if (loadExecutor_ != nullptr) {
      loadExecutor_->join();
    }
    filenames_.clear();
    CacheTTLController::testingClear();
    fileIds().testingReset();
  }

  void waitForPendingLoads() {
    while (numPendingLoads_ > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(2000)); // NOLINT
    }
  }

  void initializeMemoryManager(int64_t capacity) {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::Options options;
      options.useMmapAllocator = true;
      options.allocatorCapacity = capacity;
      options.arbitratorCapacity = capacity;
      options.trackDefaultUsage = true;
      memory::MemoryManager::initialize(options);
    }
  }

  void initializeCache(
      uint64_t maxBytes,
      int64_t ssdBytes = 0,
      uint64_t checkpointIntervalBytes = 0,
      bool eraseCheckpoint = false,
      AsyncDataCache::Options cacheOptions = {}) {
    if (cache_ != nullptr) {
      cache_->shutdown();
    }
    cache_.reset();

    std::unique_ptr<SsdCache> ssdCache;
    if (ssdBytes > 0) {
      // tmpfs does not support O_DIRECT, so turn this off for testing.
      FLAGS_velox_ssd_odirect = false;
      // Make a new tempDirectory only if one is not already set. The
      // second creation of cache must find the checkpoint of the
      // previous one.
      if (tempDirectory_ == nullptr || eraseCheckpoint) {
        tempDirectory_ = TempDirectoryPath::create();
      }
      SsdCache::Config config(
          fmt::format("{}/cache", tempDirectory_->getPath()),
          ssdBytes,
          kNumSsdShards,
          ssdExecutor(),
          checkpointIntervalBytes > 0 ? checkpointIntervalBytes : ssdBytes / 20,
          false,
          GetParam().checksumEnabled,
          GetParam().checksumVerificationEnabled);
      ssdCache = std::make_unique<SsdCache>(config);
      if (ssdCache != nullptr) {
        ssdCacheHelper_ =
            std::make_unique<cache::test::SsdCacheTestHelper>(ssdCache.get());
        ASSERT_EQ(ssdCacheHelper_->numShards(), kNumSsdShards);
      }
    }

    memory::MemoryManager::Options options;
    options.useMmapAllocator = true;
    options.allocatorCapacity = maxBytes;
    options.arbitratorCapacity = maxBytes;
    options.trackDefaultUsage = true;
    manager_ = std::make_unique<memory::MemoryManager>(options);
    allocator_ = static_cast<memory::MmapAllocator*>(manager_->allocator());
    cache_ =
        AsyncDataCache::create(allocator_, std::move(ssdCache), cacheOptions);
    asyncDataCacheHelper_ =
        std::make_unique<cache::test::AsyncDataCacheTestHelper>(cache_.get());
    if (filenames_.empty()) {
      for (auto i = 0; i < kNumFiles; ++i) {
        auto name = fmt::format("testing_file_{}", i);
        filenames_.push_back(StringIdLease(fileIds(), name));
      }
    }
    ASSERT_EQ(cache_->allocator()->kind(), MemoryAllocator::Kind::kMmap);
    ASSERT_EQ(MemoryAllocator::kindString(cache_->allocator()->kind()), "MMAP");
  }

  // Finds one entry from RAM, SSD or storage. Throws if the data
  // cannot be read or 'injectError' is true. Checks the data with
  // verifyHook and discards the pin.
  void loadOne(uint64_t fileNum, Request& request, bool injectError);

  // Brings the data for the ranges in 'requests' into cache. The individual
  // entries should be accessed with loadOne(). 'requests' are handled with one
  // TestingCoalescedSsdLoad and one TestingCoalescedLoad. Call
  // semaphore.acquire() twice if needing to wait for the two loads to finish.
  void loadBatch(
      uint64_t fileNum,
      std::vector<Request>& requests,
      bool injectError,
      Semaphore* semaphore = nullptr);

  // Gets a pin on each of 'requests' individually. This checks the contents via
  // cache_'s verifyHook.
  void checkBatch(
      uint64_t fileNum,
      std::vector<Request>& requests,
      bool injectError) {
    for (auto& request : requests) {
      loadOne(fileNum, request, injectError);
    }
  }

  void loadNFiles(int32_t numFiles, std::vector<int64_t> offsets);

  // Loads a sequence of entries from a number of files. Looks up a
  // number of entries, then loads the ones that nobody else is
  // loading. Stops after loading 'loadBytes' worth of entries. If
  // 'errorEveryNBatches' is non-0, every nth load batch will have a
  // bad read and wil be dropped. The entries of the failed batch read
  // will still be accessed one by one. If 'largeEveryNBatches' is
  // non-0, allocates and frees a single allocation of 'largeBytes'
  // every so many batches. This creates extra memory pressure, as
  // happens when allocating large hash tables in queries.
  void loadLoop(
      int64_t startOffset,
      int64_t loadBytes,
      int32_t errorEveryNBatches = 0,
      int32_t largeEveryNBatches = 0,
      int32_t largeBytes = 0);

  // Calls func on 'numThreads' in parallel.
  template <typename Func>
  void runThreads(int32_t numThreads, Func func) {
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (int32_t i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread([i, func]() { func(i); }));
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  // Checks that the contents are consistent with what is set in
  // initializeContents.
  static void checkContents(const AsyncDataCacheEntry& entry) {
    const auto& alloc = entry.nonContiguousData();
    const int32_t numBytes = entry.size();
    const int64_t expectedSequence = entry.key().fileNum.id() + entry.offset();
    int32_t bytesChecked = sizeof(int64_t);
    for (int32_t i = 0; i < alloc.numRuns(); ++i) {
      const memory::Allocation::PageRun run = alloc.runAt(i);
      int64_t* ptr = reinterpret_cast<int64_t*>(run.data());
      const int32_t numWords =
          memory::AllocationTraits::pageBytes(run.numPages()) / sizeof(void*);
      for (int32_t offset = 0; offset < numWords; ++offset) {
        ASSERT_EQ(ptr[offset], expectedSequence + offset) << fmt::format(
            "{} {} + {}", entry.toString(), expectedSequence + offset, offset);
        bytesChecked += sizeof(int64_t);
        if (bytesChecked >= numBytes) {
          return;
        }
      }
    }
  }

  static void waitForSsdWriteToFinish(const SsdCache* ssdCache) {
    while (ssdCache->writeInProgress()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100)); // NOLINT
    }
  }

  CachePin newEntry(uint64_t offset, int32_t size) {
    folly::SemiFuture<bool> wait(false);
    try {
      RawFileCacheKey key{filenames_[0].id(), offset};
      auto pin = cache_->findOrCreate(key, size, /*contiguous=*/false, &wait);
      EXPECT_FALSE(pin.empty());
      EXPECT_TRUE(pin.entry()->isExclusive());
      pin.entry()->setPrefetch();
      return pin;
    } catch (const VeloxException&) {
      return CachePin();
    };
  }

  folly::IOThreadPoolExecutor* loadExecutor() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> l(mutex);
    if (loadExecutor_ == nullptr) {
      // We have up to 20 threads. Some tests run at max 16 threads so
      // that there are threads left over for SSD background write.
      loadExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(20);
    }
    return loadExecutor_.get();
  }

  folly::IOThreadPoolExecutor* ssdExecutor() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> l(mutex);
    if (ssdExecutor_ == nullptr) {
      // We have up to 20 threads. Some tests run at max 16 threads so
      // that there are threads left over for SSD background write.
      ssdExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(20);
    }
    return ssdExecutor_.get();
  }

  void clearAllocations(std::deque<memory::Allocation>& allocations) {
    while (!allocations.empty()) {
      allocator_->freeNonContiguous(allocations.front());
      allocations.pop_front();
    }
  }

  std::shared_ptr<TempDirectoryPath> tempDirectory_;
  std::unique_ptr<memory::MemoryManager> manager_;
  memory::MemoryAllocator* allocator_;
  std::shared_ptr<AsyncDataCache> cache_;
  std::unique_ptr<cache::test::AsyncDataCacheTestHelper> asyncDataCacheHelper_;
  std::unique_ptr<cache::test::SsdCacheTestHelper> ssdCacheHelper_;
  std::vector<StringIdLease> filenames_;
  std::unique_ptr<folly::IOThreadPoolExecutor> loadExecutor_;
  std::unique_ptr<folly::IOThreadPoolExecutor> ssdExecutor_;
  int32_t numLargeRetries_{0};
  std::atomic_int64_t numPendingLoads_{0};
};

class TestingCoalescedLoad : public CoalescedLoad {
 public:
  TestingCoalescedLoad(
      std::vector<RawFileCacheKey> keys,
      std::vector<int32_t> sizes,
      const std::shared_ptr<AsyncDataCache>& cache,
      bool injectError)
      : CoalescedLoad(std::move(keys), std::move(sizes)),
        cache_(cache),
        injectError_(injectError) {}

  std::vector<CachePin> loadData(bool /*isPrefetch*/) override {
    std::vector<CachePin> pins;
    cache_->makePins(
        keys_,
        [&](int32_t index) { return sizes_[index]; },
        [&](int32_t /*index*/, CachePin pin) {
          pins.push_back(std::move(pin));
        });
    for (const auto& pin : pins) {
      auto& buffer = pin.entry()->nonContiguousData();
      AsyncDataCacheTest::initializeContents(
          pin.entry()->key().offset + pin.entry()->key().fileNum.id(), buffer);
    }
    VELOX_CHECK(!injectError_, "Testing error");
    return pins;
  }

  int64_t size() const override {
    int64_t sum = 0;
    for (auto& request : requests_) {
      sum += request.size;
    }
    return sum;
  }

  bool isSsdLoad() const override {
    return false;
  }

 protected:
  const std::shared_ptr<AsyncDataCache> cache_;
  const std::vector<Request> requests_;
  const bool injectError_{false};
};

class TestingCoalescedSsdLoad : public TestingCoalescedLoad {
 public:
  TestingCoalescedSsdLoad(
      std::vector<RawFileCacheKey> keys,
      std::vector<int32_t> sizes,
      std::vector<SsdPin> ssdPins,
      const std::shared_ptr<AsyncDataCache>& cache,
      bool injectError)
      : TestingCoalescedLoad(
            std::move(keys),
            std::move(sizes),
            cache,
            injectError),
        ssdPins_(std::move(ssdPins)) {}

  std::vector<CachePin> loadData(bool /*isPrefetch*/) override {
    const auto fileNum = keys_[0].fileNum;
    auto& file = cache_->ssdCache()->file(fileNum);
    std::vector<CachePin> pins;
    std::vector<SsdPin> toLoad;
    // We make pins for the new load but leave out the entries that may have
    // been loaded between constructing 'this' and now.
    cache_->makePins(
        keys_,
        [&](int32_t index) { return sizes_[index]; },
        [&](int32_t index, CachePin pin) {
          pins.push_back(std::move(pin));
          toLoad.push_back(std::move(ssdPins_[index]));
        });

    // This is an illustration of discarding SSD entries that could not be read.
    try {
      file.load(toLoad, pins);
      VELOX_CHECK(!injectError_, "Testing error");
    } catch (std::exception&) {
      try {
        for (const auto& ssdPin : toLoad) {
          file.erase(RawFileCacheKey{fileNum, ssdPin.run().offset()});
        }
      } catch (const std::exception&) {
        // Ignore error.
      }
      throw;
    }
    return pins;
  }

  bool isSsdLoad() const override {
    return true;
  }

 private:
  std::vector<SsdPin> ssdPins_;
};

namespace {
int64_t sizeAtOffset(int64_t offset) {
  return offset % 100'000;
}
} // namespace

void AsyncDataCacheTest::loadOne(
    uint64_t fileNum,
    Request& request,
    bool injectError) {
  // Pattern for loading one pin.
  RawFileCacheKey key{fileNum, request.offset};
  for (;;) {
    folly::SemiFuture<bool> loadFuture(false);
    auto pin = cache_->findOrCreate(
        key, request.size, /*contiguous=*/false, &loadFuture);
    if (pin.empty()) {
      // The pin was exclusive on another thread. Wait until it is no longer so
      // and retry.
      auto& exec = folly::QueuedImmediateExecutor::instance();
      std::move(loadFuture).via(&exec).wait();
      continue;
    }
    auto* entry = pin.checkedEntry();
    if (entry->isShared()) {
      // Already in RAM. Check the data.
      checkContents(*entry);
      VELOX_CHECK(!injectError, "Testing error");
      return;
    }
    // We have an uninitialized entry in exclusive mode. We fill it with data
    // and set it to shared. If we release this pin while still in exclusive
    // mode, the entry will be erased.
    if (cache_->ssdCache() != nullptr) {
      auto& ssdFile = cache_->ssdCache()->file(key.fileNum);
      auto ssdPin = ssdFile.find(key);
      if (!ssdPin.empty()) {
        std::vector<CachePin> pins;
        std::vector<SsdPin> ssdPins;
        // pin is exclusive and not copiable, so std::move.
        pins.push_back(std::move(pin));
        ssdPins.push_back(std::move(ssdPin));
        ssdFile.load(ssdPins, pins);
        entry->setExclusiveToShared();
        return;
      }
    }
    // Load from storage.
    initializeContents(
        entry->key().offset + entry->key().fileNum.id(),
        entry->nonContiguousData());
    entry->setExclusiveToShared();
    return;
  }
}

void AsyncDataCacheTest::loadBatch(
    uint64_t fileNum,
    std::vector<Request>& requests,
    bool injectError,
    Semaphore* semaphore) {
  // Pattern for loading a set of buffers from a file: Divide the requested
  // ranges between already loaded and loadable from storage.
  std::vector<Request*> fromStorage;
  std::vector<Request*> fromSsd;
  for (auto& request : requests) {
    RawFileCacheKey key{fileNum, request.offset};
    if (cache_->exists(key)) {
      continue;
    }
    // Schedule a CoalescedLoad with other keys that need loading from the same
    // source.
    if (cache_->ssdCache() != nullptr) {
      auto& file = cache_->ssdCache()->file(key.fileNum);
      request.ssdPin = file.find(key);
      if (!request.ssdPin.empty()) {
        fromSsd.push_back(&request);
        continue;
      }
    }
    fromStorage.push_back(&request);
  }

  // Make CoalescedLoads for pins from different sources.
  if (!fromStorage.empty()) {
    std::vector<RawFileCacheKey> keys;
    std::vector<int32_t> sizes;
    for (auto request : fromStorage) {
      keys.push_back(RawFileCacheKey{fileNum, request->offset});
      sizes.push_back(request->size);
    }
    auto load = std::make_shared<TestingCoalescedLoad>(
        std::move(keys), std::move(sizes), cache_, injectError);
    ++numPendingLoads_;
    loadExecutor()->add([this, load, semaphore]() {
      SCOPE_EXIT {
        --numPendingLoads_;
      };
      try {
        load->loadOrFuture(nullptr);
      } catch (const std::exception&) {
        // Expecting error, ignore.
      };
      if (semaphore) {
        semaphore->release();
      }
    });
  } else if (semaphore) {
    semaphore->release();
  }

  if (!fromSsd.empty()) {
    std::vector<SsdPin> ssdPins;
    std::vector<RawFileCacheKey> keys;
    std::vector<int32_t> sizes;
    for (auto* request : fromSsd) {
      keys.push_back(RawFileCacheKey{fileNum, request->offset});
      sizes.push_back(request->size);
      ssdPins.push_back(std::move(request->ssdPin));
    }
    auto load = std::make_shared<TestingCoalescedSsdLoad>(
        std::move(keys),
        std::move(sizes),
        std::move(ssdPins),
        cache_,
        injectError);
    ++numPendingLoads_;
    loadExecutor()->add([this, load, semaphore]() {
      SCOPE_EXIT {
        --numPendingLoads_;
      };
      try {
        load->loadOrFuture(nullptr);
      } catch (const std::exception&) {
        // Expecting error, ignore.
      };
      if (semaphore) {
        semaphore->release();
      }
    });
  } else if (semaphore) {
    semaphore->release();
  }
}

void AsyncDataCacheTest::loadNFiles(
    int32_t numFiles,
    std::vector<int64_t> offsets) {
  Semaphore semaphore(0);

  std::vector<Request> batch;
  int32_t numLoads = 0;
  for (auto file = 0; file < numFiles; ++file) {
    auto fileNum = filenames_[file].id();
    if (auto instance = CacheTTLController::getInstance()) {
      instance->addOpenFileInfo(fileNum);
    }
    for (auto i = 0; i < offsets.size() - 1; i++) {
      batch.emplace_back(offsets[i], offsets[i + 1] - offsets[i]);
      if (batch.size() == 8 || i == (offsets.size() - 2)) {
        loadBatch(fileNum, batch, false, &semaphore);
        batch.clear();
        numLoads +=
            2; // One TestingCoalescedSsdLoad and one TestingCoalescedLoad.
      }
    }
  }

  for (auto i = 0; i < numLoads; i++) {
    semaphore.acquire();
  }
}

void AsyncDataCacheTest::loadLoop(
    int64_t startOffset,
    int64_t loadBytes,
    int32_t errorEveryNBatches,
    int32_t largeEveryNBatches,
    int32_t largeAllocSize) {
  const int64_t maxOffset =
      std::max<int64_t>(100'000, (startOffset + loadBytes) / filenames_.size());
  int64_t skippedBytes = 0;
  int32_t errorCounter = 0;
  int32_t largeCounter = 0;
  std::vector<Request> batch;
  for (auto i = 0; i < filenames_.size(); ++i) {
    const auto fileNum = filenames_[i].id();
    for (uint64_t offset = 100; offset < maxOffset;
         offset += sizeAtOffset(offset)) {
      const auto size = sizeAtOffset(offset);
      if (skippedBytes < startOffset) {
        skippedBytes += size;
        continue;
      }

      batch.emplace_back(offset, size);
      if (batch.size() >= 8) {
        for (;;) {
          if (largeEveryNBatches > 0 &&
              largeCounter++ % largeEveryNBatches == 0) {
            // Many threads will allocate a single large chunk at the
            // same time. Some are expected to fail. All will
            // eventually succeed because whoever gets the allocation
            // frees it soon and without deadlocking with others..
            memory::ContiguousAllocation large;
            // Explicitly free 'large' on exit. Do not use MemoryPool for that
            // because we test the allocator's limits, not the pool/memory
            // manager  limits.
            auto guard =
                folly::makeGuard([&]() { allocator_->freeContiguous(large); });
            while (!allocator_->allocateContiguous(
                memory::AllocationTraits::numPages(largeAllocSize),
                nullptr,
                large)) {
              ++numLargeRetries_;
            }
            std::this_thread::sleep_for(
                std::chrono::microseconds(2000)); // NOLINT
          }
          const bool injectError = (errorEveryNBatches > 0) &&
              (++errorCounter % errorEveryNBatches == 0);
          loadBatch(fileNum, batch, injectError);
          try {
            checkBatch(fileNum, batch, injectError);
          } catch (std::exception&) {
            continue;
          }
          batch.clear();
          break;
        }
      }
    }
  }
}

TEST_P(AsyncDataCacheTest, pin) {
  constexpr int64_t kSize = 25000;
  initializeCache(1 << 20);
  auto& exec = folly::QueuedImmediateExecutor::instance();

  StringIdLease file(fileIds(), std::string_view("testingfile"));
  uint64_t offset = 1000;
  folly::SemiFuture<bool> wait(false);
  RawFileCacheKey key{file.id(), offset};
  auto pin = cache_->findOrCreate(key, kSize, /*contiguous=*/false, &wait);
  EXPECT_FALSE(pin.empty());
  EXPECT_TRUE(wait.isReady());
  EXPECT_TRUE(pin.entry()->isExclusive());
  pin.entry()->setPrefetch();
  EXPECT_LE(kSize, pin.entry()->nonContiguousData().byteSize());
  EXPECT_LT(0, cache_->incrementPrefetchPages(0));
  auto stats = cache_->refreshStats();
  EXPECT_EQ(1, stats.numExclusive);
  EXPECT_EQ(0, stats.largeSize);

  CachePin otherPin;
  EXPECT_THROW(otherPin = pin, VeloxException);
  EXPECT_TRUE(otherPin.empty());

  // Second reference to an exclusive entry.
  otherPin = cache_->findOrCreate(key, kSize, /*contiguous=*/false, &wait);
  EXPECT_FALSE(wait.isReady());
  EXPECT_TRUE(otherPin.empty());
  bool noLongerExclusive = false;
  std::move(wait).via(&exec).thenValue([&](bool) { noLongerExclusive = true; });
  initializeContents(
      key.fileNum + key.offset, pin.checkedEntry()->nonContiguousData());
  pin.checkedEntry()->setExclusiveToShared();
  pin.clear();
  EXPECT_TRUE(pin.empty());

  EXPECT_TRUE(noLongerExclusive);

  pin = cache_->findOrCreate(key, kSize, /*contiguous=*/false, &wait);
  EXPECT_TRUE(pin.entry()->isShared());
  EXPECT_TRUE(pin.entry()->getAndClearFirstUseFlag());
  EXPECT_FALSE(pin.entry()->getAndClearFirstUseFlag());
  checkContents(*pin.entry());
  otherPin = pin;
  EXPECT_EQ(2, pin.entry()->numPins());
  EXPECT_FALSE(pin.entry()->isPrefetch());
  auto largerPin =
      cache_->findOrCreate(key, kSize * 2, /*contiguous=*/false, &wait);
  // We expect a new uninitialized entry with a larger size to displace the
  // previous one.

  EXPECT_TRUE(largerPin.checkedEntry()->isExclusive());
  largerPin.checkedEntry()->setExclusiveToShared();
  largerPin.clear();
  pin.clear();
  otherPin.clear();
  stats = cache_->refreshStats();
  EXPECT_LE(kSize * 2, stats.largeSize);
  EXPECT_EQ(1, stats.numEntries);
  EXPECT_EQ(0, stats.numShared);
  EXPECT_EQ(0, stats.numExclusive);

  cache_->clear();
  stats = cache_->refreshStats();
  EXPECT_EQ(0, stats.largeSize);
  EXPECT_EQ(0, stats.numEntries);
  EXPECT_EQ(0, cache_->incrementPrefetchPages(0));
}

TEST_P(AsyncDataCacheTest, contiguousPin) {
  initializeCache(1 << 20);
  auto& exec = folly::QueuedImmediateExecutor::instance();

  StringIdLease file(fileIds(), std::string_view("testingfile_contiguous"));

  struct TestParam {
    int64_t size;
    bool expectTiny;
    std::string debugString() const {
      return fmt::format("size {}, expectTiny {}", size, expectTiny);
    }
  };

  std::vector<TestParam> testSettings = {
      {AsyncDataCacheEntry::kTinyDataSize / 2, true},
      {AsyncDataCacheEntry::kTinyDataSize - 1, true},
      {AsyncDataCacheEntry::kTinyDataSize, false},
      {AsyncDataCacheEntry::kTinyDataSize * 4, false},
      {25'000, false},
  };

  uint64_t offset = 1'000;
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    folly::SemiFuture<bool> wait(false);
    RawFileCacheKey key{file.id(), offset};
    offset += testData.size;

    auto pin =
        cache_->findOrCreate(key, testData.size, /*contiguous=*/true, &wait);
    ASSERT_FALSE(pin.empty());
    ASSERT_TRUE(wait.isReady());
    auto* entry = pin.checkedEntry();
    ASSERT_TRUE(entry->isExclusive());
    ASSERT_TRUE(entry->hasContiguousData());
    ASSERT_TRUE(entry->nonContiguousData().empty());
    ASSERT_EQ(entry->contiguousDataSize(), testData.size);

    test::AsyncDataCacheEntryTestHelper entryHelper(entry);
    if (testData.expectTiny) {
      ASSERT_TRUE(entryHelper.isTinyData());
      ASSERT_FALSE(entryHelper.isContiguousData());
    } else {
      ASSERT_FALSE(entryHelper.isTinyData());
      ASSERT_TRUE(entryHelper.isContiguousData());
    }

    ::memset(entry->contiguousData(), 0xCD, testData.size);
    entry->setExclusiveToShared();

    // Verify stats include the contiguous allocation.
    {
      auto stats = cache_->refreshStats();
      ASSERT_EQ(stats.numEntries, 1);
      if (testData.expectTiny) {
        ASSERT_EQ(stats.numTinyEntries, 1);
        ASSERT_EQ(stats.tinySize, testData.size);
      } else {
        ASSERT_EQ(stats.numLargeEntries, 1);
        ASSERT_EQ(stats.largeSize, testData.size);
      }
      if (!testData.expectTiny) {
        ASSERT_EQ(
            cache_->cachedPages(),
            memory::AllocationTraits::numPages(testData.size));
      }
    }

    auto pin2 =
        cache_->findOrCreate(key, testData.size, /*contiguous=*/true, &wait);
    ASSERT_FALSE(pin2.empty());
    ASSERT_TRUE(pin2.checkedEntry()->isShared());
    ASSERT_TRUE(pin2.checkedEntry()->hasContiguousData());
    ASSERT_EQ(
        static_cast<uint8_t>(pin2.checkedEntry()->contiguousData()[0]), 0xCD);
    pin2.clear();

    // Lookup with contiguous=false should return the same contiguous entry.
    auto pin3 =
        cache_->findOrCreate(key, testData.size, /*contiguous=*/false, &wait);
    ASSERT_FALSE(pin3.empty());
    ASSERT_TRUE(pin3.checkedEntry()->isShared());
    ASSERT_TRUE(pin3.checkedEntry()->hasContiguousData());
    pin.clear();
    pin3.clear();

    // Wait-future: concurrent access to an exclusive entry.
    RawFileCacheKey key2{file.id(), offset};
    offset += testData.size;
    auto exclusivePin =
        cache_->findOrCreate(key2, testData.size, /*contiguous=*/true, &wait);
    ASSERT_FALSE(exclusivePin.empty());
    ASSERT_TRUE(exclusivePin.checkedEntry()->isExclusive());

    auto waitingPin =
        cache_->findOrCreate(key2, testData.size, /*contiguous=*/true, &wait);
    ASSERT_TRUE(waitingPin.empty());
    ASSERT_FALSE(wait.isReady());

    bool notified = false;
    std::move(wait).via(&exec).thenValue([&](bool) { notified = true; });
    exclusivePin.checkedEntry()->setExclusiveToShared();
    exclusivePin.clear();
    ASSERT_TRUE(notified);

    auto retryPin =
        cache_->findOrCreate(key2, testData.size, /*contiguous=*/true, &wait);
    ASSERT_FALSE(retryPin.empty());
    ASSERT_TRUE(retryPin.checkedEntry()->isShared());
    retryPin.clear();

    cache_->clear();
    auto stats = cache_->refreshStats();
    ASSERT_EQ(stats.numEntries, 0);
    ASSERT_EQ(stats.largeSize, 0);
    ASSERT_EQ(stats.tinySize, 0);
    ASSERT_EQ(cache_->cachedPages(), 0);
  }
}

TEST_P(AsyncDataCacheTest, nonContiguousPin) {
  initializeCache(1 << 20);

  StringIdLease file(fileIds(), std::string_view("testingfile_noncontiguous"));

  struct TestParam {
    int64_t size;
    bool expectTiny;
    std::string debugString() const {
      return fmt::format("size {}, expectTiny {}", size, expectTiny);
    }
  };

  std::vector<TestParam> testSettings = {
      {AsyncDataCacheEntry::kTinyDataSize / 2, true},
      {AsyncDataCacheEntry::kTinyDataSize - 1, true},
      {AsyncDataCacheEntry::kTinyDataSize, false},
      {AsyncDataCacheEntry::kTinyDataSize * 4, false},
      {25'000, false},
  };

  uint64_t offset = 1'000;
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    folly::SemiFuture<bool> wait(false);
    RawFileCacheKey key{file.id(), offset};
    offset += testData.size;

    auto pin =
        cache_->findOrCreate(key, testData.size, /*contiguous=*/false, &wait);
    ASSERT_FALSE(pin.empty());
    ASSERT_TRUE(wait.isReady());
    auto* entry = pin.checkedEntry();
    ASSERT_TRUE(entry->isExclusive());

    test::AsyncDataCacheEntryTestHelper entryHelper(entry);
    if (testData.expectTiny) {
      ASSERT_TRUE(entryHelper.isTinyData());
      ASSERT_FALSE(entryHelper.isContiguousData());
      ASSERT_TRUE(entry->hasContiguousData());
      ASSERT_TRUE(entry->nonContiguousData().empty());
    } else {
      ASSERT_FALSE(entryHelper.isTinyData());
      ASSERT_FALSE(entryHelper.isContiguousData());
      ASSERT_FALSE(entry->hasContiguousData());
      ASSERT_FALSE(entry->nonContiguousData().empty());
    }

    entry->setExclusiveToShared();

    // Verify stats.
    {
      auto stats = cache_->refreshStats();
      ASSERT_EQ(stats.numEntries, 1);
      if (testData.expectTiny) {
        ASSERT_EQ(stats.numTinyEntries, 1);
        ASSERT_EQ(stats.tinySize, testData.size);
      } else {
        ASSERT_EQ(stats.numLargeEntries, 1);
      }
      if (!testData.expectTiny) {
        ASSERT_EQ(
            cache_->cachedPages(),
            memory::AllocationTraits::numPages(testData.size));
      }
    }

    // Shared hit with same flag.
    auto pin2 =
        cache_->findOrCreate(key, testData.size, /*contiguous=*/false, &wait);
    ASSERT_FALSE(pin2.empty());
    ASSERT_TRUE(pin2.checkedEntry()->isShared());
    pin2.clear();

    // Lookup with contiguous=true should return the same non-contiguous entry.
    auto pin3 =
        cache_->findOrCreate(key, testData.size, /*contiguous=*/true, &wait);
    ASSERT_FALSE(pin3.empty());
    ASSERT_TRUE(pin3.checkedEntry()->isShared());
    if (testData.expectTiny) {
      ASSERT_TRUE(pin3.checkedEntry()->hasContiguousData());
    } else {
      ASSERT_FALSE(pin3.checkedEntry()->hasContiguousData());
    }
    pin.clear();
    pin3.clear();

    cache_->clear();
    auto stats = cache_->refreshStats();
    ASSERT_EQ(stats.numEntries, 0);
    ASSERT_EQ(cache_->cachedPages(), 0);
  }
}

TEST_P(AsyncDataCacheTest, replace) {
  constexpr int64_t kMaxBytes = 64 << 20;
  FLAGS_velox_exception_user_stacktrace_enabled = false;
  initializeCache(kMaxBytes);
  // Load 10x the max size, inject an error every 21 batches.
  loadLoop(0, kMaxBytes * 10, 21);
  if (loadExecutor_ != nullptr) {
    loadExecutor_->join();
  }
  auto stats = cache_->refreshStats();
  EXPECT_LT(0, stats.numHit);
  EXPECT_LT(0, stats.hitBytes);
  EXPECT_LT(0, stats.numEvict);
  EXPECT_GE(
      kMaxBytes / memory::AllocationTraits::kPageSize, cache_->cachedPages());
}

TEST_P(AsyncDataCacheTest, evictAccounting) {
  constexpr int64_t kMaxBytes = 64 << 20;
  FLAGS_velox_exception_user_stacktrace_enabled = false;
  initializeCache(kMaxBytes);
  auto pool = manager_->addLeafPool("test");

  // We make allocations that we exchange for larger ones later. This will evict
  // cache. We check that the evictions are not counted on the pool even if they
  // occur as a result of action on the pool.
  memory::Allocation allocation;
  memory::ContiguousAllocation large;
  pool->allocateNonContiguous(1200, allocation);
  pool->allocateContiguous(1200, large);
  EXPECT_EQ(memory::AllocationTraits::pageBytes(2400), pool->usedBytes());
  loadLoop(0, kMaxBytes * 1.1);
  waitForPendingLoads();
  pool->allocateNonContiguous(2400, allocation);
  pool->allocateContiguous(2400, large);
  EXPECT_EQ(memory::AllocationTraits::pageBytes(4800), pool->usedBytes());
  auto stats = cache_->refreshStats();
  EXPECT_LT(0, stats.numEvict);
}

TEST_P(AsyncDataCacheTest, largeEvict) {
  constexpr int64_t kMaxBytes = 256 << 20;
  constexpr int32_t kNumThreads = 24;
  FLAGS_velox_exception_user_stacktrace_enabled = false;
  initializeCache(kMaxBytes);
  // Load 10x the max size, inject an allocation of 1/8 the capacity every 4
  // batches.
  runThreads(kNumThreads, [&](int32_t /*i*/) {
    loadLoop(0, kMaxBytes * 1.2, 0, 1, kMaxBytes / 4);
  });
  if (loadExecutor_ != nullptr) {
    loadExecutor_->join();
  }
  auto stats = cache_->refreshStats();
  EXPECT_LT(0, stats.numEvict);
  EXPECT_GE(
      kMaxBytes / memory::AllocationTraits::kPageSize, cache_->cachedPages());
  LOG(INFO) << "Reties after failed evict: " << numLargeRetries_;
}

TEST_P(AsyncDataCacheTest, outOfCapacity) {
  const int64_t kMaxBytes = 64
      << 20; // 64MB as MmapAllocator's min size is 64MB
  const int32_t kSize = 16 << 10;
  const int32_t kSizeInPages = memory::AllocationTraits::numPages(kSize);
  std::deque<CachePin> pins;
  std::deque<memory::Allocation> allocations;
  initializeCache(kMaxBytes);
  // We pin 2 16K entries and unpin 1. Eventually the whole capacity
  // is pinned and we fail making a ew entry.

  uint64_t offset = 0;
  for (;;) {
    pins.push_back(newEntry(++offset, kSize));
    pins.push_back(newEntry(++offset, kSize));
    if (pins.back().empty()) {
      break;
    }
    pins.pop_front();
  }
  memory::Allocation allocation;
  ASSERT_FALSE(allocator_->allocateNonContiguous(kSizeInPages, allocation));
  // One 4 page entry below the max size of 4K 4 page entries in 16MB of
  // capacity.
  ASSERT_EQ(16384, cache_->cachedPages());
  ASSERT_EQ(16384, cache_->incrementPrefetchPages(0));
  pins.clear();

  // We allocate the full capacity and expect the cache entries to go.
  for (;;) {
    if (!allocator_->allocateNonContiguous(kSizeInPages, allocation)) {
      break;
    }
    allocations.push_back(std::move(allocation));
  }
  EXPECT_EQ(0, cache_->cachedPages());
  EXPECT_EQ(0, cache_->incrementPrefetchPages(0));
  EXPECT_EQ(16384, allocator_->numAllocated());
  clearAllocations(allocations);
}

namespace {
// Cuts off the last 1/10th of file at 'path'.
void corruptFile(const std::string& path) {
  const auto fd = ::open(path.c_str(), O_WRONLY);
  const auto size = ::lseek(fd, 0, SEEK_END);
  const auto rc = ftruncate(fd, size / 10 * 9);
  ASSERT_EQ(rc, 0);
}
} // namespace

TEST_P(AsyncDataCacheTest, DISABLED_ssd) {
#ifdef TSAN_BUILD
  // NOTE: scale down the test data set to prevent tsan tester from running out
  // of memory.
  constexpr uint64_t kRamBytes = 16 << 20;
  constexpr uint64_t kSsdBytes = 256UL << 20;
#else
  constexpr uint64_t kRamBytes = 32 << 20;
  constexpr uint64_t kSsdBytes = 512UL << 20;
#endif
  FLAGS_velox_exception_user_stacktrace_enabled = false;
  initializeCache(kRamBytes, kSsdBytes);
  cache_->setVerifyHook(
      [&](const AsyncDataCacheEntry& entry) { checkContents(entry); });

  // Read back all writes. This increases the chance of writes falling behind
  // new entry creation.
  FLAGS_velox_ssd_verify_write = true;

  // We read kSsdBytes worth of data on 16 threads. The same data will be hit by
  // all threads. The expectation is that most of the data ends up on SSD. All
  // data may not get written if reading is faster than writing. Error out once
  // every 11 load batches.
  //
  // NOTE: loadExecutor() must have more threads so that background write does
  // not wait for the workload.
  runThreads(16, [&](int32_t /*i*/) { loadLoop(0, kSsdBytes, 11); });
  LOG(INFO) << "Stats after first pass: " << cache_->toString();
  auto ssdStats = cache_->ssdCache()->stats();
  ASSERT_LE(kRamBytes, ssdStats.bytesWritten);

  // We allow writes to proceed faster.
  FLAGS_velox_ssd_verify_write = false;

  // We read the data back. The verify hook checks correct values. Error every
  // 13 batch loads.
  runThreads(16, [&](int32_t /*i*/) { loadLoop(0, kSsdBytes, 13); });
  LOG(INFO) << "Stats after second pass:" << cache_->toString();
  ssdStats = cache_->ssdCache()->stats();
  ASSERT_LE(kRamBytes, ssdStats.bytesRead);

  // We re-read the second half and add another half capacity of new entries. We
  // expect some of the oldest entries to get evicted. Error every 17 batch
  // loads.
  runThreads(
      16, [&](int32_t /*i*/) { loadLoop(kSsdBytes / 2, kSsdBytes * 1.5, 17); });
  LOG(INFO) << "Stats after third pass:" << cache_->toString();

  // Wait for writes to finish and make a checkpoint.
  cache_->ssdCache()->shutdown();
  auto ssdStatsAfterShutdown = cache_->ssdCache()->stats();
  ASSERT_GT(ssdStatsAfterShutdown.bytesWritten, ssdStats.bytesWritten);
  ASSERT_GT(ssdStatsAfterShutdown.bytesRead, ssdStats.bytesRead);

  // Check that no pins are leaked.
  ASSERT_EQ(ssdStatsAfterShutdown.numPins, 0);

  auto ramStats = cache_->refreshStats();
  ASSERT_EQ(ramStats.numShared, 0);
  ASSERT_EQ(ramStats.numExclusive, 0);

  cache_->ssdCache()->clear();
  // We cut the tail off one of the cache shards.
  corruptFile(fmt::format("{}/cache0.cpt", tempDirectory_->getPath()));
  // We open the cache from checkpoint. Reading checks the data integrity, here
  // we check that more data was read than written.
  initializeCache(kRamBytes, kSsdBytes);
  runThreads(16, [&](int32_t /*i*/) {
    loadLoop(kSsdBytes / 2, kSsdBytes * 1.5, 113);
  });
  LOG(INFO) << "State after starting 3/4 shards from checkpoint: "
            << cache_->toString();
  const auto ssdStatsFromCP = cache_->ssdCache()->stats();
  ASSERT_EQ(ssdStatsFromCP.readCheckpointErrors, 1);
}

TEST_P(AsyncDataCacheTest, invalidSsdPath) {
  auto testPath = "hdfs:/test/prefix_";
  uint64_t ssdBytes = 256UL << 20;
  SsdCache::Config config(testPath, ssdBytes, 4, ssdExecutor(), ssdBytes / 20);
  VELOX_ASSERT_THROW(
      SsdCache(config),
      fmt::format(
          "Ssd path '{}' does not start with '/' that points to local file system.",
          testPath));
}

TEST_P(AsyncDataCacheTest, cacheStats) {
  CacheStats stats;
  stats.tinySize = 234;
  stats.largeSize = 1024;
  stats.tinyPadding = 23;
  stats.largePadding = 1344;
  stats.numEntries = 100;
  stats.numExclusive = 20;
  stats.numShared = 30;
  stats.sharedPinnedBytes = 10 << 20;
  stats.exclusivePinnedBytes = 10 << 20;
  stats.numEmptyEntries = 20;
  stats.numPrefetch = 30;
  stats.prefetchBytes = 100;
  stats.numHit = 46;
  stats.hitBytes = 1374;
  stats.numNew = 2041;
  stats.numEvict = 463;
  stats.numEvictChecks = 348;
  stats.numWaitExclusive = 244;
  stats.numAgedOut = 10;
  stats.allocClocks = 1320;
  stats.sumEvictScore = 123;
  stats.numStales = 100;
  ASSERT_EQ(
      stats.toString(),
      "Cache size: 2.56KB tinySize: 257B large size: 2.31KB\n"
      "Cache entries: 100 read pins: 30 write pins: 20 pinned shared: 10.00MB pinned exclusive: 10.00MB\n"
      " num write wait: 244 empty entries: 20\n"
      "Cache access miss: 2041 hit: 46 hit bytes: 1.34KB eviction: 463 savable eviction: 0 eviction checks: 348 aged out: 10 stales: 100\n"
      "Prefetch entries: 30 bytes: 100B\n"
      "Alloc Megaclocks 0");

  CacheStats statsDelta = stats - stats;
  ASSERT_EQ(statsDelta.tinySize, 0);
  ASSERT_EQ(statsDelta.largeSize, 0);
  ASSERT_EQ(statsDelta.tinyPadding, 0);
  ASSERT_EQ(statsDelta.largePadding, 0);
  ASSERT_EQ(statsDelta.numEntries, 0);
  ASSERT_EQ(statsDelta.numExclusive, 0);
  ASSERT_EQ(statsDelta.numShared, 0);
  ASSERT_EQ(statsDelta.sharedPinnedBytes, 0);
  ASSERT_EQ(statsDelta.exclusivePinnedBytes, 0);
  ASSERT_EQ(statsDelta.numEmptyEntries, 0);
  ASSERT_EQ(statsDelta.numPrefetch, 0);
  ASSERT_EQ(statsDelta.prefetchBytes, 0);
  ASSERT_EQ(statsDelta.numHit, 0);
  ASSERT_EQ(statsDelta.hitBytes, 0);
  ASSERT_EQ(statsDelta.numNew, 0);
  ASSERT_EQ(statsDelta.numEvict, 0);
  ASSERT_EQ(statsDelta.numEvictChecks, 0);
  ASSERT_EQ(statsDelta.numWaitExclusive, 0);
  ASSERT_EQ(statsDelta.numAgedOut, 0);
  ASSERT_EQ(statsDelta.allocClocks, 0);
  ASSERT_EQ(statsDelta.sumEvictScore, 0);
  ASSERT_EQ(statsDelta.numStales, 0);

  constexpr uint64_t kRamBytes = 32 << 20;
  constexpr uint64_t kSsdBytes = 512UL << 20;
  initializeCache(kRamBytes, kSsdBytes);
  const std::string expectedDetailedCacheOutput =
      "AsyncDataCache:\n"
      "Cache size: 0B tinySize: 0B large size: 0B\n"
      "Cache entries: 0 read pins: 0 write pins: 0 pinned shared: 0B pinned exclusive: 0B\n"
      " num write wait: 0 empty entries: 0\n"
      "Cache access miss: 0 hit: 0 hit bytes: 0B eviction: 0 savable eviction: 0 eviction checks: 0 aged out: 0 stales: 0\n"
      "Prefetch entries: 0 bytes: 0B\n"
      "Alloc Megaclocks 0\n"
      "Allocated pages: 0 cached pages: 0\n"
      "Backing: Memory Allocator[MMAP total capacity 64.00MB free capacity 64.00MB allocated pages 0 mapped pages 0 external mapped pages 0\n"
      "[size 1: 0(0MB) allocated 0 mapped]\n"
      "[size 2: 0(0MB) allocated 0 mapped]\n"
      "[size 4: 0(0MB) allocated 0 mapped]\n"
      "[size 8: 0(0MB) allocated 0 mapped]\n"
      "[size 16: 0(0MB) allocated 0 mapped]\n"
      "[size 32: 0(0MB) allocated 0 mapped]\n"
      "[size 64: 0(0MB) allocated 0 mapped]\n"
      "[size 128: 0(0MB) allocated 0 mapped]\n"
      "[size 256: 0(0MB) allocated 0 mapped]\n"
      "]\n"
      "SSD: Ssd cache IO: Write 0B read 0B Size 512.00MB Occupied 0B 0K entries.\n"
      "GroupStats: <dummy FileGroupStats>";
  ASSERT_EQ(cache_->toString(), expectedDetailedCacheOutput);
  ASSERT_EQ(cache_->toString(true), expectedDetailedCacheOutput);
  const std::string expectedShortCacheOutput =
      "AsyncDataCache:\n"
      "Cache size: 0B tinySize: 0B large size: 0B\n"
      "Cache entries: 0 read pins: 0 write pins: 0 pinned shared: 0B pinned exclusive: 0B\n"
      " num write wait: 0 empty entries: 0\n"
      "Cache access miss: 0 hit: 0 hit bytes: 0B eviction: 0 savable eviction: 0 eviction checks: 0 aged out: 0 stales: 0\n"
      "Prefetch entries: 0 bytes: 0B\n"
      "Alloc Megaclocks 0\n"
      "Allocated pages: 0 cached pages: 0\n";
  ASSERT_EQ(cache_->toString(false), expectedShortCacheOutput);
}

TEST_P(AsyncDataCacheTest, cacheStatsWithSsd) {
  CacheStats stats;
  stats.numHit = 234;
  stats.numEvict = 1024;
  stats.ssdStats = std::make_shared<SsdCacheStats>();
  stats.ssdStats->bytesWritten = 1;
  stats.ssdStats->bytesRead = 1;

  const CacheStats otherStats;
  const CacheStats deltaStats = stats - otherStats;
  ASSERT_EQ(deltaStats.numHit, 234);
  ASSERT_EQ(deltaStats.numEvict, 1024);
  ASSERT_TRUE(deltaStats.ssdStats != nullptr);
  ASSERT_EQ(deltaStats.ssdStats->bytesWritten, 1);
  ASSERT_EQ(deltaStats.ssdStats->bytesRead, 1);
  const std::string expectedDeltaCacheStats =
      "Cache size: 0B tinySize: 0B large size: 0B\nCache entries: 0 read pins: 0 write pins: 0 pinned shared: 0B pinned exclusive: 0B\n num write wait: 0 empty entries: 0\nCache access miss: 0 hit: 234 hit bytes: 0B eviction: 1024 savable eviction: 0 eviction checks: 0 aged out: 0 stales: 0\nPrefetch entries: 0 bytes: 0B\nAlloc Megaclocks 0";
  ASSERT_EQ(deltaStats.toString(), expectedDeltaCacheStats);
}

TEST_P(AsyncDataCacheTest, staleEntry) {
  constexpr uint64_t kRamBytes = 1UL << 30;
  // Disable SSD cache to test in-memory cache stale entry only.
  initializeCache(kRamBytes, 0, 0);
  StringIdLease file(fileIds(), std::string_view("staleEntry"));
  const uint64_t offset = 1000;
  const uint64_t size = 200;
  folly::SemiFuture<bool> wait(false);
  RawFileCacheKey key{file.id(), offset};
  auto pin = cache_->findOrCreate(key, size, /*contiguous=*/false, &wait);
  ASSERT_FALSE(pin.empty());
  ASSERT_TRUE(wait.isReady());
  ASSERT_TRUE(pin.entry()->isExclusive());
  pin.entry()->setExclusiveToShared();
  ASSERT_FALSE(pin.entry()->isExclusive());
  auto stats = cache_->refreshStats();
  ASSERT_EQ(stats.numStales, 0);
  ASSERT_EQ(stats.numEntries, 1);
  ASSERT_EQ(stats.numHit, 0);

  auto validPin = cache_->findOrCreate(key, size, /*contiguous=*/false, &wait);
  ASSERT_FALSE(validPin.empty());
  ASSERT_TRUE(wait.isReady());
  ASSERT_FALSE(validPin.entry()->isExclusive());
  stats = cache_->refreshStats();
  ASSERT_EQ(stats.numStales, 0);
  ASSERT_EQ(stats.numEntries, 1);
  ASSERT_EQ(stats.numHit, 1);

  // Stale cache access with large cache size.
  auto stalePin =
      cache_->findOrCreate(key, 2 * size, /*contiguous=*/false, &wait);
  ASSERT_FALSE(stalePin.empty());
  ASSERT_TRUE(wait.isReady());
  ASSERT_TRUE(stalePin.entry()->isExclusive());
  stalePin.entry()->setExclusiveToShared();
  stats = cache_->refreshStats();
  ASSERT_EQ(stats.numStales, 1);
  ASSERT_EQ(stats.numEntries, 1);
  ASSERT_EQ(stats.numHit, 1);
}

TEST_P(AsyncDataCacheTest, shrinkCache) {
  constexpr uint64_t kRamBytes = 128UL << 20;
  constexpr uint64_t kSsdBytes = 512UL << 20;
  constexpr int kTinyDataSize = AsyncDataCacheEntry::kTinyDataSize - 1;
  const int numEntries{10};
  constexpr int kLargeDataSize = kTinyDataSize * 2;
  ASSERT_LE(numEntries * (kTinyDataSize + kLargeDataSize), kRamBytes);

  std::vector<RawFileCacheKey> tinyCacheKeys;
  std::vector<RawFileCacheKey> largeCacheKeys;
  std::vector<StringIdLease> fileLeases;
  for (int i = 0; i < numEntries; ++i) {
    fileLeases.emplace_back(
        StringIdLease(fileIds(), fmt::format("shrinkCacheFile{}", i)));
    tinyCacheKeys.emplace_back(RawFileCacheKey{fileLeases.back().id(), 0});
    largeCacheKeys.emplace_back(
        RawFileCacheKey{fileLeases.back().id(), kLargeDataSize});
  }

  struct {
    bool shrinkAll;
    bool hasSsd;
    bool releaseAll;

    std::string debugString() const {
      return fmt::format(
          "shrinkAll {}, hasSsd {}, releaseAll {}",
          shrinkAll,
          hasSsd,
          releaseAll);
    }
  } testSettings[] = {
      {true, false, false},
      {true, true, false},
      {true, false, true},
      {true, true, true},
      {false, false, true},
      {false, true, true},
      {false, false, false},
      {false, true, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    initializeCache(kRamBytes, testData.hasSsd ? kSsdBytes : 0);
    std::vector<CachePin> pins;
    for (int i = 0; i < numEntries; ++i) {
      auto tinyPin = cache_->findOrCreate(tinyCacheKeys[i], kTinyDataSize);
      ASSERT_FALSE(tinyPin.empty());
      ASSERT_TRUE(tinyPin.entry()->hasContiguousData());
      ASSERT_TRUE(tinyPin.entry()->nonContiguousData().empty());
      ASSERT_FALSE(tinyPin.entry()->isPrefetch());
      ASSERT_FALSE(tinyPin.entry()->ssdSaveable());
      pins.push_back(std::move(tinyPin));
      auto largePin = cache_->findOrCreate(largeCacheKeys[i], kLargeDataSize);
      ASSERT_FALSE(largePin.entry()->hasContiguousData());
      ASSERT_FALSE(largePin.entry()->nonContiguousData().empty());
      ASSERT_FALSE(largePin.entry()->isPrefetch());
      ASSERT_FALSE(largePin.entry()->ssdSaveable());
      pins.push_back(std::move(largePin));
    }
    auto stats = cache_->refreshStats();
    ASSERT_EQ(stats.numEntries, 0);
    ASSERT_EQ(stats.numEmptyEntries, 0);
    ASSERT_EQ(stats.numExclusive, numEntries * 2);
    ASSERT_EQ(stats.numEvict, 0);
    ASSERT_EQ(stats.numHit, 0);
    ASSERT_EQ(stats.tinySize, 0);
    ASSERT_EQ(stats.largeSize, 0);
    ASSERT_EQ(stats.sharedPinnedBytes, 0);
    ASSERT_GE(
        stats.exclusivePinnedBytes,
        (kTinyDataSize + kLargeDataSize) * numEntries);
    ASSERT_EQ(stats.prefetchBytes, 0);
    ASSERT_EQ(stats.numPrefetch, 0);

    const auto numMappedPagesBeforeShrink = allocator_->numMapped();
    ASSERT_GT(numMappedPagesBeforeShrink, 0);

    // Everything gets pinged in memory.
    VELOX_ASSERT_THROW(cache_->shrink(0), "");
    ASSERT_EQ(cache_->shrink(testData.shrinkAll ? kRamBytes : 1), 0);

    if (!testData.releaseAll) {
      for (auto& pin : pins) {
        pin.entry()->setExclusiveToShared();
      }
      pins.clear();
      if (testData.shrinkAll) {
        ASSERT_GE(
            cache_->shrink(kRamBytes),
            (kLargeDataSize + kTinyDataSize) * numEntries);
      } else {
        ASSERT_GE(cache_->shrink(2 * kTinyDataSize), kTinyDataSize);
      }
    } else {
      pins.clear();
      // We expect everything has been freed.
      ASSERT_EQ(
          cache_->shrink(testData.shrinkAll ? kRamBytes : 2 * kTinyDataSize),
          0);
    }
    stats = cache_->refreshStats();
    const auto numMappedPagesAfterShrink = allocator_->numMapped();
    if (testData.shrinkAll || testData.releaseAll) {
      ASSERT_EQ(stats.numEntries, 0);
      ASSERT_EQ(stats.numEmptyEntries, 2 * numEntries);
      ASSERT_EQ(stats.numExclusive, 0);
      ASSERT_EQ(stats.numEvict, 2 * numEntries);
      ASSERT_EQ(stats.numHit, 0);
      ASSERT_EQ(stats.tinySize, 0);
      ASSERT_EQ(stats.largeSize, 0);
      ASSERT_EQ(stats.sharedPinnedBytes, 0);
      ASSERT_GE(stats.exclusivePinnedBytes, 0);
      ASSERT_EQ(stats.prefetchBytes, 0);
      ASSERT_EQ(stats.numPrefetch, 0);
      if (testData.shrinkAll) {
        ASSERT_EQ(numMappedPagesAfterShrink, 0);
      } else {
        ASSERT_LT(numMappedPagesAfterShrink, numMappedPagesBeforeShrink);
      }
    } else {
      ASSERT_LT(stats.numEntries, 2 * numEntries);
      ASSERT_GT(stats.numEntries, 0);
      ASSERT_GE(stats.numEmptyEntries, 1);
      ASSERT_EQ(stats.numExclusive, 0);
      ASSERT_GE(stats.numEvict, 1);
      ASSERT_EQ(stats.numHit, 0);
      ASSERT_GT(stats.tinySize, 0);
      ASSERT_GT(stats.largeSize, 0);
      ASSERT_EQ(stats.sharedPinnedBytes, 0);
      ASSERT_GE(stats.exclusivePinnedBytes, 0);
      ASSERT_EQ(stats.prefetchBytes, 0);
      ASSERT_EQ(stats.numPrefetch, 0);
      ASSERT_LT(numMappedPagesAfterShrink, numMappedPagesBeforeShrink);
    }
  }
}

TEST_P(AsyncDataCacheTest, shutdown) {
  constexpr uint64_t kRamBytes = 16 << 20;
  constexpr uint64_t kSsdBytes = 64UL << 20;

  initializeMemoryManager(kRamBytes);

  for (const auto asyncShutdown : {false, true}) {
    SCOPED_TRACE(fmt::format("asyncShutdown {}", asyncShutdown));
    // Initialize cache with a big checkpointIntervalBytes, giving eviction log
    // chance to survive.
    initializeCache(
        kRamBytes,
        kSsdBytes,
        /*checkpointIntervalBytes=*/(1ULL << 30) * kNumSsdShards);
    ASSERT_EQ(cache_->ssdCache()->stats().openCheckpointErrors, 4);

    // Write large amount of data, making sure eviction is triggered and the log
    // file is not zero.
    loadLoop(0, 16 * kSsdBytes);
    ASSERT_EQ(cache_->ssdCache()->stats().checkpointsWritten, 0);
    ASSERT_GT(cache_->ssdCache()->stats().regionsEvicted, 0);
    ASSERT_GT(ssdCacheHelper_->totalEvictionLogFilesSize(), 0);

    // Shutdown cache.
    if (!asyncShutdown) {
      waitForSsdWriteToFinish(cache_->ssdCache());
    }
    // NOTE: we need to wait for async load to complete before shutdown as async
    // data cache doesn't handle the cache access after the cache shutdown.
    if (loadExecutor_ != nullptr) {
      loadExecutor_->join();
      loadExecutor_.reset();
    }
    const uint64_t bytesWrittenBeforeShutdown =
        cache_->ssdCache()->stats().bytesWritten;
    cache_->ssdCache()->shutdown();
    const uint64_t bytesWrittenAfterShutdown =
        cache_->ssdCache()->stats().bytesWritten;

    if (asyncShutdown) {
      // The written bytes before shutdown is not larger than before shutdown.
      ASSERT_LE(bytesWrittenBeforeShutdown, bytesWrittenAfterShutdown);
    } else {
      // No new data has been written after shutdown.
      ASSERT_EQ(bytesWrittenBeforeShutdown, bytesWrittenAfterShutdown);
    }
    // Eviction log files have been truncated.
    ASSERT_EQ(ssdCacheHelper_->totalEvictionLogFilesSize(), 0);

    // Shutdown again making sure no issue is triggered.
    cache_->ssdCache()->shutdown();

    // New cache write attempt is blocked and triggers exception.
    VELOX_ASSERT_THROW(
        cache_->ssdCache()->startWrite(),
        "Unexpected write after SSD cache has been shutdown");

    // Re-initialize cache.
    cache_->ssdCache()->clear();
    initializeCache(kRamBytes, kSsdBytes, kSsdBytes * 10);
    // Checkpoint files are intact and readable.
    ASSERT_EQ(cache_->ssdCache()->stats().openCheckpointErrors, 0);
    ASSERT_EQ(cache_->ssdCache()->stats().readCheckpointErrors, 0);
    ssdCacheHelper_->deleteCheckpoints();
  }
}

DEBUG_ONLY_TEST_P(AsyncDataCacheTest, shrinkWithSsdWrite) {
  constexpr uint64_t kRamBytes = 128UL << 20;
  constexpr uint64_t kSsdBytes = 512UL << 20;
  constexpr int kDataSize = 4096;
  initializeCache(kRamBytes, kSsdBytes);
  const int numEntries{10};
  std::vector<CachePin> cachePins;
  uint64_t offset = 0;
  for (int i = 0; i < numEntries; ++i) {
    cachePins.push_back(newEntry(offset, kDataSize));
    offset += kDataSize;
  }
  for (auto& pin : cachePins) {
    pin.entry()->setExclusiveToShared();
  }

  std::atomic_bool writeStartFlag{false};
  folly::EventCount writeStartWait;
  std::atomic_bool writeWaitFlag{true};
  folly::EventCount writeWait;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cache::SsdCache::write",
      std::function<void(const SsdCache*)>(([&](const SsdCache* cache) {
        writeStartFlag = true;
        writeStartWait.notifyAll();
        writeWait.await([&]() { return !writeWaitFlag.load(); });
      })));

  // Starts a write thread running at background.
  std::thread ssdWriteThread([&]() {
    ASSERT_TRUE(cache_->ssdCache()->startWrite());
    cache_->saveToSsd();
  });

  // Wait for the write thread to start, and block it while do cache shrink.
  writeStartWait.await([&]() { return writeStartFlag.load(); });
  ASSERT_TRUE(cache_->ssdCache()->writeInProgress());

  cachePins.clear();
  cache_->shrink(kRamBytes);
  auto stats = cache_->refreshStats();
  // Shrink can only reclaim some entries but not all as some of the cache
  // entries have been pickup for ssd write which is not evictable.
  ASSERT_LT(stats.numEntries, numEntries);
  ASSERT_GT(stats.numEmptyEntries, 0);
  ASSERT_GT(stats.numEvict, 0);
  ASSERT_GT(stats.numShared, 0);
  ASSERT_EQ(stats.numExclusive, 0);
  ASSERT_EQ(stats.numWaitExclusive, 0);

  // Wait for write to complete.
  writeWaitFlag = false;
  writeWait.notifyAll();
  ssdWriteThread.join();
  waitForSsdWriteToFinish(cache_->ssdCache());

  stats = cache_->refreshStats();
  ASSERT_GT(stats.numEntries, stats.numEmptyEntries);

  ASSERT_GT(cache_->shrink(kRamBytes), 0);
  stats = cache_->refreshStats();
  ASSERT_EQ(stats.numEntries, 0);
  ASSERT_EQ(stats.numEmptyEntries, numEntries);
}

DEBUG_ONLY_TEST_P(AsyncDataCacheTest, ttl) {
  constexpr uint64_t kRamBytes = 32 << 20;
  constexpr uint64_t kSsdBytes = 128UL << 20;

  initializeCache(kRamBytes, kSsdBytes);
  CacheTTLController::create(*cache_);

  std::vector<int64_t> offsets(32);
  std::generate(offsets.begin(), offsets.end(), [&, n = 0]() mutable {
    return n += (kRamBytes / kNumFiles / offsets.size());
  });

  ScopedTestTime stt;
  auto loadTime1 = getCurrentTimeSec();
  auto loadTime2 = loadTime1 + 100;

  stt.setCurrentTestTimeSec(loadTime1);
  loadNFiles(filenames_.size() * 2 / 3, offsets);
  waitForSsdWriteToFinish(cache_->ssdCache());
  auto statsT1 = cache_->refreshStats();

  stt.setCurrentTestTimeSec(loadTime2);
  loadNFiles(filenames_.size(), offsets);

  runThreads(2, [&](int32_t /*i*/) {
    CacheTTLController::getInstance()->applyTTL(
        getCurrentTimeSec() - loadTime1 - 2);
  });

  auto statsTtl = cache_->refreshStats();
  EXPECT_EQ(statsTtl.numAgedOut, statsT1.numEntries);
  EXPECT_EQ(statsTtl.ssdStats->entriesAgedOut, statsT1.ssdStats->entriesCached);
}

TEST_P(AsyncDataCacheTest, makeEvictable) {
  constexpr uint64_t kRamBytes = 128UL << 20;
  constexpr uint64_t kSsdBytes = 512UL << 20;
  constexpr int kDataSize = 4096;
  for (const bool evictable : {false, true}) {
    SCOPED_TRACE(fmt::format("evictable: {}", evictable));
    initializeCache(kRamBytes, kSsdBytes);
    const int numEntries{10};
    std::vector<CachePin> cachePins;
    uint64_t offset = 0;
    for (int i = 0; i < numEntries; ++i) {
      cachePins.push_back(newEntry(offset, kDataSize));
      offset += kDataSize;
    }
    for (auto& pin : cachePins) {
      pin.entry()->setExclusiveToShared(!evictable);
    }
    if (evictable) {
      std::vector<RawFileCacheKey> keys;
      keys.reserve(cachePins.size());
      for (const auto& pin : cachePins) {
        keys.push_back(
            RawFileCacheKey{
                pin.checkedEntry()->key().fileNum.id(),
                pin.checkedEntry()->key().offset});
      }
      cachePins.clear();
      for (const auto& key : keys) {
        cache_->makeEvictable(key);
      }
    }
    const auto cacheEntries = asyncDataCacheHelper_->cacheEntries();
    for (const auto& cacheEntry : cacheEntries) {
      const auto cacheEntryHelper =
          cache::test::AsyncDataCacheEntryTestHelper(cacheEntry);
      ASSERT_EQ(cacheEntry->ssdSaveable(), !evictable);
      ASSERT_EQ(cacheEntryHelper.accessStats().numUses, 0);
      if (evictable) {
        ASSERT_EQ(cacheEntryHelper.accessStats().lastUse, 0);
      } else {
        ASSERT_NE(cacheEntryHelper.accessStats().lastUse, 0);
      }
    }
    auto* ssdCache = cache_->ssdCache();
    if (ssdCache == nullptr) {
      continue;
    }
    ssdCache->waitForWriteToFinish();
    if (evictable) {
      ASSERT_EQ(ssdCache->stats().entriesCached, 0);
    } else {
      if (asyncDataCacheHelper_->ssdSavable() == 0) {
        ASSERT_GT(ssdCache->stats().entriesCached, 0);
      } else {
        // Ssd write only gets triggered after a certain ssd space usage
        // threshold.
        ASSERT_GE(ssdCache->stats().entriesCached, 0);
      }
    }
  }
}

TEST_P(AsyncDataCacheTest, ssdWriteOptions) {
  constexpr uint64_t kRamBytes = 16UL << 20; // 16 MB
  constexpr uint64_t kSsdBytes = 64UL << 20; // 64 MB

  // Test if ssd write behavior with different settings.
  struct {
    double maxWriteRatio;
    double ssdSavableRatio;
    int32_t minSsdSavableBytes;
    bool expectedSaveToSsd;

    std::string debugString() const {
      return fmt::format(
          "maxWriteRatio {}, ssdSavableRatio {}, minSsdSavableBytes {}, expectedSaveToSsd {}",
          maxWriteRatio,
          ssdSavableRatio,
          minSsdSavableBytes,
          expectedSaveToSsd);
    }
  } testSettings[] = {
      {0.8, 0.95, 32UL << 20, false},
      {0.8, 0.95, 4UL << 20, false},
      {0.8, 0.3, 32UL << 20, false},
      {0.8, 0.3, 4UL << 20, true},
      {0.0, 0.8, 0, true}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    initializeCache(
        kRamBytes,
        kSsdBytes,
        0,
        true,
        {testData.maxWriteRatio,
         testData.ssdSavableRatio,
         testData.minSsdSavableBytes});
    // Load data half of the in-memory capacity.
    loadLoop(0, kRamBytes / 2);
    waitForPendingLoads();
    auto stats = cache_->refreshStats();
    if (testData.expectedSaveToSsd) {
      EXPECT_GT(stats.ssdStats->entriesWritten, 0);
    } else {
      EXPECT_EQ(stats.ssdStats->entriesWritten, 0);
    }
    if (testData.maxWriteRatio < 0.0001) {
      // SSD cache write stops right after the first entry in each shard.
      // Only a few entries can be written.
      EXPECT_LE(stats.ssdStats->entriesWritten, 20);
    }
  }
}

TEST_P(AsyncDataCacheTest, ssdFlushThresholdBytes) {
  constexpr uint64_t kRamBytes = 16UL << 20; // 16 MB
  constexpr uint64_t kSsdBytes = 64UL << 20; // 64 MB

  struct {
    double maxWriteRatio;
    double ssdSavableRatio;
    int32_t minSsdSavableBytes;
    uint64_t ssdFlushThresholdBytes;
    bool expectedSaveToSsd;

    std::string debugString() const {
      return fmt::format(
          "maxWriteRatio {}, ssdSavableRatio {}, minSsdSavableBytes {}, ssdFlushThresholdBytes {}, expectedSaveToSsd {}",
          maxWriteRatio,
          ssdSavableRatio,
          minSsdSavableBytes,
          ssdFlushThresholdBytes,
          expectedSaveToSsd);
    }
  } testSettings[] = {
      // Ratio-based threshold not met, ssdFlushThresholdBytes disabled (0).
      // No flush expected.
      {0.8, 0.95, 32 << 20, 0, false},
      // Ratio-based threshold not met, but ssdFlushThresholdBytes is small
      // (1MB).
      // Flush expected due to absolute threshold.
      {0.8, 0.95, 32 << 20, 1UL << 20, true},
      // Ratio-based threshold met. ssdFlushThresholdBytes disabled.
      // Flush expected due to ratio.
      {0.8, 0.3, 4 << 20, 0, true},
      // Both thresholds could trigger. Flush expected.
      {0.8, 0.3, 4 << 20, 1UL << 20, true}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    initializeCache(
        kRamBytes,
        kSsdBytes,
        0,
        true,
        AsyncDataCache::Options(
            testData.maxWriteRatio,
            testData.ssdSavableRatio,
            testData.minSsdSavableBytes,
            AsyncDataCache::kDefaultNumShards,
            testData.ssdFlushThresholdBytes));
    // Load data half of the in-memory capacity.
    loadLoop(0, kRamBytes / 2);
    waitForPendingLoads();
    auto stats = cache_->refreshStats();
    if (testData.expectedSaveToSsd) {
      EXPECT_GT(stats.ssdStats->entriesWritten, 0);
    } else {
      EXPECT_EQ(stats.ssdStats->entriesWritten, 0);
    }
  }
}

TEST_P(AsyncDataCacheTest, appendSsdSaveable) {
  constexpr uint64_t kRamBytes = 64UL << 20; // 64 MB
  constexpr uint64_t kSsdBytes = 128UL << 20; // 128 MB

  // Test if ssd write behavior with different settings.
  struct {
    double maxWriteRatio;
    double ssdSavableRatio;
    int32_t minSsdSavableBytes;
    bool appendAll;

    std::string debugString() const {
      return fmt::format(
          "maxWriteRatio {}, ssdSavableRatio {}, minSsdSavableBytes {}, appendAll {}",
          maxWriteRatio,
          ssdSavableRatio,
          minSsdSavableBytes,
          appendAll);
    }
  } testSettings[] = {
      {0.0, 10000.0, 1ULL << 30, true}, {0.0, 10000.0, 1UL << 30, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    initializeCache(
        kRamBytes,
        kSsdBytes,
        /*checkpointIntervalBytes=*/1UL << 30,
        /*eraseCheckpoint=*/true,
        {testData.maxWriteRatio,
         testData.ssdSavableRatio,
         testData.minSsdSavableBytes});
    // Load data half of the in-memory capacity.
    loadLoop(0, kRamBytes / 2);
    waitForPendingLoads();
    auto stats = cache_->refreshStats();

    ASSERT_TRUE(cache_->ssdCache()->startWrite());
    cache_->saveToSsd(testData.appendAll);

    cache_->ssdCache()->waitForWriteToFinish();
    stats = cache_->refreshStats();
    if (testData.appendAll) {
      // There might be some cache evictions.
      ASSERT_GE(stats.ssdStats->entriesWritten, stats.numEntries);
    } else {
      ASSERT_EQ(
          stats.ssdStats->entriesWritten, asyncDataCacheHelper_->numShards());
    }
  }
}

TEST_P(AsyncDataCacheTest, checkpoint) {
  constexpr uint64_t kRamBytes = 16UL << 20; // 16 MB
  constexpr uint64_t kSsdBytes = 64UL << 20; // 64 MB

  initializeMemoryManager(kRamBytes);
  initializeCache(
      kRamBytes,
      kSsdBytes,
      /*checkpointIntervalBytes=*/1ULL << 30,
      /*eraseCheckpoint=*/true);
  // Load data half of the in-memory capacity.
  loadLoop(0, kRamBytes / 2);
  waitForPendingLoads();
  auto stats = cache_->refreshStats();
  ASSERT_EQ(stats.ssdStats->checkpointsWritten, 0);
  ASSERT_TRUE(cache_->ssdCache()->startWrite());
  cache_->ssdCache()->checkpoint();
  cache_->ssdCache()->waitForWriteToFinish();
  stats = cache_->refreshStats();
  ASSERT_EQ(stats.ssdStats->checkpointsWritten, kNumSsdShards);
}

// TODO: add concurrent fuzzer test.

TEST_P(AsyncDataCacheTest, numShardsDefault) {
  constexpr uint64_t kRamBytes = 16UL << 20;

  initializeCache(kRamBytes);
  ASSERT_EQ(
      asyncDataCacheHelper_->numShards(), AsyncDataCache::kDefaultNumShards);
}

TEST_P(AsyncDataCacheTest, numShardsInvalid) {
  constexpr uint64_t kRamBytes = 16UL << 20;

  // Non-power-of-2 should fail.
  for (int32_t numShards : {3, 5, 6, 7, 9, 10}) {
    AsyncDataCache::Options options;
    options.numShards = numShards;
    VELOX_ASSERT_THROW(
        initializeCache(kRamBytes, 0, 0, false, options),
        "numShards must be a power of 2");
  }

  // Zero should fail.
  {
    AsyncDataCache::Options options;
    options.numShards = 0;
    VELOX_ASSERT_THROW(
        initializeCache(kRamBytes, 0, 0, false, options),
        "numShards must be positive");
  }

  // Negative should fail.
  {
    AsyncDataCache::Options options;
    options.numShards = -1;
    VELOX_ASSERT_THROW(
        initializeCache(kRamBytes, 0, 0, false, options),
        "numShards must be positive");
  }
}

TEST_P(AsyncDataCacheTest, findMiss) {
  constexpr int64_t kRamBytes = 32 << 20;
  initializeMemoryManager(kRamBytes);
  initializeCache(kRamBytes);

  RawFileCacheKey key{filenames_[0].id(), 0};
  auto result = cache_->find(key);
  ASSERT_FALSE(result.has_value());
}

TEST_P(AsyncDataCacheTest, findHit) {
  constexpr int64_t kRamBytes = 32 << 20;
  constexpr int32_t kEntrySize = 4096;
  initializeMemoryManager(kRamBytes);
  initializeCache(kRamBytes);

  RawFileCacheKey key{filenames_[0].id(), 1000};

  // Populate the entry via findOrCreate.
  {
    auto pin = cache_->findOrCreate(key, kEntrySize);
    ASSERT_FALSE(pin.empty());
    ASSERT_TRUE(pin.entry()->isExclusive());
    initializeContents(
        key.offset + key.fileNum, pin.entry()->nonContiguousData());
    pin.entry()->setExclusiveToShared();
  }

  // find should return a shared pin with correct data.
  auto result = cache_->find(key);
  ASSERT_TRUE(result.has_value());
  ASSERT_FALSE(result->empty());
  auto* entry = result->checkedEntry();
  ASSERT_TRUE(entry->isShared());
  ASSERT_EQ(entry->size(), kEntrySize);
  checkContents(*entry);
}

TEST_P(AsyncDataCacheTest, findExclusiveWithWait) {
  constexpr int64_t kRamBytes = 32 << 20;
  constexpr int32_t kEntrySize = 4096;
  initializeMemoryManager(kRamBytes);
  initializeCache(kRamBytes);

  RawFileCacheKey key{filenames_[0].id(), 2000};

  // Create an exclusive entry.
  auto exclusivePin = cache_->findOrCreate(key, kEntrySize);
  ASSERT_FALSE(exclusivePin.empty());
  ASSERT_TRUE(exclusivePin.entry()->isExclusive());

  // find without wait returns empty pin (entry exists but is exclusive).
  {
    auto result = cache_->find(key);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->empty());
  }

  // find with wait returns empty pin and sets a future.
  folly::SemiFuture<bool> waitFuture(false);
  {
    auto result = cache_->find(key, &waitFuture);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->empty());
  }

  // The future should not be ready while the entry is exclusive.
  ASSERT_FALSE(waitFuture.isReady());

  // Timed wait should time out while the entry is still exclusive.
  {
    auto waitCopy = std::move(waitFuture);
    auto timedResult =
        std::move(waitCopy).via(&folly::QueuedImmediateExecutor::instance());
    ASSERT_FALSE(
        std::move(timedResult).wait(std::chrono::seconds(1)).isReady());
  }

  // Re-issue find with wait after the timed-out future was consumed.
  waitFuture = folly::SemiFuture<bool>(false);
  {
    auto result = cache_->find(key, &waitFuture);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->empty());
  }
  ASSERT_FALSE(waitFuture.isReady());

  // Transition to shared makes the future ready.
  initializeContents(
      key.offset + key.fileNum, exclusivePin.entry()->nonContiguousData());
  exclusivePin.entry()->setExclusiveToShared();
  exclusivePin.clear();

  auto& exec = folly::QueuedImmediateExecutor::instance();
  ASSERT_TRUE(std::move(waitFuture).via(&exec).wait().isReady());

  // Now find should return a shared pin.
  auto result = cache_->find(key);
  ASSERT_TRUE(result.has_value());
  ASSERT_FALSE(result->empty());
  checkContents(*result->checkedEntry());
}

TEST_P(AsyncDataCacheTest, fuzz) {
  constexpr int64_t kRamBytes = 64 << 20;
  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumFiles = 10;
  constexpr int32_t kNumOffsets = 20;
  constexpr int32_t kEntrySize = 4096;
  constexpr int32_t kTestDurationMs = 10'000;

  initializeMemoryManager(kRamBytes);
  initializeCache(kRamBytes);

  std::atomic_bool stop{false};

  // Worker threads: findOrCreate/find entries, verify content.
  auto workerFunc = [&](int32_t threadId) {
    std::mt19937 rng(threadId);
    while (!stop.load(std::memory_order_relaxed)) {
      const auto fileIdx = rng() % kNumFiles;
      const auto offsetIdx = rng() % kNumOffsets;
      const uint64_t offset = offsetIdx * kEntrySize;
      RawFileCacheKey key{filenames_[fileIdx].id(), offset};

      // Randomly choose between find and findOrCreate.
      if (rng() % 3 == 0) {
        // find: lookup only.
        auto result = cache_->find(key);
        if (result.has_value() && !result->empty()) {
          checkContents(*result->checkedEntry());
        }
      } else {
        // findOrCreate: populate if new.
        folly::SemiFuture<bool> waitFuture(false);
        auto pin = cache_->findOrCreate(
            key, kEntrySize, /*contiguous=*/false, &waitFuture);
        if (pin.empty()) {
          auto& exec = folly::QueuedImmediateExecutor::instance();
          std::move(waitFuture).via(&exec).wait();
          continue;
        }
        auto* entry = pin.checkedEntry();
        if (entry->isExclusive()) {
          initializeContents(
              key.offset + key.fileNum, entry->nonContiguousData());
          entry->setExclusiveToShared();
        }
        checkContents(*entry);
      }
    }
  };

  // Eviction thread: periodically remove a subset of files.
  auto evictFunc = [&]() {
    std::mt19937 rng(kNumThreads + 1);
    while (!stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50)); // NOLINT
      folly::F14FastSet<uint64_t> filesToRemove;
      // Remove a random subset of files.
      const auto numToRemove = (rng() % kNumFiles) + 1;
      for (uint32_t i = 0; i < numToRemove; ++i) {
        filesToRemove.insert(filenames_[rng() % kNumFiles].id());
      }
      folly::F14FastSet<uint64_t> filesRetained;
      cache_->removeFileEntries(filesToRemove, filesRetained);
    }
  };

  std::vector<std::thread> threads;
  for (int32_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(workerFunc, i);
  }
  threads.emplace_back(evictFunc);

  std::this_thread::sleep_for(
      std::chrono::milliseconds(kTestDurationMs)); // NOLINT
  stop.store(true, std::memory_order_relaxed);

  for (auto& thread : threads) {
    thread.join();
  }

  auto stats = cache_->refreshStats();
  LOG(INFO) << "fuzz stats: " << stats.numEntries << " entries, "
            << stats.numHit << " hits, " << stats.numNew << " new, "
            << stats.numEvict << " evicts";
}

TEST_P(AsyncDataCacheTest, dataRanges) {
  constexpr uint64_t kRamBytes = 64UL << 20;
  initializeCache(kRamBytes);

  struct TestParam {
    int32_t size;
    std::string debugString() const {
      return fmt::format("size {}", size);
    }
  };

  std::vector<TestParam> testSettings = {
      // Tiny entry (< kTinyDataSize).
      {AsyncDataCacheEntry::kTinyDataSize - 1},
      // Allocation-backed entry (>= kTinyDataSize).
      {AsyncDataCacheEntry::kTinyDataSize * 4},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    auto pin = newEntry(testData.size, testData.size);
    ASSERT_FALSE(pin.empty());
    auto* entry = pin.checkedEntry();
    ASSERT_TRUE(entry->isExclusive());

    auto ranges = entry->dataRanges(testData.size);
    ASSERT_FALSE(ranges.empty());

    // Verify total bytes across all ranges covers the entry size.
    uint64_t totalBytes = 0;
    for (const auto& range : ranges) {
      ASSERT_GT(range.size(), 0);
      totalBytes += range.size();
    }
    ASSERT_EQ(totalBytes, testData.size);

    // Verify ranges are writable by writing a pattern and reading it back.
    uint8_t pattern = 0;
    for (auto& range : ranges) {
      ::memset(range.data(), pattern, range.size());
      ++pattern;
    }
    pattern = 0;
    for (const auto& range : ranges) {
      for (size_t i = 0; i < range.size(); ++i) {
        ASSERT_EQ(static_cast<uint8_t>(range.data()[i]), pattern);
      }
      ++pattern;
    }

    if (testData.size < AsyncDataCacheEntry::kTinyDataSize) {
      // Tiny entry: single range backed by tinyData.
      ASSERT_EQ(ranges.size(), 1);
      ASSERT_TRUE(entry->hasContiguousData());
      ASSERT_EQ(ranges[0].data(), entry->contiguousData());
    } else {
      // Allocation-backed: one range per run.
      ASSERT_EQ(ranges.size(), entry->nonContiguousData().numRuns());
    }

    entry->setExclusiveToShared();
  }
}

TEST_P(AsyncDataCacheTest, acquiredMemory) {
  constexpr uint64_t kRamBytes = 64UL << 20;
  initializeCache(kRamBytes);
  auto* allocator = cache_->allocator();

  // Default is empty with zero bytes.
  {
    AcquiredMemory acquired;
    ASSERT_TRUE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), 0);
    acquired.free(allocator);
    ASSERT_TRUE(acquired.empty());
  }

  // Non-contiguous only.
  {
    AcquiredMemory acquired;
    memory::Allocation allocation;
    ASSERT_TRUE(allocator->allocateNonContiguous(10, allocation));
    const auto expectedBytes = allocation.byteSize();
    acquired.nonContiguous.appendMove(allocation);

    ASSERT_FALSE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), expectedBytes);

    acquired.free(allocator);
    ASSERT_TRUE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), 0);
  }

  // Contiguous only.
  {
    AcquiredMemory acquired;
    auto* ptr1 = allocator->allocateBytes(1'024);
    auto* ptr2 = allocator->allocateBytes(2'048);
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    acquired.contiguous.emplace_back(ptr1, 1'024);
    acquired.contiguous.emplace_back(ptr2, 2'048);

    ASSERT_FALSE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), 3'072);

    acquired.free(allocator);
    ASSERT_TRUE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), 0);
  }

  // Mixed non-contiguous and contiguous.
  {
    AcquiredMemory acquired;
    memory::Allocation allocation;
    ASSERT_TRUE(allocator->allocateNonContiguous(10, allocation));
    const auto nonContiguousBytes = allocation.byteSize();
    acquired.nonContiguous.appendMove(allocation);

    constexpr uint64_t kContiguousSize = 4'096;
    auto* ptr = allocator->allocateBytes(kContiguousSize);
    ASSERT_NE(ptr, nullptr);
    acquired.contiguous.emplace_back(ptr, kContiguousSize);

    ASSERT_FALSE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), nonContiguousBytes + kContiguousSize);

    acquired.free(allocator);
    ASSERT_TRUE(acquired.empty());
    ASSERT_EQ(acquired.totalBytes(), 0);
  }
}

TEST_P(AsyncDataCacheTest, eviction) {
  constexpr uint64_t kRamBytes = 64UL << 20;
  initializeCache(kRamBytes);

  constexpr int32_t kEntrySize = 64 * 1'024;
  constexpr int32_t kNumEntries = 32;

  enum class EntryType { kContiguous, kNonContiguous, kMixed };

  struct TestParam {
    EntryType entryType;
    std::string debugString() const {
      switch (entryType) {
        case EntryType::kContiguous:
          return "contiguous";
        case EntryType::kNonContiguous:
          return "nonContiguous";
        case EntryType::kMixed:
          return "mixed";
      }
      VELOX_UNREACHABLE();
    }
  };

  std::vector<TestParam> testSettings = {
      {EntryType::kContiguous},
      {EntryType::kNonContiguous},
      {EntryType::kMixed},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::vector<CachePin> pins;
    for (int i = 0; i < kNumEntries; ++i) {
      bool contiguous = false;
      switch (testData.entryType) {
        case EntryType::kContiguous:
          contiguous = true;
          break;
        case EntryType::kNonContiguous:
          contiguous = false;
          break;
        case EntryType::kMixed:
          contiguous = (i % 2 == 0);
          break;
      }
      RawFileCacheKey key{
          filenames_[0].id(), static_cast<uint64_t>(i) * kEntrySize};
      auto pin = cache_->findOrCreate(key, kEntrySize, contiguous);
      ASSERT_FALSE(pin.empty());
      auto* entry = pin.checkedEntry();
      ASSERT_TRUE(entry->isExclusive());
      if (contiguous) {
        ASSERT_TRUE(entry->hasContiguousData());
        ::memset(entry->contiguousData(), static_cast<int>(i), kEntrySize);
      }
      entry->setExclusiveToShared();
      pins.push_back(std::move(pin));
    }
    ASSERT_EQ(pins.size(), kNumEntries);

    auto statsBefore = cache_->refreshStats();
    ASSERT_EQ(statsBefore.numEntries, kNumEntries);
    const auto evictsBefore = statsBefore.numEvict;

    pins.clear();

    auto freed = cache_->shrink(kRamBytes);
    ASSERT_GT(freed, 0);

    auto statsAfter = cache_->refreshStats();
    ASSERT_EQ(statsAfter.numEntries, 0);
    ASSERT_GE(statsAfter.numEvict - evictsBefore, kNumEntries);
  }
}

TEST_P(AsyncDataCacheTest, retryAllocation) {
  constexpr uint64_t kRamBytes = 64UL << 20;
  initializeCache(kRamBytes);

  constexpr int32_t kEntrySize = 64 * 1'024;
  constexpr int32_t kNumEntries = 512;

  enum class EntryType { kContiguous, kNonContiguous, kMixed };

  struct TestParam {
    EntryType entryType;
    std::string debugString() const {
      switch (entryType) {
        case EntryType::kContiguous:
          return "contiguous";
        case EntryType::kNonContiguous:
          return "nonContiguous";
        case EntryType::kMixed:
          return "mixed";
      }
      VELOX_UNREACHABLE();
    }
  };

  std::vector<TestParam> testSettings = {
      {EntryType::kContiguous},
      {EntryType::kNonContiguous},
      {EntryType::kMixed},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    // Fill the cache with entries.
    std::vector<CachePin> pins;
    for (int i = 0; i < kNumEntries; ++i) {
      bool contiguous = false;
      switch (testData.entryType) {
        case EntryType::kContiguous:
          contiguous = true;
          break;
        case EntryType::kNonContiguous:
          contiguous = false;
          break;
        case EntryType::kMixed:
          contiguous = (i % 2 == 0);
          break;
      }
      RawFileCacheKey key{
          filenames_[0].id(), static_cast<uint64_t>(i) * kEntrySize};
      auto pin = cache_->findOrCreate(key, kEntrySize, contiguous);
      ASSERT_FALSE(pin.empty());
      auto* entry = pin.checkedEntry();
      if (contiguous) {
        ::memset(entry->contiguousData(), 0xAB, kEntrySize);
      }
      entry->setExclusiveToShared();
      pins.push_back(std::move(pin));
    }

    // Unpin so entries are evictable.
    pins.clear();

    auto statsBefore = cache_->refreshStats();
    ASSERT_EQ(statsBefore.numEntries, kNumEntries);

    auto* allocator = cache_->allocator();

    // Allocate non-contiguous pages through the allocator directly.
    // Request more than what's free to force eviction via makeSpace.
    constexpr uint64_t kAllocBytes = 48UL << 20;
    memory::Allocation allocation;
    ASSERT_TRUE(allocator->allocateNonContiguous(
        memory::AllocationTraits::numPages(kAllocBytes), allocation));

    auto statsAfter = cache_->refreshStats();
    ASSERT_GT(statsAfter.numEvict, statsBefore.numEvict);

    allocator->freeNonContiguous(allocation);

    cache_->clear();
    auto statsFinal = cache_->refreshStats();
    ASSERT_EQ(statsFinal.numEntries, 0);
  }
}

TEST_P(AsyncDataCacheTest, makePins) {
  constexpr uint64_t kRamBytes = 64UL << 20;
  initializeCache(kRamBytes);

  constexpr int32_t kEntrySize = 8'192;

  struct TestParam {
    int numEntries;
    bool contiguous;
    std::string debugString() const {
      return fmt::format(
          "numEntries {}, contiguous {}", numEntries, contiguous);
    }
  };

  std::vector<TestParam> testSettings = {
      {1, false},
      {1, true},
      {10, false},
      {10, true},
      {100, false},
      {100, true},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::vector<RawFileCacheKey> keys;
    keys.reserve(testData.numEntries);
    for (int i = 0; i < testData.numEntries; ++i) {
      keys.push_back(
          {filenames_[0].id(), static_cast<uint64_t>(i) * kEntrySize});
    }

    std::vector<CachePin> pins;
    cache_->makePins(
        keys,
        [&](size_t /*i*/) { return kEntrySize; },
        [&](size_t /*i*/, CachePin&& pin) { pins.push_back(std::move(pin)); },
        testData.contiguous);

    ASSERT_EQ(pins.size(), testData.numEntries);
    for (auto& pin : pins) {
      auto* entry = pin.checkedEntry();
      ASSERT_TRUE(entry->isExclusive());
      test::AsyncDataCacheEntryTestHelper entryHelper(entry);
      if (testData.contiguous) {
        ASSERT_TRUE(entryHelper.isContiguousData());
        ASSERT_TRUE(entry->hasContiguousData());
        ASSERT_EQ(entry->contiguousDataSize(), kEntrySize);
      } else {
        ASSERT_FALSE(entryHelper.isContiguousData());
        ASSERT_FALSE(entry->hasContiguousData());
      }
      entry->setExclusiveToShared();
    }
    pins.clear();
    cache_->clear();
  }
}

TEST_P(AsyncDataCacheTest, removeFileEntries) {
  constexpr uint64_t kRamBytes = 64UL << 20;
  initializeCache(kRamBytes);

  constexpr int32_t kEntrySize = 8'192;
  constexpr int kNumEntries = 10;

  struct TestParam {
    bool contiguous;
    std::string debugString() const {
      return fmt::format("contiguous {}", contiguous);
    }
  };

  std::vector<TestParam> testSettings = {{false}, {true}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    for (int i = 0; i < kNumEntries; ++i) {
      RawFileCacheKey key{
          filenames_[0].id(), static_cast<uint64_t>(i) * kEntrySize};
      auto pin = cache_->findOrCreate(key, kEntrySize, testData.contiguous);
      ASSERT_FALSE(pin.empty());
      auto* entry = pin.checkedEntry();
      if (testData.contiguous) {
        ::memset(entry->contiguousData(), 0xCD, kEntrySize);
      }
      entry->setExclusiveToShared();
    }

    auto statsBefore = cache_->refreshStats();
    ASSERT_EQ(statsBefore.numEntries, kNumEntries);
    ASSERT_GT(statsBefore.largeSize, 0);

    folly::F14FastSet<uint64_t> filesToRemove;
    filesToRemove.insert(filenames_[0].id());
    folly::F14FastSet<uint64_t> filesRetained;
    cache_->removeFileEntries(filesToRemove, filesRetained);
    ASSERT_TRUE(filesRetained.empty());

    auto statsAfter = cache_->refreshStats();
    ASSERT_EQ(statsAfter.numEntries, 0);
    ASSERT_EQ(statsAfter.largeSize, 0);
    ASSERT_EQ(statsAfter.tinySize, 0);
  }
}

TEST_P(AsyncDataCacheTest, mixedBufferFuzz) {
  constexpr uint64_t kRamBytes = 256UL << 20;
  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumKeys = 200;
  constexpr int32_t kTestDurationMs = 20'000;

  initializeCache(kRamBytes);

  // Per-key properties are deterministic so all threads agree.
  struct KeyProps {
    int64_t size;
    bool contiguous;
  };
  std::vector<KeyProps> keyProps(kNumKeys);
  {
    std::mt19937 rng(42);
    for (int i = 0; i < kNumKeys; ++i) {
      switch (i % 3) {
        case 0:
          keyProps[i].size = 512 + (rng() % 1'024);
          break;
        case 1:
          keyProps[i].size =
              AsyncDataCacheEntry::kTinyDataSize + (rng() % (64 * 1'024));
          break;
        default:
          // Large entries ensure total data exceeds cache capacity,
          // triggering eviction.
          keyProps[i].size = 1'024 * 1'024 + (rng() % (3 * 1'024 * 1'024));
          break;
      }
      keyProps[i].contiguous = (i % 2 == 0);
    }
  }

  struct EntryState {
    uint8_t pattern{0};
  };

  std::mutex stateMutex;
  std::unordered_map<uint64_t, EntryState> entryStates;

  std::atomic_bool stop{false};

  auto workerFunc = [&](int32_t threadId) {
    std::mt19937 rng(threadId * 7 + 13);
    while (!stop.load(std::memory_order_relaxed)) {
      const auto keyIdx = rng() % kNumKeys;
      const uint64_t offset = keyIdx * 128 * 1'024;
      RawFileCacheKey key{filenames_[0].id(), offset};

      const auto& props = keyProps[keyIdx];
      const uint8_t pattern = static_cast<uint8_t>(rng());

      folly::SemiFuture<bool> waitFuture(false);
      auto pin =
          cache_->findOrCreate(key, props.size, props.contiguous, &waitFuture);
      if (pin.empty()) {
        continue;
      }
      auto* entry = pin.checkedEntry();
      ASSERT_EQ(entry->size(), props.size);

      if (entry->isExclusive()) {
        if (entry->hasContiguousData()) {
          ::memset(entry->contiguousData(), pattern, entry->size());
        } else {
          for (auto& range : entry->dataRanges(entry->size())) {
            ::memset(range.data(), pattern, range.size());
          }
        }
        {
          // Record state before making shared so other threads always
          // find the state when they get a shared pin.
          std::lock_guard<std::mutex> l(stateMutex);
          entryStates[offset] = {pattern};
        }
        entry->setExclusiveToShared();
      } else {
        EntryState expected;
        {
          std::lock_guard<std::mutex> l(stateMutex);
          auto it = entryStates.find(offset);
          ASSERT_NE(it, entryStates.end());
          expected = it->second;
        }

        if (entry->hasContiguousData()) {
          const auto* data = entry->contiguousData();
          for (int i = 0; i < entry->size(); ++i) {
            ASSERT_EQ(static_cast<uint8_t>(data[i]), expected.pattern)
                << "Data mismatch at offset " << offset << " byte " << i;
          }
        } else {
          int byteIdx = 0;
          for (const auto& range : entry->dataRanges(entry->size())) {
            for (size_t i = 0; i < range.size(); ++i) {
              ASSERT_EQ(static_cast<uint8_t>(range.data()[i]), expected.pattern)
                  << "Data mismatch at offset " << offset << " byte "
                  << byteIdx;
              ++byteIdx;
            }
          }
        }
      }
      // Release pin immediately so eviction can reclaim it.
    }
  };

  std::vector<std::thread> threads;
  for (int32_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(workerFunc, i);
  }

  std::this_thread::sleep_for(
      std::chrono::milliseconds(kTestDurationMs)); // NOLINT
  stop.store(true, std::memory_order_relaxed);

  for (auto& thread : threads) {
    thread.join();
  }

  auto stats = cache_->refreshStats();
  LOG(INFO) << "contiguousFuzz stats: " << stats.numEntries << " entries, "
            << stats.numHit << " hits, " << stats.numNew << " new, "
            << stats.numEvict << " evicts";

  // Verify remaining entries have correct data.
  int32_t verified = 0;
  int32_t evicted = 0;
  {
    std::lock_guard<std::mutex> l(stateMutex);
    for (const auto& [offset, expected] : entryStates) {
      RawFileCacheKey key{filenames_[0].id(), offset};
      auto result = cache_->find(key);
      if (!result.has_value()) {
        ++evicted;
        continue;
      }
      ASSERT_FALSE(result->empty());
      auto* entry = result->checkedEntry();
      // Verify the data pattern.
      if (entry->hasContiguousData()) {
        const auto* data = entry->contiguousData();
        for (int i = 0; i < entry->size(); ++i) {
          ASSERT_EQ(static_cast<uint8_t>(data[i]), expected.pattern);
        }
      } else {
        for (const auto& range : entry->dataRanges(entry->size())) {
          for (size_t i = 0; i < range.size(); ++i) {
            ASSERT_EQ(static_cast<uint8_t>(range.data()[i]), expected.pattern);
          }
        }
      }
      ++verified;
    }
  }

  LOG(INFO) << "Verified " << verified << " entries, " << evicted
            << " evicted from tracked state";
  if (evicted > 0) {
    ASSERT_GT(stats.numEvict, 0);
  }

  cache_->clear();
  auto finalStats = cache_->refreshStats();
  ASSERT_EQ(finalStats.numEntries, 0);
}

INSTANTIATE_TEST_SUITE_P(
    AsyncDataCacheTest,
    AsyncDataCacheTest,
    ::testing::ValuesIn(AsyncDataCacheTest::getTestParams()));
