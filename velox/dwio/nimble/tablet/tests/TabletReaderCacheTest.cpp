/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/tablet/TabletReaderCache.h"

#include <random>
#include <thread>

#include <gtest/gtest.h>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/synchronization/Latch.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::nimble;
using namespace facebook;

namespace {
// InMemoryReadFile with a configurable name for cache key differentiation.
class NamedInMemoryReadFile : public velox::InMemoryReadFile {
 public:
  NamedInMemoryReadFile(std::string name, std::string_view data)
      : velox::InMemoryReadFile(data), name_{std::move(name)} {}

  std::string getName() const override {
    return name_;
  }

 private:
  const std::string name_;
};
} // namespace

class TabletReaderCacheTest : public ::testing::Test {
 protected:
  // Default test schema: ROW({"col0"}, {BIGINT()}).
  static inline const velox::RowTypePtr kVeloxSchema =
      velox::ROW({"col0"}, {velox::BIGINT()});

  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  }

  void TearDown() override {
    TabletReaderCache::testingReset();
  }

  std::string writeTestFile(uint32_t numRows = 100) {
    std::string file;
    velox::test::VectorMaker vectorMaker(pool_.get());
    auto vector = vectorMaker.rowVector(
        {"col0"},
        {vectorMaker.flatVector<int64_t>(numRows, [](auto i) { return i; })});

    WriterOptions writerOptions;
    auto writer = std::make_unique<Writer>(
        kVeloxSchema,
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *pool_,
        std::move(writerOptions));
    writer->write(vector);
    writer->close();
    return file;
  }

  void verifySchema(const CachedTabletReader& cached) {
    ASSERT_NE(cached.nimbleSchema, nullptr);
    EXPECT_TRUE(cached.nimbleSchema->isRow());
    ASSERT_NE(cached.veloxSchema, nullptr);
    EXPECT_TRUE(cached.veloxSchema->equivalent(*kVeloxSchema));
  }

  std::shared_ptr<NamedInMemoryReadFile> makeReadFile(
      const std::string& name,
      const std::string& data) {
    return std::make_shared<NamedInMemoryReadFile>(name, data);
  }

  TabletReaderCache::Options makeOptions(
      uint32_t numShards = 2,
      size_t maxEntries = 100) {
    return {
        .numShards = numShards,
        .maxEntries = maxEntries,
        .executor = executor_,
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
};

TEST_F(TabletReaderCacheTest, optionsToString) {
  TabletReaderCache::Options opts;
  opts.numShards = 8;
  opts.maxEntries = 2'000;
  opts.expireDurationMs = 60'000;
  opts.executor = executor_;
  EXPECT_EQ(
      opts.toString(),
      "numShards=8, maxEntries=2000, expireDuration=60.00s, executor=set");

  opts.executor = nullptr;
  EXPECT_EQ(
      opts.toString(),
      "numShards=8, maxEntries=2000, expireDuration=60.00s, executor=null");
}

TEST_F(TabletReaderCacheTest, invalidOptions) {
  NIMBLE_ASSERT_THROW(
      TabletReaderCache(makeOptions(/*numShards=*/3)),
      "numShards must be a power of 2");

  TabletReaderCache::Options opts;
  opts.numShards = 2;
  opts.maxEntries = 100;
  opts.executor = nullptr;
  NIMBLE_ASSERT_THROW(TabletReaderCache(opts), "");
}

TEST_F(TabletReaderCacheTest, cacheHitAndMiss) {
  TabletReaderCache cache(makeOptions());

  const auto file1 = writeTestFile(100);
  const auto file2 = writeTestFile(200);
  auto readFile1 = makeReadFile("file1", file1);
  auto readFile2 = makeReadFile("file2", file2);

  // First get is a miss.
  auto cached1 = cache.get(readFile1, {});
  EXPECT_EQ(cache.stats().numLookups, 1);
  EXPECT_EQ(cache.stats().numHits, 0);
  EXPECT_EQ(cache.stats().numElements, 1);

  // Same file is a hit — returns identical pointers.
  auto cached2 = cache.get(readFile1, {});
  EXPECT_EQ(cached1.tablet.get(), cached2.tablet.get());
  EXPECT_EQ(cached1.nimbleSchema.get(), cached2.nimbleSchema.get());
  EXPECT_EQ(cached1.veloxSchema.get(), cached2.veloxSchema.get());
  EXPECT_EQ(cache.stats().numLookups, 2);
  EXPECT_EQ(cache.stats().numHits, 1);
  EXPECT_EQ(cache.stats().numElements, 1);

  // Different file is a miss — returns different pointers.
  auto cached3 = cache.get(readFile2, {});
  EXPECT_NE(cached1.tablet.get(), cached3.tablet.get());
  EXPECT_NE(cached1.nimbleSchema.get(), cached3.nimbleSchema.get());
  EXPECT_NE(cached1.veloxSchema.get(), cached3.veloxSchema.get());
  EXPECT_EQ(cache.stats().numLookups, 3);
  EXPECT_EQ(cache.stats().numHits, 1);
  EXPECT_EQ(cache.stats().numElements, 2);
}

TEST_F(TabletReaderCacheTest, eviction) {
  TabletReaderCache cache(makeOptions(/*numShards=*/2, /*maxEntries=*/2));

  const auto file = writeTestFile();

  // Fill cache to capacity.
  for (int i = 0; i < 2; ++i) {
    auto readFile = makeReadFile("file" + std::to_string(i), file);
    cache.get(readFile, {});
  }
  EXPECT_EQ(cache.stats().numElements, 2);

  // Adding a third entry should evict the least-recently used one.
  auto readFile = makeReadFile("file2", file);
  cache.get(readFile, {});
  EXPECT_EQ(cache.stats().numElements, 2);
}

TEST_F(TabletReaderCacheTest, expiration) {
  constexpr size_t kTtlMs = 100;
  TabletReaderCache::Options opts;
  opts.numShards = 2;
  opts.maxEntries = 100;
  opts.expireDurationMs = kTtlMs;
  opts.executor = executor_;
  TabletReaderCache cache(opts);

  const auto file = writeTestFile();
  auto readFile = makeReadFile("file1", file);

  auto cached1 = cache.get(readFile, {});
  EXPECT_EQ(cache.stats().numElements, 1);
  EXPECT_EQ(cache.stats().numHits, 0);

  // Before expiration, same file is a cache hit.
  auto cached2 = cache.get(readFile, {});
  EXPECT_EQ(cached1.tablet.get(), cached2.tablet.get());
  EXPECT_EQ(cache.stats().numHits, 1);

  // Wait for TTL to expire.
  std::this_thread::sleep_for(std::chrono::milliseconds(kTtlMs + 50));

  // After expiration, same file is a cache miss — new tablet created.
  auto cached3 = cache.get(readFile, {});
  EXPECT_NE(cached1.tablet.get(), cached3.tablet.get());
  EXPECT_EQ(cache.stats().numHits, 1);
}

TEST_F(TabletReaderCacheTest, concurrentGet) {
  constexpr int kNumThreads = 32;
  constexpr int kNumFiles = 64;
  constexpr int kMaxCacheEntries = 16;
  constexpr auto kTestDuration = std::chrono::seconds(10);

  TabletReaderCache cache(
      makeOptions(/*numShards=*/4, /*maxEntries=*/kMaxCacheEntries));

  // Pre-generate file contents — each file has a distinct row count so
  // we can verify identity via tabletRowCount().
  std::vector<std::string> fileContents(kNumFiles);
  for (int i = 0; i < kNumFiles; ++i) {
    fileContents[i] = writeTestFile(/*numRows=*/100 + i);
  }

  std::atomic_bool stop{false};
  std::atomic<uint64_t> totalGets{0};
  folly::Latch latch(kNumThreads);

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(t);
      std::uniform_int_distribution<int> dist(0, kNumFiles - 1);
      latch.count_down();
      latch.wait();
      while (!stop.load(std::memory_order_relaxed)) {
        const int fileIdx = dist(rng);
        const auto name = "file" + std::to_string(fileIdx);
        auto readFile = makeReadFile(name, fileContents[fileIdx]);
        auto cached = cache.get(readFile, {});
        ASSERT_NE(cached.tablet, nullptr);
        ASSERT_EQ(
            cached.tablet->tabletRowCount(),
            static_cast<uint64_t>(100 + fileIdx));
        ++totalGets;
      }
    });
  }

  std::this_thread::sleep_for(kTestDuration);
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : threads) {
    t.join();
  }

  const auto stats = cache.stats();
  EXPECT_EQ(stats.maxSize, kMaxCacheEntries);
  EXPECT_LE(stats.numElements, kMaxCacheEntries);
  EXPECT_GT(stats.numLookups, 0);
  EXPECT_GT(stats.numHits, 0);

  // Verify all cached entries are valid.
  size_t numCached = 0;
  for (int i = 0; i < kNumFiles; ++i) {
    const auto name = "file" + std::to_string(i);
    auto entry = cache.testingGet(name);
    if (!entry.has_value()) {
      continue;
    }
    ++numCached;
    ASSERT_NE(entry->tablet, nullptr);
    EXPECT_EQ(entry->tablet->tabletRowCount(), static_cast<uint64_t>(100 + i));
    verifySchema(*entry);
  }
  EXPECT_EQ(numCached, stats.numElements);

  LOG(INFO) << "concurrentGet: " << totalGets.load() << " gets, "
            << stats.numHits << "/" << stats.numLookups << " hits, "
            << numCached << " cached";
}

TEST_F(TabletReaderCacheTest, singleton) {
  // Ensure clean state.
  TabletReaderCache::testingReset();

  // getInstance() before initialize() should throw.
  NIMBLE_ASSERT_THROW(
      TabletReaderCache::getInstance(),
      "TabletReaderCache::initialize() must be called before getInstance()");

  // Initialize with specific options.
  const auto opts = makeOptions(/*numShards=*/4, /*maxEntries=*/50);
  TabletReaderCache::initialize(opts);

  // getInstance() returns a functional instance.
  auto& instance = TabletReaderCache::getInstance();
  const auto file = writeTestFile();
  auto readFile = makeReadFile("singleton_test", file);
  {
    auto cached = instance.get(readFile, {});
    ASSERT_NE(cached.tablet, nullptr);
    EXPECT_EQ(cached.tablet->fileSize(), readFile->size());
    EXPECT_EQ(cached.tablet->stripeCount(), 1);
    EXPECT_EQ(cached.tablet->tabletRowCount(), 100);
    verifySchema(cached);
  }

  // Verify cache stats reflect the configured maxEntries.
  EXPECT_EQ(instance.stats().maxSize, 50);

  // Double initialize() should throw.
  NIMBLE_ASSERT_THROW(
      TabletReaderCache::initialize(makeOptions()),
      "TabletReaderCache::initialize() must only be called once");

  // Reset allows re-initialization.
  TabletReaderCache::testingReset();
  NIMBLE_ASSERT_THROW(
      TabletReaderCache::getInstance(),
      "TabletReaderCache::initialize() must be called before getInstance()");

  TabletReaderCache::initialize(
      makeOptions(/*numShards=*/2, /*maxEntries=*/200));
  EXPECT_EQ(TabletReaderCache::getInstance().stats().maxSize, 200);

  TabletReaderCache::testingReset();
}
