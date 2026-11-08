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

#include "velox/dwio/nimble/index/StreamReaderCache.h"

#include <memory>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble {
namespace {

class StreamReaderCacheTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("stream_reader_cache_test");
    schema_ = std::make_shared<const ScalarType>(
        StreamDescriptor{0, ScalarKind::Int64});
  }

  // Returns an entry with a new reader of the schema on a new pool of 'cache',
  // as a row iterator builds one when the cache has no idle entry.
  std::unique_ptr<StreamReaderCache::Entry> makeEntry(
      StreamReaderCache& cache) const {
    auto pool = cache.addReaderPool();
    auto reader = std::make_unique<StreamReader>(schema_, pool.get());
    return std::make_unique<StreamReaderCache::Entry>(
        std::move(pool), std::move(reader));
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<const Type> schema_;
};

TEST_F(StreamReaderCacheTest, reusesTheEntryReleasedLast) {
  StreamReaderCache cache{rootPool_, /*maxReaders=*/2};
  EXPECT_EQ(cache.get(), nullptr);

  std::vector<const StreamReader*> readers;
  readers.reserve(2);
  for (int i = 0; i < 2; ++i) {
    auto entry = makeEntry(cache);
    readers.emplace_back(entry->streamReader.get());
    cache.release(std::move(entry));
  }

  std::vector<const StreamReader*> reused;
  reused.reserve(2);
  for (int i = 0; i < 2; ++i) {
    reused.emplace_back(cache.get()->streamReader.get());
  }
  EXPECT_EQ(cache.get(), nullptr);
  EXPECT_THAT(reused, ::testing::ElementsAre(readers[1], readers[0]));
}

TEST_F(StreamReaderCacheTest, destroysEntriesBeyondTheMaximum) {
  StreamReaderCache cache{rootPool_, /*maxReaders=*/2};
  std::vector<const StreamReader*> readers;
  readers.reserve(2);
  for (int i = 0; i < 2; ++i) {
    auto entry = makeEntry(cache);
    readers.emplace_back(entry->streamReader.get());
    cache.release(std::move(entry));
  }

  // The cache already keeps 2 idle entries, so it destroys the third, with the
  // pool of its reader.
  auto entry = makeEntry(cache);
  const std::weak_ptr<velox::memory::MemoryPool> destroyedPool = entry->pool;
  cache.release(std::move(entry));
  EXPECT_TRUE(destroyedPool.expired());
  std::vector<const StreamReader*> reused;
  reused.reserve(2);
  for (int i = 0; i < 2; ++i) {
    reused.emplace_back(cache.get()->streamReader.get());
  }
  EXPECT_EQ(cache.get(), nullptr);
  EXPECT_THAT(reused, ::testing::ElementsAre(readers[1], readers[0]));

  StreamReaderCache noIdleEntries{rootPool_, /*maxReaders=*/0};
  noIdleEntries.release(makeEntry(noIdleEntries));
  EXPECT_EQ(noIdleEntries.get(), nullptr);
}

TEST_F(StreamReaderCacheTest, givesEachReaderItsOwnLeafPool) {
  StreamReaderCache cache{rootPool_, /*maxReaders=*/2};
  const auto first = cache.addReaderPool();
  const auto second = cache.addReaderPool();
  EXPECT_NE(first, second);
  for (const auto* pool : {first.get(), second.get()}) {
    EXPECT_TRUE(pool->isLeaf());
    EXPECT_TRUE(pool->threadSafe());
    EXPECT_EQ(pool->parent(), rootPool_.get());
  }
}

TEST_F(StreamReaderCacheTest, sharesEntriesAcrossThreads) {
  StreamReaderCache cache{rootPool_, /*maxReaders=*/4};
  auto entry = makeEntry(cache);
  const auto* reader = entry->streamReader.get();

  // An entry that one thread releases serves the next get() on any thread.
  std::thread{[&] { cache.release(std::move(entry)); }}.join();
  const auto reused = cache.get();
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->streamReader.get(), reader);

  // Threads get and release entries at once, and build one when none is idle.
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&] {
      for (int i = 0; i < 100; ++i) {
        auto threadEntry = cache.get();
        if (threadEntry == nullptr) {
          threadEntry = makeEntry(cache);
        }
        cache.release(std::move(threadEntry));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  int numIdleEntries{0};
  while (cache.get() != nullptr) {
    ++numIdleEntries;
  }
  EXPECT_GE(numIdleEntries, 1);
  EXPECT_LE(numIdleEntries, 4);
}

TEST_F(StreamReaderCacheTest, rejectsALeafPool) {
  const auto leafPool = rootPool_->addLeafChild("leaf", /*threadSafe=*/true);
  NIMBLE_ASSERT_THROW(
      StreamReaderCache(leafPool, /*maxReaders=*/1),
      "The pool of a stream reader cache must be an aggregate pool");
}

} // namespace
} // namespace facebook::nimble
