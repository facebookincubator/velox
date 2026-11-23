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
#include "velox/dwio/nimble/tablet/MetadataCache.h"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "folly/Synchronized.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "velox/dwio/common/ExecutorBarrier.h"

using namespace facebook::nimble;

class MetadataCacheTest : public ::testing::Test {
 protected:
  int buildCount_{0};

  MetadataCache<uint32_t, std::string>::ValueBuilder builder() {
    return [this](uint32_t key) {
      ++buildCount_;
      return std::make_shared<std::string>("value_" + std::to_string(key));
    };
  }
};

TEST_F(MetadataCacheTest, basicGet) {
  MetadataCache<uint32_t, std::string> cache(builder());

  auto val = cache.getOrCreate(0);
  EXPECT_EQ(*val, "value_0");
  EXPECT_EQ(buildCount_, 1);
  EXPECT_TRUE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 1);
}

TEST_F(MetadataCacheTest, weakExpiration) {
  MetadataCache<uint32_t, std::string> cache(builder());

  // Get creates a weak entry. Dropping the shared_ptr should expire it.
  {
    auto val = cache.getOrCreate(0);
    EXPECT_EQ(*val, "value_0");
    EXPECT_TRUE(cache.hasCacheEntry(0));
  }
  // Weak pointer expired after shared_ptr goes out of scope.
  EXPECT_FALSE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 0);

  // Getting again should rebuild.
  auto val2 = cache.getOrCreate(0);
  EXPECT_EQ(*val2, "value_0");
  EXPECT_EQ(buildCount_, 2);
}

TEST_F(MetadataCacheTest, expiredWeakEntryReplacedOnReinsert) {
  int buildCounter = 0;
  auto versionedBuilder = [&](uint32_t key) {
    return std::make_shared<std::string>(
        fmt::format("value_{}_{}", key, buildCounter++));
  };
  MetadataCache<uint32_t, std::string> cache(versionedBuilder);

  // Create a weak cache entry.
  auto val1 = cache.getOrCreate(0);
  EXPECT_EQ(*val1, "value_0_0");
  EXPECT_EQ(buildCounter, 1);
  EXPECT_TRUE(cache.hasCacheEntry(0));

  // Drop the only shared_ptr — the weak entry in cacheEntries_ is now expired,
  // but the key still exists in the map.
  val1.reset();
  EXPECT_FALSE(cache.hasCacheEntry(0));

  // Request the same key again. The read-lock path finds the expired weak_ptr
  // and falls through. The write-lock path's emplace finds the stale key
  // (!inserted), lock() returns nullptr, and it->second is updated with the
  // newly built element.
  auto val2 = cache.getOrCreate(0);
  EXPECT_EQ(*val2, "value_0_1");
  EXPECT_EQ(buildCounter, 2);
  EXPECT_TRUE(cache.hasCacheEntry(0));
}

TEST_F(MetadataCacheTest, weakAlive) {
  MetadataCache<uint32_t, std::string> cache(builder());

  auto val = cache.getOrCreate(0);
  EXPECT_EQ(buildCount_, 1);

  // Getting again while holding the shared_ptr should return the same entry.
  auto val2 = cache.getOrCreate(0);
  EXPECT_EQ(buildCount_, 1);
  EXPECT_EQ(val.get(), val2.get());
}

TEST_F(MetadataCacheTest, pinnedEntries) {
  MetadataCache<uint32_t, std::string> cache(builder(), /*pinEntries=*/true);

  {
    auto val = cache.getOrCreate(0);
    EXPECT_EQ(*val, "value_0");
    EXPECT_EQ(buildCount_, 1);
  }
  // Pinned entries should not expire even after dropping the shared_ptr.
  EXPECT_TRUE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 1);

  auto val2 = cache.getOrCreate(0);
  EXPECT_EQ(*val2, "value_0");
  EXPECT_EQ(buildCount_, 1);
}

TEST_F(MetadataCacheTest, pin) {
  struct TestParam {
    bool pinEntries;
    std::string debugString() const {
      return fmt::format("pinEntries {}", pinEntries);
    }
  };
  std::vector<TestParam> testSettings = {{false}, {true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    buildCount_ = 0;
    MetadataCache<uint32_t, std::string> cache(builder(), testData.pinEntries);

    cache.pin(0, std::make_shared<std::string>("pinned_0"));

    // Add another key and verify the pinned key is still there.
    auto val1 = cache.getOrCreate(1);
    EXPECT_EQ(*val1, "value_1");
    EXPECT_EQ(buildCount_, 1);
    EXPECT_TRUE(cache.hasCacheEntry(0));
    EXPECT_EQ(cache.testingCacheCount(), 2);

    // Get the pinned key — should return the pinned value without calling
    // the builder.
    {
      auto val0 = cache.getOrCreate(0);
      EXPECT_EQ(*val0, "pinned_0");
      EXPECT_EQ(buildCount_, 1);
      EXPECT_TRUE(cache.hasCacheEntry(0));
      EXPECT_EQ(cache.testingCacheCount(), 2);
    }

    // After dropping the local shared reference, the pinned entry should still
    // be valid regardless of the pinEntries setting.
    EXPECT_TRUE(cache.hasCacheEntry(0));
    EXPECT_EQ(cache.testingCacheCount(), 2);

    // Retrieving again returns the same pinned value, no rebuild.
    auto val0Again = cache.getOrCreate(0);
    EXPECT_EQ(*val0Again, "pinned_0");
    EXPECT_EQ(buildCount_, 1);
  }
}

TEST_F(MetadataCacheTest, pinFoundByGetOrCreateWithoutPinMode) {
  MetadataCache<uint32_t, std::string> cache(builder());

  // Pin an entry in non-pin mode.
  cache.pin(0, std::make_shared<std::string>("pinned_0"));

  // getOrCreate should find the pinned entry without building.
  auto val = cache.getOrCreate(0);
  EXPECT_EQ(*val, "pinned_0");
  EXPECT_EQ(buildCount_, 0);
  EXPECT_TRUE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 1);

  // Drop local reference — pinned entry persists as a strong reference.
  val.reset();
  EXPECT_TRUE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 1);

  // Re-fetch still returns the pinned value, no rebuild.
  auto val2 = cache.getOrCreate(0);
  EXPECT_EQ(*val2, "pinned_0");
  EXPECT_EQ(buildCount_, 0);
}

TEST_F(MetadataCacheTest, pinFoundByGetOrCreateWithPinMode) {
  MetadataCache<uint32_t, std::string> cache(builder(), /*pinEntries=*/true);

  // Pin an entry in pin mode.
  cache.pin(0, std::make_shared<std::string>("pinned_0"));

  // getOrCreate should find the pinned entry without building.
  auto val = cache.getOrCreate(0);
  EXPECT_EQ(*val, "pinned_0");
  EXPECT_EQ(buildCount_, 0);
  EXPECT_TRUE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 1);

  // Drop local reference — pinned entry persists as a strong reference.
  val.reset();
  EXPECT_TRUE(cache.hasCacheEntry(0));
  EXPECT_EQ(cache.testingCacheCount(), 1);

  // Re-fetch still returns the pinned value, no rebuild.
  auto val2 = cache.getOrCreate(0);
  EXPECT_EQ(*val2, "pinned_0");
  EXPECT_EQ(buildCount_, 0);
}

TEST_F(MetadataCacheTest, customBuilder) {
  struct TestParam {
    bool pinEntries;
    std::string debugString() const {
      return fmt::format("pinEntries {}", pinEntries);
    }
  };
  std::vector<TestParam> testSettings = {{false}, {true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    buildCount_ = 0;
    MetadataCache<uint32_t, std::string> cache(builder(), testData.pinEntries);

    auto customBuilder = [](uint32_t key) {
      return std::make_shared<std::string>("custom_" + std::to_string(key));
    };

    // First call with custom builder should use the custom builder.
    auto val = cache.getOrCreate(0, customBuilder);
    EXPECT_EQ(*val, "custom_0");
    EXPECT_EQ(buildCount_, 0);

    // Second call with a different builder should return the cached entry,
    // not rebuild with the new builder.
    auto val2 = cache.getOrCreate(0, [](uint32_t key) {
      return std::make_shared<std::string>("other_" + std::to_string(key));
    });
    EXPECT_EQ(*val2, "custom_0");
    EXPECT_EQ(val.get(), val2.get());
    EXPECT_EQ(buildCount_, 0);
  }
}

TEST_F(MetadataCacheTest, multipleKeys) {
  MetadataCache<uint32_t, std::string> cache(builder(), /*pinEntries=*/true);

  auto val0 = cache.getOrCreate(0);
  auto val1 = cache.getOrCreate(1);
  auto val2 = cache.getOrCreate(2);

  EXPECT_EQ(*val0, "value_0");
  EXPECT_EQ(*val1, "value_1");
  EXPECT_EQ(*val2, "value_2");
  EXPECT_EQ(buildCount_, 3);
  EXPECT_EQ(cache.testingCacheCount(), 3);
}

TEST_F(MetadataCacheTest, pinDoesNotOverwrite) {
  struct TestParam {
    bool pinEntries;
    std::string debugString() const {
      return fmt::format("pinEntries {}", pinEntries);
    }
  };
  std::vector<TestParam> testSettings = {{false}, {true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    buildCount_ = 0;
    MetadataCache<uint32_t, std::string> cache(builder(), testData.pinEntries);

    cache.pin(0, std::make_shared<std::string>("first"));
    // emplace does not overwrite existing key.
    cache.pin(0, std::make_shared<std::string>("second"));

    auto val = cache.getOrCreate(0);
    EXPECT_EQ(*val, "first");
    EXPECT_EQ(buildCount_, 0);
  }
}

TEST_F(MetadataCacheTest, concurrentFuzzer) {
  struct TestParam {
    bool pinEntries;
    std::string debugString() const {
      return fmt::format("pinEntries {}", pinEntries);
    }
  };
  std::vector<TestParam> testSettings = {{false}, {true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    constexpr uint32_t kNumKeys = 100;
    constexpr int kNumThreads = 8;
    constexpr auto kDuration = std::chrono::seconds(10);

    // All builders produce the same deterministic value per key so we can
    // verify key-value consistency regardless of which builder runs.
    auto makeValue = [](uint32_t key) {
      return std::make_shared<std::string>("value_" + std::to_string(key));
    };

    MetadataCache<uint32_t, std::string> cache(makeValue, testData.pinEntries);

    std::atomic_bool stop{false};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      threads.emplace_back([&, t]() {
        std::mt19937 rng(t);
        std::vector<std::shared_ptr<std::string>> held;

        while (!stop.load(std::memory_order_relaxed)) {
          uint32_t key = rng() % kNumKeys;
          int action = rng() % 5;

          std::shared_ptr<std::string> val;
          switch (action) {
            case 0:
              val = cache.getOrCreate(key);
              break;
            case 1:
              val = cache.getOrCreate(key, makeValue);
              break;
            case 2:
              cache.pin(
                  key,
                  std::make_shared<std::string>(
                      "value_" + std::to_string(key)));
              continue;
            case 3:
              // Drop some held references to let weak entries expire.
              if (!held.empty()) {
                held.erase(held.begin(), held.begin() + (held.size() / 2));
              }
              continue;
            case 4:
              // Exercise read-only methods.
              cache.hasCacheEntry(key);
              cache.testingCacheCount();
              continue;
          }

          // Verify the value matches the key.
          ASSERT_EQ(*val, "value_" + std::to_string(key));

          // Randomly hold or drop the reference.
          if (rng() % 2 == 0) {
            held.emplace_back(std::move(val));
          }
        }
      });
    }

    std::this_thread::sleep_for(kDuration);
    stop.store(true, std::memory_order_relaxed);

    for (auto& thread : threads) {
      thread.join();
    }

    // Sanity check: all valid entries have correct key-value mapping.
    auto entries = cache.testingGetAllEntries();
    for (const auto& [key, value] : entries) {
      EXPECT_EQ(*value, "value_" + std::to_string(key));
    }
  }
}

namespace {

enum class ActionEnum { kCreated, kDestroyed };

using Action = std::pair<ActionEnum, int>;
using Actions = std::vector<Action>;

class Guard {
 public:
  Guard(int id, Actions& actions) : id_{id}, actions_{actions} {
    actions_.emplace_back(ActionEnum::kCreated, id_);
  }

  ~Guard() {
    actions_.emplace_back(ActionEnum::kDestroyed, id_);
  }

  Guard(const Guard&) = delete;
  Guard(Guard&&) = delete;
  Guard& operator=(const Guard&) = delete;
  Guard& operator=(Guard&&) = delete;

  int id() const {
    return id_;
  }

 private:
  int id_;
  Actions& actions_;
};

} // namespace

TEST_F(MetadataCacheTest, lifetimeTracking) {
  Actions actions;
  MetadataCache<int, Guard> cache{
      [&](int id) { return std::make_shared<Guard>(id, actions); }};

  auto e1 = cache.getOrCreate(0);
  EXPECT_EQ(e1->id(), 0);
  EXPECT_EQ(actions, Actions({{ActionEnum::kCreated, 0}}));

  auto e2 = cache.getOrCreate(0);
  EXPECT_EQ(e2->id(), 0);
  EXPECT_EQ(actions, Actions({{ActionEnum::kCreated, 0}}));
  e2.reset();
  EXPECT_EQ(actions, Actions({{ActionEnum::kCreated, 0}}));

  auto e3 = cache.getOrCreate(1);
  EXPECT_EQ(e3->id(), 1);
  EXPECT_EQ(
      actions, Actions({{ActionEnum::kCreated, 0}, {ActionEnum::kCreated, 1}}));

  e1.reset();
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0}}));

  auto e4 = e3;
  EXPECT_EQ(e4->id(), 1);
  e3.reset();
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0}}));

  e4.reset();
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0},
           {ActionEnum::kDestroyed, 1}}));

  auto e5 = cache.getOrCreate(1);
  EXPECT_EQ(e5->id(), 1);
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0},
           {ActionEnum::kDestroyed, 1},
           {ActionEnum::kCreated, 1}}));

  auto e6 = cache.getOrCreate(0);
  EXPECT_EQ(e6->id(), 0);
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0},
           {ActionEnum::kDestroyed, 1},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kCreated, 0}}));

  e5.reset();
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0},
           {ActionEnum::kDestroyed, 1},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kCreated, 0},
           {ActionEnum::kDestroyed, 1}}));

  e6.reset();
  EXPECT_EQ(
      actions,
      Actions(
          {{ActionEnum::kCreated, 0},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kDestroyed, 0},
           {ActionEnum::kDestroyed, 1},
           {ActionEnum::kCreated, 1},
           {ActionEnum::kCreated, 0},
           {ActionEnum::kDestroyed, 1},
           {ActionEnum::kDestroyed, 0}}));
}

TEST_F(MetadataCacheTest, stressParallelDuplicates) {
  std::atomic_int counter{0};
  MetadataCache<int, int> cache{[&](int id) {
    ++counter;
    return std::make_shared<int>(id);
  }};
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  facebook::velox::dwio::common::ExecutorBarrier barrier(executor);
  constexpr int kEntryIds = 100;
  constexpr int kEntryDuplicates = 10;
  for (int i = 0; i < kEntryIds; ++i) {
    for (int n = 0; n < kEntryDuplicates; ++n) {
      barrier.add([i, &cache]() {
        auto e = cache.getOrCreate(i);
        EXPECT_EQ(*e, i);
      });
    }
  }
  barrier.waitAll();
  EXPECT_GE(counter.load(), kEntryIds);
}

TEST_F(MetadataCacheTest, stressParallelDuplicatesSaveEntries) {
  std::atomic_int counter{0};
  MetadataCache<int, int> cache{[&](int id) {
    ++counter;
    return std::make_shared<int>(id);
  }};
  folly::Synchronized<std::vector<std::shared_ptr<int>>> entries;
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  facebook::velox::dwio::common::ExecutorBarrier barrier(executor);
  constexpr int kEntryIds = 100;
  constexpr int kEntryDuplicates = 10;
  for (int i = 0; i < kEntryIds; ++i) {
    for (int n = 0; n < kEntryDuplicates; ++n) {
      barrier.add([i, &cache, &entries]() {
        auto e = cache.getOrCreate(i);
        EXPECT_EQ(*e, i);
        entries.wlock()->push_back(e);
      });
    }
  }
  barrier.waitAll();
  // Duplicate builds may occur under contention (double-check lock pattern).
  EXPECT_GE(counter.load(), kEntryIds);
}

TEST_F(MetadataCacheTest, stress) {
  std::atomic_int counter{0};
  MetadataCache<int, int> cache{[&](int id) {
    ++counter;
    return std::make_shared<int>(id);
  }};
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  facebook::velox::dwio::common::ExecutorBarrier barrier(executor);
  constexpr int kEntryIds = 100;
  constexpr int kEntryDuplicates = 10;
  for (int n = 0; n < kEntryDuplicates; ++n) {
    for (int i = 0; i < kEntryIds; ++i) {
      barrier.add([i, &cache]() {
        auto e = cache.getOrCreate(i);
        EXPECT_EQ(*e, i);
      });
    }
  }
  barrier.waitAll();
  EXPECT_GE(counter.load(), kEntryIds);
}

TEST_F(MetadataCacheTest, stressSaveEntries) {
  std::atomic_int counter{0};
  MetadataCache<int, int> cache{[&](int id) {
    ++counter;
    return std::make_shared<int>(id);
  }};
  folly::Synchronized<std::vector<std::shared_ptr<int>>> entries;
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  facebook::velox::dwio::common::ExecutorBarrier barrier(executor);
  constexpr int kEntryIds = 100;
  constexpr int kEntryDuplicates = 10;
  for (int n = 0; n < kEntryDuplicates; ++n) {
    for (int i = 0; i < kEntryIds; ++i) {
      barrier.add([i, &cache, &entries]() {
        auto e = cache.getOrCreate(i);
        EXPECT_EQ(*e, i);
        entries.wlock()->push_back(e);
      });
    }
  }
  barrier.waitAll();
  // Duplicate builds may occur under contention (double-check lock pattern).
  EXPECT_GE(counter.load(), kEntryIds);
}
