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

#include "velox/common/base/AsyncSource.h"
#include <fmt/format.h>
#include <folly/Random.h>
#include <folly/Synchronized.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>
#include <thread>
#include "velox/common/base/Exceptions.h"

using namespace facebook::velox;
using namespace std::chrono_literals;

// A sample class to be constructed via AsyncSource.
struct Gizmo {
  explicit Gizmo(int32_t _id) : id(_id) {}

  const int32_t id;
};

TEST(AsyncSourceTest, basic) {
  AsyncSource<Gizmo> gizmo([]() { return std::make_unique<Gizmo>(11); });
  EXPECT_FALSE(gizmo.hasValue());
  gizmo.prepare();
  EXPECT_TRUE(gizmo.hasValue());
  auto value = gizmo.move();
  EXPECT_FALSE(gizmo.hasValue());
  EXPECT_EQ(11, value->id);
  EXPECT_EQ(1, gizmo.prepareTiming().count);

  AsyncSource<Gizmo> error(
      []() -> std::unique_ptr<Gizmo> { VELOX_USER_FAIL("Testing error"); });
  EXPECT_THROW(error.move(), VeloxException);
  EXPECT_TRUE(error.hasValue());
}

TEST(AsyncSourceTest, threads) {
  constexpr int32_t kNumThreads = 10;
  constexpr int32_t kNumGizmos = 2000;
  folly::Synchronized<std::unordered_set<int32_t>> results;
  std::vector<std::shared_ptr<AsyncSource<Gizmo>>> gizmos;
  for (auto i = 0; i < kNumGizmos; ++i) {
    gizmos.push_back(std::make_shared<AsyncSource<Gizmo>>([i]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); // NOLINT
      return std::make_unique<Gizmo>(i);
    }));
  }

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t threadIndex = 0; threadIndex < kNumThreads; ++threadIndex) {
    threads.push_back(std::thread([threadIndex, &gizmos, &results]() {
      if (threadIndex < kNumThreads / 2) {
        // The first half of the threads prepare Gizmos in the background.
        for (auto i = 0; i < kNumGizmos; ++i) {
          gizmos[i]->prepare();
        }
      } else {
        // The rest of the threads first get random Gizmos and then do a pass
        // over all the Gizmos to make sure all get collected. We assert that
        // each Gizmo is obtained once.
        folly::Random::DefaultGenerator rng;
        for (auto i = 0; i < kNumGizmos / 3; ++i) {
          auto gizmo =
              gizmos[folly::Random::rand32(rng) % gizmos.size()]->move();
          if (gizmo) {
            results.withWLock([&](auto& set) {
              EXPECT_TRUE(set.find(gizmo->id) == set.end());
              set.insert(gizmo->id);
            });
          }
        }
        for (auto i = 0; i < gizmos.size(); ++i) {
          auto gizmo = gizmos[i]->move();
          if (gizmo) {
            results.withWLock([&](auto& set) {
              EXPECT_TRUE(set.find(gizmo->id) == set.end());
              set.insert(gizmo->id);
            });
          }
        }
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  results.withRLock([&](auto& set) {
    for (auto i = 0; i < kNumGizmos; ++i) {
      EXPECT_TRUE(set.find(i) != set.end());
    }
  });
}

TEST(AsyncSourceTest, errorsWithThreads) {
  constexpr int32_t kNumGizmos = 50;
  constexpr int32_t kNumThreads = 10;
  std::vector<std::shared_ptr<AsyncSource<Gizmo>>> gizmos;
  std::atomic<int32_t> numErrors{0};
  for (auto i = 0; i < kNumGizmos; ++i) {
    gizmos.push_back(
        std::make_shared<AsyncSource<Gizmo>>([]() -> std::unique_ptr<Gizmo> {
          std::this_thread::sleep_for(std::chrono::milliseconds(1)); // NOLINT
          VELOX_USER_FAIL("Testing error");
        }));
  }

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t threadIndex = 0; threadIndex < kNumThreads; ++threadIndex) {
    threads.push_back(std::thread([threadIndex, &gizmos, &numErrors]() {
      if (threadIndex < kNumThreads / 2) {
        // The first half of the threads prepare Gizmos in the background.
        for (auto i = 0; i < kNumGizmos; ++i) {
          gizmos[i]->prepare();
        }
      } else {
        // The rest of the threads get random gizmos. They are
        // expected to produce an error or nullptr in the event
        // another thread is already waiting for the same gizmo.
        folly::Random::DefaultGenerator rng;
        for (auto i = 0; i < kNumGizmos / 3; ++i) {
          try {
            auto gizmo =
                gizmos[folly::Random::rand32(rng) % gizmos.size()]->move();
            EXPECT_EQ(nullptr, gizmo);
          } catch (std::exception&) {
            ++numErrors;
          }
        }
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // There will always be errors since the first to wait for any given
  // gizmo is sure to get an error.
  EXPECT_LT(0, numErrors);
  for (auto& source : gizmos) {
    source->close();
  }
}

class DataCounter {
 public:
  DataCounter() {
    objectNumber_ = ++numCreatedDataCounters_;
  }

  ~DataCounter() {
    ++numDeletedDataCounters_;
  }

  static uint64_t numCreatedDataCounters() {
    return numCreatedDataCounters_;
  }

  static uint64_t numDeletedDataCounters() {
    return numDeletedDataCounters_;
  }

  static void reset() {
    numCreatedDataCounters_ = 0;
    numDeletedDataCounters_ = 0;
  }

  uint64_t objectNumber() const {
    return objectNumber_;
  }

 private:
  static std::atomic<uint64_t> numCreatedDataCounters_;
  static std::atomic<uint64_t> numDeletedDataCounters_;

  uint64_t objectNumber_{0};
};

std::atomic<uint64_t> DataCounter::numCreatedDataCounters_ = 0;

std::atomic<uint64_t> DataCounter::numDeletedDataCounters_ = 0;

TEST(AsyncSourceTest, close) {
  // If 'prepare()' is not executed within the thread pool, invoking 'close()'
  // will set 'make_' to nullptr. The deletion of 'dateCounter' is used as a
  // verification for this behavior.
  auto dateCounter = std::make_shared<DataCounter>();
  AsyncSource<uint64_t> countAsyncSource([dateCounter]() {
    return std::make_unique<uint64_t>(dateCounter->objectNumber());
  });
  dateCounter.reset();
  EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  EXPECT_EQ(DataCounter::numDeletedDataCounters(), 0);

  countAsyncSource.close();
  EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
  DataCounter::reset();

  // If 'prepare()' is executed within the thread pool but 'move()' is not
  // invoked, invoking 'close()' will set 'item_' to nullptr. The deletion of
  // 'dateCounter' is used as a verification for this behavior.
  auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
      []() { return std::make_unique<DataCounter>(); });
  asyncSource->prepare();
  EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  EXPECT_EQ(DataCounter::numDeletedDataCounters(), 0);

  asyncSource->close();
  EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
  DataCounter::reset();

  // If 'prepare()' is currently being executed within the thread pool,
  // 'close()' should wait for the completion of 'prepare()' and set 'item_' to
  // nullptr.
  folly::Baton<> baton;
  auto sleepAsyncSource =
      std::make_shared<AsyncSource<DataCounter>>([&baton]() {
        baton.post();
        return std::make_unique<DataCounter>();
      });
  auto thread1 =
      std::thread([&sleepAsyncSource] { sleepAsyncSource->prepare(); });
  EXPECT_TRUE(baton.try_wait_for(1s));
  sleepAsyncSource->close();
  EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
  thread1.join();
}

void verifyContexts(
    const std::string& expectedPoolName,
    const std::string& expectedTaskId) {
  EXPECT_EQ(process::GetThreadDebugInfo()->taskId_, expectedTaskId);
}

TEST(AsyncSourceTest, emptyContexts) {
  EXPECT_EQ(process::GetThreadDebugInfo(), nullptr);

  AsyncSource<bool> src([]() {
    // The Contexts at the time this was created were null so we should inherit
    // them from the caller.
    verifyContexts("test", "task_id");

    return std::make_unique<bool>(true);
  });

  process::ThreadDebugInfo debugInfo{"query_id", "task_id", nullptr};
  process::ScopedThreadDebugInfo scopedDebugInfo(debugInfo);

  verifyContexts("test", "task_id");

  ASSERT_TRUE(*src.move());

  verifyContexts("test", "task_id");
}

TEST(AsyncSourceTest, setContexts) {
  process::ThreadDebugInfo debugInfo1{"query_id1", "task_id1", nullptr};

  std::unique_ptr<AsyncSource<bool>> src;
  process::ScopedThreadDebugInfo scopedDebugInfo1(debugInfo1);

  verifyContexts("test1", "task_id1");

  src = std::make_unique<AsyncSource<bool>>(([]() {
    // The Contexts at the time this was created were set so we should have
    // the same contexts when this is executed.
    verifyContexts("test1", "task_id1");

    return std::make_unique<bool>(true);
  }));

  process::ThreadDebugInfo debugInfo2{"query_id2", "task_id2", nullptr};
  process::ScopedThreadDebugInfo scopedDebugInfo2(debugInfo2);

  verifyContexts("test2", "task_id2");

  ASSERT_TRUE(*src->move());

  verifyContexts("test2", "task_id2");
}

TEST(AsyncSourceTest, cancel) {
  DataCounter::reset();

  // Cancel before prepare() - task should not run and resources cleaned up
  {
    auto dataCounter = std::make_shared<DataCounter>();
    auto asyncSource = std::make_shared<AsyncSource<uint64_t>>([dataCounter]() {
      return std::make_unique<uint64_t>(dataCounter->objectNumber());
    });
    dataCounter.reset();

    asyncSource->cancel();
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
  }
  DataCounter::reset();

  // Cancel while task is running in prepare() - should not affect the
  // prepare() result
  {
    folly::Baton<> startBaton;
    folly::Baton<> finishBaton;
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        [&startBaton, &finishBaton]() {
          startBaton.post();
          finishBaton.wait();
          return std::make_unique<DataCounter>();
        });

    auto thread = std::thread([&asyncSource] { asyncSource->prepare(); });
    EXPECT_TRUE(
        startBaton.try_wait_for(1s)); // Make sure prepare() gets lock first

    asyncSource->cancel(); // Should be no-op

    finishBaton.post();
    thread.join();

    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_TRUE(asyncSource->hasValue());
  }
  DataCounter::reset();

  // Cancel after prepare() completes - should not destroy the result
  {
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        []() { return std::make_unique<DataCounter>(); });
    asyncSource->prepare();

    asyncSource->cancel(); // Should be no-op since make_ was taken

    EXPECT_TRUE(asyncSource->hasValue());
    EXPECT_NE(asyncSource->move(), nullptr);
  }
  DataCounter::reset();

  // prepare() and move() are no-ops after cancel()
  {
    std::atomic<bool> taskExecuted{false};
    auto asyncSource =
        std::make_shared<AsyncSource<DataCounter>>([&taskExecuted]() {
          taskExecuted = true;
          return std::make_unique<DataCounter>();
        });

    asyncSource->cancel();
    asyncSource->prepare(); // No-op
    EXPECT_FALSE(taskExecuted);
    EXPECT_FALSE(asyncSource->hasValue());

    EXPECT_EQ(asyncSource->move(), nullptr); // No-op
    EXPECT_FALSE(taskExecuted);
  }

  // Multiple cancel calls are idempotent
  {
    auto dataCounter = std::make_shared<DataCounter>();
    auto asyncSource = std::make_shared<AsyncSource<uint64_t>>([dataCounter]() {
      return std::make_unique<uint64_t>(dataCounter->objectNumber());
    });
    dataCounter.reset();

    asyncSource->cancel();
    asyncSource->cancel(); // Should be safe
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
  }
  DataCounter::reset();

  // Cancel called during move() execution - should be no-op
  {
    folly::Baton<> moveStarted;
    folly::Baton<> continueMove;
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        [&moveStarted, &continueMove]() {
          moveStarted.post();
          continueMove.wait();
          return std::make_unique<DataCounter>();
        });

    // move() will execute the lambda inline since prepare() wasn't called
    auto moveThread = std::thread([&asyncSource] {
      auto result = asyncSource->move();
      EXPECT_NE(result, nullptr);
    });

    EXPECT_TRUE(moveStarted.try_wait_for(1s)); // Wait for move to start

    asyncSource->cancel(); // Should be no-op - make_ already taken

    continueMove.post();
    moveThread.join();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  }
  DataCounter::reset();

  // Cancel called after move() completes - should be no-op
  {
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        []() { return std::make_unique<DataCounter>(); });

    auto result = asyncSource->move(); // Complete move
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);

    asyncSource->cancel(); // Should be no-op - moved_ is true, make_ is null

    // Item was already consumed
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  }
  DataCounter::reset();
}
