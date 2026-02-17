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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"

using namespace facebook::velox;
using namespace std::chrono_literals;

namespace {
struct Gizmo {
  explicit Gizmo(int32_t _id) : id(_id) {}

  const int32_t id;
};

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
  inline static std::atomic<uint64_t> numCreatedDataCounters_{0};
  inline static std::atomic<uint64_t> numDeletedDataCounters_{0};

  uint64_t objectNumber_{0};
};

void verifyContexts(
    const std::string& expectedPoolName,
    const std::string& expectedTaskId) {
  EXPECT_EQ(process::GetThreadDebugInfo()->taskId_, expectedTaskId);
}
} // namespace

class AsyncSourceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    common::testutil::TestValue::enable();
  }

  void SetUp() override {
    DataCounter::reset();
  }

  void TearDown() override {
    DataCounter::reset();
  }
};

TEST_F(AsyncSourceTest, basic) {
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
  VELOX_ASSERT_USER_THROW(error.move(), "Testing error");
  EXPECT_TRUE(error.hasValue());
}

TEST_F(AsyncSourceTest, close) {
  {
    auto dateCounter = std::make_shared<DataCounter>();
    AsyncSource<uint64_t> countAsyncSource([dateCounter]() {
      return std::make_unique<uint64_t>(dateCounter->objectNumber());
    });
    dateCounter.reset();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 0);
    EXPECT_EQ(0, countAsyncSource.prepareTiming().count);

    countAsyncSource.close();

    EXPECT_EQ(0, countAsyncSource.prepareTiming().count);
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
  }
  DataCounter::reset();

  {
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        []() { return std::make_unique<DataCounter>(); });
    asyncSource->prepare();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 0);
    EXPECT_EQ(1, asyncSource->prepareTiming().count);

    asyncSource->close();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
    EXPECT_EQ(1, asyncSource->prepareTiming().count);
  }
  DataCounter::reset();

  {
    folly::Baton<> baton;
    auto sleepAsyncSource =
        std::make_shared<AsyncSource<DataCounter>>([&baton]() {
          baton.post();
          return std::make_unique<DataCounter>();
        });
    auto thread =
        std::thread([&sleepAsyncSource] { sleepAsyncSource->prepare(); });
    EXPECT_TRUE(baton.try_wait_for(1s));
    sleepAsyncSource->close();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
    EXPECT_EQ(1, sleepAsyncSource->prepareTiming().count);
    thread.join();
  }
}

TEST_F(AsyncSourceTest, emptyContexts) {
  EXPECT_EQ(process::GetThreadDebugInfo(), nullptr);

  AsyncSource<bool> src([]() {
    verifyContexts("test", "task_id");
    return std::make_unique<bool>(true);
  });

  process::ThreadDebugInfo debugInfo{"query_id", "task_id", nullptr};
  process::ScopedThreadDebugInfo scopedDebugInfo(debugInfo);

  verifyContexts("test", "task_id");

  ASSERT_TRUE(*src.move());

  verifyContexts("test", "task_id");
}

TEST_F(AsyncSourceTest, setContexts) {
  process::ThreadDebugInfo debugInfo1{"query_id1", "task_id1", nullptr};

  std::unique_ptr<AsyncSource<bool>> src;
  process::ScopedThreadDebugInfo scopedDebugInfo1(debugInfo1);

  verifyContexts("test1", "task_id1");

  src = std::make_unique<AsyncSource<bool>>(([]() {
    verifyContexts("test1", "task_id1");
    return std::make_unique<bool>(true);
  }));

  process::ThreadDebugInfo debugInfo2{"query_id2", "task_id2", nullptr};
  process::ScopedThreadDebugInfo scopedDebugInfo2(debugInfo2);

  verifyContexts("test2", "task_id2");

  ASSERT_TRUE(*src->move());

  verifyContexts("test2", "task_id2");
}

TEST_F(AsyncSourceTest, cancel) {
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
    EXPECT_TRUE(startBaton.try_wait_for(1s));

    asyncSource->cancel();

    finishBaton.post();
    thread.join();

    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
    EXPECT_TRUE(asyncSource->hasValue());
    asyncSource->close();
    EXPECT_EQ(1, asyncSource->prepareTiming().count);
  }
  DataCounter::reset();

  {
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        []() { return std::make_unique<DataCounter>(); });
    asyncSource->prepare();

    asyncSource->cancel();

    EXPECT_FALSE(asyncSource->hasValue());
    EXPECT_EQ(asyncSource->move(), nullptr);
    EXPECT_EQ(1, asyncSource->prepareTiming().count);
  }
  DataCounter::reset();

  {
    std::atomic_bool taskExecuted{false};
    auto asyncSource =
        std::make_shared<AsyncSource<DataCounter>>([&taskExecuted]() {
          taskExecuted = true;
          return std::make_unique<DataCounter>();
        });

    asyncSource->cancel();
    asyncSource->prepare();
    EXPECT_FALSE(taskExecuted);
    EXPECT_FALSE(asyncSource->hasValue());

    EXPECT_EQ(asyncSource->move(), nullptr);
    EXPECT_FALSE(taskExecuted);
  }

  {
    auto dataCounter = std::make_shared<DataCounter>();
    auto asyncSource = std::make_shared<AsyncSource<uint64_t>>([dataCounter]() {
      return std::make_unique<uint64_t>(dataCounter->objectNumber());
    });
    dataCounter.reset();

    asyncSource->cancel();
    asyncSource->cancel();
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
    EXPECT_EQ(0, asyncSource->prepareTiming().count);
  }
  DataCounter::reset();

  {
    folly::Baton<> moveStarted;
    folly::Baton<> continueMove;
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        [&moveStarted, &continueMove]() {
          moveStarted.post();
          continueMove.wait();
          return std::make_unique<DataCounter>();
        });

    auto moveThread = std::thread([&asyncSource] {
      auto result = asyncSource->move();
      EXPECT_NE(result, nullptr);
    });

    EXPECT_TRUE(moveStarted.try_wait_for(1s));

    asyncSource->cancel();

    continueMove.post();
    moveThread.join();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  }
  DataCounter::reset();

  {
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        []() { return std::make_unique<DataCounter>(); });

    auto result = asyncSource->move();
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);

    asyncSource->cancel();

    EXPECT_EQ(1, asyncSource->prepareTiming().count);
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);
  }
  DataCounter::reset();

  // Cancel called after close() - should be no-op.
  {
    auto asyncSource = std::make_shared<AsyncSource<DataCounter>>(
        []() { return std::make_unique<DataCounter>(); });
    asyncSource->prepare();
    EXPECT_EQ(DataCounter::numCreatedDataCounters(), 1);

    asyncSource->close();
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);

    asyncSource->cancel();
    EXPECT_EQ(DataCounter::numDeletedDataCounters(), 1);
    EXPECT_EQ(1, asyncSource->prepareTiming().count);
  }
}

TEST_F(AsyncSourceTest, multithreadedPrepareAndMove) {
  constexpr int32_t kNumThreads = 10;
  constexpr int32_t kNumGizmos = 2000;
  folly::Synchronized<std::unordered_set<int32_t>> results;
  std::vector<std::shared_ptr<AsyncSource<Gizmo>>> gizmos;
  gizmos.reserve(kNumGizmos);
  for (auto i = 0; i < kNumGizmos; ++i) {
    gizmos.push_back(std::make_shared<AsyncSource<Gizmo>>([i]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); // NOLINT
      return std::make_unique<Gizmo>(i);
    }));
  }

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t threadIndex = 0; threadIndex < kNumThreads; ++threadIndex) {
    threads.emplace_back([threadIndex, &gizmos, &results]() {
      if (threadIndex < kNumThreads / 2) {
        for (auto i = 0; i < kNumGizmos; ++i) {
          gizmos[i]->prepare();
        }
      } else {
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
    });
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

TEST_F(AsyncSourceTest, multithreadedErrorHandling) {
  constexpr int32_t kNumGizmos = 50;
  constexpr int32_t kNumThreads = 10;
  std::vector<std::shared_ptr<AsyncSource<Gizmo>>> gizmos;
  std::atomic<int32_t> numErrors{0};
  gizmos.reserve(kNumGizmos);
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
    threads.emplace_back([threadIndex, &gizmos, &numErrors]() {
      if (threadIndex < kNumThreads / 2) {
        for (auto i = 0; i < kNumGizmos; ++i) {
          gizmos[i]->prepare();
        }
      } else {
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
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_LT(0, numErrors);
  for (auto& source : gizmos) {
    source->close();
  }
}

DEBUG_ONLY_TEST_F(AsyncSourceTest, concurrentMoveSteal) {
  // Test scenario: first move() waits for making
  // it gets signaled through promises, but a second move() comes in
  // between the wait completion and lock re-acquisition and steals the item.
  // The first move() should get nothing.
  folly::Baton<> makingStarted;
  folly::Baton<> makingContinue;
  folly::Baton<> firstMoveWaiting;
  folly::Baton<> secondMoveComplete;

  auto asyncSource =
      std::make_shared<AsyncSource<Gizmo>>([&makingStarted, &makingContinue]() {
        makingStarted.post();
        makingContinue.wait();
        return std::make_unique<Gizmo>(42);
      });

  std::atomic<Gizmo*> firstMoveResult{nullptr};
  std::atomic<Gizmo*> secondMoveResult{nullptr};
  std::unique_ptr<Gizmo> firstMoveHolder;
  std::unique_ptr<Gizmo> secondMoveHolder;

  SCOPED_TESTVALUE_SET(
      "facebook::velox::AsyncSource::makeWait",
      std::function<void(AsyncSource<Gizmo>*)>([&](AsyncSource<Gizmo>* source) {
        // Signal that first move is about to re-acquire lock.
        firstMoveWaiting.post();
        // Wait for second move to complete and steal the item.
        secondMoveComplete.wait();
      }));

  // Thread 1: First move() - will wait for making and then get blocked by
  // TestValue.
  auto firstMoveThread = std::thread([&]() {
    firstMoveHolder = asyncSource->move();
    firstMoveResult = firstMoveHolder.get();
  });

  // Thread 2: prepare() - starts making the item.
  auto prepareThread = std::thread([&]() { asyncSource->prepare(); });

  // Wait for making to start.
  ASSERT_TRUE(makingStarted.try_wait_for(1s));

  // Let making complete - this will signal the first move's promise.
  makingContinue.post();

  // Wait for first move to be signaled and about to re-acquire lock.
  ASSERT_TRUE(firstMoveWaiting.try_wait_for(1s));

  // Thread 3: Second move() - steals the item while first move is blocked.
  auto secondMoveThread = std::thread([&]() {
    secondMoveHolder = asyncSource->move();
    secondMoveResult = secondMoveHolder.get();
    secondMoveComplete.post();
  });

  firstMoveThread.join();
  secondMoveThread.join();
  prepareThread.join();

  // Second move should have stolen the item.
  EXPECT_NE(secondMoveResult.load(), nullptr);
  EXPECT_EQ(secondMoveResult.load()->id, 42);

  // First move should get nothing because second move stole the item.
  EXPECT_EQ(firstMoveResult.load(), nullptr);
}

DEBUG_ONLY_TEST_F(AsyncSourceTest, concurrentMoveCloseRace) {
  // Test scenario: Tests the race condition where move()
  // preparation and close() sneaks in to grab the item first.
  //
  // Timeline:
  //   1. prepare() starts making the item in a background thread
  //   2. move() enters and waits for preparation to complete (blocked on
  //   promise)
  //   3. prepare() completes and signals the promise
  //   4. move() wakes up but TestValue blocks it before re-acquiring the lock
  //   5. close() runs and transitions state to kFinished, clearing the item
  //   6. move() finally re-acquires lock but finds state is kFinished
  //   7. move() returns nullptr
  //
  // Expected: move() gets nullptr because close() grabbed the item first.
  folly::Baton<> makingStarted;
  folly::Baton<> makingContinue;
  folly::Baton<> moveWaiting;
  folly::Baton<> closeComplete;

  auto asyncSource =
      std::make_shared<AsyncSource<Gizmo>>([&makingStarted, &makingContinue]() {
        makingStarted.post();
        makingContinue.wait();
        return std::make_unique<Gizmo>(42);
      });

  std::atomic<Gizmo*> moveResult{nullptr};
  std::unique_ptr<Gizmo> moveHolder;

  SCOPED_TESTVALUE_SET(
      "facebook::velox::AsyncSource::makeWait",
      std::function<void(AsyncSource<Gizmo>*)>([&](AsyncSource<Gizmo>* source) {
        // Signal that move is about to re-acquire lock.
        moveWaiting.post();
        // Wait for close to complete.
        closeComplete.wait();
      }));

  // Thread 1: move() - will wait for making and then get blocked by TestValue.
  auto moveThread = std::thread([&]() {
    moveHolder = asyncSource->move();
    moveResult = moveHolder.get();
  });

  // Thread 2: prepare() - starts making the item.
  auto prepareThread = std::thread([&]() { asyncSource->prepare(); });

  // Wait for making to start.
  ASSERT_TRUE(makingStarted.try_wait_for(1s));

  // Let making complete - this will signal move's promise.
  makingContinue.post();

  // Wait for move to be signaled and about to re-acquire lock.
  ASSERT_TRUE(moveWaiting.try_wait_for(1s));

  // close() comes in and closes the item while move is blocked.
  asyncSource->close();
  closeComplete.post();

  moveThread.join();
  prepareThread.join();

  // move() should get nothing because close() grabbed the item first.
  EXPECT_EQ(moveResult.load(), nullptr);
}

DEBUG_ONLY_TEST_F(AsyncSourceTest, concurrentCloseMoveRace) {
  // Test scenario: Tests the race condition where close()
  // preparation and move() sneaks in to grab the item first.
  //
  // Timeline:
  //   1. prepare() starts making the item in a background thread
  //   2. close() enters and waits for preparation to complete (blocked on
  //   promise)
  //   3. prepare() completes and signals the promise
  //   4. close() wakes up but TestValue blocks it before re-acquiring the lock
  //   5. move() runs and transitions state to kFinished, taking the item
  //   6. close() finally re-acquires lock and finds state is kFinished
  //   7. close() completes successfully (nothing to close)
  //
  // Expected: move() gets the item, close() finds nothing to close.
  folly::Baton<> makingStarted;
  folly::Baton<> makingContinue;
  folly::Baton<> closeWaiting;
  folly::Baton<> moveComplete;

  auto asyncSource =
      std::make_shared<AsyncSource<Gizmo>>([&makingStarted, &makingContinue]() {
        makingStarted.post();
        makingContinue.wait();
        return std::make_unique<Gizmo>(42);
      });

  std::atomic<Gizmo*> moveResult{nullptr};
  std::unique_ptr<Gizmo> moveHolder;

  SCOPED_TESTVALUE_SET(
      "facebook::velox::AsyncSource::makeWait",
      std::function<void(AsyncSource<Gizmo>*)>([&](AsyncSource<Gizmo>* source) {
        // Signal that close is about to re-acquire lock.
        closeWaiting.post();
        // Wait for move to complete and take the item.
        moveComplete.wait();
      }));

  // Thread 1: prepare() - starts making the item.
  auto prepareThread = std::thread([&]() { asyncSource->prepare(); });

  // Wait for making to start.
  ASSERT_TRUE(makingStarted.try_wait_for(1s));

  // Thread 2: close() - will wait for making and then get blocked by TestValue.
  auto closeThread = std::thread([&]() { asyncSource->close(); });

  // Wait for close to be signaled and about to re-acquire lock.
  ASSERT_TRUE(closeWaiting.try_wait_for(1s));

  // Let making complete - this will signal close's promise.
  makingContinue.post();

  std::this_thread::sleep_for(std::chrono::seconds(1)); // NOLINT

  // move() comes in and takes the item while close is blocked.
  moveHolder = asyncSource->move();
  moveResult = moveHolder.get();
  moveComplete.post();

  closeThread.join();
  prepareThread.join();

  // move() should have taken the item.
  EXPECT_NE(moveResult.load(), nullptr);
  EXPECT_EQ(moveResult.load()->id, 42);
}

TEST_F(AsyncSourceTest, prepareTiming) {
  auto asyncSource = std::make_shared<AsyncSource<Gizmo>>([]() {
    std::this_thread::sleep_for(1s);
    return std::make_unique<Gizmo>(42);
  });

  asyncSource->prepare();

  const auto& timing = asyncSource->prepareTiming();
  EXPECT_EQ(timing.count, 1);
  EXPECT_GE(timing.wallNanos, 1'000'000'000);
  asyncSource->close();
}

TEST_F(AsyncSourceTest, itemMakerReturnsNull) {
  // Test when itemMaker returns nullptr via prepare().
  {
    auto asyncSource = std::make_shared<AsyncSource<Gizmo>>(
        []() -> std::unique_ptr<Gizmo> { return nullptr; });
    asyncSource->prepare();
    EXPECT_FALSE(asyncSource->hasValue());
    auto result = asyncSource->move();
    EXPECT_EQ(result, nullptr);
  }

  // Test when itemMaker returns nullptr via move() (inline making).
  {
    auto asyncSource = std::make_shared<AsyncSource<Gizmo>>(
        []() -> std::unique_ptr<Gizmo> { return nullptr; });
    auto result = asyncSource->move();
    EXPECT_EQ(result, nullptr);
  }
}
