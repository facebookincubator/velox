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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "velox/dwio/common/ExecutorBarrier.h"
#include "velox/dwio/common/exception/Exception.h"

using namespace ::testing;
using namespace ::facebook::velox::dwio::common;

namespace {

class MockExecutor : public folly::Executor {
 public:
  ~MockExecutor() override = default;

  MOCK_METHOD(void, add, (folly::Func func), (override));
  MOCK_METHOD(void, addWithPriority, (folly::Func, int8_t), (override));
  MOCK_METHOD(uint8_t, getNumPriorities, (), (const, override));
};

} // namespace

TEST(ExecutorBarrierTest, GetNumPriorities) {
  const uint8_t kNumPriorities = 5;
  auto executor =
      std::make_shared<folly::CPUThreadPoolExecutor>(10, kNumPriorities);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);
  EXPECT_EQ(barrier->getNumPriorities(), kNumPriorities);
}

TEST(ExecutorBarrierTest, CanOwn) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  {
    auto barrier = std::make_shared<ExecutorBarrier>(executor);
    EXPECT_EQ(executor.use_count(), 2);
  }
  EXPECT_EQ(executor.use_count(), 1);
}

TEST(ExecutorBarrierTest, ThrowsIfLambdaNotExecuted) {
  MockExecutor mockExecutor;
  auto barrier = std::make_shared<ExecutorBarrier>(mockExecutor);
  // Do nothing on call, so the lambda will never be executed.
  EXPECT_CALL(mockExecutor, add(_)).Times(1);
  std::atomic<bool> executed{false};
  barrier->add([&executed]() { executed = true; });
  EXPECT_THROW(
      barrier->waitAll(),
      facebook::velox::dwio::common::exception::LoggedException);
  EXPECT_FALSE(executed);
}

TEST(ExecutorBarrierTest, CanAwaitMultipleTimes) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);
  for (int time = 0, multipleTimes = 10; time < multipleTimes; ++time) {
    barrier->waitAll();
  }
}

TEST(ExecutorBarrierTest, AddCanBeReused) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([&]() { ++count; });
  }
  barrier->waitAll();
  EXPECT_EQ(count, kCalls);

  for (int i = 0; i < kCalls; ++i) {
    barrier->add([&]() { ++count; });
  }
  barrier->waitAll();
  EXPECT_EQ(count, (2 * kCalls));
}

TEST(ExecutorBarrierTest, AddWithPriorityCanBeReused) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  const int8_t kPriority = 4;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority([&]() { ++count; }, kPriority);
  }
  barrier->waitAll();
  EXPECT_EQ(count, kCalls);

  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority([&]() { ++count; }, kPriority);
  }
  barrier->waitAll();
  EXPECT_EQ(count, (2 * kCalls));
}

TEST(ExecutorBarrierTest, AddCanBeReusedAfterException) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([&count]() {
      ++count;
      throw std::runtime_error("");
    });
  }
  EXPECT_THROW(barrier->waitAll(), std::runtime_error);
  EXPECT_EQ(count, kCalls);

  for (int i = 0; i < kCalls; ++i) {
    barrier->add([&]() { ++count; });
  }
  barrier->waitAll();
  EXPECT_EQ(count, (2 * kCalls));
}

TEST(ExecutorBarrierTest, AddWithPriorityCanBeReusedAfterException) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  const int8_t kPriority = 4;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority(
        [&count]() {
          ++count;
          throw std::runtime_error("");
        },
        kPriority);
  }
  EXPECT_THROW(barrier->waitAll(), std::runtime_error);
  EXPECT_EQ(count, kCalls);

  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority([&]() { ++count; }, kPriority);
  }
  barrier->waitAll();
  EXPECT_EQ(count, (2 * kCalls));
}

TEST(ExecutorBarrierTest, Add) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([&]() { ++count; });
  }
  barrier->waitAll();
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddWithPriority) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  const int8_t kPriority = 4;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority([&]() { ++count; }, kPriority);
  }
  barrier->waitAll();
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddCanIgnore) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([]() {});
  }
  // Discard: barrier->waitAll();
}

TEST(ExecutorBarrierTest, AddWithPriorityCanIgnore) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority([]() {}, i);
  }
  // Discard: barrier->waitAll();
}

TEST(ExecutorBarrierTest, DestructorDoesntThrow) {
  const int kCalls = 30;
  std::atomic<int> count{0};
  {
    auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
    auto barrier = std::make_shared<ExecutorBarrier>(*executor);

    for (int i = 0; i < kCalls; ++i) {
      barrier->add([shouldThrow = (i == 0), &count]() {
        ++count;
        if (shouldThrow) {
          throw std::runtime_error("");
        }
      });
    }
  } // executor awaits but doesn't throw
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddException) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([shouldThrow = (i == 0), &count]() {
      ++count;
      if (shouldThrow) {
        throw std::runtime_error("");
      }
    });
  }
  EXPECT_THROW(barrier->waitAll(), std::runtime_error);
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddWithPriorityException) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  const int8_t kPriority = 4;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority(
        [shouldThrow = (i == 0), &count]() {
          ++count;
          if (shouldThrow) {
            throw std::runtime_error("");
          }
        },
        kPriority);
  }
  EXPECT_THROW(barrier->waitAll(), std::runtime_error);
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddNonStdException) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([shouldThrow = (i == 0), &count]() {
      ++count;
      if (shouldThrow) {
        // @lint-ignore CLANGTIDY facebook-hte-ThrowNonStdExceptionIssue
        throw 1;
      }
    });
  }
  EXPECT_THROW(barrier->waitAll(), int);
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddWithPriorityNonStdException) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  const int8_t kPriority = 4;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority(
        [shouldThrow = (i == 0), &count]() {
          ++count;
          if (shouldThrow) {
            // @lint-ignore CLANGTIDY facebook-hte-ThrowNonStdExceptionIssue
            throw 1;
          }
        },
        kPriority);
  }
  EXPECT_THROW(barrier->waitAll(), int);
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddExceptions) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->add([&]() {
      ++count;
      throw std::runtime_error("");
    });
  }
  EXPECT_THROW(barrier->waitAll(), std::runtime_error);
  EXPECT_EQ(count, kCalls);
}

TEST(ExecutorBarrierTest, AddWithPriorityExceptions) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(10);
  auto barrier = std::make_shared<ExecutorBarrier>(*executor);

  const int kCalls = 30;
  const int8_t kPriority = 4;
  std::atomic<int> count{0};
  for (int i = 0; i < kCalls; ++i) {
    barrier->addWithPriority(
        [&]() {
          ++count;
          throw std::runtime_error("");
        },
        kPriority);
  }
  EXPECT_THROW(barrier->waitAll(), std::runtime_error);
  EXPECT_EQ(count, kCalls);
}
