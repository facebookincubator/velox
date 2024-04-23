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

#include <atomic>
#include <chrono>
#include <mutex>

#include "folly/Synchronized.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "velox/dwio/common/Budget.h"
#include "velox/dwio/common/ExecutorBarrier.h"

using namespace ::testing;
using namespace ::facebook::velox::dwio::common;

TEST(BudgetTests, CanUseZero) {
  Budget budget([&]() { return 0; });
  EXPECT_FALSE(budget.waitForBudget(0).has_value());
}

TEST(BudgetTests, CanUseHalf) {
  Budget budget([&]() { return 10; });
  EXPECT_FALSE(budget.waitForBudget(5).has_value());
}

TEST(BudgetTests, CanUseAll) {
  Budget budget([&]() { return 10; });
  EXPECT_FALSE(budget.waitForBudget(10).has_value());
}

TEST(BudgetTests, TwoCanUse) {
  Budget budget([&]() { return 10; });
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
}

TEST(BudgetTests, TwoCanUseAll) {
  Budget budget([&]() { return 10; });
  EXPECT_FALSE(budget.waitForBudget(5).has_value());
  EXPECT_FALSE(budget.waitForBudget(5).has_value());
}

TEST(BudgetTests, WillRefresh) {
  uint64_t budgetVar = 0;
  Budget budget([&]() { return budgetVar; });
  budgetVar = 1;
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  budgetVar += 1;
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  budgetVar += 1;
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  budgetVar += 1;
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  budgetVar += 1;
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  budgetVar += 1;
  EXPECT_FALSE(budget.waitForBudget(1).has_value());
  budget.releaseBudget(6);
  EXPECT_FALSE(budget.waitForBudget(6).has_value());
}

TEST(BudgetTests, WillUnblock) {
  folly::CPUThreadPoolExecutor executor(1);
  ExecutorBarrier barrier(executor);
  std::atomic_uint64_t budgetVar = 0;
  Budget budget([&]() { return budgetVar.load(); });
  barrier.add([&]() { EXPECT_TRUE(budget.waitForBudget(1).has_value()); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  budgetVar = 1;
  barrier.waitAll();
}

TEST(BudgetTests, OutOfLimits) {
  uint64_t budgetVar = std::numeric_limits<uint64_t>::max() - 1;
  auto budgetBig = [&]() { return budgetVar; };
  EXPECT_THAT(
      [&]() { Budget budget(budgetBig); },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::failingExpression,
          HasSubstr("value <= std::numeric_limits<int64_t>::max()"))));

  budgetVar = 1;
  Budget budget(budgetBig);
  budgetVar = std::numeric_limits<uint64_t>::max() - 1;
  EXPECT_THAT(
      [&]() { budget.waitForBudget(1); },
      Throws<facebook::velox::VeloxRuntimeError>(Property(
          &facebook::velox::VeloxRuntimeError::failingExpression,
          HasSubstr("value <= std::numeric_limits<int64_t>::max()"))));
}

TEST(BudgetTests, ContentionForBudget) {
  std::atomic_uint64_t simultaneous = 0;
  std::atomic_uint64_t maxSimultaneous = 0;
  std::atomic_uint64_t timesDidntWaitForBudget = 0;
  folly::CPUThreadPoolExecutor executor(3);
  ExecutorBarrier barrier(executor);
  Budget budget([]() { return 2; });
  folly::Synchronized<std::chrono::high_resolution_clock::duration>
      timeWaitingForBudget;
  // I tested with i = [0 1] and it already achieved maxSimultaneous = 2. So
  // setting 100 should be enough for this test not to be flaky, and to try to
  // run 3 simultaneously if it were possible (if there's a bug in the budget).
  for (int i = 0; i < 100; ++i) {
    barrier.add([&]() {
      auto timeWaiting = budget.waitForBudget(1);
      ++simultaneous;
      if (timeWaiting.has_value()) {
        timeWaitingForBudget.withWLock([&](auto& timeWaitingForBudget) {
          timeWaitingForBudget += timeWaiting.value();
        });
      } else {
        timesDidntWaitForBudget += 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      maxSimultaneous = std::max(maxSimultaneous.load(), simultaneous.load());
      --simultaneous;
      budget.releaseBudget(1);
    });
  }
  barrier.waitAll();
  EXPECT_EQ(maxSimultaneous.load(), 2);
  EXPECT_GT(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          *timeWaitingForBudget.rlock())
          .count(),
      0);
  EXPECT_GT(timesDidntWaitForBudget.load(), 0);
}
