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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <thread>
#include "dwio/common/exception/Exception.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/executors/InlineExecutor.h"
#include "velox/dwio/common/ExecutorBarrier.h"
#include "velox/dwio/common/PercentUnitLoader.h"
#include "velox/dwio/common/UnitLoader.h"
#include "velox/dwio/common/UnitLoaderTools.h"
#include "velox/dwio/common/tests/utils/UnitLoaderTestTools.h"

using namespace ::testing;
using namespace ::facebook::velox::dwio::common;
using facebook::velox::dwio::common::test::getUnitsLoadedWithFalse;
using facebook::velox::dwio::common::test::LoadUnitMock;
using facebook::velox::dwio::common::test::ReaderMock;

namespace {

class StepByStepExecutor : public folly::Executor {
 public:
  ~StepByStepExecutor() override = default;

  /// Enqueue a function to be executed by this executor. This and all
  /// variants must be threadsafe.
  void add(folly::Func f) override {
    tasks_.push_back(std::move(f));
  }

  /// Enqueue a function with a given priority, where 0 is the medium priority
  /// This is up to the implementation to enforce
  void addWithPriority(folly::Func /* f */, int8_t /* priority */) override {
    DWIO_RAISE("NYI");
  }

  uint8_t getNumPriorities() const override {
    return 1;
  }

  bool runOne() {
    if (tasks_.empty()) {
      return false;
    }
    tasks_.front()();
    tasks_.pop_front();
    return true;
  }

 public:
  std::deque<folly::Func> tasks_;
};

class PercentUnitLoaderPTests : public ::testing::TestWithParam<uint32_t> {};

class PercentUnitLoaderEdgeTests : public ::testing::TestWithParam<uint32_t> {};

class MockLoadUnit : public LoadUnit {
 public:
  MOCK_METHOD(void, load, (), ());
  MOCK_METHOD(void, unload, (), ());
  MOCK_METHOD(uint64_t, getNumRows, (), ());
  MOCK_METHOD(uint64_t, getIoSize, (), ());
};

} // namespace

TEST(PercentUnitLoaderTests, IsExceptionSafe) {
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      300, // prefetch 3 units
      nullptr,
      firstLoadExecutor,
      loadExecutor,
      nullptr,
      nullptr,
      nullptr);

  std::vector<std::unique_ptr<facebook::velox::dwio::common::LoadUnit>>
      loadUnits;
  loadUnits.reserve(3);

  auto unit1 = std::make_unique<MockLoadUnit>();
  auto unit2 = std::make_unique<MockLoadUnit>();
  auto unit3 = std::make_unique<MockLoadUnit>();

  EXPECT_CALL(*unit1, load())
      .Times(1)
      .WillOnce(Throw(std::runtime_error("Can't load unit 1")));
  EXPECT_CALL(*unit2, load()).Times(1);
  EXPECT_CALL(*unit3, load())
      .Times(1)
      .WillOnce(Throw(std::runtime_error("Can't load unit 3")));

  EXPECT_CALL(*unit1, getIoSize()).WillRepeatedly(Return(1));
  EXPECT_CALL(*unit2, getIoSize()).WillRepeatedly(Return(1));
  EXPECT_CALL(*unit3, getIoSize()).WillRepeatedly(Return(1));
  EXPECT_CALL(*unit1, getNumRows()).WillRepeatedly(Return(1));
  EXPECT_CALL(*unit2, getNumRows()).WillRepeatedly(Return(1));
  EXPECT_CALL(*unit3, getNumRows()).WillRepeatedly(Return(1));

  loadUnits.push_back(std::move(unit1));
  loadUnits.push_back(std::move(unit2));
  loadUnits.push_back(std::move(unit3));

  auto unitLoader = factory.create(std::move(loadUnits));
  // Prefetch 2 stripes. No exceptions should be thrown.
  EXPECT_NO_THROW(firstLoadExecutor.runOne());
  EXPECT_NO_THROW(firstLoadExecutor.runOne());

  // I should get the exceptions when I try to retrieve those stripes
  EXPECT_THAT(
      [&]() { unitLoader->getLoadedUnit(0); },
      ThrowsMessage<std::runtime_error>("Can't load unit 1"));
  unitLoader->getLoadedUnit(1);

  // Delay the load of the 3rd unit so we test the path that waits for the load.
  folly::CPUThreadPoolExecutor executor(1);
  executor.add([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    EXPECT_NO_THROW(firstLoadExecutor.runOne());
  });
  EXPECT_THAT(
      [&]() { unitLoader->getLoadedUnit(2); },
      ThrowsMessage<std::runtime_error>("Can't load unit 3"));
}

TEST(PercentUnitLoaderTests, WontTryToPrefetchMoreUnitsThanItHas) {
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      500, // try to prefetch 5 units
      nullptr,
      firstLoadExecutor,
      loadExecutor,
      nullptr,
      nullptr,
      nullptr);

  std::vector<std::unique_ptr<facebook::velox::dwio::common::LoadUnit>>
      loadUnits;
  loadUnits.reserve(3);

  auto unit1 = std::make_unique<MockLoadUnit>();
  auto unit2 = std::make_unique<MockLoadUnit>();
  auto unit3 = std::make_unique<MockLoadUnit>();

  EXPECT_NO_THROW(factory.create(std::move(loadUnits)));
}

TEST(PercentUnitLoaderTests, Case100) {
  auto budget = std::make_shared<Budget>([]() { return 8; });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      100,
      budget,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock{{10, 20, 30}, {1, 2, 4}, factory}; // schedule(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(5)); // Unit: 0, rows: 0-4, schedule(1)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(5)); // Unit: 0, rows: 5-9
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(5)); // Unit: 1, rows: 0-4, unload(1), schedule(2)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(15)); // Unit: 1, rows: 5-19
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);
}

TEST(PercentUnitLoaderTests, Case1000Exceeds) {
  // 1000% exceeds the number of stripes. Make sure it's supported
  auto budget = std::make_shared<Budget>([]() { return 8; });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      1000,
      budget,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock{
      {10, 20, 30},
      {1, 2, 4},
      factory}; // schedule(0), schedule(1), schedule(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, true}));
  EXPECT_FALSE(firstLoadExecutor.runOne()); // no-op
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, true}));
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(5)); // Unit: 0, rows: 0-4,
  EXPECT_EQ(noMoreIO, 1);
}

TEST(PercentUnitLoaderTests, Case100SharedBudget) {
  uint64_t budgetVar = 3;
  auto budget = std::make_shared<Budget>([&]() { return budgetVar; });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  facebook::velox::dwio::common::ExecutorBarrier executor(
      std::make_shared<folly::CPUThreadPoolExecutor>(10));
  PercentUnitLoaderFactory factory(
      100,
      budget,
      executor,
      executor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock1{{10, 20, 30}, {1, 10, 100}, factory}; // schedule(0)
  ReaderMock readerMock2{{10, 20, 30}, {1, 10, 100}, factory}; // schedule(0)
  ReaderMock readerMock3{{10, 20, 30}, {1, 10, 100}, factory}; // schedule(0)
  executor.waitAll();
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({true, false, false}));

  // No budget
  readerMock1.read(10); // schedule(1)
  readerMock2.read(10); // schedule(1)
  readerMock3.read(10); // schedule(1)
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(noMoreIO, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds{100});

  // Budget for one
  budgetVar += 10;

  std::this_thread::sleep_for(std::chrono::milliseconds{100});

  // Budget for two
  budgetVar += 20;

  executor.waitAll();
  EXPECT_GE(blockedOnBudgetCount, 2);
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(noMoreIO, 0);

  // Budget for one
  budgetVar += 99;
  readerMock1.read(20);
  executor.waitAll();
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(noMoreIO, 0);

  // No budget
  readerMock2.read(20);
  readerMock3.read(20);
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({false, true, false}));

  std::this_thread::sleep_for(std::chrono::milliseconds{100});

  // Budget for two
  budgetVar += 198;

  executor.waitAll();
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(noMoreIO, 1);
}

TEST(PercentUnitLoaderTests, Case100NoCallbacks) {
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      100, nullptr, firstLoadExecutor, loadExecutor, nullptr, nullptr, nullptr);
  ReaderMock readerMock{{10, 20, 30}, {1, 2, 4}, factory}; // schedule(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));

  EXPECT_TRUE(readerMock.read(5)); // Unit: 0, rows: 0-4, schedule(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(5)); // Unit: 0, rows: 5-9
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(5)); // Unit: 1, rows: 0-4, unload(0), schedule(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));

  EXPECT_TRUE(readerMock.read(15)); // Unit: 1, rows: 5-19
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
}

TEST(PercentUnitLoaderTests, Case70) {
  auto budget = std::make_shared<Budget>([]() { return 8; });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      70,
      budget,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock{{10, 20, 30}, {1, 2, 4}, factory}; // schedule(0)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5, schedule(1)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(5)); // Unit: 1, rows: 0-4, unload(0)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(15)); // Unit: 1, rows: 5-19, schedule(2)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);
}

TEST(PercentUnitLoaderTests, Case30) {
  auto budget = std::make_shared<Budget>([]() { return 8; });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      30,
      budget,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock{{10, 20, 30}, {1, 2, 4}, factory}; // schedule(0)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9, schedule(1)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(14)); // Unit: 1, rows: 0-13, unload(0)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  // will only read 5 rows, no more rows in unit 1
  EXPECT_TRUE(readerMock.read(10)); // Unit: 1, rows: 14-19, schedule(2)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);
}

TEST(PercentUnitLoaderTests, Case0) {
  auto budget = std::make_shared<Budget>([]() { return 8; });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      0,
      budget,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock{{10, 20, 30}, {1, 2, 4}, factory};
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_FALSE(firstLoadExecutor.runOne()); // no-op
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2, load(0)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 1);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 1);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 1);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(14)); // Unit: 1, rows: 0-13, unload(0), load(1)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(blockedOnIoCount, 2);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  // will only read 5 rows, no more rows in unit 1
  EXPECT_TRUE(readerMock.read(10)); // Unit: 1, rows: 14-19
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(blockedOnIoCount, 2);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1), load(2)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 3);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 3);
  EXPECT_EQ(blockedOnBudgetCount, 0);
}

TEST(PercentUnitLoaderTests, Case0SharedBudget) {
  std::atomic_uint64_t budgetVar = 1;
  auto budget = std::make_shared<Budget>([&]() { return budgetVar.load(); });
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  facebook::velox::dwio::common::ExecutorBarrier executor(
      std::make_shared<folly::CPUThreadPoolExecutor>(10));
  PercentUnitLoaderFactory factory(
      0,
      budget,
      executor,
      executor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock1{{10, 20, 30}, {1, 10, 100}, factory};
  ReaderMock readerMock2{{10, 20, 30}, {1, 10, 100}, factory};
  ReaderMock readerMock3{{10, 20, 30}, {1, 10, 100}, factory};
  executor.waitAll(); // nothing scheduled
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);
  EXPECT_EQ(
      readerMock1.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_EQ(
      readerMock2.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_EQ(
      readerMock3.unitsLoaded(), std::vector<bool>({false, false, false}));

  facebook::velox::dwio::common::ExecutorBarrier readExecutor(
      std::make_shared<folly::CPUThreadPoolExecutor>(3));

  // Budget just for one
  readExecutor.add([&]() {
    readerMock1.read(10);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    budgetVar += 1;
  });
  readExecutor.add([&]() {
    readerMock2.read(10);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    budgetVar += 1;
  });
  readExecutor.add([&]() {
    readerMock3.read(10);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    budgetVar += 1;
  });
  readExecutor.waitAll();

  // Two had to wait. But they could have waited multiple times. So at least 2.
  EXPECT_GE(blockedOnBudgetCount, 2);
  const size_t oldBlockedOnBudgetCount = blockedOnBudgetCount;
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({true, false, false}));
  EXPECT_EQ(noMoreIO, 0);

  // Budget for one
  budgetVar += 9;
  readerMock1.read(20);
  executor.waitAll();
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(noMoreIO, 0);

  // No budget
  readExecutor.add([&]() {
    readerMock2.read(20);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    budgetVar += 9;
  });
  readExecutor.add([&]() {
    readerMock3.read(20);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    budgetVar += 9;
  });

  budgetVar += 9;

  readExecutor.waitAll();
  EXPECT_GT(blockedOnBudgetCount, oldBlockedOnBudgetCount);
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(noMoreIO, 0);

  // Budget for three
  budgetVar += 300;
  readerMock1.read(20);
  readerMock2.read(20);
  readerMock3.read(20);
  executor.waitAll();
  EXPECT_EQ(readerMock1.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(readerMock2.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(readerMock3.unitsLoaded(), std::vector<bool>({false, false, true}));
  EXPECT_EQ(noMoreIO, 1);
}

TEST(PercentUnitLoaderTests, Case0NoCallbacks) {
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      0, nullptr, firstLoadExecutor, loadExecutor, nullptr, nullptr, nullptr);
  ReaderMock readerMock{{10, 20, 30}, {0, 0, 0}, factory};
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));
  EXPECT_FALSE(firstLoadExecutor.runOne()); // no-op
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, false}));

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2, load(0)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, false, false}));

  EXPECT_TRUE(readerMock.read(14)); // Unit: 1, rows: 0-13, unload(0), load(1)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));

  // will only read 5 rows, no more rows in unit 1
  EXPECT_TRUE(readerMock.read(10)); // Unit: 1, rows: 14-19
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, false}));

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1), load(2)
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
}

TEST(PercentUnitLoaderTests, Case150) {
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  std::atomic_size_t blockedOnBudgetCount(0);
  StepByStepExecutor firstLoadExecutor;
  StepByStepExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      150,
      nullptr,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      [&](auto) { ++blockedOnBudgetCount; });
  ReaderMock readerMock{
      {10, 20, 30, 40}, {0, 0, 0, 0}, factory}; // schedule(0), schedule(1)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(
      readerMock.unitsLoaded(),
      std::vector<bool>({false, false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(0)
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({true, false, false, false}));
  EXPECT_TRUE(firstLoadExecutor.runOne()); // fetch(1)
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({true, true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({true, true, false, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5, schedule(2)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({true, true, false, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(2)
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({true, true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({true, true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(5)); // Unit: 1, rows: 0-4, unload(0)
  EXPECT_EQ(noMoreIO, 0);
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({false, true, true, false}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(15)); // Unit: 1, rows: 5-19, schedule(3)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({false, true, true, false}));
  EXPECT_TRUE(loadExecutor.runOne()); // fetch(3)
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({false, true, true, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({false, false, true, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_TRUE(readerMock.read(40)); // Unit: 3, rows: 0-39, unload(2)
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({false, false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);

  EXPECT_FALSE(readerMock.read(1)); // No more data
  EXPECT_EQ(noMoreIO, 1); // No more IO
  EXPECT_EQ(
      readerMock.unitsLoaded(), std::vector<bool>({false, false, false, true}));
  EXPECT_EQ(blockedOnIoCount, 0);
  EXPECT_EQ(blockedOnBudgetCount, 0);
}

TEST(PercentUnitLoaderTests, WillEmitOneNoMoreIo) {
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  folly::InlineExecutor firstLoadExecutor;
  folly::InlineExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      100,
      nullptr,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      nullptr);

  ReaderMock readerMock1{{10}, {0}, factory}; // schedule(0)
  ReaderMock readerMock2{{11}, {0}, factory}; // schedule(0)
  ReaderMock readerMock3{{12}, {0}, factory}; // schedule(0)
  EXPECT_EQ(noMoreIO, 0);
  readerMock1.read(1);
  EXPECT_EQ(noMoreIO, 0);
  readerMock2.read(1);
  EXPECT_EQ(noMoreIO, 0);
  readerMock3.read(1); // The last one emits
  EXPECT_EQ(noMoreIO, 1);
}

TEST(PercentUnitLoaderTests, WillEmitOneNoMoreIoWhenLastReadersDestructed) {
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  folly::InlineExecutor firstLoadExecutor;
  folly::InlineExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      100,
      nullptr,
      firstLoadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      nullptr);

  {
    ReaderMock readerMock1{{10}, {0}, factory}; // schedule(0)
    {
      ReaderMock readerMock2{{11}, {0}, factory}; // schedule(0)
      {
        ReaderMock readerMock3{{12}, {0}, factory}; // schedule(0)
        EXPECT_EQ(noMoreIO, 0);
      }
      EXPECT_EQ(noMoreIO, 0);
    }
    EXPECT_EQ(noMoreIO, 0);
  } // The last destruction emits
  EXPECT_EQ(noMoreIO, 1);
}

TEST_P(PercentUnitLoaderPTests, UnitOutOfRange) {
  folly::InlineExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      GetParam(),
      nullptr,
      loadExecutor,
      loadExecutor,
      nullptr,
      nullptr,
      nullptr);
  {
    std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(1));
    std::vector<std::unique_ptr<LoadUnit>> units;
    units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));

    auto unitLoader = factory.create(std::move(units));
    unitLoader->getLoadedUnit(0);
    EXPECT_THAT(
        [&]() { unitLoader->getLoadedUnit(1); },
        Throws<
            facebook::velox::dwio::common::exception::LoggedException>(Property(
            &facebook::velox::dwio::common::exception::LoggedException::message,
            HasSubstr("Unit out of range"))));
  }
  {
    std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(1));
    std::vector<std::unique_ptr<LoadUnit>> units;
    units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));

    auto unitLoader = factory.create(std::move(units));
    unitLoader->onRead(0, 0, 1);
    EXPECT_THAT(
        [&]() { unitLoader->onRead(1, 0, 1); },
        Throws<
            facebook::velox::dwio::common::exception::LoggedException>(Property(
            &facebook::velox::dwio::common::exception::LoggedException::message,
            HasSubstr("Unit out of range"))));
  }
}

TEST_P(PercentUnitLoaderPTests, CanRequestUnitMultipleTimes) {
  folly::InlineExecutor loadExecutor;
  PercentUnitLoaderFactory factory(
      GetParam(),
      nullptr,
      loadExecutor,
      loadExecutor,
      nullptr,
      nullptr,
      nullptr);
  std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(1));
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));

  auto unitLoader = factory.create(std::move(units));
  unitLoader->getLoadedUnit(0);
  unitLoader->getLoadedUnit(0);
  unitLoader->getLoadedUnit(0);
}

INSTANTIATE_TEST_SUITE_P(
    OneHundredPercent,
    PercentUnitLoaderPTests,
    ValuesIn({100U}));

INSTANTIATE_TEST_SUITE_P(ZeroPercent, PercentUnitLoaderPTests, ValuesIn({0U}));

TEST_P(PercentUnitLoaderEdgeTests, WillReadAllRowsInEdgeCases) {
  std::atomic_size_t noMoreIO = 0;
  std::atomic_size_t blockedOnIoCount(0);
  folly::CPUThreadPoolExecutor loadExecutor(1);
  PercentUnitLoaderFactory factory(
      GetParam(),
      nullptr,
      loadExecutor,
      loadExecutor,
      [&]() { ++noMoreIO; },
      [&](auto) { ++blockedOnIoCount; },
      nullptr);
  std::vector<uint64_t> rowsInStripe;
  rowsInStripe.reserve(10);
  uint64_t rowsToRead = 0;
  for (uint64_t rows = 1; rows < 1024; rows *= 2) {
    rowsInStripe.push_back(rows);
    rowsToRead += rows;
  }
  std::vector<uint64_t> stripeBytes(rowsInStripe.size(), 0);
  ReaderMock readerMock{
      std::move(rowsInStripe), std::move(stripeBytes), factory};

  uint64_t rowsRead = 0;
  while (readerMock.read(1)) {
    ++rowsRead;
  }

  EXPECT_EQ(rowsToRead, rowsRead);
  EXPECT_TRUE(noMoreIO);
}

INSTANTIATE_TEST_SUITE_P(
    EdgeCases,
    PercentUnitLoaderEdgeTests,
    ValuesIn(
        // Percentages
        {0U,  1U,  2U,  3U,  4U,  5U,   6U,   7U,   8U,   9U,  10U,
         20U, 25U, 50U, 90U, 99U, 100U, 101U, 199U, 200U, 201U}));
