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

#include "velox/dwio/common/ParallelUnitLoader.h"
#include "velox/dwio/common/tests/UnitLoaderBaseTest.h"
#include "velox/dwio/common/tests/utils/UnitLoaderTestTools.h"

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/ManualExecutor.h>
#include <folly/futures/Future.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>
#include <array>
#include <future>
#include "velox/common/testutil/TestValue.h"

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::common::test;

namespace {

struct RetainedUnitState {
  size_t allocations{0};
  size_t loads{0};
  size_t unloads{0};
  bool failIoSize{false};
  bool failLoad{false};
};

// Models units that initialize retained resources while estimating IO size.
class StatefulLoadUnit : public LoadUnit {
 public:
  explicit StatefulLoadUnit(std::shared_ptr<RetainedUnitState> state)
      : state_(std::move(state)) {}

  ~StatefulLoadUnit() override {
    releaseResources();
  }

  void load() override {
    initialize();
    ++state_->loads;
    if (state_->failLoad) {
      VELOX_FAIL("Load failed after initialization");
    }
  }

  void unload() override {
    ++state_->unloads;
    releaseResources();
  }

  uint64_t getNumRows() override {
    return 10;
  }

  uint64_t getIoSize() override {
    initialize();
    if (state_->failIoSize) {
      VELOX_FAIL("IO size failed after initialization");
    }
    return 100;
  }

 private:
  void initialize() {
    if (!resources_) {
      resources_ = std::make_unique<std::array<char, 4'096>>();
      ++state_->allocations;
    }
  }

  void releaseResources() {
    if (resources_) {
      resources_.reset();
      --state_->allocations;
    }
  }

  const std::shared_ptr<RetainedUnitState> state_;
  std::unique_ptr<std::array<char, 4'096>> resources_;
};

std::vector<std::unique_ptr<LoadUnit>> makeStatefulUnits(
    const std::shared_ptr<RetainedUnitState>& state,
    bool includeForeground = true) {
  std::vector<std::unique_ptr<LoadUnit>> units;
  if (includeForeground) {
    units.push_back(
        std::make_unique<StatefulLoadUnit>(
            std::make_shared<RetainedUnitState>()));
  }
  units.push_back(std::make_unique<StatefulLoadUnit>(state));
  return units;
}

} // namespace

class ParallelUnitLoaderPrefetchTest
    : public UnitLoaderBaseTest<ParallelUnitLoaderFactory> {
 protected:
  ParallelUnitLoaderFactory createFactory() override {
    return createPrefetchFactory(ioExecutor_.get(), 2);
  }

  static ParallelUnitLoaderFactory createPrefetchFactory(
      folly::Executor* executor,
      size_t maxConcurrentLoads,
      ParallelUnitLoaderOptions options = {}) {
    return ParallelUnitLoaderFactory(
        executor, maxConcurrentLoads, std::move(options));
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> ioExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(10);
};

TEST_F(ParallelUnitLoaderPrefetchTest, noUnitButSkip) {
  testNoUnitButSkip();
}

TEST_F(ParallelUnitLoaderPrefetchTest, initialSkip) {
  testInitialSkip();
}

TEST_F(ParallelUnitLoaderPrefetchTest, canRequestUnitMultipleTimes) {
  testCanRequestUnitMultipleTimes();
}

TEST_F(ParallelUnitLoaderPrefetchTest, unitOutOfRange) {
  testUnitOutOfRange();
}

TEST_F(ParallelUnitLoaderPrefetchTest, seekOutOfRange) {
  testSeekOutOfRange();
}

TEST_F(ParallelUnitLoaderPrefetchTest, seekOutOfRangeReaderError) {
  testSeekOutOfRangeReaderError();
}

TEST_F(ParallelUnitLoaderPrefetchTest, loadsCorrectlyWithReader) {
  folly::ManualExecutor executor;
  auto factory = createPrefetchFactory(&executor, 2);
  ReaderMock readerMock{{10, 20, 30}, {0, 0, 0}, factory, 0};

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2, load(0)
  executor.drain();
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(14)); // Unit: 1, rows: 0-13, unload(0), load(1)
  executor.drain();
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));

  // will only read 5 rows, no more rows in unit 1
  EXPECT_TRUE(readerMock.read(10)); // Unit: 1, rows: 14-19
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1), load(2)
  executor.drain();
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
}

TEST_F(ParallelUnitLoaderPrefetchTest, destructionUnloadsLoadedUnits) {
  auto loaded = getUnitsLoadedWithFalse(1);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(
      std::make_unique<LoadUnitMock>(
          10, 100, loaded, 0, std::chrono::milliseconds(0)));
  auto factory = createPrefetchFactory(ioExecutor_.get(), 1);
  {
    auto loader = factory.create(std::move(units), 0);
    loader->getLoadedUnit(0);
    ASSERT_TRUE(loaded[0]);
  }
  EXPECT_FALSE(loaded[0]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, prefetchWindowAndSkippedQueuedUnit) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(4);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  auto factory = createPrefetchFactory(&executor, 2);
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  EXPECT_TRUE(loaded[0]);
  EXPECT_FALSE(loaded[1]);
  loader->getLoadedUnit(2);
  executor.drain();
  EXPECT_FALSE(loaded[0]);
  EXPECT_FALSE(loaded[1]);
  EXPECT_TRUE(loaded[2]);
  EXPECT_TRUE(loaded[3]);
  loader.reset();
  EXPECT_FALSE(loaded[2]);
  EXPECT_FALSE(loaded[3]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, requiresBothBudgetLimits) {
  folly::ManualExecutor executor;
  VELOX_ASSERT_THROW(
      createPrefetchFactory(&executor, 2, {.maxPrefetchLoads = 1}),
      "Set both prefetch limits");
  VELOX_ASSERT_THROW(
      createPrefetchFactory(&executor, 2, {.maxPrefetchBytes = 100}),
      "Set both prefetch limits");
}

TEST_F(ParallelUnitLoaderPrefetchTest, retainedBytesReleasedOnConsumption) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(3);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  auto factory = createPrefetchFactory(
      &executor, 3, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  executor.drain();
  EXPECT_TRUE(loaded[1]);
  loader->onRead(0, 0, 1);
  executor.drain();
  EXPECT_FALSE(loaded[2]);
  loader->getLoadedUnit(1);
  executor.drain();
  EXPECT_TRUE(loaded[2]);
}

TEST_F(
    ParallelUnitLoaderPrefetchTest,
    budgetSharedAndQueuedReservationsReturned) {
  folly::ManualExecutor executor;
  ParallelUnitLoaderFactory factory(
      &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
  auto firstLoaded = getUnitsLoadedWithFalse(2);
  auto secondLoaded = getUnitsLoadedWithFalse(2);
  auto makeLoader = [&](auto& loaded) {
    std::vector<std::unique_ptr<LoadUnit>> units;
    units.reserve(loaded.size());
    for (size_t index = 0; index < loaded.size(); ++index) {
      units.push_back(
          std::make_unique<LoadUnitMock>(
              10, 100, loaded, index, std::chrono::milliseconds(0)));
    }
    return factory.create(std::move(units), 0);
  };
  auto first = makeLoader(firstLoaded);
  auto second = makeLoader(secondLoaded);
  first->getLoadedUnit(0);
  second->getLoadedUnit(0);
  EXPECT_TRUE(secondLoaded[0]);
  EXPECT_FALSE(secondLoaded[1]);
  first.reset();
  second->onRead(0, 0, 1);
  executor.drain();
  EXPECT_FALSE(firstLoaded[1]);
  EXPECT_TRUE(secondLoaded[1]);
}

TEST_F(
    ParallelUnitLoaderPrefetchTest,
    completedLoadsReleaseSlotsButRetainBytes) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(4);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  auto factory = createPrefetchFactory(
      &executor, 4, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 200});
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  executor.drain();
  EXPECT_TRUE(loaded[1]);
  EXPECT_FALSE(loaded[2]);
  loader->onRead(0, 0, 1);
  executor.drain();
  EXPECT_TRUE(loaded[2]);
  loader->onRead(0, 0, 1);
  executor.drain();
  EXPECT_FALSE(loaded[3]);
  loader->getLoadedUnit(1);
  executor.drain();
  EXPECT_TRUE(loaded[3]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, rejectedPrefetchReleasesEstimatedState) {
  folly::ManualExecutor executor;
  auto state = std::make_shared<RetainedUnitState>();
  auto factory = createPrefetchFactory(
      &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 10});
  auto loader = factory.create(makeStatefulUnits(state), 0);
  loader->getLoadedUnit(0);
  EXPECT_EQ(state->allocations, 0);
  EXPECT_EQ(state->loads, 0);
  EXPECT_GT(state->unloads, 0);
  loader->onRead(0, 0, 1);
  EXPECT_EQ(state->allocations, 0);
  loader->getLoadedUnit(1);
  EXPECT_EQ(state->allocations, 1);
  EXPECT_EQ(state->loads, 1);
  loader.reset();
  EXPECT_EQ(state->allocations, 0);
  executor.drain();
}

TEST_F(ParallelUnitLoaderPrefetchTest, submissionErrorReleasesEstimatedState) {
  class RejectingExecutor : public folly::Executor {
   public:
    void add(folly::Func task) override {
      if (reject) {
        VELOX_FAIL("Executor rejected submission");
      }
      task();
      reject = true;
    }
    bool reject{false};
  } executor;
  auto state = std::make_shared<RetainedUnitState>();
  auto factory = createPrefetchFactory(
      &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
  auto loader = factory.create(makeStatefulUnits(state), 0);
  loader->getLoadedUnit(0);
  EXPECT_EQ(state->allocations, 0);
  EXPECT_EQ(state->loads, 0);
  EXPECT_GT(state->unloads, 0);
  executor.reject = false;
  loader->onRead(0, 0, 1);
  loader->getLoadedUnit(1);
  EXPECT_EQ(state->loads, 1);
}

TEST_F(
    ParallelUnitLoaderPrefetchTest,
    queuedCancellationReleasesEstimatedState) {
  folly::ManualExecutor executor;
  auto state = std::make_shared<RetainedUnitState>();
  auto factory = createPrefetchFactory(
      &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
  auto loader = factory.create(makeStatefulUnits(state), 0);
  loader->getLoadedUnit(0);
  EXPECT_EQ(state->allocations, 1);
  EXPECT_EQ(state->loads, 0);
  loader.reset();
  EXPECT_EQ(state->allocations, 0);
  EXPECT_EQ(state->unloads, 1);
  executor.drain();
  EXPECT_EQ(state->loads, 0);
  EXPECT_EQ(state->unloads, 1);
}

TEST_F(ParallelUnitLoaderPrefetchTest, ioSizeFailureReleasesInitializedState) {
  folly::ManualExecutor executor;
  auto state = std::make_shared<RetainedUnitState>();
  state->failIoSize = true;
  auto factory = createPrefetchFactory(
      &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
  auto loader = factory.create(makeStatefulUnits(state), 0);
  VELOX_ASSERT_THROW(
      loader->getLoadedUnit(0), "IO size failed after initialization");
  EXPECT_EQ(state->allocations, 0);
  EXPECT_EQ(state->unloads, 1);
  loader.reset();
  executor.drain();
}

TEST_F(ParallelUnitLoaderPrefetchTest, loadFailureReleasesInitializedState) {
  for (bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch);
    folly::ManualExecutor executor;
    auto state = std::make_shared<RetainedUnitState>();
    state->failLoad = true;
    auto factory = createPrefetchFactory(
        &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
    auto loader = factory.create(makeStatefulUnits(state, prefetch), 0);
    if (prefetch) {
      loader->getLoadedUnit(0);
      executor.drain();
    }
    VELOX_ASSERT_THROW(
        loader->getLoadedUnit(prefetch ? 1 : 0),
        "Load failed after initialization");
    loader.reset();
    EXPECT_EQ(state->allocations, 0);
    EXPECT_EQ(state->unloads, 1);
    executor.drain();
    EXPECT_EQ(state->loads, 1);
  }
}

TEST_F(ParallelUnitLoaderPrefetchTest, oversizedUnitsUseForegroundPath) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(2);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  size_t rejectedPrefetches{0};
  size_t prefetchErrors{0};
  auto factory = createPrefetchFactory(
      &executor,
      2,
      {.maxPrefetchLoads = 1,
       .maxPrefetchBytes = 10,
       .metricCallback = [&](auto event, auto) {
         if (event == UnitLoader::Event::kPrefetchRejected) {
           ++rejectedPrefetches;
         } else if (event == UnitLoader::Event::kPrefetchError) {
           ++prefetchErrors;
         }
       }});
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  executor.drain();
  EXPECT_FALSE(loaded[1]);
  EXPECT_GT(rejectedPrefetches, 0);
  EXPECT_EQ(prefetchErrors, 0);
  loader->getLoadedUnit(1);
  EXPECT_TRUE(loaded[1]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, prefetchSubmissionErrorCanBeRetried) {
  class RejectingExecutor : public folly::Executor {
   public:
    void add(folly::Func task) override {
      if (reject) {
        VELOX_FAIL("Executor rejected submission");
      }
      task();
      reject = true;
    }

    bool reject{false};
  };
  RejectingExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(2);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  size_t rejectedPrefetches{0};
  size_t prefetchErrors{0};
  auto factory = createPrefetchFactory(
      &executor,
      2,
      {.maxPrefetchLoads = 1,
       .maxPrefetchBytes = 100,
       .metricCallback = [&](auto event, auto) {
         if (event == UnitLoader::Event::kPrefetchRejected) {
           ++rejectedPrefetches;
         } else if (event == UnitLoader::Event::kPrefetchError) {
           ++prefetchErrors;
         }
       }});
  auto loader = factory.create(std::move(units), 0);

  ASSERT_NO_THROW(loader->getLoadedUnit(0));
  EXPECT_TRUE(loaded[0]);
  EXPECT_FALSE(loaded[1]);
  EXPECT_EQ(rejectedPrefetches, 0);
  EXPECT_GT(prefetchErrors, 0);
  executor.reject = false;
  loader->onRead(0, 0, 1);
  EXPECT_TRUE(loaded[1]);
  EXPECT_EQ(rejectedPrefetches, 0);
  ASSERT_NO_THROW(loader->getLoadedUnit(1));
  EXPECT_FALSE(loaded[0]);
  EXPECT_TRUE(loaded[1]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, rejectedForegroundSubmissionFailsRead) {
  class RejectingExecutor : public folly::Executor {
   public:
    void add(folly::Func /* task */) override {
      VELOX_FAIL("Executor rejected submission");
    }
  };
  RejectingExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(1);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(
      std::make_unique<LoadUnitMock>(
          10, 100, loaded, 0, std::chrono::milliseconds(0)));
  auto factory = createPrefetchFactory(&executor, 1);
  auto loader = factory.create(std::move(units), 0);

  VELOX_ASSERT_THROW(loader->getLoadedUnit(0), "Executor rejected submission");
  EXPECT_FALSE(loaded[0]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, failedPrefetchReturnsAdmission) {
  class FailingLoadUnit : public LoadUnitMock {
   public:
    using LoadUnitMock::LoadUnitMock;
    void load() override {
      VELOX_FAIL("Stripe load failed");
    }
  };
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(2);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(
      std::make_unique<LoadUnitMock>(
          10, 100, loaded, 0, std::chrono::milliseconds(0)));
  units.push_back(std::make_unique<FailingLoadUnit>(10, 100, loaded, 1));
  auto factory = createPrefetchFactory(
      &executor, 2, {.maxPrefetchLoads = 1, .maxPrefetchBytes = 100});
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  executor.drain();
  VELOX_ASSERT_THROW(loader->getLoadedUnit(1), "Stripe load failed");
  loader.reset();
  units.clear();
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  auto retry = factory.create(std::move(units), 0);
  retry->getLoadedUnit(0);
  executor.drain();
  EXPECT_TRUE(loaded[1]);
}

TEST_F(ParallelUnitLoaderPrefetchTest, readinessAndQueryWaitCallbacks) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(2);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  std::vector<UnitLoader::Event> events;
  size_t waits{0};
  auto factory = createPrefetchFactory(
      &executor,
      2,
      {.metricCallback = [&](auto event, auto) { events.push_back(event); },
       .blockedOnIoCallback =
           [&](auto duration) {
             EXPECT_GE(duration.count(), 0);
             ++waits;
           }});
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  executor.drain();
  loader->getLoadedUnit(1);
  EXPECT_EQ(waits, 1);
  EXPECT_NE(
      std::find(
          events.begin(), events.end(), UnitLoader::Event::kPrefetchReady),
      events.end());
}

TEST_F(
    ParallelUnitLoaderPrefetchTest,
    factoryCallbackAndBudgetSharedAcrossLoaders) {
  folly::ManualExecutor executor;
  size_t factoryWaits{0};
  size_t rejectedPrefetches{0};
  auto factory = createPrefetchFactory(
      &executor,
      2,
      {.maxPrefetchLoads = 1,
       .maxPrefetchBytes = 100,
       .metricCallback =
           [&](auto event, auto) {
             if (event == UnitLoader::Event::kPrefetchRejected) {
               ++rejectedPrefetches;
             }
           },
       .blockedOnIoCallback = [&](auto) { ++factoryWaits; }});
  auto firstLoaded = getUnitsLoadedWithFalse(2);
  auto secondLoaded = getUnitsLoadedWithFalse(2);
  auto defaultLoaded = getUnitsLoadedWithFalse(2);
  auto makeUnits = [&](auto& loaded) {
    std::vector<std::unique_ptr<LoadUnit>> units;
    units.reserve(loaded.size());
    for (size_t index = 0; index < loaded.size(); ++index) {
      units.push_back(
          std::make_unique<LoadUnitMock>(
              10, 100, loaded, index, std::chrono::milliseconds(0)));
    }
    return units;
  };
  auto first = factory.create(makeUnits(firstLoaded), 0);
  auto second = factory.create(makeUnits(secondLoaded), 0);
  auto defaultLoader = factory.create(makeUnits(defaultLoaded), 0);
  first->getLoadedUnit(0);
  second->getLoadedUnit(0);
  defaultLoader->getLoadedUnit(0);
  EXPECT_EQ(factoryWaits, 3);
  EXPECT_GT(rejectedPrefetches, 0);
  executor.drain();
  EXPECT_TRUE(firstLoaded[1]);
  EXPECT_FALSE(secondLoaded[1]);
  EXPECT_FALSE(defaultLoaded[1]);
  first.reset();
  second->onRead(0, 0, 1);
  second->getLoadedUnit(1);
  EXPECT_TRUE(secondLoaded[1]);
  EXPECT_EQ(factoryWaits, 4);
}

TEST_F(ParallelUnitLoaderPrefetchTest, repeatedLoadedUnitDoesNotReportWait) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(2);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  size_t waitSamples{0};
  size_t blockedWaits{0};
  auto factory = createPrefetchFactory(
      &executor,
      2,
      {.metricCallback =
           [&](auto event, auto) {
             if (event == UnitLoader::Event::kWaitNanos) {
               ++waitSamples;
             }
           },
       .blockedOnIoCallback = [&](auto) { ++blockedWaits; }});
  auto loader = factory.create(std::move(units), 0);

  for (uint32_t unit = 0; unit < loaded.size(); ++unit) {
    auto& loadedUnit = loader->getLoadedUnit(unit);
    EXPECT_TRUE(loaded[unit]);
    EXPECT_EQ(waitSamples, unit + 1);
    EXPECT_EQ(blockedWaits, 1);
    for (size_t repeat = 0; repeat < 3; ++repeat) {
      EXPECT_EQ(&loader->getLoadedUnit(unit), &loadedUnit);
    }
    EXPECT_EQ(waitSamples, unit + 1);
    EXPECT_EQ(blockedWaits, 1);
    executor.drain();
  }
}

TEST_F(ParallelUnitLoaderPrefetchTest, destructionWaitsForRunningPrefetch) {
  class BlockingLoadUnit : public LoadUnit {
   public:
    BlockingLoadUnit(folly::Baton<>& started, folly::Baton<>& release)
        : started_(started), release_(release) {}
    void load() override {
      started_.post();
      release_.wait();
    }
    void unload() override {}
    uint64_t getNumRows() override {
      return 10;
    }
    uint64_t getIoSize() override {
      return 100;
    }

   private:
    folly::Baton<>& started_;
    folly::Baton<>& release_;
  };
  folly::Baton<> started;
  folly::Baton<> release;
  auto loaded = getUnitsLoadedWithFalse(1);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(
      std::make_unique<LoadUnitMock>(
          10, 100, loaded, 0, std::chrono::milliseconds(0)));
  units.push_back(std::make_unique<BlockingLoadUnit>(started, release));
  auto factory = createFactory();
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  started.wait();
  auto destroy = std::async(
      std::launch::async,
      [loader = std::move(loader)]() mutable { loader.reset(); });
  EXPECT_EQ(
      destroy.wait_for(std::chrono::seconds(1)), std::future_status::timeout);
  release.post();
  destroy.get();
}

TEST_F(ParallelUnitLoaderPrefetchTest, backwardSeekDiscardsFutureWindow) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(4);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  auto factory = createPrefetchFactory(&executor, 2);
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(2);
  executor.drain();
  EXPECT_TRUE(loaded[2]);
  EXPECT_TRUE(loaded[3]);
  loader->onSeek(0, 0);
  loader->getLoadedUnit(0);
  executor.drain();
  EXPECT_TRUE(loaded[0]);
  EXPECT_TRUE(loaded[1]);
  EXPECT_FALSE(loaded[2]);
  EXPECT_FALSE(loaded[3]);
}

#ifndef NDEBUG
TEST_F(ParallelUnitLoaderPrefetchTest, cleanupVisitsOnlyDiscardedWindow) {
  folly::ManualExecutor executor;
  auto loaded = getUnitsLoadedWithFalse(1'000);
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.reserve(loaded.size());
  for (size_t index = 0; index < loaded.size(); ++index) {
    units.push_back(
        std::make_unique<LoadUnitMock>(
            10, 100, loaded, index, std::chrono::milliseconds(0)));
  }
  size_t cleanupVisits{0};
  facebook::velox::common::testutil::TestValue::enable();
  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwio::common::ParallelUnitLoader::unload",
      std::function<void(size_t*)>([&](size_t*) { ++cleanupVisits; }));
  auto factory = createPrefetchFactory(&executor, 2);
  auto loader = factory.create(std::move(units), 0);
  loader->getLoadedUnit(0);
  EXPECT_EQ(cleanupVisits, 0);
  for (size_t repeat = 0; repeat < 10; ++repeat) {
    loader->getLoadedUnit(0);
  }
  EXPECT_EQ(cleanupVisits, 0);
  loader->getLoadedUnit(1);
  EXPECT_EQ(cleanupVisits, 1);
  loader->getLoadedUnit(900);
  EXPECT_EQ(cleanupVisits, 3);
  loader->getLoadedUnit(0);
  EXPECT_EQ(cleanupVisits, 5);
  loader.reset();
  EXPECT_EQ(cleanupVisits, 7);
  executor.drain();
}
#endif

DEBUG_ONLY_TEST_F(
    ParallelUnitLoaderPrefetchTest,
    reservationCallbacksFinishBeforeLoaderDestruction) {
  facebook::velox::common::testutil::TestValue::enable();
  for (bool consume : {false, true}) {
    for (bool failLoad : {false, true}) {
      SCOPED_TRACE(consume);
      SCOPED_TRACE(failLoad);
      std::atomic<uint64_t> releasedBytes{0};
      std::atomic<size_t> lateCallbacks{0};
      std::atomic<size_t> readerReleases{0};
      std::atomic<bool> loaderGone{false};
      const auto readerThread = std::this_thread::get_id();
      {
        folly::ManualExecutor executor;
        folly::Baton<> published;
        folly::Baton<> resume;
        SCOPED_TESTVALUE_SET(
            "facebook::velox::AsyncSource::makeItem::afterNotify",
            std::function<void(void*)>([&](void*) {
              if (std::this_thread::get_id() != readerThread) {
                published.post();
                resume.wait();
              }
            }));
        auto state = std::make_shared<RetainedUnitState>();
        state->failLoad = failLoad;
        auto factory = createPrefetchFactory(
            &executor,
            2,
            {.maxPrefetchLoads = 1,
             .maxPrefetchBytes = 100,
             .metricCallback = [&](auto event, auto value) {
               if (loaderGone.load()) {
                 ++lateCallbacks;
               }
               if (event == UnitLoader::Event::kBytesReleased) {
                 releasedBytes += value;
                 if (std::this_thread::get_id() == readerThread) {
                   ++readerReleases;
                 }
               }
             }});
        auto loader = factory.create(makeStatefulUnits(state), 0);
        loader->getLoadedUnit(0);
        auto worker = std::async(std::launch::async, [&] { executor.drain(); });
        SCOPE_EXIT {
          resume.post();
          worker.get();
        };
        ASSERT_TRUE(published.try_wait_for(std::chrono::seconds(5)));
        if (consume) {
          if (failLoad) {
            VELOX_ASSERT_THROW(
                loader->getLoadedUnit(1), "Load failed after initialization");
          } else {
            loader->getLoadedUnit(1);
            EXPECT_EQ(releasedBytes.load(), 100);
          }
        }
        loader.reset();
        loaderGone.store(true);
        EXPECT_EQ(releasedBytes.load(), 100);
        EXPECT_EQ(readerReleases.load(), 1);
      }
      EXPECT_EQ(releasedBytes.load(), 100);
      EXPECT_EQ(lateCallbacks.load(), 0);
    }
  }
}
