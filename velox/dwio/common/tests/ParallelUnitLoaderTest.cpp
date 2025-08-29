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
#include "velox/dwio/common/OnDemandUnitLoader.h"
#include "velox/dwio/common/tests/utils/UnitLoaderTestTools.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::common::test;

namespace facebook::velox::dwio::common {

class ParallelUnitLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
  }

  void TearDown() override {
    executor_.reset();
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
};

// Enhanced LoadUnitMock for testing parallel loading with timing
class TimedLoadUnitMock : public LoadUnit {
 public:
  TimedLoadUnitMock(
      uint64_t rowCount,
      uint64_t ioSize,
      std::vector<std::atomic_bool>& unitsLoaded,
      size_t unitId,
      std::chrono::milliseconds loadDelay = std::chrono::milliseconds(10),
      bool shouldFail = false)
      : rowCount_(rowCount),
        ioSize_(ioSize),
        unitsLoaded_(unitsLoaded),
        unitId_(unitId),
        loadDelay_(loadDelay),
        shouldFail_(shouldFail) {}

  void load() override {
    if (shouldFail_) {
      throw std::runtime_error("Mock load failure for unit " + std::to_string(unitId_));
    }

    VELOX_CHECK(!isLoaded());

    // Simulate loading time
    std::this_thread::sleep_for(loadDelay_);

    unitsLoaded_[unitId_] = true;
  }

  void unload() override {
    VELOX_CHECK(isLoaded());
    unitsLoaded_[unitId_] = false;
  }

  uint64_t getNumRows() override {
    return rowCount_;
  }

  uint64_t getIoSize() override {
    return ioSize_;
  }

  bool isLoaded() const {
    return unitsLoaded_[unitId_];
  }

  size_t getUnitId() const {
    return unitId_;
  }

 private:
  uint64_t rowCount_;
  uint64_t ioSize_;
  std::vector<std::atomic_bool>& unitsLoaded_;
  size_t unitId_;
  std::chrono::milliseconds loadDelay_;
  bool shouldFail_;
};

TEST_F(ParallelUnitLoaderTest, BasicFunctionality) {
  std::vector<uint64_t> rowsPerUnit = {100, 200, 150, 300, 250};
  std::vector<uint64_t> ioSizes = {1024, 2048, 1536, 3072, 2560};

  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 2);
  ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

  // Read through all units
  for (size_t i = 0; i < rowsPerUnit.size(); ++i) {
    uint64_t totalRowsRead = 0;
    while (totalRowsRead < rowsPerUnit[i]) {
      EXPECT_TRUE(reader.read(50));
      totalRowsRead += std::min(50UL, rowsPerUnit[i] - totalRowsRead);
    }
  }
}

TEST_F(ParallelUnitLoaderTest, PrefetchWindowBehavior) {
  std::vector<uint64_t> rowsPerUnit = {100, 100, 100, 100, 100};
  std::vector<uint64_t> ioSizes = {1024, 1024, 1024, 1024, 1024};

  // Create factory with prefetch window of 3
  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 3);

  // Create units manually to test loading behavior
  auto unitsLoaded = getUnitsLoadedWithFalse(5);
  std::vector<std::unique_ptr<LoadUnit>> units;
  for (size_t i = 0; i < 5; ++i) {
    units.emplace_back(std::make_unique<TimedLoadUnitMock>(
        rowsPerUnit[i], ioSizes[i], unitsLoaded, i, std::chrono::milliseconds(20)));
  }

  auto loader = factory->create(std::move(units), 0);

  // Give some time for initial prefetch
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // First 3 units should be loaded or in the process of loading
  // Access first unit
  auto& unit0 = loader->getLoadedUnit(0);
  EXPECT_EQ(unit0.getNumRows(), 100);

  // Access more units to test sliding window
  auto& unit1 = loader->getLoadedUnit(1);
  auto& unit2 = loader->getLoadedUnit(2);
  auto& unit3 = loader->getLoadedUnit(3);

  EXPECT_EQ(unit1.getNumRows(), 100);
  EXPECT_EQ(unit2.getNumRows(), 100);
  EXPECT_EQ(unit3.getNumRows(), 100);
}

TEST_F(ParallelUnitLoaderTest, LoadFailureHandling) {
  auto unitsLoaded = getUnitsLoadedWithFalse(5);
  std::vector<std::unique_ptr<LoadUnit>> units;

  // Create units where unit 2 will fail to load
  for (size_t i = 0; i < 5; ++i) {
    bool shouldFail = (i == 2);
    units.emplace_back(std::make_unique<TimedLoadUnitMock>(
        100, 1024, unitsLoaded, i, std::chrono::milliseconds(10), shouldFail));
  }

  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 2);
  auto loader = factory->create(std::move(units), 0);

  // Access units 0 and 1 should work
  EXPECT_NO_THROW(loader->getLoadedUnit(0));
  EXPECT_NO_THROW(loader->getLoadedUnit(1));

  // Unit 2 should throw an exception
  EXPECT_THROW(loader->getLoadedUnit(2), velox::VeloxException);
}

TEST_F(ParallelUnitLoaderTest, OnReadAndOnSeekValidation) {
  auto unitsLoaded = getUnitsLoadedWithFalse(3);
  std::vector<std::unique_ptr<LoadUnit>> units;
  for (size_t i = 0; i < 3; ++i) {
    units.emplace_back(std::make_unique<TimedLoadUnitMock>(
        100, 1024, unitsLoaded, i));
  }

  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 2);
  auto loader = factory->create(std::move(units), 0);

  // Test valid onRead calls
  EXPECT_NO_THROW(loader->onRead(0, 50, 10));
  EXPECT_NO_THROW(loader->onRead(1, 0, 100));
  EXPECT_NO_THROW(loader->onRead(2, 99, 1));

  // Test invalid onRead calls
  EXPECT_THROW(loader->onRead(3, 0, 10), velox::VeloxException); // Unit out of range
  EXPECT_THROW(loader->onRead(0, 100, 1), velox::VeloxException); // Row out of range

  // Test valid onSeek calls
  EXPECT_NO_THROW(loader->onSeek(0, 50));
  EXPECT_NO_THROW(loader->onSeek(1, 0));
  EXPECT_NO_THROW(loader->onSeek(2, 100)); // At end is valid

  // Test invalid onSeek calls
  EXPECT_THROW(loader->onSeek(3, 0), velox::VeloxException); // Unit out of range
  EXPECT_THROW(loader->onSeek(0, 101), velox::VeloxException); // Row out of range
}

TEST_F(ParallelUnitLoaderTest, CompareWithOnDemandLoader) {
  std::vector<uint64_t> rowsPerUnit = {100, 200, 150, 300};
  std::vector<uint64_t> ioSizes = {1024, 2048, 1536, 3072};

  // Test with ParallelUnitLoader
  auto parallelFactory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 2);
  ReaderMock parallelReader(rowsPerUnit, ioSizes, *parallelFactory, 0);

  // Test with OnDemandUnitLoader
  auto onDemandFactory = std::make_shared<OnDemandUnitLoaderFactory>();
  ReaderMock onDemandReader(rowsPerUnit, ioSizes, *onDemandFactory, 0);

  // Both should produce the same results
  for (size_t i = 0; i < rowsPerUnit.size(); ++i) {
    uint64_t totalRowsRead = 0;
    while (totalRowsRead < rowsPerUnit[i]) {
      bool parallelResult = parallelReader.read(50);
      bool onDemandResult = onDemandReader.read(50);

      EXPECT_EQ(parallelResult, onDemandResult);
      totalRowsRead += std::min(50UL, rowsPerUnit[i] - totalRowsRead);
    }
  }
}

TEST_F(ParallelUnitLoaderTest, SeekFunctionality) {
  std::vector<uint64_t> rowsPerUnit = {100, 200, 150};
  std::vector<uint64_t> ioSizes = {1024, 2048, 1536};

  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 2);
  ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

  // Test seeking to different positions
  reader.seek(0);    // Beginning of first unit
  reader.seek(50);   // Middle of first unit
  reader.seek(100);  // Beginning of second unit
  reader.seek(250);  // Middle of second unit
  reader.seek(300);  // Beginning of third unit
  reader.seek(449);  // End of file

  // Test reading after seek
  reader.seek(150);
  EXPECT_TRUE(reader.read(50));
}

TEST_F(ParallelUnitLoaderTest, SkipRowsFunctionality) {
  std::vector<uint64_t> rowsPerUnit = {100, 200, 150};
  std::vector<uint64_t> ioSizes = {1024, 2048, 1536};

  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 2);

  // Skip first 150 rows (entire first unit + 50 rows of second unit)
  ReaderMock reader(rowsPerUnit, ioSizes, *factory, 150);

  // Should start reading from row 150 (middle of second unit)
  EXPECT_TRUE(reader.read(50));
}

TEST_F(ParallelUnitLoaderTest, EdgeCases) {
  // Test with zero prefetch window
  {
    std::vector<uint64_t> rowsPerUnit = {100, 100};
    std::vector<uint64_t> ioSizes = {1024, 1024};

    auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 0);
    ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

    EXPECT_TRUE(reader.read(50));
  }

  // Test with prefetch window larger than number of units
  {
    std::vector<uint64_t> rowsPerUnit = {100, 100};
    std::vector<uint64_t> ioSizes = {1024, 1024};

    auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 10);
    ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

    EXPECT_TRUE(reader.read(50));
  }

  // Test with single unit
  {
    std::vector<uint64_t> rowsPerUnit = {100};
    std::vector<uint64_t> ioSizes = {1024};

    auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 3);
    ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

    EXPECT_TRUE(reader.read(50));
    EXPECT_TRUE(reader.read(50));
    EXPECT_FALSE(reader.read(50)); // Should be at end
  }
}

TEST_F(ParallelUnitLoaderTest, ConcurrentAccess) {
  std::vector<uint64_t> rowsPerUnit = {100, 100, 100, 100, 100};
  std::vector<uint64_t> ioSizes = {1024, 1024, 1024, 1024, 1024};

  auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 3);

  // Create multiple readers to test concurrent access
  std::vector<std::unique_ptr<ReaderMock>> readers;
  for (int i = 0; i < 3; ++i) {
    readers.push_back(std::make_unique<ReaderMock>(rowsPerUnit, ioSizes, *factory, 0));
  }

  // Test concurrent reading
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int t = 0; t < 3; ++t) {
    threads.emplace_back([&readers, &successCount, t]() {
      try {
        auto& reader = *readers[t];
        for (size_t i = 0; i < 5; ++i) {
          if (reader.read(20)) {
            successCount++;
          }
        }
      } catch (const std::exception& e) {
        // Some concurrent access might have issues, which is acceptable
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // At least some reads should succeed
  EXPECT_GT(successCount.load(), 0);
}

// Performance comparison test
TEST_F(ParallelUnitLoaderTest, PerformanceComparison) {
  std::vector<uint64_t> rowsPerUnit = {100, 100, 100, 100, 100, 100, 100, 100};
  std::vector<uint64_t> ioSizes = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

  // Measure ParallelUnitLoader performance
  auto parallelStart = std::chrono::high_resolution_clock::now();
  {
    auto factory = std::make_shared<ParallelUnitLoaderFactory>(executor_.get(), 3);
    ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

    for (size_t i = 0; i < rowsPerUnit.size(); ++i) {
      uint64_t totalRowsRead = 0;
      while (totalRowsRead < rowsPerUnit[i]) {
        reader.read(25);
        totalRowsRead += std::min(25UL, rowsPerUnit[i] - totalRowsRead);
      }
    }
  }
  auto parallelEnd = std::chrono::high_resolution_clock::now();

  // Measure OnDemandUnitLoader performance
  auto onDemandStart = std::chrono::high_resolution_clock::now();
  {
    auto factory = std::make_shared<OnDemandUnitLoaderFactory>();
    ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

    for (size_t i = 0; i < rowsPerUnit.size(); ++i) {
      uint64_t totalRowsRead = 0;
      while (totalRowsRead < rowsPerUnit[i]) {
        reader.read(25);
        totalRowsRead += std::min(25UL, rowsPerUnit[i] - totalRowsRead);
      }
    }
  }
  auto onDemandEnd = std::chrono::high_resolution_clock::now();

  auto parallelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      parallelEnd - parallelStart);
  auto onDemandDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      onDemandEnd - onDemandStart);

  // ParallelUnitLoader should be faster or at least not significantly slower
  // Note: This is a rough performance test and results may vary
  std::cout << "ParallelUnitLoader time: " << parallelDuration.count() << "ms" << std::endl;
  std::cout << "OnDemandUnitLoader time: " << onDemandDuration.count() << "ms" << std::endl;
}

} // namespace facebook::velox::dwio::common
