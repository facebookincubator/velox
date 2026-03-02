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
#include "velox/dwio/common/tests/UnitLoaderBaseTest.h"
#include "velox/dwio/common/tests/utils/UnitLoaderTestTools.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::common::test;

class ParallelUnitLoaderTest
    : public UnitLoaderBaseTest<ParallelUnitLoaderFactory> {
 protected:
  ParallelUnitLoaderFactory createFactory() override {
    return ParallelUnitLoaderFactory(ioExecutor_.get(), 2);
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> ioExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(10);
};

TEST_F(ParallelUnitLoaderTest, NoUnitButSkip) {
  testNoUnitButSkip();
}

TEST_F(ParallelUnitLoaderTest, InitialSkip) {
  testInitialSkip();
}

TEST_F(ParallelUnitLoaderTest, CanRequestUnitMultipleTimes) {
  testCanRequestUnitMultipleTimes();
}

TEST_F(ParallelUnitLoaderTest, UnitOutOfRange) {
  testUnitOutOfRange();
}

TEST_F(ParallelUnitLoaderTest, SeekOutOfRange) {
  testSeekOutOfRange();
}

TEST_F(ParallelUnitLoaderTest, SeekOutOfRangeReaderError) {
  testSeekOutOfRangeReaderError();
}

TEST_F(ParallelUnitLoaderTest, LoadsCorrectlyWithReader) {
  auto factory = createFactory();
  ReaderMock readerMock{{10, 20, 30}, {0, 0, 0}, factory, 0};

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 0-2, load(0)
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(3)); // Unit: 0, rows: 3-5
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(4)); // Unit: 0, rows: 6-9
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({true, true, false}));

  EXPECT_TRUE(readerMock.read(14)); // Unit: 1, rows: 0-13, unload(0), load(1)
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));

  // will only read 5 rows, no more rows in unit 1
  EXPECT_TRUE(readerMock.read(10)); // Unit: 1, rows: 14-19
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, true, true}));

  EXPECT_TRUE(readerMock.read(30)); // Unit: 2, rows: 0-29, unload(1), load(2)
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));

  EXPECT_FALSE(readerMock.read(30)); // No more data
  EXPECT_EQ(readerMock.unitsLoaded(), std::vector<bool>({false, false, true}));
}

// Performance comparison test
TEST_F(ParallelUnitLoaderTest, PerformanceComparison) {
  std::vector<uint64_t> rowsPerUnit = {100, 100, 100, 100, 100, 100, 100, 100};
  std::vector<uint64_t> ioSizes = {
      1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

  // Measure ParallelUnitLoader performance
  auto parallelStart = std::chrono::high_resolution_clock::now();
  {
    auto factory = createFactory();
    ReaderMock reader(rowsPerUnit, ioSizes, factory, 0);

    for (size_t i = 0; i < rowsPerUnit.size(); ++i) {
      uint64_t totalRowsRead = 0;
      while (totalRowsRead < rowsPerUnit[i]) {
        reader.read(25);
        int nextRead = rowsPerUnit[i] - totalRowsRead;
        totalRowsRead += std::min(25, nextRead);
      }
    }
  }
  auto parallelEnd = std::chrono::high_resolution_clock::now();

  // Measure OnDemandUnitLoader performance
  auto onDemandStart = std::chrono::high_resolution_clock::now();
  {
    auto factory = std::make_shared<OnDemandUnitLoaderFactory>(nullptr);
    ReaderMock reader(rowsPerUnit, ioSizes, *factory, 0);

    for (size_t i = 0; i < rowsPerUnit.size(); ++i) {
      uint64_t totalRowsRead = 0;
      while (totalRowsRead < rowsPerUnit[i]) {
        reader.read(25);
        int nextRead = rowsPerUnit[i] - totalRowsRead;
        totalRowsRead += std::min(25, nextRead);
      }
    }
  }
  auto onDemandEnd = std::chrono::high_resolution_clock::now();

  auto parallelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      parallelEnd - parallelStart);
  auto onDemandDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      onDemandEnd - onDemandStart);

  // ParallelUnitLoader should be faster
  EXPECT_GT(onDemandDuration.count(), parallelDuration.count());
}
