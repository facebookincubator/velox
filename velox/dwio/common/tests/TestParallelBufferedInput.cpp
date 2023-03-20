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

#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include "velox/dwio/common/ParallelBufferedInput.h"
#include "velox/exec/tests/utils/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

class TestParallelBufferedInput : public testing::Test {
 protected:
  void SetUp() override {
    executor_ = std::make_unique<folly::IOThreadPoolExecutor>(kThreads);
    pool_ = memory::getDefaultMemoryPool();
  }
  const int32_t kOneKB = 1 * 1024;
  const int32_t kOneMB = 1 * 1024 * 1024;
  const int32_t kThreads = 4;

  std::unique_ptr<folly::IOThreadPoolExecutor> executor_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(TestParallelBufferedInput, parallelLoad) {
  auto ioStats = std::make_shared<IoStatistics>();
  int32_t loadQuantum = kOneKB;

  std::string fileData;
  {
    InMemoryWriteFile writeFile(&fileData);
    writeFile.append(std::string(kOneMB, 'a'));
    writeFile.append(std::string(2 * kOneMB, 'b'));
    writeFile.append(std::string(2 * kOneMB, 'c'));
    writeFile.append(std::string(kOneMB, 'd'));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(fileData);

  ParallelBufferedInput input(
      readFile,
      *pool_,
      MetricsLog::voidLog(),
      ioStats,
      executor_.get(),
      loadQuantum);

  // Avoid coalescing.
  std::vector<Region> regions = {
      {0, static_cast<uint64_t>(kOneKB)},
      {static_cast<uint64_t>(2 * kOneMB), 10},
      {static_cast<uint64_t>(4 * kOneMB), readFile->size() - (4 * kOneMB)}};
  std::vector<std::unique_ptr<SeekableInputStream>> streams;
  for (auto& region : regions) {
    streams.push_back(input.enqueue(region));
  }

  input.load(LogType::TEST);

  for (size_t i = 0; i < regions.size(); i++) {
    const auto& region = regions[i];
    const auto& stream = streams[i];

    auto buffer =
        std::make_shared<dwio::common::DataBuffer<char>>(*pool_, region.length);
    stream->readFully(buffer->data(), region.length);

    // Verify data correctness.
    auto expect = readFile->pread(region.offset, region.length);
    ASSERT_EQ(buffer->size(), expect.size());
    ASSERT_EQ(buffer->data(), expect);
  }

  int32_t totalLength = 0;
  for (const auto& region : regions) {
    totalLength += region.length;
  }
  ASSERT_EQ(totalLength, ioStats->rawBytesRead());

  ASSERT_EQ(0, ioStats->rawOverreadBytes());
}

TEST_F(TestParallelBufferedInput, mergeRegions) {
  auto ioStats = std::make_shared<IoStatistics>();
  int32_t loadQuantum = kOneMB;

  std::string fileData;
  {
    InMemoryWriteFile writeFile(&fileData);
    writeFile.append(std::string(kOneMB, 'a'));
    writeFile.append(std::string(2 * kOneMB, 'b'));
    writeFile.append(std::string(2 * kOneMB, 'c'));
    writeFile.append(std::string(kOneMB, 'd'));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(fileData);

  ParallelBufferedInput input(
      readFile,
      *pool_,
      MetricsLog::voidLog(),
      ioStats,
      executor_.get(),
      loadQuantum);

  // First two regions will be merged by default kMergeDistance(1.5MB).
  std::vector<Region> regions = {
      {0, static_cast<uint64_t>(kOneKB)},
      {static_cast<uint64_t>(kOneMB), 10},
      {static_cast<uint64_t>(4 * kOneMB), readFile->size() - (4 * kOneMB)}};
  std::vector<std::unique_ptr<SeekableInputStream>> streams;
  for (auto& region : regions) {
    streams.push_back(input.enqueue(region));
  }
  int64_t sizeBeforeMerge = 0;
  for (const auto& region : regions) {
    sizeBeforeMerge += region.length;
  }

  input.load(LogType::TEST);

  for (size_t i = 0; i < regions.size(); i++) {
    const auto& region = regions[i];
    const auto& stream = streams[i];

    auto buffer =
        std::make_shared<dwio::common::DataBuffer<char>>(*pool_, region.length);
    stream->readFully(buffer->data(), region.length);

    // Verify data correctness.
    auto expect = readFile->pread(region.offset, region.length);
    ASSERT_EQ(buffer->size(), expect.size());
    ASSERT_EQ(buffer->data(), expect);
  }

  auto overreadBytes = ioStats->rawOverreadBytes();
  ASSERT_EQ(sizeBeforeMerge + overreadBytes, ioStats->rawBytesRead());
  ASSERT_EQ(overreadBytes, kOneMB - kOneKB);
}

class TestReadFile : public InMemoryReadFile {
 public:
  explicit TestReadFile(std::string_view file) : InMemoryReadFile(file) {}
  std::string_view pread(uint64_t offset, uint64_t length, void* buf)
      const override {
    // Assume read latency is 1 millisecond per MB.
    int64_t latency = length >> 20;
    std::this_thread::sleep_for(std::chrono::milliseconds(latency));
    return {static_cast<char*>(buf), length};
  }
};

TEST_F(TestParallelBufferedInput, simpleBenchmark) {
  std::string filename = "test-file";
  int32_t loadQuantum = 8 * kOneMB;
  auto readFile = std::make_shared<TestReadFile>(filename);

  std::vector<Region> regions;

  // 10 regions, each has 2MB length and 2MB distance, eg [0, 2MB], [4MB,
  // 2MB]...
  for (uint64_t iter = 0, cursor = 0, length = 2 * kOneMB; iter < 10;
       ++iter, cursor += 4 * kOneMB) {
    regions.emplace_back(cursor, length);
  }
  // 20 regions, each has 14MB length and 2MB distance, eg [0, 14MB], [16MB,
  // 14MB]...
  for (uint64_t iter = 0, cursor = 0, length = 14 * kOneMB; iter < 20;
       ++iter, cursor += 16 * kOneMB) {
    regions.emplace_back(cursor, length);
  }

  auto stats = std::make_shared<IoStatistics>();
  auto parallelStats = std::make_shared<IoStatistics>();
  auto parallelFunc = [&]() {
    ParallelBufferedInput parallelInput(
        readFile,
        *pool_,
        MetricsLog::voidLog(),
        parallelStats,
        executor_.get(),
        loadQuantum);
    for (const auto& region : regions) {
      parallelInput.enqueue({region.offset, region.length});
    }
    parallelInput.load(LogType::TEST);
  };
  auto normalFunc = [&]() {
    BufferedInput input(readFile, *pool_, MetricsLog::voidLog(), stats.get());
    for (const auto& region : regions) {
      input.enqueue({region.offset, region.length});
    }
    input.load(LogType::TEST);
  };

  // Execute 4 times test in alternating order, compare ioStats.
  bool parallelFirst = false;
  for (int32_t iter = 0; iter < 4; iter++) {
    if (iter % 2 == 0) {
      parallelFirst = true;
    } else {
      parallelFirst = false;
    }
    if (parallelFirst) {
      parallelFunc();
      normalFunc();
    } else {
      normalFunc();
      parallelFunc();
    }
    ASSERT_EQ(stats->rawBytesRead(), parallelStats->rawBytesRead());
    ASSERT_GT(stats->totalScanTime(), parallelStats->totalScanTime());

    // Reset stats.
    stats = std::make_shared<IoStatistics>();
    parallelStats = std::make_shared<IoStatistics>();
  }
}
