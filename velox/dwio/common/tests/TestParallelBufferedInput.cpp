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

TEST(TestParallelBufferedInput, parallelLoad) {
  const int32_t kOneKB = 1 * 1024;
  const int32_t threads = 4;

  auto ioStats = std::make_shared<IoStatistics>();
  auto executor = std::make_unique<folly::IOThreadPoolExecutor>(threads);
  int32_t loadQuantum = kOneKB;

  std::string fileData;
  {
    InMemoryWriteFile writeFile(&fileData);
    writeFile.append(std::string(5, 'a'));
    writeFile.append(std::string(kOneKB, 'b'));
    writeFile.append(std::string(kOneKB, 'c'));
    writeFile.append(std::string(5, 'd'));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(fileData);
  auto pool = memory::getDefaultMemoryPool();
  ParallelBufferedInput input(
      readFile,
      *pool,
      MetricsLog::voidLog(),
      ioStats,
      executor.get(),
      loadQuantum,
      1);

  std::vector<Region> regions = {
      {0, kOneKB},
      {kOneKB, 10},
      {kOneKB + 20, readFile->size() - (kOneKB + 20)}};
  std::vector<std::unique_ptr<SeekableInputStream>> streams;
  for (auto& region : regions) {
    streams.push_back(input.enqueue(region));
  }

  input.load(LogType::TEST);

  for (size_t i = 0; i < regions.size(); i++) {
    const auto& region = regions[i];
    const auto& stream = streams[i];

    auto buffer =
        std::make_shared<dwio::common::DataBuffer<char>>(*pool, region.length);
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

  // Merge distance is 1, should not have overread bytes.
  ASSERT_EQ(0, ioStats->rawOverreadBytes());
}

class TestReadFile : public ReadFile {
 public:
  explicit TestReadFile(std::string_view file) : file_(file) {}
  std::string_view pread(uint64_t offset, uint64_t length, void* buf)
      const override {
    // Assume read latency is 1 millisecond per MB.
    int64_t latency = length >> 20;
    std::this_thread::sleep_for(std::chrono::milliseconds(latency));
    return {static_cast<char*>(buf), length};
  }
  bool shouldCoalesce() const override {
    return false;
  }
  uint64_t size() const override {
    return 1 << 30;
  }
  uint64_t memoryUsage() const override {
    return 0;
  }
  std::string getName() const override {
    return "dummy";
  }
  uint64_t getNaturalReadSize() const override {
    return 1 << 20;
  }

 private:
  std::string_view file_;
};

TEST(TestParallelBufferedInput, simpleBenchmark) {
  std::string filename = "test-file";
  const int32_t kOneMB = 1 * 1024 * 1024;
  const int32_t threads = 4;
  int32_t loadQuantum = 8 * kOneMB;
  auto executor = std::make_unique<folly::IOThreadPoolExecutor>(threads);
  auto pool = memory::getDefaultMemoryPool();
  auto readFile = std::make_shared<TestReadFile>(filename);

  std::vector<Region> regions;

  // 10 regions, each has 2MB length and 1MB distance, eg [0, 2MB], [3MB,
  // 2MB]...
  for (uint64_t iter = 0, cursor = 0, length = 2 * kOneMB; iter < 10;
       ++iter, cursor += 3 * kOneMB) {
    regions.emplace_back(cursor, length);
  }
  // 20 regions, each has 14MB length and 1MB distance, eg [0, 14MB], [15MB,
  // 14MB]...
  for (uint64_t iter = 0, cursor = 0, length = 14 * kOneMB; iter < 20;
       ++iter, cursor += 15 * kOneMB) {
    regions.emplace_back(cursor, length);
  }

  auto stats = std::make_shared<IoStatistics>();
  auto parallelStats = std::make_shared<IoStatistics>();
  auto parallelFunc = [&]() {
    ParallelBufferedInput parallelInput(
        readFile,
        *pool,
        MetricsLog::voidLog(),
        parallelStats,
        executor.get(),
        loadQuantum,
        1);
    for (const auto& region : regions) {
      parallelInput.enqueue({region.offset, region.length});
    }
    parallelInput.load(LogType::TEST);
  };
  auto normalFunc = [&]() {
    BufferedInput input(readFile, *pool, MetricsLog::voidLog(), stats.get(), 1);
    for (const auto& region : regions) {
      input.enqueue({region.offset, region.length});
    }
    input.load(LogType::TEST);
  };

  // Execute 4 times test in alternating order, compare ioStats.
  bool parallelFist = false;
  for (int32_t iter = 0; iter < 4; iter++) {
    if (iter % 2 == 0) {
      parallelFist = true;
    } else {
      parallelFist = false;
    }
    if (parallelFist) {
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
    ASSERT_EQ(stats->rawBytesRead(), 0);
    ASSERT_EQ(stats->totalScanTime(), 0);
    ASSERT_EQ(parallelStats->rawBytesRead(), 0);
    ASSERT_EQ(parallelStats->totalScanTime(), 0);
  }
}