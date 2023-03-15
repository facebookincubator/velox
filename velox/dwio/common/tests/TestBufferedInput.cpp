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
#include "velox/dwio/common/BufferedInput.h"

DECLARE_bool(load_parallel);

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

TEST(TestBufferedInput, ZeroLengthStream) {
  auto readFile =
      std::make_shared<facebook::velox::InMemoryReadFile>(std::string());
  auto pool = facebook::velox::memory::getDefaultMemoryPool();
  BufferedInput input(readFile, *pool);
  auto ret = input.enqueue({0, 0});
  EXPECT_NE(ret, nullptr);
  const void* buf = nullptr;
  int32_t size = 1;
  EXPECT_FALSE(ret->Next(&buf, &size));
  EXPECT_EQ(size, 0);
}

TEST(TestBufferedInput, ParallelLoad) {
  const int32_t kOneKB = 1 * 1024;
  const int32_t threads = 4;
  FLAGS_load_parallel = true;

  std::shared_ptr<IoStatistics> ioStats = std::make_shared<IoStatistics>();
  std::unique_ptr<folly::Executor> executor =
      std::make_unique<folly::IOThreadPoolExecutor>(threads);
  int32_t loadQuantum = kOneKB;

  std::string fileData;
  {
    InMemoryWriteFile writeFile(&fileData);
    writeFile.append(std::string(5, 'a'));
    writeFile.append(std::string(kOneKB, 'b'));
    writeFile.append(std::string(kOneKB, 'c'));
    writeFile.append(std::string(5, 'd'));
  }
  auto readFile = std::make_shared<facebook::velox::InMemoryReadFile>(fileData);
  auto pool = facebook::velox::memory::getDefaultMemoryPool();
  BufferedInput input(
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

    // verify data correctness.
    auto expect = readFile->pread(region.offset, region.length);
    ASSERT_EQ(buffer->size(), expect.size());
    ASSERT_EQ(buffer->data(), expect);
  }

  int32_t totalLength = 0;
  for (const auto& region : regions) {
    totalLength += region.length;
  }
  ASSERT_EQ(totalLength, ioStats->rawBytesRead());

  // merge distance is 1, should not have overread bytes.
  ASSERT_EQ(0, ioStats->rawOverreadBytes());
}