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

#include <filesystem>
#include <fstream>
#include <limits>

#include <fmt/format.h>
#include <folly/hash/Checksum.h>
#include <folly/testing/TestUtil.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/Memory.h"
#include "velox/exec/FileExchangeFormat.h"
#include "velox/exec/FileExchangeSink.h"
#include "velox/exec/FileExchangeSource.h"

namespace facebook::velox::exec {
namespace {

class FileExchangeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
    filesystems::registerLocalFileSystem();
    tests::utils::registerFaultyFileSystem();
  }

  void SetUp() override {
    temporaryDirectory_ =
        std::make_unique<folly::test::TemporaryDirectory>("file_exchange_test");
    rootDirectory_ = temporaryDirectory_->path().string();
    pool_ = memory::memoryManager()->addRootPool("file_exchange_test");
  }

  static std::string pageData(const SerializedPageBase& page) {
    auto data = page.getIOBuf();
    data->coalesce();
    return std::string(
        reinterpret_cast<const char*>(data->data()), data->length());
  }

  static std::string fileContents(
      std::initializer_list<std::string_view> pages) {
    std::string contents;
    for (const auto page : pages) {
      const auto encodedPageSize = file_exchange::encodePageSize(page.size());
      contents.append(
          reinterpret_cast<const char*>(&encodedPageSize),
          sizeof(file_exchange::PageSize));
      contents.append(page);
    }
    return contents;
  }

  static uint32_t checksum(std::string_view contents) {
    return folly::crc32c(
        reinterpret_cast<const uint8_t*>(contents.data()), contents.size());
  }

  static std::string fileLocation(
      std::string_view path,
      std::string_view contents) {
    const file_exchange::ExchangeOutputFile outputFile{
        .path = std::string{path},
        .size = contents.size(),
        .checksum = checksum(contents),
    };
    return outputFile.serialize();
  }

  std::unique_ptr<folly::test::TemporaryDirectory> temporaryDirectory_;
  std::string rootDirectory_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(FileExchangeTest, multiplePagesShareOnePartitionFile) {
  FileExchangeSink sink(rootDirectory_, "exchange", "task", 1);
  sink.append(0, "first page");
  sink.append(0, "second page");
  const auto output = sink.finish();

  const auto expectedPath =
      fmt::format("{}/exchange/partition_0/task.bin", rootDirectory_);
  const auto expectedContents = fileContents({"first page", "second page"});
  ASSERT_EQ(
      output.locations.at(0), fileLocation(expectedPath, expectedContents));
  ASSERT_TRUE(std::filesystem::exists(expectedPath));

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    queue->addSourceLocked();
  }
  queue->noMoreSources();

  FileExchangeSource source(output.locations.at(0), 0, queue, pool_.get());
  const auto response =
      source.request(std::numeric_limits<uint32_t>::max(), {}).get();
  EXPECT_TRUE(response.atEnd);

  bool atEnd = false;
  ContinueFuture future;
  auto stalePromise = ContinuePromise::makeEmpty();
  std::vector<std::unique_ptr<SerializedPageBase>> pages;
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    pages = queue->dequeueLocked(
        0,
        std::numeric_limits<uint32_t>::max(),
        &atEnd,
        &future,
        &stalePromise);
  }

  ASSERT_TRUE(atEnd);
  ASSERT_EQ(pages.size(), 2);
  EXPECT_EQ(pageData(*pages[0]), "first page");
  EXPECT_EQ(pageData(*pages[1]), "second page");
}

TEST_F(FileExchangeTest, exchangeOutputFileRoundTrip) {
  const file_exchange::ExchangeOutputFile expected{
      .path = R"(test://root/path?size=inside&note="quoted")",
      .size = 123,
      .checksum = 456,
  };

  const auto serializedOutput = expected.serialize();
  EXPECT_NE(serializedOutput.find(R"("checksum":456)"), std::string::npos);
  EXPECT_EQ(serializedOutput.find(R"("crc32c")"), std::string::npos);
  EXPECT_TRUE(file_exchange::ExchangeOutputFile::serialized(serializedOutput));
  EXPECT_FALSE(file_exchange::ExchangeOutputFile::serialized("file://path"));
  const auto actual =
      file_exchange::ExchangeOutputFile::deserialize(serializedOutput);
  EXPECT_EQ(actual.path, expected.path);
  EXPECT_EQ(actual.size, expected.size);
  EXPECT_EQ(actual.checksum, expected.checksum);
}

TEST_F(FileExchangeTest, pageSizeRoundTrip) {
  for (const uint64_t size : std::initializer_list<uint64_t>{
           0,
           1,
           1'024,
           std::numeric_limits<uint64_t>::max(),
       }) {
    EXPECT_EQ(
        file_exchange::decodePageSize(file_exchange::encodePageSize(size)),
        size);
  }
}

TEST_F(FileExchangeTest, usesRegisteredFileSystemForStoragePath) {
  const auto rootDirectory = fmt::format("faulty:{}", rootDirectory_);
  FileExchangeSink sink(rootDirectory, "exchange", "task", 1);
  sink.append(0, "page");
  const auto output = sink.finish();

  EXPECT_EQ(
      file_exchange::ExchangeOutputFile::deserialize(output.locations.at(0))
          .path,
      fmt::format("faulty:{}/exchange/partition_0/task.bin", rootDirectory_));
  EXPECT_TRUE(
      std::filesystem::exists(
          fmt::format("{}/exchange/partition_0/task.bin", rootDirectory_)));

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    queue->addSourceLocked();
  }
  queue->noMoreSources();

  FileExchangeSource source(output.locations.at(0), 0, queue, pool_.get());
  const auto response =
      source.request(std::numeric_limits<uint32_t>::max(), {}).get();
  EXPECT_TRUE(response.atEnd);

  bool atEnd = false;
  ContinueFuture future;
  auto stalePromise = ContinuePromise::makeEmpty();
  std::vector<std::unique_ptr<SerializedPageBase>> pages;
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    pages = queue->dequeueLocked(
        0,
        std::numeric_limits<uint32_t>::max(),
        &atEnd,
        &future,
        &stalePromise);
  }
  ASSERT_TRUE(atEnd);
  ASSERT_EQ(pages.size(), 1);
  EXPECT_EQ(pageData(*pages[0]), "page");
}

TEST_F(FileExchangeTest, chainedPageIsOnePage) {
  FileExchangeSink sink(rootDirectory_, "exchange", "task", 1);
  auto page = folly::IOBuf::copyBuffer("first ", 6);
  page->appendToChain(folly::IOBuf::copyBuffer("second", 6));
  sink.append(0, std::move(page));
  const auto output = sink.finish();

  const auto stats = sink.stats();
  EXPECT_EQ(stats.at("totalBytesWritten"), 12);
  EXPECT_EQ(stats.at("totalPages"), 1);

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    queue->addSourceLocked();
  }
  queue->noMoreSources();

  FileExchangeSource source(output.locations.at(0), 0, queue, pool_.get());
  const auto response =
      source.request(std::numeric_limits<uint32_t>::max(), {}).get();
  EXPECT_TRUE(response.atEnd);

  bool atEnd = false;
  ContinueFuture future;
  auto stalePromise = ContinuePromise::makeEmpty();
  std::vector<std::unique_ptr<SerializedPageBase>> pages;
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    pages = queue->dequeueLocked(
        0,
        std::numeric_limits<uint32_t>::max(),
        &atEnd,
        &future,
        &stalePromise);
  }

  ASSERT_TRUE(atEnd);
  ASSERT_EQ(pages.size(), 1);
  EXPECT_EQ(pageData(*pages[0]), "first second");
}

TEST_F(FileExchangeTest, closePreventsReads) {
  FileExchangeSink sink(rootDirectory_, "exchange", "task", 1);
  sink.append(0, "page");
  const auto output = sink.finish();

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  FileExchangeSource source(output.locations.at(0), 0, queue, pool_.get());

  source.close();

  const auto response =
      source.request(std::numeric_limits<uint32_t>::max(), {}).get();
  EXPECT_EQ(response.bytes, 0);
  EXPECT_FALSE(response.atEnd);
}

TEST_F(FileExchangeTest, emptyPartitionsHaveNoCommittedLocation) {
  FileExchangeSink sink(rootDirectory_, "exchange", "task", 2);
  sink.append(1, "page");
  const auto output = sink.finish();

  EXPECT_EQ(output.locations.size(), 1);
  EXPECT_EQ(output.locations.count(0), 0);
  EXPECT_EQ(output.locations.count(1), 1);
}

TEST_F(FileExchangeTest, missingFileIsRejected) {
  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  const file_exchange::ExchangeOutputFile outputFile{
      .path = fmt::format("{}/missing.bin", rootDirectory_),
      .size = 0,
      .checksum = 0,
  };
  FileExchangeSource source(outputFile.serialize(), 0, queue, pool_.get());

  EXPECT_THROW(
      source.request(std::numeric_limits<uint32_t>::max(), {}),
      VeloxRuntimeError);
}

TEST_F(FileExchangeTest, corruptedFileIsRejected) {
  FileExchangeSink sink(rootDirectory_, "exchange", "task", 1);
  sink.append(0, "page");
  const auto output = sink.finish();

  const auto filePath =
      fmt::format("{}/exchange/partition_0/task.bin", rootDirectory_);
  std::fstream file(filePath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(sizeof(file_exchange::PageSize));
  file.put('P');
  file.close();

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    queue->addSourceLocked();
  }
  queue->noMoreSources();

  FileExchangeSource source(output.locations.at(0), 0, queue, pool_.get());
  EXPECT_THROW(
      source.request(std::numeric_limits<uint32_t>::max(), {}),
      VeloxRuntimeError);
}

TEST_F(FileExchangeTest, incorrectFileSizeIsRejected) {
  FileExchangeSink sink(rootDirectory_, "exchange", "task", 1);
  sink.append(0, "page");
  sink.finish();

  const auto filePath =
      fmt::format("{}/exchange/partition_0/task.bin", rootDirectory_);
  const auto contents = fileContents({"page"});
  const file_exchange::ExchangeOutputFile outputFile{
      .path = filePath,
      .size = contents.size() + 1,
      .checksum = checksum(contents),
  };
  const auto location = outputFile.serialize();

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    queue->addSourceLocked();
  }
  queue->noMoreSources();

  FileExchangeSource source(location, 0, queue, pool_.get());
  EXPECT_THROW(
      source.request(std::numeric_limits<uint32_t>::max(), {}),
      VeloxRuntimeError);
}

TEST_F(FileExchangeTest, truncatedPageIsRejected) {
  const auto filePath = fmt::format("{}/truncated.bin", rootDirectory_);
  std::ofstream file(filePath, std::ios::binary);
  const auto encodedPageSize = file_exchange::encodePageSize(10);
  file.write(
      reinterpret_cast<const char*>(&encodedPageSize),
      sizeof(file_exchange::PageSize));
  file.write("short", 5);
  file.close();

  auto queue = std::make_shared<ExchangeQueue>(1, 0);
  {
    std::lock_guard<std::mutex> lock(queue->mutex());
    queue->addSourceLocked();
  }
  queue->noMoreSources();

  FileExchangeSource source(
      fileLocation(filePath, fileContents({"short"})), 0, queue, pool_.get());
  EXPECT_THROW(
      source.request(std::numeric_limits<uint32_t>::max(), {}),
      VeloxRuntimeError);
}

} // namespace
} // namespace facebook::velox::exec
